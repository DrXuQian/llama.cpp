#ifdef GGML_NCP_MOE
#    include "mxfp4-ncp.cuh"
#    include "mxfp4-ncp-layout.h"
#    include "common.cuh"
#    include "ncp-lib.h"
#    include "ggml-backend-impl.h"

#    include <algorithm>
#    include <map>
#    include <mutex>
#    include <vector>

namespace {
constexpr size_t copy_limit = 8 * 1024 * 1024;

bool valid_weight(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_MXFP4 && t->ne[3] == 1 && ggml_is_contiguous(t) &&
           ggml_ncp_mxfp4::shape(t->ne[1], t->ne[0], t->ne[2]);
}

struct type_context {
    int                      device;
    std::string              name;
    ggml_backend_buffer_type type;
};

struct buffer_context {
    int                   device;
    ggml_backend_buffer_t backing;
};

// Bulk uploads own complete words. Only unaligned byte-range updates need CAS.
static __global__ void pack_tiles(const uint8_t * raw,
                                  uint32_t *      weights,
                                  uint8_t *       scales,
                                  int             n,
                                  int             k,
                                  uint64_t        first_tile,
                                  uint64_t        tiles) {
    const uint64_t i     = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t words = uint64_t(64) * k / 8, units = uint64_t(64) * k / 32;
    if (i < tiles * words) {
        const uint64_t local = i % words, tile = first_tile + i / words;
        const uint64_t expert = tile / (n / 64), row = (tile % (n / 64)) * 64;
        weights[(expert * (k / 16) + local / 128) * (2 * n) + 2 * row + local % 128] =
            ggml_ncp_mxfp4::tile_word(raw + (i / words) * units * 17, k, local);
    }
    if (i < tiles * units) {
        const uint64_t local = i % units, tile = first_tile + i / units;
        const unsigned row = local / (k / 32), kb = local % (k / 32);
        scales[ggml_ncp_mxfp4::scale_index(n, k, tile / (n / 64), (tile % (n / 64)) * 64 + row, kb)] =
            raw[(i / units) * units * 17 + local * 17];
    }
}

static __device__ void store_code(uint32_t * words, uint64_t index, unsigned code) {
    auto *         target = words + index / 8;
    const unsigned shift = 4 * (index % 8), mask = 15u << shift;
    uint32_t       old = atomicCAS(target, 0, 0), assumed;
    do {
        assumed = old;
        old     = atomicCAS(target, assumed, (assumed & ~mask) | (code << shift));
    } while (old != assumed);
}

static __global__ void
pack_bytes(const uint8_t * raw, uint8_t * packed, int n, int k, int experts, uint64_t offset, uint64_t bytes) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= bytes) {
        return;
    }
    const uint64_t block = (offset + i) / 17;
    const unsigned byte = (offset + i) % 17, kb = block % (k / 32), row = (block / (k / 32)) % n;
    const uint64_t expert = block / (uint64_t(n) * (k / 32));
    if (byte == 0) {
        packed[uint64_t(experts) * n * k / 2 + ggml_ncp_mxfp4::scale_index(n, k, expert, row, kb)] = raw[i];
    } else {
        auto * words = reinterpret_cast<uint32_t *>(packed);
        store_code(words, ggml_ncp_mxfp4::code_index(n, k, expert, row, kb * 32 + byte - 1), raw[i] & 15);
        store_code(words, ggml_ncp_mxfp4::code_index(n, k, expert, row, kb * 32 + byte - 1 + 16), raw[i] >> 4);
    }
}

static __global__ void
unpack_bytes(const uint8_t * packed, uint8_t * raw, int n, int k, int experts, uint64_t offset, uint64_t bytes) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) {
        raw[i] = ggml_ncp_mxfp4::raw_byte(packed, n, k, experts, offset + i);
    }
}

const char * type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<type_context *>(buft->context)->name.c_str();
}

void free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    ggml_cuda_set_device(ctx->device);
    ggml_backend_buffer_free(ctx->backing);
    delete ctx;
}

void * get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(static_cast<buffer_context *>(buffer->context)->backing);
}

ggml_status init_tensor(ggml_backend_buffer_t, ggml_tensor * tensor) {
    return valid_weight(tensor) && (!tensor->view_src || ggml_cuda_ncp_mxfp4_alias(tensor)) ? GGML_STATUS_SUCCESS :
                                                                                              GGML_STATUS_FAILED;
}

void transfer(ggml_backend_buffer_t buffer,
              const ggml_tensor *   tensor,
              void *                host,
              size_t                offset,
              size_t                size,
              bool                  upload) {
    GGML_ASSERT(valid_weight(tensor) && (!tensor->view_src || ggml_cuda_ncp_mxfp4_alias(tensor)));
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    if (!size) {
        return;
    }
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    ggml_cuda_set_device(ctx->device);
    const int    n = tensor->ne[1], k = tensor->ne[0], experts = tensor->ne[2];
    const size_t tile_bytes = size_t(64) * (k / 32) * 17;
    const size_t capacity   = std::min(copy_limit, size);
    uint8_t *    temporary  = nullptr;
    CUDA_CHECK(cudaMalloc(&temporary, capacity));
    auto * packed     = static_cast<uint8_t *>(tensor->data);
    auto * host_bytes = static_cast<uint8_t *>(host);
    while (size) {
        size_t     count = std::min(capacity, size);
        const bool bulk  = upload && offset % tile_bytes == 0 && count >= tile_bytes;
        if (bulk) {
            count -= count % tile_bytes;
        } else if (upload && offset % tile_bytes) {
            count = std::min(count, tile_bytes - offset % tile_bytes);
        }
        if (upload) {
            CUDA_CHECK(cudaMemcpyAsync(temporary, host_bytes, count, cudaMemcpyHostToDevice, cudaStreamPerThread));
            if (bulk) {
                const size_t tiles = count / tile_bytes, words = tiles * 64 * k / 8;
                pack_tiles<<<(words + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    temporary, reinterpret_cast<uint32_t *>(packed), packed + uint64_t(experts) * n * k / 2, n, k,
                    offset / tile_bytes, tiles);
            } else {
                pack_bytes<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(temporary, packed, n, k, experts,
                                                                                 offset, count);
            }
        } else {
            unpack_bytes<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(packed, temporary, n, k, experts, offset,
                                                                               count);
        }
        CUDA_CHECK(cudaGetLastError());
        if (!upload) {
            CUDA_CHECK(cudaMemcpyAsync(host_bytes, temporary, count, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        offset += count;
        host_bytes += count;
        size -= count;
    }
    CUDA_CHECK(cudaFree(temporary));
}

void set_tensor(ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    transfer(b, t, const_cast<void *>(data), offset, size, true);
}

void get_tensor(ggml_backend_buffer_t b, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    transfer(b, t, data, offset, size, false);
}

void memset_tensor(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    std::vector<uint8_t> raw(std::min(copy_limit, size), value);
    while (size) {
        const size_t count = std::min(raw.size(), size);
        set_tensor(b, t, raw.data(), offset, count);
        offset += count;
        size -= count;
    }
}

void set_tensor_2d(ggml_backend_buffer_t b,
                   ggml_tensor *         t,
                   const void *          data,
                   size_t                offset,
                   size_t                size,
                   size_t                copies,
                   size_t                stride_tensor,
                   size_t                stride_data) {
    if (!size || !copies) {
        return;
    }
    if (stride_tensor == size) {
        // K shards arrive as strided GGUF rows. Batch their upload without a second resident weight copy.
        const size_t         rows = std::min(copies, std::max(size_t(1), copy_limit / size));
        std::vector<uint8_t> raw(rows * size);
        for (size_t first = 0; first < copies; first += rows) {
            const size_t count = std::min(rows, copies - first);
            for (size_t j = 0; j < count; ++j) {
                memcpy(raw.data() + j * size, static_cast<const uint8_t *>(data) + (first + j) * stride_data, size);
            }
            set_tensor(b, t, raw.data(), offset + first * size, count * size);
        }
        return;
    }
    for (size_t i = 0; i < copies; ++i) {
        set_tensor(b, t, static_cast<const uint8_t *>(data) + i * stride_data, offset + i * stride_tensor, size);
    }
}

void get_tensor_2d(ggml_backend_buffer_t b,
                   const ggml_tensor *   t,
                   void *                data,
                   size_t                offset,
                   size_t                size,
                   size_t                copies,
                   size_t                stride_tensor,
                   size_t                stride_data) {
    for (size_t i = 0; i < copies; ++i) {
        get_tensor(b, t, static_cast<uint8_t *>(data) + i * stride_data, offset + i * stride_tensor, size);
    }
}

void clear_buffer(ggml_backend_buffer_t b, uint8_t value) {
    // Buffer clear initializes storage, not a sequence of GGUF tensor writes.
    ggml_backend_buffer_clear(static_cast<buffer_context *>(b->context)->backing, value);
}

ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * type    = static_cast<type_context *>(buft->context);
    auto   backing = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(type->device), size);
    if (!backing) {
        return nullptr;
    }
    static const ggml_backend_buffer_i iface = {free_buffer, get_base,     init_tensor,   memset_tensor,
                                                set_tensor,  get_tensor,   set_tensor_2d, get_tensor_2d,
                                                nullptr,     clear_buffer, nullptr};
    return ggml_backend_buffer_init(buft, iface, new buffer_context{type->device, backing}, size);
}

size_t alignment(ggml_backend_buffer_type_t) {
    return 256;
}

size_t alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
}
}  // namespace

bool ggml_cuda_buft_is_ncp_mxfp4(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == type_name;
}

bool ggml_cuda_ncp_mxfp4_alias(const ggml_tensor * tensor) {
    if (!tensor || !tensor->view_src) {
        return false;
    }
    while (tensor->view_src) {
        switch (tensor->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                return false;
        }
        const auto * source = tensor->view_src;
        if (tensor->view_offs || !ggml_are_same_layout(tensor, source) ||
            (tensor->buffer && tensor->buffer != source->buffer) || (tensor->data && tensor->data != source->data)) {
            return false;
        }
        tensor = source;
    }
    return tensor->buffer && ggml_cuda_buft_is_ncp_mxfp4(tensor->buffer->buft) && valid_weight(tensor);
}

bool ggml_cuda_ncp_mxfp4_tensor(const ggml_tensor * tensor) {
    return tensor && (tensor->view_src ?
                          ggml_cuda_ncp_mxfp4_alias(tensor) :
                          tensor->buffer && ggml_cuda_buft_is_ncp_mxfp4(tensor->buffer->buft) && valid_weight(tensor));
}

ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_weight_buft(ggml_backend_dev_t  dev,
                                                           const ggml_tensor * weight,
                                                           ggml_op             op) {
    if (op != GGML_OP_MUL_MAT_ID || !valid_weight(weight) || weight->view_src || !ggml_ncp_lib_moe_w4fa16_available()) {
        return nullptr;
    }
    return ggml_cuda_ncp_mxfp4_buffer_type(dev);
}

ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_buffer_type(ggml_backend_dev_t dev) {
    auto * reg    = ggml_backend_dev_backend_reg(dev);
    int    device = -1;
    for (size_t i = 0; reg && i < ggml_backend_reg_dev_count(reg); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            device = i;
            break;
        }
    }
    if (device < 0) {
        return nullptr;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc != GGML_CUDA_CC_AMPERE && cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return nullptr;
    }
    static std::mutex                                 mutex;
    static std::map<ggml_backend_dev_t, type_context> types;
    std::lock_guard<std::mutex>                       lock(mutex);
    auto                                              found = types.find(dev);
    if (found == types.end()) {
        auto & ctx = types[dev];
        ctx.device = device;
        ctx.name   = std::string(ggml_backend_dev_name(dev)) + "_MXFP4";
        ctx.type   = {{type_name, alloc_buffer, alignment, nullptr, alloc_size, nullptr}, dev, &ctx};
        found      = types.find(dev);
    }
    return &found->second.type;
}
#endif  // GGML_NCP_MOE
