#ifdef GGML_NCP_QUACTLIZE
#    include "quactlize-buft.cuh"
#    include "common.cuh"
#    include "ggml-backend-impl.h"

#    include <climits>
#    include <cstring>
#    include <dlfcn.h>
#    include <map>

namespace {

struct pack_library {
    void *                                                                  handle      = nullptr;
    decltype(&quactlize_ppu_kpack_canonical_arrangement_v1)                 arrangement = nullptr;
    decltype(&quactlize_ppu_kpack_sizes_for_arrangement_v1)                 sizes       = nullptr;
    decltype(&quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2) pack        = nullptr;

    pack_library() {
        const char * path = getenv("QUACTLIZE_PPU_PACK_LIBRARY");
        handle            = dlopen(path && *path ? path : "libquactlize_ppu_pack.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            GGML_LOG_WARN("quactlize: packer unavailable: %s\n", dlerror());
            return;
        }
        arrangement =
            reinterpret_cast<decltype(arrangement)>(dlsym(handle, "quactlize_ppu_kpack_canonical_arrangement_v1"));
        sizes = reinterpret_cast<decltype(sizes)>(dlsym(handle, "quactlize_ppu_kpack_sizes_for_arrangement_v1"));
        pack  = reinterpret_cast<decltype(pack)>(
            dlsym(handle, "quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2"));
        if (!arrangement || !sizes || !pack) {
            GGML_LOG_WARN("quactlize: packer has an incomplete ABI\n");
            dlclose(handle);
            handle = nullptr;
        }
    }
};

const pack_library & library() {
    static const pack_library lib;
    return lib;
}

struct tensor_state {
    ggml_quactlize_packed_layout layout{};
    bool                         loaded = false;
};

struct buffer_context {
    int                                         device;
    void *                                      data     = nullptr;
    cudaStream_t                                stream   = nullptr;
    cudaEvent_t                                 uploaded = nullptr;
    void *                                      scratch  = nullptr;
    size_t                                      capacity = 0;
    const ggml_tensor *                         pending  = nullptr;
    size_t                                      received = 0;
    std::map<const ggml_tensor *, tensor_state> tensors;
};

struct buft_context {
    int                        device;
    std::string                name;
    ggml_backend_buffer_type   type{};
    ggml_backend_buffer_type_t extra[2]{};
};

const char * buffer_name(ggml_backend_buffer_type_t buft) {
    return static_cast<buft_context *>(buft->context)->name.c_str();
}

const ggml_tensor * packed_source(const ggml_tensor * tensor) {
    while (tensor) {
        if (tensor->buffer && ggml_cuda_buft_is_quactlize(tensor->buffer->buft)) {
            return tensor;
        }
        tensor = tensor->view_src;
    }
    return nullptr;
}

void free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaEventDestroy(ctx->uploaded));
    CUDA_CHECK(cudaStreamDestroy(ctx->stream));
    CUDA_CHECK(cudaFree(ctx->scratch));
    CUDA_CHECK(cudaFree(ctx->data));
    delete ctx;
}

void * get_base(ggml_backend_buffer_t buffer) {
    return static_cast<buffer_context *>(buffer->context)->data;
}

ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    tensor_state state;
    if (!ggml_quactlize_can_load(tensor, &state.layout)) {
        GGML_LOG_ERROR("quactlize: unsupported load: %s (%s)\n", tensor->name, ggml_type_name(tensor->type));
        return GGML_STATUS_FAILED;
    }
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    ctx->tensors.emplace(tensor, state);
    return GGML_STATUS_SUCCESS;
}

void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx   = static_cast<buffer_context *>(buffer->context);
    auto   found = ctx->tensors.find(tensor);
    GGML_ASSERT(found != ctx->tensors.end() && !found->second.loaded);
    GGML_ASSERT(data && size && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    // Ordinary loaders may split a tensor into ordered chunks. Interleaved or strided uploads are not supported.
    GGML_ASSERT((!ctx->pending && offset == 0) || (ctx->pending == tensor && offset == ctx->received));
    ggml_cuda_set_device(ctx->device);
    const auto & layout = found->second.layout;
    if (!ctx->pending) {
        if (ctx->capacity < layout.sizes.raw_bytes) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaFree(ctx->scratch));
            CUDA_CHECK(cudaMalloc(&ctx->scratch, layout.sizes.raw_bytes));
            ctx->capacity = layout.sizes.raw_bytes;
        }
        ctx->pending  = tensor;
        ctx->received = 0;
    }
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(ctx->scratch) + offset, data, size, cudaMemcpyHostToDevice,
                               ctx->stream));
    CUDA_CHECK(cudaEventRecord(ctx->uploaded, ctx->stream));
    ctx->received += size;
    if (ctx->received == layout.sizes.raw_bytes) {
        auto *    dst = static_cast<uint8_t *>(tensor->data);
        const int rc =
            library().pack(static_cast<const uint8_t *>(ctx->scratch), dst,
                           layout.sizes.high_bytes ? dst + layout.sizes.low_bytes : nullptr,
                           dst + layout.sizes.low_bytes + layout.sizes.high_bytes, tensor->ne[1], tensor->ne[0],
                           tensor->ne[2] * tensor->ne[3], tensor->type, &layout.arrangement, ctx->stream);
        if (rc) {
            GGML_ABORT("quactlize: GPU shuffle failed for %s (rc=%d)", tensor->name, rc);
        }
        found->second.loaded = true;
        ctx->pending         = nullptr;
        ctx->received        = 0;
    }
    // The caller may release its host bytes on return. The final GPU pack need not finish yet.
    CUDA_CHECK(cudaEventSynchronize(ctx->uploaded));
}

void clear_buffer(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    GGML_ASSERT(!ctx->pending);
    for (const auto & item : ctx->tensors) {
        GGML_ASSERT(!item.second.loaded && "cannot clear packed weights");
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->data, value, buffer->size, ctx->stream));
}

const ggml_backend_buffer_i buffer_interface = {
    free_buffer, get_base, init_tensor, nullptr, set_tensor, nullptr, nullptr, nullptr, nullptr, clear_buffer, nullptr,
};

ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const auto * type = static_cast<buft_context *>(buft->context);
    ggml_cuda_set_device(type->device);
    void * data = nullptr;
    if (cudaMalloc(&data, size) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    auto * ctx  = new buffer_context{};
    ctx->device = type->device;
    ctx->data   = data;
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ctx->uploaded, cudaEventDisableTiming));
    return ggml_backend_buffer_init(buft, buffer_interface, ctx, size);
}

size_t alignment(ggml_backend_buffer_type_t) {
    return 128;
}

size_t alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
}

}  // namespace

bool ggml_quactlize_can_load(const ggml_tensor * tensor, ggml_quactlize_packed_layout * out) {
    if (!tensor || tensor->view_src || !ggml_is_contiguous(tensor) || !library().handle) {
        return false;
    }
    if (tensor->type != GGML_TYPE_Q8_0 && (tensor->type < GGML_TYPE_Q2_K || tensor->type > GGML_TYPE_Q6_K)) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0 || tensor->ne[d] > INT_MAX) {
            return false;
        }
    }
    if (tensor->ne[2] > INT_MAX / tensor->ne[3]) {
        return false;
    }
    ggml_quactlize_packed_layout layout{};
    auto &                       a  = layout.arrangement;
    const uint64_t expected_mapping = tensor->type == GGML_TYPE_Q4_K ? QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID :
                                      tensor->type == GGML_TYPE_Q8_0 ? QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID :
                                                                       QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID;
    if (library().arrangement(tensor->type, &a) || a.version != 2 || a.artifact_tile_k || a.reserved ||
        a.mapping_id != expected_mapping ||
        library().sizes(tensor->ne[1], tensor->ne[0], tensor->ne[2] * tensor->ne[3], tensor->type, &a, &layout.sizes)) {
        return false;
    }
    const auto & s = layout.sizes;
    if (s.raw_bytes != ggml_nbytes(tensor) || !s.low_bytes || !s.units_bytes || s.low_bytes > s.raw_bytes ||
        s.high_bytes > s.raw_bytes - s.low_bytes || s.units_bytes != s.raw_bytes - s.low_bytes - s.high_bytes) {
        return false;
    }
    if (out) {
        *out = layout;
    }
    return true;
}

bool ggml_cuda_buft_is_quactlize(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == buffer_name;
}

ggml_backend_buffer_type_t * ggml_cuda_quactlize_extra_bufts(ggml_backend_dev_t dev) {
    if (!library().handle) {
        return nullptr;
    }
    static std::mutex                                 mutex;
    static std::map<ggml_backend_dev_t, buft_context> types;
    std::lock_guard<std::mutex>                       lock(mutex);
    auto                                              found = types.find(dev);
    if (found == types.end()) {
        int    index = -1;
        auto * reg   = ggml_backend_dev_backend_reg(dev);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            if (ggml_backend_reg_dev_get(reg, i) == dev) {
                index = static_cast<int>(i);
                break;
            }
        }
        GGML_ASSERT(index >= 0);
        auto & ctx   = types[dev];
        ctx.device   = index;
        ctx.name     = std::string(ggml_backend_dev_name(dev)) + "_KPACK";
        ctx.type     = {{buffer_name, alloc_buffer, alignment, nullptr, alloc_size, nullptr}, dev, &ctx};
        ctx.extra[0] = &ctx.type;
        found        = types.find(dev);
    }
    return found->second.extra;
}

bool ggml_quactlize_read_packed(const ggml_tensor * tensor, void * data, size_t size) {
    if (!tensor || tensor->view_src || !packed_source(tensor) || !data || size != ggml_nbytes(tensor)) {
        return false;
    }
    auto * ctx   = static_cast<buffer_context *>(tensor->buffer->context);
    auto   found = ctx->tensors.find(tensor);
    if (found == ctx->tensors.end() || !found->second.loaded) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, tensor->data, size, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return true;
}

bool ggml_quactlize_set_packed(ggml_tensor *                        tensor,
                               const ggml_quactlize_packed_layout * layout,
                               const void *                         data,
                               size_t                               size) {
    if (!tensor || tensor->view_src || !packed_source(tensor) || !layout || !data || size != ggml_nbytes(tensor)) {
        return false;
    }
    auto * ctx   = static_cast<buffer_context *>(tensor->buffer->context);
    auto   found = ctx->tensors.find(tensor);
    if (ctx->pending || found == ctx->tensors.end() || found->second.loaded ||
        memcmp(layout, &found->second.layout, sizeof(*layout))) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, data, size, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    found->second.loaded = true;
    return true;
}

void ggml_quactlize_assert_no_compute(const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (packed_source(node)) {
            GGML_ABORT("quactlize: K-pack is load-only; inference is not enabled (%s)", node->name);
        }
        for (const auto * src : node->src) {
            if (packed_source(src)) {
                GGML_ABORT("quactlize: K-pack is load-only; inference is not enabled (%s reads %s)", node->name,
                           src->name);
            }
        }
    }
}
#endif
