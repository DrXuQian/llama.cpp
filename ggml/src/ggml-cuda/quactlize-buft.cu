#ifdef GGML_NCP_QUACTLIZE
#    include "quactlize-buft.cuh"
#    include "common.cuh"
#    include "ggml-backend-impl.h"

#    include <atomic>
#    include <cinttypes>
#    include <climits>
#    include <cstring>
#    include <dlfcn.h>
#    include <map>
#    include <iterator>

namespace {

// Once a buffer type escapes, keep checking even after its buffers are freed.
std::atomic<bool> buffer_type_exposed{false};

// Validate loading and execution together before admitting packed weights.
struct quactlize_library {
    void *                                                                  handle      = nullptr;
    decltype(&quactlize_ppu_kpack_canonical_arrangement_v1)                 arrangement = nullptr;
    decltype(&quactlize_ppu_kpack_sizes_for_arrangement_v1)                 sizes       = nullptr;
    decltype(&quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2) pack        = nullptr;

    decltype(&quactlize_ppu_dense_supported_v1) dense_supported = nullptr;
    decltype(&quactlize_ppu_dense_create_v1)    dense_create    = nullptr;
    decltype(&quactlize_ppu_dense_prepare_v1)   dense_prepare   = nullptr;
    decltype(&quactlize_ppu_dense_run_v1)       dense_run       = nullptr;
    decltype(&quactlize_ppu_dense_destroy_v1)   dense_destroy   = nullptr;
    decltype(&quactlize_ppu_dense_error_v1)     dense_error     = nullptr;
    decltype(&quactlize_ppu_moe_matmul_supported_v1) moe_supported   = nullptr;
    decltype(&quactlize_ppu_moe_matmul_create_v1)    moe_create      = nullptr;
    decltype(&quactlize_ppu_moe_matmul_prepare_v1)   moe_prepare     = nullptr;
    decltype(&quactlize_ppu_moe_matmul_run_v1)       moe_run         = nullptr;
    decltype(&quactlize_ppu_moe_matmul_destroy_v1)   moe_destroy     = nullptr;
    decltype(&quactlize_ppu_moe_matmul_error_v1)     moe_error       = nullptr;

    quactlize_library() {
        const char * path = getenv("QUACTLIZE_PPU_LIBRARY");
        if (!path || !*path) {
            path = "libncp_quactlize.so";
        }
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            GGML_LOG_WARN("quactlize: library %s unavailable: %s\n", path, dlerror());
            return;
        }
#    define QZ_BIND(field, symbol)                                                     \
        field = reinterpret_cast<decltype(field)>(dlsym(handle, #symbol));             \
        if (!field) {                                                                  \
            GGML_LOG_WARN("quactlize: library %s missing symbol %s\n", path, #symbol); \
            dlclose(handle);                                                           \
            handle = nullptr;                                                          \
            return;                                                                    \
        }
        QZ_BIND(arrangement, quactlize_ppu_kpack_canonical_arrangement_v1);
        QZ_BIND(sizes, quactlize_ppu_kpack_sizes_for_arrangement_v1);
        QZ_BIND(pack, quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2);
        QZ_BIND(dense_supported, quactlize_ppu_dense_supported_v1);
        QZ_BIND(dense_create, quactlize_ppu_dense_create_v1);
        QZ_BIND(dense_prepare, quactlize_ppu_dense_prepare_v1);
        QZ_BIND(dense_run, quactlize_ppu_dense_run_v1);
        QZ_BIND(dense_destroy, quactlize_ppu_dense_destroy_v1);
        QZ_BIND(dense_error, quactlize_ppu_dense_error_v1);
        QZ_BIND(moe_supported, quactlize_ppu_moe_matmul_supported_v1);
        QZ_BIND(moe_create, quactlize_ppu_moe_matmul_create_v1);
        QZ_BIND(moe_prepare, quactlize_ppu_moe_matmul_prepare_v1);
        QZ_BIND(moe_run, quactlize_ppu_moe_matmul_run_v1);
        QZ_BIND(moe_destroy, quactlize_ppu_moe_matmul_destroy_v1);
        QZ_BIND(moe_error, quactlize_ppu_moe_matmul_error_v1);
#    undef QZ_BIND
        GGML_LOG_INFO("[quactlize] loaded %s\n", path);
    }
};

const quactlize_library & library() {
    static const quactlize_library lib;
    return lib;
}

struct dense_plan {
    void *                           handle = nullptr;
    quactlize_ppu_dense_plan_info_v1 info{};
};

using dense_key = std::array<int64_t, 3>;

struct moe_plan {
    void *                                handle = nullptr;
    quactlize_ppu_moe_matmul_plan_info_v1 info{};
};

using moe_key = std::array<int64_t, 7>;

moe_key key_for(const quactlize_ppu_moe_matmul_params_v1 & p) {
    return {p.tokens, p.topk, p.channels, p.input_row_stride, p.input_token_stride, p.ids_stride, p.output_stride};
}

const ggml_tensor * weight_owner(const ggml_tensor * tensor) {
    while (tensor && tensor->view_src) {
        switch (tensor->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                return nullptr;
        }
        const auto * source = tensor->view_src;
        if (tensor->view_offs != 0 || !ggml_are_same_layout(tensor, source) ||
            (tensor->buffer && tensor->buffer != source->buffer) || (tensor->data && tensor->data != source->data)) {
            return nullptr;
        }
        tensor = source;
    }
    return tensor;
}

bool dense_params(const ggml_tensor * op, quactlize_ppu_dense_params_v1 & params) {
    if (!op || op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const auto *                 weight = weight_owner(op->src[0]);
    const auto *                 input  = op->src[1];
    ggml_quactlize_packed_layout layout{};
    if (!weight || !input || weight->ne[2] != 1 || weight->ne[3] != 1 || input->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(input) || !ggml_quactlize_can_load(weight, &layout)) {
        return false;
    }
    // Meta leaves global widths on zero-row outputs; their packed weights are already local shards.
    const int64_t rows = ggml_nrows(input);
    if (rows < 0 || rows > INT_MAX || ggml_nrows(op) != rows ||
        (rows != 0 && (input->ne[0] != weight->ne[0] || op->ne[0] != weight->ne[1]))) {
        return false;
    }
    params = {1,
              sizeof(params),
              int32_t(weight->type),
              int32_t(weight->ne[1]),
              int32_t(weight->ne[0]),
              int32_t(rows),
              int64_t(input->nb[1] / sizeof(float)),
              int64_t(op->nb[1] / sizeof(float)),
              layout.arrangement,
              layout.sizes};
    return true;
}

bool moe_params(const ggml_tensor * op, quactlize_ppu_moe_matmul_params_v1 & params) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const auto *                 weight = weight_owner(op->src[0]);
    const auto *                 input  = op->src[1];
    const auto *                 ids    = op->src[2];
    ggml_quactlize_packed_layout layout{};
    if (!weight || !input || !ids || !ggml_quactlize_can_load(weight, &layout) || weight->ne[3] != 1 ||
        input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || input->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        (ids->ne[1] != 0 && (input->ne[0] != weight->ne[0] || op->ne[0] != weight->ne[1])) || op->ne[1] != ids->ne[0] ||
        op->ne[2] != ids->ne[1] || input->ne[2] != ids->ne[1] || op->ne[3] != 1 || input->ne[1] < 1 ||
        input->ne[1] > 32 || ids->ne[0] < 1 || ids->ne[0] > 32 || ids->ne[1] < 0 || ids->ne[1] > INT_MAX ||
        input->nb[1] % sizeof(float) || input->nb[2] % sizeof(float) || ids->nb[1] % sizeof(int32_t)) {
        return false;
    }
    params = {1,
              sizeof(params),
              int32_t(weight->type),
              int32_t(weight->ne[1]),
              int32_t(weight->ne[0]),
              int32_t(weight->ne[2]),
              int32_t(ids->ne[1]),
              int32_t(ids->ne[0]),
              int32_t(input->ne[1]),
              0,
              int64_t(input->nb[1] / sizeof(float)),
              int64_t(input->nb[2] / sizeof(float)),
              int64_t(ids->nb[1] / sizeof(int32_t)),
              int64_t(op->nb[1] / sizeof(float)),
              layout.arrangement,
              layout.sizes};
    if (params.tokens == 0) {
        // Meta can retain global strides on an empty local output.
        params.input_row_stride   = params.k;
        params.input_token_stride = int64_t(params.k) * params.channels;
        params.output_stride      = params.n;
    }
    return true;
}

struct tensor_state {
    ggml_quactlize_packed_layout    layout{};
    bool                            loaded        = false;
    cudaEvent_t                     ready         = nullptr;
    bool                            compute_ready = false;
    std::map<dense_key, dense_plan> plans;
    std::map<moe_key, moe_plan>     moe_plans;
};

constexpr size_t cache_chunk_bytes = 8 * 1024 * 1024;

struct upload_slot {
    void *      host     = nullptr;
    cudaEvent_t reusable = nullptr;
    bool        pending  = false;
};

struct buffer_context {
    int                                         device;
    void *                                      data        = nullptr;
    cudaStream_t                                stream      = nullptr;
    cudaStream_t                                copy_stream = nullptr;
    cudaEvent_t                                 uploaded    = nullptr;
    void *                                      scratch     = nullptr;
    size_t                                      capacity    = 0;
    const ggml_tensor *                         pending     = nullptr;
    std::map<size_t, size_t>                    received;
    upload_slot                                 uploads[2];
    unsigned                                    upload_index = 0;
    std::map<const ggml_tensor *, tensor_state> tensors;
    std::mutex                                  compute_mutex;
    void *                                      compute = nullptr;
    void *                                      moe_compute = nullptr;
};

void prepare_ready(ggml_backend_cuda_context & cuda_ctx, tensor_state & state) {
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(cuda_ctx.stream(), &capture));
    GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
    // Retire the one-time upload before capture; no event wait is inserted in replay.
    if (!state.compute_ready) {
        CUDA_CHECK(cudaEventSynchronize(state.ready));
        state.compute_ready = true;
    }
}

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
    CUDA_CHECK(cudaStreamSynchronize(ctx->copy_stream));
    if (ctx->compute || ctx->moe_compute) {
        // Compute plans reference these weights and may be captured on other streams.
        CUDA_CHECK(cudaDeviceSynchronize());
        if (ctx->compute) {
            library().dense_destroy(ctx->compute);
        }
        if (ctx->moe_compute) {
            library().moe_destroy(ctx->moe_compute);
        }
    }
    for (const auto & item : ctx->tensors) {
        if (item.second.ready) {
            CUDA_CHECK(cudaEventDestroy(item.second.ready));
        }
    }
    for (auto & slot : ctx->uploads) {
        if (slot.reusable) {
            CUDA_CHECK(cudaEventDestroy(slot.reusable));
        }
        if (slot.host) {
            CUDA_CHECK(cudaFreeHost(slot.host));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(ctx->copy_stream));
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
    auto * ctx = static_cast<buffer_context *>(buffer->context);
    if (tensor->view_src) {
        const auto * owner = weight_owner(tensor);
        if (owner && owner->buffer == buffer && ctx->tensors.count(owner)) {
            // The owner alone holds upload events and compute plans.
            return GGML_STATUS_SUCCESS;
        }
        GGML_LOG_ERROR("quactlize: unsupported weight view: %s\n", tensor->name);
        return GGML_STATUS_FAILED;
    }
    if (ggml_nelements(tensor) == 0) {
        return GGML_STATUS_SUCCESS;
    }
    tensor_state state;
    if (!ggml_quactlize_can_load(tensor, &state.layout)) {
        GGML_LOG_ERROR("quactlize: unsupported load: %s (%s)\n", tensor->name, ggml_type_name(tensor->type));
        return GGML_STATUS_FAILED;
    }
    ctx->tensors.emplace(tensor, state);
    return GGML_STATUS_SUCCESS;
}

void set_tensor_2d(ggml_backend_buffer_t buffer,
                   ggml_tensor *         tensor,
                   const void *          data,
                   size_t                offset,
                   size_t                width,
                   size_t                rows,
                   size_t                dst_pitch,
                   size_t                src_pitch) {
    auto * ctx   = static_cast<buffer_context *>(buffer->context);
    auto   found = ctx->tensors.find(tensor);
    GGML_ASSERT(found != ctx->tensors.end() && !found->second.loaded);
    const size_t total = ggml_nbytes(tensor);
    GGML_ASSERT(data && width && rows && src_pitch >= width && dst_pitch >= width);
    GGML_ASSERT(offset <= total && width <= total - offset && rows - 1 <= (total - offset - width) / dst_pitch);
    GGML_ASSERT(rows - 1 <= (SIZE_MAX - width) / src_pitch);
    GGML_ASSERT(!ctx->pending || ctx->pending == tensor);
    // Track local byte coverage; segments may arrive in a strided or nonsequential order.
    const size_t range_rows  = dst_pitch == width ? 1 : rows;
    const size_t range_width = dst_pitch == width ? width * rows : width;
    for (size_t row = 0; row < range_rows; ++row) {
        const size_t begin = offset + row * dst_pitch, end = begin + range_width;
        auto         next = ctx->received.lower_bound(begin);
        GGML_ASSERT((next == ctx->received.end() || next->first >= end) &&
                    (next == ctx->received.begin() || std::prev(next)->second <= begin));
    }
    for (size_t row = 0; row < range_rows; ++row) {
        size_t begin = offset + row * dst_pitch, end = begin + range_width;
        auto   next = ctx->received.lower_bound(begin);
        if (next != ctx->received.begin() && std::prev(next)->second == begin) {
            auto previous = std::prev(next);
            begin         = previous->first;
            ctx->received.erase(previous);
        }
        if (next != ctx->received.end() && next->first == end) {
            end = next->second;
            ctx->received.erase(next);
        }
        ctx->received.emplace(begin, end);
    }
    ggml_cuda_set_device(ctx->device);
    const auto & layout = found->second.layout;
    if (!ctx->pending) {
        if (ctx->capacity < layout.sizes.raw_bytes) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaFree(ctx->scratch));
            CUDA_CHECK(cudaMalloc(&ctx->scratch, layout.sizes.raw_bytes));
            ctx->capacity = layout.sizes.raw_bytes;
        }
        ctx->pending = tensor;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<uint8_t *>(ctx->scratch) + offset, dst_pitch, data, src_pitch, width, rows,
                                 cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaEventRecord(ctx->uploaded, ctx->stream));
    if (ctx->received.size() == 1 && ctx->received.begin()->first == 0 && ctx->received.begin()->second == total) {
        auto *    dst = static_cast<uint8_t *>(tensor->data);
        const int rc =
            library().pack(static_cast<const uint8_t *>(ctx->scratch), dst,
                           layout.sizes.high_bytes ? dst + layout.sizes.low_bytes : nullptr,
                           dst + layout.sizes.low_bytes + layout.sizes.high_bytes, tensor->ne[1], tensor->ne[0],
                           tensor->ne[2] * tensor->ne[3], tensor->type, &layout.arrangement, ctx->stream);
        if (rc) {
            GGML_ABORT("quactlize: GPU shuffle failed for %s (rc=%d)", tensor->name, rc);
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&found->second.ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(found->second.ready, ctx->stream));
        found->second.loaded = true;
        ctx->pending         = nullptr;
        ctx->received.clear();
    }
    // The caller may release its host bytes on return. The final GPU pack need not finish yet.
    CUDA_CHECK(cudaEventSynchronize(ctx->uploaded));
}

void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    set_tensor_2d(buffer, tensor, data, offset, size, 1, size, size);
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
    free_buffer,   get_base, init_tensor, nullptr,      set_tensor, nullptr,
    set_tensor_2d, nullptr,  nullptr,     clear_buffer, nullptr,
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
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->copy_stream, cudaStreamNonBlocking));
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

bool ggml_quactlize_is_alias(const ggml_tensor * tensor) {
    const auto * owner = weight_owner(tensor);
    return tensor && tensor->view_src && owner && owner->buffer && ggml_cuda_buft_is_quactlize(owner->buffer->buft);
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
    buffer_type_exposed.store(true, std::memory_order_release);
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
    if (!ctx->uploads[0].host) {
        for (auto & slot : ctx->uploads) {
            CUDA_CHECK(cudaMallocHost(&slot.host, cache_chunk_bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming));
        }
    }
    for (size_t offset = 0; offset < size;) {
        auto & slot = ctx->uploads[ctx->upload_index];
        if (slot.pending) {
            CUDA_CHECK(cudaEventSynchronize(slot.reusable));
        }
        const size_t bytes = std::min(cache_chunk_bytes, size - offset);
        memcpy(slot.host, static_cast<const uint8_t *>(data) + offset, bytes);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(tensor->data) + offset, slot.host, bytes,
                                   cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaEventRecord(slot.reusable, ctx->stream));
        slot.pending = true;
        ctx->upload_index ^= 1;
        offset += bytes;
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&found->second.ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(found->second.ready, ctx->stream));
    // Host input is consumed into owned slots; the last uploads may still be in flight.
    found->second.loaded = true;
    return true;
}

bool ggml_quactlize_tensor_is_kpack(const ggml_tensor * tensor) {
    return tensor && !tensor->view_src && packed_source(tensor);
}

bool ggml_quactlize_copy_range_async(const ggml_tensor *  tensor,
                                     void *               pinned,
                                     size_t               offset,
                                     size_t               bytes,
                                     ggml_backend_event_t completion) {
    if (!ggml_quactlize_tensor_is_kpack(tensor) || !pinned || !completion || !completion->context || !bytes ||
        offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset ||
        completion->device != ggml_backend_buft_get_device(tensor->buffer->buft)) {
        return false;
    }
    auto *     ctx   = static_cast<buffer_context *>(tensor->buffer->context);
    const auto found = ctx->tensors.find(tensor);
    if (found == ctx->tensors.end() || !found->second.loaded) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    cudaPointerAttributes attributes{};
    cudaError_t           rc = cudaPointerGetAttributes(&attributes, pinned);
    if (rc != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (rc != cudaSuccess) {
            (void) cudaGetLastError();
        }
        return false;
    }
    rc = cudaStreamWaitEvent(ctx->copy_stream, found->second.ready, 0);
    if (rc == cudaSuccess) {
        rc = cudaMemcpyAsync(pinned, static_cast<const uint8_t *>(tensor->data) + offset, bytes, cudaMemcpyDeviceToHost,
                             ctx->copy_stream);
    }
    if (rc == cudaSuccess) {
        rc = cudaEventRecord((cudaEvent_t) completion->context, ctx->copy_stream);
    }
    if (rc == cudaSuccess) {
        return true;
    }
    // A failed enqueue may already reference the destination. Drain this stream before returning it.
    CUDA_CHECK(cudaStreamSynchronize(ctx->copy_stream));
    (void) cudaGetLastError();
    return false;
}

bool ggml_quactlize_copy_range_wait(ggml_backend_event_t completion) {
    if (!completion || !completion->context) {
        return false;
    }
    const cudaError_t rc = cudaEventSynchronize((cudaEvent_t) completion->context);
    if (rc != cudaSuccess) {
        (void) cudaGetLastError();
    }
    return rc == cudaSuccess;
}

bool ggml_quactlize_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_QUACTLIZE");
        if (!value || !*value || strcmp(value, "1") == 0) {
            return true;
        }
        if (strcmp(value, "0") == 0) {
            return false;
        }
        GGML_ABORT("GGML_QUACTLIZE must be 0 or 1");
    }();
    return enabled;
}

ggml_backend_buffer_type_t ggml_cuda_quactlize_weight_buft(ggml_backend_dev_t  dev,
                                                           const ggml_tensor * weight,
                                                           ggml_op             op) {
    if (!ggml_quactlize_enabled() || (op != GGML_OP_MUL_MAT && op != GGML_OP_MUL_MAT_ID) || !weight ||
        (op == GGML_OP_MUL_MAT && weight->ne[2] != 1) || weight->ne[3] != 1 ||
        (weight->type != GGML_TYPE_Q8_0 && (weight->type < GGML_TYPE_Q2_K || weight->type > GGML_TYPE_Q6_K))) {
        return nullptr;
    }
    if (!library().handle) {
        GGML_ABORT("quactlize: library unavailable for automatic placement");
    }
    ggml_quactlize_packed_layout layout{};
    if (!ggml_quactlize_can_load(weight, &layout)) {
        return nullptr;
    }
    if (op == GGML_OP_MUL_MAT_ID) {
        const quactlize_ppu_moe_matmul_params_v1 params{1,
                                                        sizeof(params),
                                                        int32_t(weight->type),
                                                        int32_t(weight->ne[1]),
                                                        int32_t(weight->ne[0]),
                                                        int32_t(weight->ne[2]),
                                                        0,
                                                        1,
                                                        1,
                                                        0,
                                                        weight->ne[0],
                                                        weight->ne[0],
                                                        1,
                                                        weight->ne[1],
                                                        layout.arrangement,
                                                        layout.sizes};
        const int                                supported = library().moe_supported(&params);
        if (supported < 0 || supported > 1) {
            GGML_ABORT("quactlize: MoE weight admission %s: %s", weight->name, library().moe_error());
        }
        return supported ? ggml_cuda_quactlize_extra_bufts(dev)[0] : nullptr;
    }
    quactlize_ppu_dense_params_v1 params{
        1, sizeof(params), int32_t(weight->type), int32_t(weight->ne[1]), int32_t(weight->ne[0]),
        0, weight->ne[0],  weight->ne[1],         layout.arrangement,     layout.sizes};
    const auto & api       = library();
    const int    supported = api.dense_supported(&params);
    if (supported < 0 || supported > 1) {
        GGML_ABORT("quactlize: dense weight admission %s: %s", weight->name, api.dense_error());
    }
    return supported ? ggml_cuda_quactlize_extra_bufts(dev)[0] : nullptr;
}

bool ggml_quactlize_can_compute(const ggml_tensor * op) {
    if (op && op->op == GGML_OP_MUL_MAT_ID) {
        quactlize_ppu_moe_matmul_params_v1 params{};
        return ggml_quactlize_enabled() && moe_params(op, params) && library().moe_supported(&params) == 1;
    }
    quactlize_ppu_dense_params_v1 params{};
    return ggml_quactlize_enabled() && dense_params(op, params) && library().dense_supported(&params) == 1;
}

void ggml_quactlize_prepare_graph(ggml_backend_cuda_context & cuda_ctx, const ggml_cgraph * graph) {
    if (!buffer_type_exposed.load(std::memory_order_acquire)) {
        return;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (node->op == GGML_OP_NONE) {
            continue;
        }
        if (ggml_quactlize_is_alias(node)) {
            GGML_ASSERT(static_cast<buffer_context *>(weight_owner(node)->buffer->context)->device == cuda_ctx.device);
            continue;
        }
        if (packed_source(node)) {
            GGML_ABORT("quactlize: K-pack is load-only; inference is not enabled (%s)", node->name);
        }
        bool packed = false;
        for (const auto * src : node->src) {
            if (packed_source(src)) {
                if (!ggml_quactlize_enabled()) {
                    GGML_ABORT("quactlize: K-pack is load-only; inference is not enabled (%s reads %s)", node->name,
                               src->name);
                }
                if (src != node->src[0] || !ggml_quactlize_can_compute(node)) {
                    const auto * input = node->src[1];
                    GGML_ABORT(
                        "quactlize: unsupported packed operation %s (%s reads %s) device=%d "
                        "rows(input/output)=%" PRId64 "/%" PRId64 " n(weight/output)=%" PRId64 "/%" PRId64
                        " k(weight/input)=%" PRId64 "/%" PRId64,
                        ggml_op_name(node->op), node->name, src->name, cuda_ctx.device, input ? ggml_nrows(input) : -1,
                        ggml_nrows(node), src->ne[1], node->ne[0], src->ne[0], input ? input->ne[0] : -1);
                }
                packed = true;
            }
        }
        // Empty outputs stay on this backend, but need no compute plan.
        if (!packed || ggml_is_empty(node)) {
            continue;
        }
        const auto * weight = weight_owner(node->src[0]);
        auto *       ctx    = static_cast<buffer_context *>(weight->buffer->context);
        GGML_ASSERT(ctx->device == cuda_ctx.device);
        std::lock_guard<std::mutex> lock(ctx->compute_mutex);
        auto                        found = ctx->tensors.find(weight);
        GGML_ASSERT(found != ctx->tensors.end() && found->second.loaded);
        auto &                        state = found->second;
        if (node->op == GGML_OP_MUL_MAT_ID) {
            quactlize_ppu_moe_matmul_params_v1 params{};
            GGML_ASSERT(moe_params(node, params));
            const auto key = key_for(params);
            if (state.moe_plans.count(key)) {
                continue;
            }
            prepare_ready(cuda_ctx, state);
            const auto & api = library();
            if (!ctx->moe_compute && (api.moe_create(ctx->device, &ctx->moe_compute) || !ctx->moe_compute)) {
                GGML_ABORT("quactlize: MoE context: %s", api.moe_error());
            }
            const auto * low   = static_cast<const uint8_t *>(weight->data);
            const auto & sizes = state.layout.sizes;
            moe_plan     plan;
            plan.info.version = 1;
            plan.info.size    = sizeof(plan.info);
            if (api.moe_prepare(ctx->moe_compute, &params, low, sizes.high_bytes ? low + sizes.low_bytes : nullptr,
                                low + sizes.low_bytes + sizes.high_bytes, &plan.handle, &plan.info)) {
                GGML_ABORT("quactlize: MoE prepare %s: %s", weight->name, api.moe_error());
            }
            if (!plan.handle || plan.info.version != 1 || plan.info.size != sizeof(plan.info) ||
                plan.info.workspace_bytes > SIZE_MAX || plan.info.route < 0 || plan.info.route > 2 ||
                plan.info.split < 1 || plan.info.compute_type != 1 || plan.info.reserved || !plan.info.config[0] ||
                !memchr(plan.info.config, '\0', sizeof(plan.info.config))) {
                GGML_ABORT("quactlize: invalid MoE plan ABI for %s", weight->name);
            }
            state.moe_plans.emplace(key, plan);
            GGML_LOG_INFO(
                "[quactlize-moe] tensor=%s device=%d q=%d m=%d n=%d k=%d e=%d topk=%d channels=%d "
                "route=%d policy=%d source_m=%d split=%d compute=%d prepare=%d/%d workspace=%" PRIu64 " config=%s\n",
                weight->name, ctx->device, params.qtype, params.tokens, params.n, params.k, params.experts, params.topk,
                params.channels, plan.info.route, plan.info.policy, plan.info.source_tokens, plan.info.split,
                plan.info.compute_type, plan.info.prepare_algorithm, plan.info.prepare_helper,
                plan.info.workspace_bytes, plan.info.config);
            continue;
        }
        quactlize_ppu_dense_params_v1 params{};
        GGML_ASSERT(dense_params(node, params));
        const dense_key key{params.m, params.input_stride, params.output_stride};
        if (state.plans.count(key)) {
            continue;
        }
        prepare_ready(cuda_ctx, state);
        const auto & api = library();
        if (!ctx->compute && (api.dense_create(ctx->device, &ctx->compute) || !ctx->compute)) {
            GGML_ABORT("quactlize: dense context: %s", api.dense_error());
        }
        const auto * low   = static_cast<const uint8_t *>(weight->data);
        const auto & sizes = state.layout.sizes;
        dense_plan   plan;
        plan.info.version = 1;
        plan.info.size    = sizeof(plan.info);
        if (api.dense_prepare(ctx->compute, &params, low, sizes.high_bytes ? low + sizes.low_bytes : nullptr,
                              low + sizes.low_bytes + sizes.high_bytes, &plan.handle, &plan.info)) {
            GGML_ABORT("quactlize: dense prepare %s: %s", weight->name, api.dense_error());
        }
        if (!plan.handle || plan.info.version != 1 || plan.info.size != sizeof(plan.info) ||
            plan.info.workspace_bytes > SIZE_MAX || plan.info.route < 0 || plan.info.route > 3 || plan.info.split < 1 ||
            plan.info.compute_type < 0 || plan.info.compute_type > 1 || !plan.info.config[0] ||
            !memchr(plan.info.config, '\0', sizeof(plan.info.config))) {
            GGML_ABORT("quactlize: invalid dense plan ABI for %s", weight->name);
        }
        state.plans.emplace(key, plan);
        GGML_LOG_INFO(
            "[quactlize-dense] tensor=%s device=%d q=%d m=%d n=%d k=%d route=%d policy=%d split=%d "
            "compute=%d workspace=%" PRIu64 " config=%s\n",
            weight->name, ctx->device, params.qtype, params.m, params.n, params.k, plan.info.route, plan.info.policy,
            plan.info.split, plan.info.compute_type, plan.info.workspace_bytes, plan.info.config);
    }
}

void ggml_quactlize_mul_mat(ggml_backend_cuda_context & cuda_ctx, ggml_tensor * dst) {
    quactlize_ppu_dense_params_v1 params{};
    GGML_ASSERT(ggml_quactlize_enabled() && dense_params(dst, params));
    const auto * weight = weight_owner(dst->src[0]);
    auto *       ctx    = static_cast<buffer_context *>(weight->buffer->context);
    GGML_ASSERT(ctx->device == cuda_ctx.device);
    dense_plan plan;
    {
        std::lock_guard<std::mutex> lock(ctx->compute_mutex);
        const auto                  tensor = ctx->tensors.find(weight);
        GGML_ASSERT(tensor != ctx->tensors.end() && tensor->second.compute_ready);
        const auto found = tensor->second.plans.find({params.m, params.input_stride, params.output_stride});
        GGML_ASSERT(found != tensor->second.plans.end());
        plan = found->second;
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(cuda_ctx.pool());
    if (plan.info.workspace_bytes) {
        workspace.alloc(plan.info.workspace_bytes);
    }
    const auto & api = library();
    if (api.dense_run(plan.handle, static_cast<const float *>(dst->src[1]->data), static_cast<float *>(dst->data),
                      workspace.get(), plan.info.workspace_bytes, cuda_ctx.stream())) {
        GGML_ABORT("quactlize: dense run %s: %s", dst->name, api.dense_error());
    }
}

void ggml_quactlize_mul_mat_id(ggml_backend_cuda_context & cuda_ctx, ggml_tensor * dst) {
    quactlize_ppu_moe_matmul_params_v1 params{};
    GGML_ASSERT(ggml_quactlize_enabled() && moe_params(dst, params));
    const auto * weight = weight_owner(dst->src[0]);
    auto *       ctx    = static_cast<buffer_context *>(weight->buffer->context);
    GGML_ASSERT(ctx->device == cuda_ctx.device);
    moe_plan plan;
    {
        std::lock_guard<std::mutex> lock(ctx->compute_mutex);
        const auto                  tensor = ctx->tensors.find(weight);
        GGML_ASSERT(tensor != ctx->tensors.end() && tensor->second.compute_ready);
        const auto found = tensor->second.moe_plans.find(key_for(params));
        GGML_ASSERT(found != tensor->second.moe_plans.end());
        plan = found->second;
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(cuda_ctx.pool());
    if (plan.info.workspace_bytes) {
        workspace.alloc(plan.info.workspace_bytes);
    }
    const auto & api = library();
    if (api.moe_run(plan.handle, static_cast<const float *>(dst->src[1]->data),
                    static_cast<const int32_t *>(dst->src[2]->data), static_cast<float *>(dst->data), workspace.get(),
                    plan.info.workspace_bytes, cuda_ctx.stream())) {
        GGML_ABORT("quactlize: MoE run %s: %s", dst->name, api.moe_error());
    }
}
#endif
