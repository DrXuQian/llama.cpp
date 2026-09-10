// The K-pack extra buffer type. See quactlize-buft.cuh for why this is a buffer type and not a cache.

#include "quactlize-buft.cuh"

#include "ggml-backend-impl.h"

#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#ifdef GGML_NCP_QUACTLIZE


struct qz_plane_sizes {
    int64_t low;
    int64_t high;
    int64_t units;
};

// The three resident planes, laid out [low][high][units] inside the tensor's own allocation. The sizing itself is
// host-only arithmetic and lives in the loader translation unit so it can be tested off-box; what belongs here is
// the identity it has to satisfy, because only this side knows what ggml allocated.
static bool qz_plane_sizes_for(
        const ggml_tensor * t, const quactlize_ppu_placed_arrangement_v2 & arr, qz_plane_sizes * out) {
    if (t->ne[0] <= 0 || t->ne[0] > INT32_MAX || t->ne[1] <= 0 || t->ne[1] > INT32_MAX ||
        t->ne[2] <= 0 || t->ne[3] <= 0 || t->ne[2] > INT32_MAX / t->ne[3]) {
        return false;
    }
    if (t->ne[0] > INT64_MAX / (t->ne[2] * t->ne[3]) / t->ne[1] / 8) {
        return false;
    }
    if (!ggml_quactlize_plane_sizes((int) t->type, t->ne[1], t->ne[0], t->ne[2]*t->ne[3], &arr,
                                    &out->low, &out->high, &out->units)) {
        return false;
    }
    // K-pack is byte-neutral, and that is the whole reason this path can own the weight buffer without growing the
    // resident footprint: the three planes must add up to exactly what ggml already allocates. This is where the
    // registry, the library's units_bytes and ggml's block size have to agree.
    return out->low + out->high + out->units == (int64_t) ggml_nbytes(t);
}

// ---- the buffer ----

static constexpr size_t qz_upload_chunk_bytes = 8 * 1024 * 1024;

struct qz_upload_slot {
    uint8_t * host = nullptr;
    cudaEvent_t reusable = nullptr;
    bool pending = false;
};

struct qz_buffer_context {
    int    device;
    void * dev_ptr;
    std::map<const ggml_tensor *, ggml_quactlize_artifact> artifacts;
    cudaStream_t pack_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t upload_done = nullptr;
    std::vector<cudaEvent_t> ready_events;
    uint8_t * scratch = nullptr;
    size_t scratch_bytes = 0;
    std::vector<void *> scratch_allocations;
    std::array<qz_upload_slot, 2> upload_slots;
    unsigned upload_slot = 0;
};

static void qz_init_streams(qz_buffer_context * ctx) {
    if (!ctx->pack_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->pack_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx->upload_done, cudaEventDisableTiming));
    }
}

static cudaEvent_t qz_record_ready(qz_buffer_context * ctx) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(event, ctx->pack_stream));
    ctx->ready_events.push_back(event);
    return event;
}

static void qz_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    qz_buffer_context * ctx = (qz_buffer_context *) buffer->context;
    if (ctx->dev_ptr) {
        ggml_cuda_set_device(ctx->device);
        // Only teardown drains backcopies, never an inference launch.
        if (ctx->pack_stream) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->pack_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx->copy_stream));
        }
        for (cudaEvent_t event : ctx->ready_events) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
        for (void * scratch : ctx->scratch_allocations) {
            CUDA_CHECK(cudaFree(scratch));
        }
        for (auto & slot : ctx->upload_slots) {
            if (slot.reusable) { CUDA_CHECK(cudaEventDestroy(slot.reusable)); }
            if (slot.host) { CUDA_CHECK(cudaFreeHost(slot.host)); }
        }
        if (ctx->pack_stream) {
            CUDA_CHECK(cudaEventDestroy(ctx->upload_done));
            CUDA_CHECK(cudaStreamDestroy(ctx->pack_stream));
            CUDA_CHECK(cudaStreamDestroy(ctx->copy_stream));
        }
        CUDA_CHECK(cudaFree(ctx->dev_ptr));
    }
    delete ctx;
}

static void * qz_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((qz_buffer_context *) buffer->context)->dev_ptr;
}

static enum ggml_status qz_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // No views: a view of a K-pack tensor would be a view of a layout its reader does not describe.
    GGML_ASSERT(tensor->view_src == nullptr);
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

// Fill one owned slot while the other uploads. The caller's pageable inputs are
// consumed before return; only the reusable pinned slots remain in flight.
static void qz_upload_planes(qz_buffer_context * ctx, uint8_t * dst, const ggml_quactlize_planes & p) {
    if (!ctx->upload_slots[0].host) {
        for (auto & slot : ctx->upload_slots) {
            CUDA_CHECK(cudaMallocHost((void **) &slot.host, qz_upload_chunk_bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming));
        }
        GGML_LOG_DEBUG("[quactlize] cache H2D pipeline: device=%d slots=2 slot_MiB=8 pinned_MiB=16\n", ctx->device);
    }
    const uint8_t * sources[] = {p.low, p.high, p.units};
    const size_t sizes[] = {p.low_bytes, p.high_bytes, p.units_bytes};
    const size_t total = p.low_bytes + p.high_bytes + p.units_bytes;
    size_t plane = 0, plane_offset = 0;
    for (size_t offset = 0; offset < total; ) {
        auto & slot = ctx->upload_slots[ctx->upload_slot];
        if (slot.pending) { CUDA_CHECK(cudaEventSynchronize(slot.reusable)); }
        const size_t bytes = std::min(qz_upload_chunk_bytes, total - offset);
        size_t copied = 0;
        while (copied < bytes) {
            if (plane_offset == sizes[plane]) { ++plane; plane_offset = 0; continue; }
            const size_t part = std::min(bytes - copied, sizes[plane] - plane_offset);
            memcpy(slot.host + copied, sources[plane] + plane_offset, part);
            copied += part;
            plane_offset += part;
        }
        CUDA_CHECK(cudaMemcpyAsync(dst + offset, slot.host, bytes, cudaMemcpyHostToDevice, ctx->pack_stream));
        CUDA_CHECK(cudaEventRecord(slot.reusable, ctx->pack_stream));
        slot.pending = true;
        ctx->upload_slot ^= 1;
        offset += bytes;
    }
}

// Install cached host planes into the tensor's resident allocation. Ready is
// recorded after all chunks and retained for both eager and graph consumers.
static void qz_install_planes(qz_buffer_context * ctx, ggml_tensor * tensor, const ggml_quactlize_planes & p,
                              int qtype, int64_t n, int64_t k, int64_t experts) {
    uint8_t * dst = (uint8_t *) tensor->data;
    GGML_ASSERT(ctx->artifacts.count(tensor) == 0);
    ggml_cuda_set_device(ctx->device);
    qz_init_streams(ctx);
    qz_upload_planes(ctx, dst, p);

    ggml_quactlize_artifact art;
    art.low         = dst;
    art.high        = p.high_bytes ? dst + p.low_bytes : nullptr;
    art.units       = dst + p.low_bytes + p.high_bytes;
    art.arrangement = p.arrangement;
    art.qtype       = qtype;
    art.n           = n;
    art.k           = k;
    art.experts     = experts;
    art.ready       = qz_record_ready(ctx);
    ctx->artifacts[tensor] = art;
}

bool ggml_quactlize_tensor_is_kpack(const ggml_tensor * tensor) {
    return tensor && tensor->buffer && ggml_backend_buft_is_cuda_quactlize(tensor->buffer->buft);
}

bool ggml_quactlize_plane_layout(const ggml_tensor * tensor, size_t * low_bytes, size_t * high_bytes,
                                 size_t * units_bytes, quactlize_ppu_placed_arrangement_v2 * arrangement) {
    quactlize_ppu_placed_arrangement_v2 arr;
    if (!tensor || !ggml_quactlize_arrangement_for((int) tensor->type, &arr)) {
        return false;
    }
    qz_plane_sizes ps = {};
    if (!qz_plane_sizes_for(tensor, arr, &ps)) {
        return false;
    }
    if (low_bytes)   { *low_bytes   = (size_t) ps.low; }
    if (high_bytes)  { *high_bytes  = (size_t) ps.high; }
    if (units_bytes) { *units_bytes = (size_t) ps.units; }
    if (arrangement) { *arrangement = arr; }
    return true;
}

void ggml_quactlize_set_planes(ggml_tensor * tensor, const ggml_quactlize_planes * planes) {
    GGML_ASSERT(planes != nullptr);
    if (!ggml_quactlize_tensor_is_kpack(tensor)) {
        GGML_ABORT("[quactlize] %s: set_planes on a tensor that is not in the K-pack buffer type", tensor->name);
    }
    size_t low = 0, high = 0, units = 0;
    quactlize_ppu_placed_arrangement_v2 arr;
    if (!ggml_quactlize_plane_layout(tensor, &low, &high, &units, &arr)) {
        GGML_ABORT("[quactlize] %s: no K-pack layout for %s", tensor->name, ggml_type_name(tensor->type));
    }
    // Sizes and descriptor are the sidecar's claims; they are checked against the library's here because a
    // manifest that lies about either would otherwise put bytes under a reader that describes different ones.
    if (!planes->low || !planes->units || (high && !planes->high) ||
        planes->low_bytes != low || planes->high_bytes != high || planes->units_bytes != units ||
        memcmp(&planes->arrangement, &arr, sizeof(arr)) != 0) {
        GGML_ABORT("[quactlize] %s: sidecar planes (%zu/%zu/%zu) or descriptor disagree with the library's layout "
                   "(%zu/%zu/%zu)", tensor->name, planes->low_bytes, planes->high_bytes, planes->units_bytes,
                   low, high, units);
    }
    qz_buffer_context * ctx = (qz_buffer_context *) tensor->buffer->context;
    qz_install_planes(ctx, tensor, *planes, (int) tensor->type, tensor->ne[1], tensor->ne[0],
                      tensor->ne[2] * tensor->ne[3]);
}

static void qz_set_raw(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, const void * up, size_t offset, size_t size) {
    qz_buffer_context * ctx = (qz_buffer_context *) buffer->context;

    // Whole tensor in one call. A non-default buffer type is excluded from the loader's chunked async upload
    // (llama-model-loader.cpp), which is what makes this hold -- and it must, because the K-pack address map is
    // over the whole (n, k) plane and a chunk boundary would split a transport word.
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    const int qtype = (int) tensor->type;

    quactlize_ppu_placed_arrangement_v2 arr;
    if (!ggml_quactlize_arrangement_for(qtype, &arr)) {
        GGML_ABORT("[quactlize] %s took the K-pack buffer type but no arrangement is available for %s -- "
                   "supports_op and set_tensor disagree", tensor->name, ggml_type_name(tensor->type));
    }

    qz_plane_sizes ps = {};
    if (!qz_plane_sizes_for(tensor, arr, &ps)) {
        GGML_ABORT("[quactlize] %s: K-pack planes (%" PRId64 " + %" PRId64 " + %" PRId64 ") do not add up to "
                   "ggml_nbytes (%zu) -- the artifact is not byte-neutral for this descriptor",
                   tensor->name, ps.low, ps.high, ps.units, ggml_nbytes(tensor));
    }

    const int64_t k       = tensor->ne[0];
    const int64_t n       = tensor->ne[1];
    const int64_t experts = tensor->ne[2] * tensor->ne[3];

    GGML_ASSERT(n <= INT32_MAX && k <= INT32_MAX && experts <= INT32_MAX);
    GGML_ASSERT(ctx->artifacts.count(tensor) == 0);
    quactlize_ppu_kpack_sizes_v1 sizes = {};
    if (ggml_quactlize_device_pack_sizes(qtype, n, k, experts, &arr, &sizes) != 0 ||
        sizes.raw_bytes != size) {
        GGML_ABORT("[quactlize] %s: GPU producer and resident layout disagree", tensor->name);
    }

    ggml_cuda_set_device(ctx->device);
    qz_init_streams(ctx);
    const size_t expert_bytes = size / experts;
    const size_t budget = std::max(expert_bytes, size_t(64) * 1024 * 1024);
    const int64_t batch = std::min(experts, (int64_t) (budget / expert_bytes));
    const size_t needed = batch * expert_bytes;
    if (needed > ctx->scratch_bytes) {
        const size_t capacity = std::max(needed, ctx->scratch_bytes * 2);
        void * scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&scratch, capacity));
        // Retain older buffers until teardown; cudaFree can wait for unrelated streams.
        ctx->scratch_allocations.push_back(scratch);
        ctx->scratch = (uint8_t *) scratch;
        ctx->scratch_bytes = capacity;
    }
    uint8_t * dst = (uint8_t *) tensor->data;
    for (int64_t first = 0; first < experts; first += batch) {
        const int count = (int) std::min(batch, experts - first);
        const size_t source_expert = up ? expert_bytes/2 : expert_bytes;
        const size_t source_batch = count*source_expert;
        CUDA_CHECK(cudaMemcpyAsync(ctx->scratch, (const uint8_t *) data + first*source_expert,
                                   source_batch, cudaMemcpyHostToDevice, ctx->pack_stream));
        if (up) CUDA_CHECK(cudaMemcpyAsync(ctx->scratch+source_batch, (const uint8_t *) up+first*source_expert,
                                          source_batch, cudaMemcpyHostToDevice, ctx->pack_stream));
        if (first + count == experts) {
            CUDA_CHECK(cudaEventRecord(ctx->upload_done, ctx->pack_stream));
        }
        const int rc = up ? ggml_quactlize_prepare_device_pair(
            qtype,ctx->scratch,ctx->scratch+source_batch,dst+first*(ps.low/experts),
            ps.high?dst+ps.low+first*(ps.high/experts):nullptr,
            dst+ps.low+ps.high+first*(ps.units/experts),n/2,k,count,&arr,ctx->pack_stream) : ggml_quactlize_prepare_device(
            qtype, ctx->scratch, dst + first * (ps.low / experts),
            ps.high ? dst + ps.low + first * (ps.high / experts) : nullptr,
            dst + ps.low + ps.high + first * (ps.units / experts),
            n, k, count, &arr, ctx->pack_stream);
        if (rc != 0) {
            GGML_ABORT("[quactlize] %s: GPU pack failed (rc=%d)", tensor->name, rc);
        }
    }
    ggml_quactlize_artifact art;
    art.low = dst;
    art.high = ps.high ? dst + ps.low : nullptr;
    art.units = dst + ps.low + ps.high;
    art.arrangement = arr;
    art.qtype = qtype;
    art.n = n;
    art.k = k;
    art.experts = experts;
    art.ready = qz_record_ready(ctx);
    // The caller may release raw CPU bytes on return. This waits only for H2D,
    // not for the last pack, the backcopy stream, or a persistence worker.
    CUDA_CHECK(cudaEventSynchronize(ctx->upload_done));
    ctx->artifacts[tensor] = art;
    GGML_LOG_DEBUG("[quactlize] %s: GPU pack queued, %.1f MiB, expert batch=%" PRId64 "\n",
                   tensor->name, size / 1048576.0, batch);
}

static void qz_buffer_set_tensor(ggml_backend_buffer_t buffer,ggml_tensor * tensor,
    const void * data,size_t offset,size_t size) {
    qz_set_raw(buffer,tensor,data,nullptr,offset,size);
}
bool ggml_quactlize_pair_supported(ggml_backend_buffer_type_t buft,const ggml_tensor * merged) {
    return buft && merged && ggml_backend_buft_is_cuda_quactlize(buft) && merged->ne[1]%2==0 &&
        ggml_quactlize_device_pair_available(merged->type) && ggml_quactlize_can_serve(merged,GGML_OP_MUL_MAT_ID);
}
void ggml_quactlize_set_gate_up(ggml_tensor * merged,const void * gate,const void * up,size_t bytes_each) {
    GGML_ASSERT(merged && merged->buffer && gate && up &&
        ggml_quactlize_pair_supported(merged->buffer->buft,merged) && bytes_each==ggml_nbytes(merged)/2);
    qz_set_raw(merged->buffer,merged,gate,up,0,ggml_nbytes(merged));
}

bool ggml_quactlize_copy_range_async(const ggml_tensor * tensor, void * pinned,
                                    size_t offset, size_t bytes, ggml_backend_event_t completion) {
    ggml_quactlize_artifact art;
    if (!pinned || !completion || !completion->context || !bytes ||
        !ggml_quactlize_artifact_for(tensor, &art) || offset > ggml_nbytes(tensor) ||
        bytes > ggml_nbytes(tensor) - offset ||
        completion->device != ggml_backend_buft_get_device(tensor->buffer->buft)) {
        return false;
    }
    auto * ctx = (qz_buffer_context *) tensor->buffer->context;
    ggml_cuda_set_device(ctx->device);
    cudaPointerAttributes attributes = {};
    cudaError_t rc = cudaPointerGetAttributes(&attributes, pinned);
    if (rc != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (rc != cudaSuccess) { (void) cudaGetLastError(); }
        return false;
    }
    rc = cudaStreamWaitEvent(ctx->copy_stream, art.ready, 0);
    if (rc == cudaSuccess) {
        rc = cudaMemcpyAsync(pinned, art.low + offset, bytes, cudaMemcpyDeviceToHost, ctx->copy_stream);
    }
    if (rc == cudaSuccess) { rc = cudaEventRecord((cudaEvent_t) completion->context, ctx->copy_stream); }
    if (rc == cudaSuccess) { return true; }
    // The caller may release its destination after false. Drain any partial
    // submission on this stream only, not unrelated compute streams.
    CUDA_CHECK(cudaStreamSynchronize(ctx->copy_stream));
    (void) cudaGetLastError();
    return false;
}

bool ggml_quactlize_copy_range_wait(ggml_backend_event_t completion) {
    if (!completion || !completion->context) { return false; }
    const cudaError_t rc = cudaEventSynchronize((cudaEvent_t) completion->context);
    if (rc != cudaSuccess) { (void) cudaGetLastError(); }
    return rc == cudaSuccess;
}

static void qz_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    qz_buffer_context * ctx = (qz_buffer_context *) buffer->context;
    if (!ctx->dev_ptr || buffer->size == 0) {
        return;
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
}

// get_tensor and cpy_tensor are null on purpose: the bytes here are not the tensor's declared format any more, so
// there is no correct answer to "read this tensor back". ggml-cpu's repack buffer does exactly the same.
static const ggml_backend_buffer_i qz_buffer_interface = {
    /* .free_buffer     = */ qz_buffer_free_buffer,
    /* .get_base        = */ qz_buffer_get_base,
    /* .init_tensor     = */ qz_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ qz_buffer_set_tensor,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ qz_buffer_clear,
    /* .reset           = */ NULL,
};

// ---- the buffer type ----

struct qz_buft_context {
    int         device;
    std::string name;
};

static const char * qz_buft_get_name(ggml_backend_buffer_type_t buft) {
    return ((qz_buft_context *) buft->context)->name.c_str();
}

bool ggml_backend_buft_is_cuda_quactlize(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == qz_buft_get_name;
}

static ggml_backend_buffer_t qz_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    qz_buft_context * buft_ctx = (qz_buft_context *) buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    // Plain cudaMalloc, like the split buffer: ggml_cuda_device_malloc's unified-memory arm is a fallback for
    // oversubscribed weights, and a K-pack tensor that migrated to host memory would be read by a PPU kernel.
    //
    // Size 0 is not a degenerate case to guard against but the normal one during buffer-type selection:
    // weight_buft_supported allocates a zero-byte buffer purely so supports_op can see which buft a weight would
    // land in. Allocating nothing keeps that probe from depending on how a driver answers cudaMalloc(0).
    void * dev_ptr = nullptr;
    cudaError_t err = size ? cudaMalloc(&dev_ptr, size) : cudaSuccess;
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d failed: %s\n",
                       __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    qz_buffer_context * ctx = new qz_buffer_context{};
    ctx->device = buft_ctx->device;
    ctx->dev_ptr = dev_ptr;

    return ggml_backend_buffer_init(buft, qz_buffer_interface, ctx, size);
}

static size_t qz_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

// Exactly ggml_nbytes -- no MATRIX_ROW_PADDING tail. The whole point of owning the buffer is that the resident
// footprint does not grow, and the K-pack reader does not use a padded row the way the inline quantised kernels do.
static size_t qz_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool qz_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i qz_buft_interface = {
    /* .get_name         = */ qz_buft_get_name,
    /* .alloc_buffer     = */ qz_buft_alloc_buffer,
    /* .get_alignment    = */ qz_buft_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ qz_buft_get_alloc_size,
    /* .is_host          = */ qz_buft_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_quactlize_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<int, ggml_backend_buffer_type> bufts;

    auto it = bufts.find(device);
    if (it == bufts.end()) {
        ggml_backend_buffer_type buft;
        buft.iface   = qz_buft_interface;
        buft.device  = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device);
        buft.context = new qz_buft_context{ device, GGML_CUDA_NAME + std::to_string(device) + "_KPACK" };
        it = bufts.emplace(device, buft).first;
    }

    return &it->second;
}

// ---- capability ----

bool ggml_quactlize_can_serve(const ggml_tensor * weight, ggml_op op) {
    if (weight == nullptr || (op != GGML_OP_MUL_MAT && op != GGML_OP_MUL_MAT_ID)) {
        return false;
    }

    const int qtype = (int) weight->type;
    if (!ggml_quactlize_available(qtype)) {
        return false;
    }

    quactlize_ppu_placed_arrangement_v2 arr;
    if (!ggml_quactlize_arrangement_for(qtype, &arr)) {
        return false;
    }

    // Without an in-process conversion there is no way to fill this buffer, and set_tensor has no second chance.
    if (!ggml_quactlize_device_pack_available(qtype)) {
        return false;
    }

    qz_plane_sizes ps;
    if (!qz_plane_sizes_for(weight, arr, &ps)) {
        return false;
    }

    const int k       = (int) weight->ne[0];
    const int n       = (int) weight->ne[1];
    const int experts = (int) (weight->ne[2] * weight->ne[3]);

    quactlize_ppu_kpack_sizes_v1 device_sizes = {};
    if (ggml_quactlize_device_pack_sizes(qtype, n, k, experts, &arr, &device_sizes) != 0 ||
        device_sizes.raw_bytes != ggml_nbytes(weight)) {
        return false;
    }

    // The library answers for every runtime M at once. That is the only form of the question this function can
    // ask -- it runs at load, before any M exists, and the GGUF bytes do not survive a yes -- and no shape policy is
    // mirrored beside it: the handoff makes the library the admission authority, and a rule copied here would be a
    // second source that drifts.
    if (op == GGML_OP_MUL_MAT_ID) {
        return experts > 1 && ggml_quactlize_grouped_any_m_valid(qtype, n, k, experts, &arr) == 1;
    }

    // Dense: one matrix, no expert axis. ggml hands src1 over as [k, m] with ne[0] contiguous, which is already
    // the row-major [m, k] the dense entry wants.
    return experts == 1 && ggml_quactlize_dense_any_m_valid(qtype, n, k, &arr) == 1;
}

bool ggml_quactlize_artifact_for(const ggml_tensor * tensor, ggml_quactlize_artifact * out) {
    if (tensor == nullptr || out == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    if (!ggml_backend_buft_is_cuda_quactlize(tensor->buffer->buft)) {
        return false;
    }
    qz_buffer_context * ctx = (qz_buffer_context *) tensor->buffer->context;
    auto it = ctx->artifacts.find(tensor);
    if (it == ctx->artifacts.end()) {
        return false;   // in a K-pack buffer but never converted -- must not be read as K-pack
    }
    *out = it->second;
    return true;
}


bool ggml_quactlize_node_reads_artifact(const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src && src->buffer && ggml_backend_buft_is_cuda_quactlize(src->buffer->buft)) {
            return true;
        }
    }
    return false;
}

#else  // quactlize off


ggml_backend_buffer_type_t ggml_backend_cuda_quactlize_buffer_type(int device) {
    GGML_UNUSED(device);
    return nullptr;
}

bool ggml_backend_buft_is_cuda_quactlize(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

bool ggml_quactlize_can_serve(const ggml_tensor * weight, ggml_op op) {
    GGML_UNUSED(weight);
    GGML_UNUSED(op);
    return false;
}

bool ggml_quactlize_artifact_for(const ggml_tensor * tensor, ggml_quactlize_artifact * out) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(out);
    return false;
}

bool ggml_quactlize_node_reads_artifact(const ggml_tensor * node) {
    GGML_UNUSED(node);
    return false;
}

#endif // GGML_NCP_QUACTLIZE
