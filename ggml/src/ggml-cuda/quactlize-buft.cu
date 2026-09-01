// The K-pack extra buffer type. See quactlize-buft.cuh for why this is a buffer type and not a cache.

#include "quactlize-buft.cuh"

#include "ggml-backend-impl.h"

#include <cinttypes>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#ifdef GGML_NCP_QUACTLIZE

// M values the tactic inventory is required to cover before a tensor is allowed into this buffer type.
//
// supports_op runs at model load, when nothing knows what M the graph will present, and taking this buffer type
// throws the GGUF bytes away -- so the question "can the library serve this tensor" has to be answered for every M
// at once. The library's inventory is per-M, so it is asked along a ladder from decode to a large prefill and all
// of them must answer yes. This is a conservatism policy owned here, not a format policy: it can only refuse
// tensors the library would have served, never admit one it would not.
static const int    g_m_ladder[]  = { 1, 8, 64, 512, 4096 };
static const size_t g_m_ladder_n  = sizeof(g_m_ladder) / sizeof(g_m_ladder[0]);

struct qz_plane_sizes {
    int64_t low;
    int64_t high;
    int64_t units;
};

// The three resident planes, laid out [low][high][units] inside the tensor's own allocation. K-pack is byte-neutral,
// so they must add up to exactly ggml_nbytes: that identity is the check that the descriptor and the tensor agree,
// and it is asserted rather than assumed -- a wrong `bits` would otherwise silently shorten the units plane.
static bool qz_plane_sizes_for(
        const ggml_tensor * t, const quactlize_ppu_placed_arrangement_v2 & arr, qz_plane_sizes * out) {
    if (!ggml_quactlize_plane_sizes((int) t->type, t->ne[1], t->ne[0], t->ne[2]*t->ne[3], &arr,
                                    &out->low, &out->high, &out->units)) {
        return false;
    }
    // The byte-neutrality identity, checked against what ggml itself allocates for this tensor. This is where the
    // registry, the library's units_bytes and ggml's block size have to agree; anything else is a descriptor that
    // would silently shorten one of the planes.
    return out->low + out->high + out->units == (int64_t) ggml_nbytes(t);
}

// ---- the buffer ----

struct qz_buffer_context {
    int    device;
    void * dev_ptr;
    std::map<const ggml_tensor *, ggml_quactlize_artifact> artifacts;
};

static void qz_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    qz_buffer_context * ctx = (qz_buffer_context *) buffer->context;
    if (ctx->dev_ptr) {
        ggml_cuda_set_device(ctx->device);
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

static void qz_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
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

    qz_plane_sizes ps;
    if (!qz_plane_sizes_for(tensor, arr, &ps)) {
        GGML_ABORT("[quactlize] %s: K-pack planes (%" PRId64 " + %" PRId64 " + %" PRId64 ") do not add up to "
                   "ggml_nbytes (%zu) -- the artifact is not byte-neutral for this descriptor",
                   tensor->name, ps.low, ps.high, ps.units, ggml_nbytes(tensor));
    }

    const int64_t k       = tensor->ne[0];
    const int64_t n       = tensor->ne[1];
    const int64_t experts = tensor->ne[2] * tensor->ne[3];

    // Host side of the conversion. The GGUF bytes are already here -- the loader is handing us its mmap -- so this
    // is the only new cost the K-pack path adds to a model load: no D2H, and the H2D below was going to happen.
    std::vector<uint8_t> low  ((size_t) ps.low);
    std::vector<uint8_t> high ((size_t) ps.high);
    std::vector<uint8_t> units((size_t) ps.units);

    const int rc = ggml_quactlize_prepare(qtype, (const uint8_t *) data,
                                          low.data(), ps.high ? high.data() : nullptr, units.data(),
                                          (int) n, (int) k, (int) experts, &arr);
    if (rc != 0) {
        GGML_ABORT("[quactlize] %s: prepare returned %d for n=%" PRId64 " k=%" PRId64 " experts=%" PRId64
                   " -- the GGUF bytes are the only copy and they are not going to survive this", 
                   tensor->name, rc, n, k, experts);
    }

    // Byte-exact round trip. Separate from dequantisation on purpose: it tells a packing bug from an arithmetic
    // one, and it runs before anything downstream can mistake a misplaced plane for a bad kernel.
    {
        std::vector<uint8_t> recovered(size);
        const int rrc = ggml_quactlize_recover(qtype, low.data(), ps.high ? high.data() : nullptr, units.data(),
                                               recovered.data(), (int) n, (int) k, (int) experts, &arr);
        if (rrc != 0 || memcmp(recovered.data(), data, size) != 0) {
            GGML_ABORT("[quactlize] %s: K-pack round trip failed (recover rc=%d) -- refusing to keep an artifact "
                       "that does not reproduce its own input", tensor->name, rrc);
        }
    }

    uint8_t * dst = (uint8_t *) tensor->data;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy(dst,                        low.data(),   (size_t) ps.low,   cudaMemcpyHostToDevice));
    if (ps.high) {
        CUDA_CHECK(cudaMemcpy(dst + ps.low,           high.data(),  (size_t) ps.high,  cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dst + ps.low + ps.high,     units.data(), (size_t) ps.units, cudaMemcpyHostToDevice));

    ggml_quactlize_artifact art;
    art.low         = dst;
    art.high        = ps.high ? dst + ps.low : nullptr;
    art.units       = dst + ps.low + ps.high;
    art.arrangement = arr;
    art.qtype       = qtype;
    art.n           = n;
    art.k           = k;
    art.experts     = experts;

    ctx->artifacts[tensor] = art;
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

    qz_buffer_context * ctx = new qz_buffer_context{ buft_ctx->device, dev_ptr, {} };

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
    if (!ggml_quactlize_conversion_available(qtype)) {
        return false;
    }

    qz_plane_sizes ps;
    if (!qz_plane_sizes_for(weight, arr, &ps)) {
        return false;
    }

    const int k       = (int) weight->ne[0];
    const int n       = (int) weight->ne[1];
    const int experts = (int) (weight->ne[2] * weight->ne[3]);

    if (op == GGML_OP_MUL_MAT_ID) {
        if (experts <= 1) {
            return false;
        }
        for (size_t i = 0; i < g_m_ladder_n; ++i) {
            const int total_rows = g_m_ladder[i];
            // max_rows is the largest per-expert row count; the load-time bound is "every row went to one expert".
            if (ggml_quactlize_list_grouped_configs(qtype, nullptr, 0, total_rows, n, k, arr.group_size,
                                                    experts, total_rows, &arr) <= 0) {
                return false;
            }
        }
        return true;
    }

    // Dense: one matrix, no expert axis. ggml hands src1 over as [k, m] with ne[0] contiguous, which is already
    // the row-major [m, k] the dense entry wants, so the ladder is over the batch dimension and nothing else.
    if (experts != 1) {
        return false;
    }
    for (size_t i = 0; i < g_m_ladder_n; ++i) {
        if (ggml_quactlize_list_dense_configs(qtype, nullptr, 0, g_m_ladder[i], n, k, arr.group_size, &arr) <= 0) {
            return false;
        }
    }
    return true;
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
