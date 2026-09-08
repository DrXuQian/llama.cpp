#include "quactlize-execution.cuh"
#include "quactlize-execution-lib.h"
#include "quactlize-buft.cuh"

#ifdef GGML_NCP_QUACTLIZE
#include "mmid.cuh"
#include "ncp-route.cuh"
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace {
enum class RouteMode { Auto, Fq, Sf, Gemv };
RouteMode route_mode() {
    static const RouteMode mode = [] {
        const char * s = getenv("QUACTLIZE_KPACK_ROUTE");
        if (!s || !*s || !strcmp(s, "auto")) return RouteMode::Auto;
        if (!strcmp(s, "fq")) return RouteMode::Fq;
        if (!strcmp(s, "sf")) return RouteMode::Sf;
        if (!strcmp(s, "gemv")) return RouteMode::Gemv;
        GGML_ABORT("[quactlize] QUACTLIZE_KPACK_ROUTE must be auto, fq, sf or gemv");
    }();
    return mode;
}

__global__ void gather(const float * src, half * dst, const int32_t * ids, int width, int64_t stride) {
    int64_t row = blockIdx.x;
    int64_t from = ids ? ids[row] : row;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[row * width + i] = __float2half(src[from * stride + i]);
}
__global__ void scatter(const half * src, float * dst, const int32_t * ids, int width, int64_t stride) {
    int64_t row = blockIdx.x;
    int64_t to = ids ? ids[row] : row;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[to * stride + i] = __half2float(src[row * width + i]);
}

size_t align256(size_t n) {
    GGML_ASSERT(n <= SIZE_MAX - 255);
    return (n + 255) & ~size_t(255);
}
using Key = std::array<uint64_t, 26>;
struct Plan {
    const ggml_quactlize_execution_api * api;
    ggml_quactlize_artifact art{};
    qks_choice_v1 choice{};
    qkg_call_v1 gemv{};
    qkg_config_v1 gemv_config{};
    void * handle = nullptr;
    half * a = nullptr;
    half * out = nullptr;
    int32_t * ids_src = nullptr;
    int32_t * ids_dst = nullptr;
    int32_t * bounds = nullptr;
    bool direct = false, legacy = false, sf = false;
    int rows = 0, tokens = 0, topk = 0;
    ~Plan() { if (handle) api->destroy(handle); }
};
struct Scratch { void * pointer = nullptr; size_t capacity = 0; };

struct Execution {
    const ggml_quactlize_execution_api * api;
    void * runtime = nullptr;
    int device;
    std::map<Key, std::unique_ptr<Plan>> plans;
    std::map<cudaStream_t, Scratch> scratch;
    std::vector<void *> allocations;

    Execution(const ggml_quactlize_execution_api * api, int device) : api(api), device(device) {
        if (api->open(api->root, &runtime) != QKS_OK) GGML_ABORT("[quactlize] dispatch open: %s", api->error());
    }
    ~Execution() {
        ggml_cuda_set_device(device);
        // Context teardown only. Retain previous scratch generations while
        // their handles or CUDA graphs can still reference them.
        for (auto & s : scratch) CUDA_CHECK(cudaStreamSynchronize(s.first));
        plans.clear();
        for (void * p : allocations) CUDA_CHECK(cudaFree(p));
        api->close(runtime);
    }
    uint8_t * storage(cudaStream_t stream, size_t need) {
        auto & s = scratch[stream];
        if (!need) return nullptr;
        if (need > s.capacity) {
            size_t capacity = 1 << 20;
            while (capacity < need) { GGML_ASSERT(capacity <= SIZE_MAX / 2); capacity *= 2; }
            void * p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, capacity));
            allocations.push_back(p);
            s = {p, capacity};
        }
        return (uint8_t *) s.pointer;
    }
};

Execution * execution(ggml_backend_cuda_context & ctx) {
    auto api = ggml_quactlize_execution_library();
    if (!api) return nullptr;
    if (!ctx.quactlize_execution) ctx.quactlize_execution = std::make_shared<Execution>(api, ctx.device);
    return static_cast<Execution *>(ctx.quactlize_execution.get());
}

Key make_key(const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * ids,
             const ggml_tensor * output, const ggml_quactlize_artifact & art, cudaStream_t stream) {
    return {uint64_t(uintptr_t(weight)), uint64_t(uintptr_t(art.low)), uint64_t(uintptr_t(art.ready)),
        uint64_t(uintptr_t(input->data)), uint64_t(uintptr_t(output->data)),
        uint64_t(uintptr_t(ids ? ids->data : nullptr)), uint64_t(uintptr_t(stream)),
        uint64_t(art.qtype), uint64_t(art.n), uint64_t(art.k), uint64_t(art.experts),
        uint64_t(input->ne[1]), uint64_t(input->ne[2]), uint64_t(input->ne[3]),
        input->nb[1], input->nb[2], output->nb[1], output->nb[2],
        uint64_t(ids ? ids->ne[0] : 0), uint64_t(ids ? ids->ne[1] : 0), ids ? ids->nb[1] : 0,
        uint64_t(input->type), uint64_t(output->type), uint64_t(route_mode()),
        uint64_t(output->ne[1]), uint64_t(output->ne[2])};
}

Plan & prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
               const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output) {
    auto & owner = *execution(ctx);
    ggml_quactlize_artifact art{};
    GGML_ASSERT(ggml_quactlize_artifact_for(weight, &art));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(output));
    GGML_ASSERT(input->ne[0] == art.k && output->ne[0] == art.n);
    cudaStream_t stream = ctx.stream();
    auto key = make_key(weight, input, ids, output, art, stream);
    auto found = owner.plans.find(key);
    if (found != owner.plans.end()) return *found->second;
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        GGML_ABORT("[quactlize] %s: native plan was not prepared before capture", weight->name);
    }
    int64_t tokens = ids ? input->ne[2] : input->ne[1] * input->ne[2] * input->ne[3];
    int64_t topk = ids ? ids->ne[0] : 1;
    GGML_ASSERT(tokens > 0 && topk > 0 && tokens <= INT32_MAX / topk);
    GGML_ASSERT(art.n <= INT32_MAX && art.k <= INT32_MAX && art.experts <= INT32_MAX);
    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
        GGML_ASSERT(ids->ne[1] == tokens && input->ne[3] == 1 && output->ne[1] == topk && output->ne[2] == tokens);
        GGML_ASSERT(input->ne[1] > 0 && topk <= art.experts);
    } else {
        GGML_ASSERT(art.experts == 1 && ggml_nelements(output) / art.n == tokens);
    }
    auto p = std::make_unique<Plan>();
    p->api = owner.api; p->art = art; p->rows = tokens * topk; p->tokens = tokens; p->topk = topk;
    const RouteMode mode = route_mode();
    p->direct = mode == RouteMode::Gemv || (mode == RouteMode::Auto && tokens == 1);
    if (p->direct) {
        auto & c = p->gemv;
        c.version = 1; c.size = sizeof(c); c.qtype = art.qtype; c.n = art.n; c.k = art.k;
        c.experts = art.experts; c.rows = p->rows; c.mode = ids ? QKG_INDEXED : QKG_DENSE;
        c.input_type = QKG_F32; c.channels = ids ? input->ne[1] : 1; c.topk = topk;
        c.a_row_stride = input->nb[1] / sizeof(float); c.a_token_stride = ids ? input->nb[2] / sizeof(float) : 0;
        c.ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0; c.out_row_stride = art.n;
        c.a = input->data; c.low = art.low; c.high = art.high; c.units = art.units;
        c.ids = ids ? (const int32_t *) ids->data : nullptr; c.output = (float *) output->data; c.stream = stream;
        p->direct = ggml_quactlize_gemv_config(c, &p->gemv_config);
        if (!p->direct && mode == RouteMode::Gemv)
            GGML_ABORT("[quactlize] %s: forced GEMV has no measured recipe for this request", weight->name);
    }
    if (p->direct) {
        auto & c = p->gemv;
        qkg_sizes_v1 sizes{};
        int rc = owner.api->gemv_query(&c, &p->gemv_config, &art.arrangement, &sizes);
        if (rc != QKG_OK) GGML_ABORT("[quactlize] %s: GEMV query failed rc=%d", weight->name, rc);
        c.workspace = owner.storage(stream, sizes.workspace_bytes); c.workspace_bytes = sizes.workspace_bytes;
        GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=gemv q=%d rows=%d n=%" PRId64 " k=%" PRId64
            " columns=%d warps=%d split=%d selection=MEASURED_GEMV_POOL\n", weight->name, ids ? "grouped" : "dense",
            art.qtype, p->rows, art.n, art.k, p->gemv_config.columns, p->gemv_config.warps, p->gemv_config.split);
    } else {
        qks_request_v1 r{1, sizeof(r), art.qtype, ids ? QK_GROUPED_FQ : QK_DENSE_FQ,
            p->rows, int(art.n), int(art.k), int(art.experts), int(tokens), art.arrangement.mapping_id};
        int status = QKS_MISS;
        const int prefill_choice = mode == RouteMode::Auto && tokens > 1 ? ggml_quactlize_prefill_route(r) : -1;
        if (mode == RouteMode::Sf || prefill_choice == 1) {
            r.route = ids ? QK_GROUPED_SF : QK_DENSE_SF;
            status = owner.api->query(owner.runtime, &r, &p->choice);
            if (status == QKS_OK && ggml_quactlize_prepare_scales(weight)) {
                p->sf = true;
                GGML_ASSERT(ggml_quactlize_artifact_for(weight, &p->art));
            } else if (status != QKS_OK && status != QKS_MISS) {
                GGML_ABORT("[quactlize] %s: SF selection failed: %s", weight->name, owner.api->error());
            }
        }
        if (!p->sf) {
            if (mode == RouteMode::Sf) GGML_ABORT("[quactlize] %s: forced SF has no admitted resources/module", weight->name);
            r.route = ids ? QK_GROUPED_FQ : QK_DENSE_FQ;
            status = owner.api->query(owner.runtime, &r, &p->choice);
        }
        if (status == QKS_MISS) {
            p->legacy = true;
            GGML_LOG_INFO("[quactlize] %s: native policy miss, retain legacy K-pack FQ (%s)\n", weight->name, owner.api->error());
        } else {
            if (status != QKS_OK) GGML_ABORT("[quactlize] %s: native selection: %s", weight->name, owner.api->error());
            size_t a_bytes = align256(size_t(p->rows) * art.k * sizeof(half));
            size_t out_bytes = align256(size_t(p->rows) * art.n * sizeof(half));
            size_t index_bytes = ids ? align256(size_t(p->rows) * sizeof(int32_t)) : 0;
            size_t bound_bytes = ids ? align256(size_t(art.experts + 1) * sizeof(int32_t)) : 0;
            size_t head = a_bytes + out_bytes + 2 * index_bytes + bound_bytes;
            GGML_ASSERT(p->choice.workspace_bytes <= SIZE_MAX - head);
            uint8_t * storage = owner.storage(stream, head + p->choice.workspace_bytes);
            p->a = (half *) storage; p->out = (half *) (storage + a_bytes);
            if (ids) {
                p->ids_src = (int32_t *) (storage + a_bytes + out_bytes);
                p->ids_dst = (int32_t *) ((uint8_t *) p->ids_src + index_bytes);
                p->bounds = (int32_t *) ((uint8_t *) p->ids_dst + index_bytes);
            }
            qk_call_v1 c{};
            c.version = 1; c.size = sizeof(c); c.m = p->rows; c.n = art.n; c.k = art.k; c.experts = art.experts;
            c.group_size = art.arrangement.group_size; c.device = p->choice.device; c.compute_units = p->choice.compute_units;
            GGML_ASSERT(c.device == ctx.device);
            c.mapping_id = art.arrangement.mapping_id; c.a = p->a; c.low = art.low; c.high = art.high;
            c.metadata = p->sf ? (const void *) p->art.scale : (const void *) art.units;
            c.zero = p->sf ? p->art.zero : nullptr; c.output = p->out; c.offsets_device = p->bounds;
            c.workspace = storage + head; c.workspace_bytes = p->choice.workspace_bytes; c.stream = stream;
            if (owner.api->prepare(owner.runtime, &p->choice, &c, &p->handle) != QKS_OK)
                GGML_ABORT("[quactlize] %s: native prepare: %s", weight->name, owner.api->error());
            GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=%s q=%d rows=%d n=%" PRId64 " k=%" PRId64
                " parent=%s build=%s algorithm=%d split=%d grid=%d policy=%d prefill_choice=%d\n", weight->name, ids ? "grouped" : "dense",
                p->sf ? "sf" : "fq", art.qtype, p->rows, art.n, art.k, p->choice.parent, p->choice.build_key,
                p->choice.algorithm, p->choice.split, p->choice.grid, p->choice.policy, prefill_choice);
        }
    }
    auto * result = p.get();
    owner.plans.emplace(key, std::move(p));
    return *result;
}
} // namespace

void ggml_quactlize_execution_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph) {
    if (!execution(ctx)) return;
    const int saved_stream = ctx.curr_stream_no;
    bool concurrent = !ctx.stream_context().concurrent_events.empty();
    for (const auto & event : ctx.stream_context().concurrent_events) concurrent &= event.second.is_valid();
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) continue;
        ggml_quactlize_artifact art{};
        if (!ggml_quactlize_artifact_for(node->src[0], &art)) continue;
        if (ggml_nelements(node) == 0) continue;
        ctx.curr_stream_no = 0;
        if (concurrent) for (const auto & event : ctx.stream_context().concurrent_events) {
            auto stream = event.second.stream_mapping.find(node);
            if (stream != event.second.stream_mapping.end()) { ctx.curr_stream_no = stream->second; break; }
        }
        prepare(ctx, node->src[0], node->src[1], node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr, node);
    }
    ctx.curr_stream_no = saved_stream;
}

bool ggml_quactlize_execution_run(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
    const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output) {
    if (!execution(ctx)) return false;
    if (ggml_nelements(output) == 0) return true;
    auto & p = prepare(ctx, weight, input, ids, output);
    if (p.legacy) return false;
    cudaStream_t stream = ctx.stream();
    auto dependency = p.art;
    if (p.sf) dependency.ready = p.art.scale_ready;
    ggml_quactlize_wait_ready(dependency, stream);
    if (p.direct) {
        if (p.api->gemv_run(&p.gemv, &p.gemv_config, &p.art.arrangement) != QKG_OK)
            GGML_ABORT("[quactlize] %s: GEMV launch failed", weight->name);
        ggml_ncp_route_log(output, "so-quactlize-kpack-gemv", nullptr);
        return true;
    }
    if (ids) {
        const size_t smem = ggml_cuda_info().devices[ctx.device].smpbo;
        GGML_ASSERT(size_t(p.tokens) * sizeof(int32_t) <= smem && p.tokens < (1 << 22) && p.topk < (1 << 10));
        GGML_ASSERT(ids->nb[1] / sizeof(int32_t) <= INT32_MAX && input->nb[2] / input->nb[1] <= INT32_MAX);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, p.ids_src, p.ids_dst, p.bounds,
            p.art.experts, p.tokens, p.topk, input->ne[1], ids->nb[1] / sizeof(int32_t), input->nb[2] / input->nb[1], stream);
        CUDA_CHECK(cudaGetLastError());
    }
    gather<<<p.rows, 256, 0, stream>>>((const float *) input->data, p.a, p.ids_src, p.art.k, input->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    if (p.api->run(p.handle, stream) != QKS_OK) GGML_ABORT("[quactlize] %s: native GEMM launch failed", weight->name);
    scatter<<<p.rows, 256, 0, stream>>>(p.out, (float *) output->data, p.ids_dst, p.art.n, output->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    ggml_ncp_route_log(output, p.sf ? "so-quactlize-kpack-sf" : "so-quactlize-kpack-fq-selected", nullptr);
    return true;
}
#else
void ggml_quactlize_execution_prepare_graph(ggml_backend_cuda_context &, ggml_cgraph *) {}
bool ggml_quactlize_execution_run(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
                                  const ggml_tensor *, ggml_tensor *) { return false; }
#endif
