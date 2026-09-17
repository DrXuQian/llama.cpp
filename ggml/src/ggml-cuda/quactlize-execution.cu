#include "quactlize-execution.cuh"
#include "quactlize-execution-lib.h"
#include "quactlize-buft.cuh"

#ifdef GGML_NCP_QUACTLIZE
#include "mmid.cuh"
#include "ncp-route.cuh"
#include "ggml-impl.h"
#include "quactlize/moe_graph.hpp"
#include "quactlize/gate_up_graph.hpp"
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <unistd.h>
#include <cuda_bf16.h>

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

int compute_type(const ggml_tensor * ids) {
    static const bool bf16 = [] {
        const char * s = getenv("QUACTLIZE_KPACK_COMPUTE");
        if (!s || !*s || !strcmp(s, "fp16")) return false;
        if (!strcmp(s, "bf16")) return true;
        GGML_ABORT("[quactlize] QUACTLIZE_KPACK_COMPUTE must be fp16 or bf16");
    }();
    // BF16 applies to grouped projections only. Dense keeps its FP16 policy.
    return bf16 && ids ? QK_COMPUTE_BF16 : QK_COMPUTE_F16;
}
const char * compute_name(int type) { return type == QK_COMPUTE_BF16 ? "BF16" : "FP16"; }

template<class T> __global__ void gather(const float * src, T * dst, const int32_t * ids, int width, int64_t stride) {
    int64_t row = blockIdx.x;
    int64_t from = ids ? ids[row] : row;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[row * width + i] = T(src[from * stride + i]);
}
template<class T> __global__ void scatter(const T * src, float * dst, const int32_t * ids, int width, int64_t stride) {
    int64_t row = blockIdx.x;
    int64_t to = ids ? ids[row] : row;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[to * stride + i] = float(src[row * width + i]);
}

size_t align256(size_t n) {
    GGML_ASSERT(n <= SIZE_MAX - 255);
    return (n + 255) & ~size_t(255);
}
using Key = std::array<uint64_t, 28>;
struct Plan {
    const ggml_quactlize_execution_api * api;
    ggml_quactlize_artifact art{};
    qks_choice_v1 choice{};
    qkg_call_v1 gemv{};
    qkg_config_v1 gemv_config{};
    qkg_q4_decode_config_v1 q4_config{};
    qks_smallm_choice_v1 smallm{};
    void * handle = nullptr;
    void * a = nullptr;
    void * out = nullptr;
    uint16_t * scale = nullptr;
    uint16_t * zero = nullptr;
    uint64_t sf_plane_bytes = 0, units_bytes = 0;
    int32_t * ids_src = nullptr;
    int32_t * ids_dst = nullptr;
    int32_t * bounds = nullptr;
    bool direct = false, legacy = false, sf = false, scale_resident = false;
    bool indexed = false;
    bool q4_decode = false, q4_tc = false;
    bool reuse = false, table_tc = false, matched = false;
    bool dense_io = false;
    int compute = QK_COMPUTE_F16;
    void * full_handle = nullptr;
    int sf_config = -1;
    qzd_call_v1 expansion{};
    int rows = 0, tokens = 0, topk = 0;
    ~Plan() {
        if (handle) api->destroy(handle);
        if (full_handle) api->full_destroy(full_handle);
    }
};
bool apply_matched(Plan & p, const qks_smallm_choice_v2 & selected) {
    const auto & c = selected.base;
    if (selected.version != 2 || selected.size != sizeof(selected) || selected.compute_type != p.compute ||
        c.version != 1 || c.size != sizeof(c) || c.policy < QKS_MATCHED_EXACT || c.policy > QKS_Q8_VECTOR_MEASURED ||
        c.kind < QKS_SMALLM_TC || c.kind > QKS_SMALLM_Q4) return false;
    if (c.policy == QKS_Q8_VECTOR_MEASURED && (c.kind != QKS_SMALLM_SIMT || c.simt.variant < 4 || c.simt.variant > 5)) return false;
    if (c.kind == QKS_SMALLM_TC && (c.tc.version != 1 || c.tc.size != sizeof(c.tc) || !c.tc.ticket || c.tc.policy != c.policy)) return false;
    if (c.kind == QKS_SMALLM_SIMT && (c.simt.version != 1 || c.simt.size != sizeof(c.simt))) return false;
    if (c.kind == QKS_SMALLM_Q4 && (selected.q4.version != 1 || selected.q4.size != sizeof(selected.q4))) return false;
    p.smallm = c;
    p.matched = true;
    p.direct = c.kind != QKS_SMALLM_TC;
    p.table_tc = c.kind == QKS_SMALLM_TC;
    p.reuse = c.kind == QKS_SMALLM_SIMT;
    p.q4_decode = c.kind == QKS_SMALLM_Q4;
    if (p.table_tc) p.choice = c.tc;
    if (p.q4_decode) p.q4_config = selected.q4;
    return true;
}
const char * matched_name(int policy) {
    if (policy == QKS_MATCHED_EXACT) return "MATCHED_EXACT";
    if (policy == QKS_MATCHED_BUCKET) return "MATCHED_BUCKET_PREDICTED";
    if (policy == QKS_Q8_VECTOR_MEASURED) return "Q8_VECTOR_MEASURED";
    return "MATCHED_ROUTER_MINIMAX";
}
struct Scratch { void * pointer = nullptr; size_t capacity = 0; };
struct PairedWeights {
    qkg_gate_up_layout_v1 layout{};
    uint8_t * low = nullptr;
    uint8_t * units = nullptr;
};
struct SharedPlan {
    PairedWeights * weights = nullptr;
    qkg_gate_up_call_v1 call{};
    qkg_gate_up_config_v1 config{};
};
struct MoePlan {
    const ggml_quactlize_execution_api * api;
    Plan * gate, * up, * down;
    void * handle = nullptr;
    uint32_t simt_mask = 0;
    PairedWeights * paired = nullptr;
    ~MoePlan() { if (handle) api->moe_destroy(handle); }
};

struct Execution {
    const ggml_quactlize_execution_api * api;
    void * runtime = nullptr;
    int device;
    std::map<Key, std::unique_ptr<Plan>> plans;
    std::map<std::array<uint64_t, 9>, PairedWeights> paired_weights;
    std::map<std::pair<Key, Key>, SharedPlan> shared_plans;
    std::map<std::tuple<Plan *,Plan *,Plan *,const ggml_tensor *,const ggml_tensor *>, std::unique_ptr<MoePlan>> moe_plans;
    std::map<cudaStream_t, Scratch> scratch;
    std::vector<void *> allocations;

    Execution(const ggml_quactlize_execution_api * api, int device) : api(api), device(device) {
        if (api->open(api->root, &runtime) != QKS_OK) GGML_ABORT("[quactlize] dispatch open: %s", api->error());
        if (api->enable_jit) {
            qks_jit_options_v1 options{1, sizeof(options), getenv("QUACTLIZE_KPACK_JIT_PYTHON"),
                getenv("QUACTLIZE_KPACK_JIT_HELPER"), getenv("PPU_SDK"), getenv("QUACTLIZE_KPACK_JIT_CACHE")};
            if (api->enable_jit(runtime, &options) != QKS_OK)
                GGML_ABORT("[quactlize] JIT setup requires absolute PYTHON/HELPER/PPU_SDK/CACHE paths: %s", api->error());
        }
    }
    ~Execution() {
        ggml_cuda_set_device(device);
        // Context teardown only. Retain previous scratch generations while
        // their handles or CUDA graphs can still reference them.
        for (auto & s : scratch) CUDA_CHECK(cudaStreamSynchronize(s.first));
        moe_plans.clear(); plans.clear();
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
    uint8_t * private_storage(size_t need) {
        void * pointer = nullptr;
        CUDA_CHECK(cudaMalloc(&pointer, need));
        allocations.push_back(pointer);
        return static_cast<uint8_t *>(pointer);
    }
};

Execution * execution(ggml_backend_cuda_context & ctx) {
    auto api = ggml_quactlize_execution_library();
    if (!api) return nullptr;
    if (!ctx.quactlize_execution) ctx.quactlize_execution = std::make_shared<Execution>(api, ctx.device);
    return static_cast<Execution *>(ctx.quactlize_execution.get());
}

void paired_log(const char * tensor, const char * op, int q, int tokens, int experts,
                int compute, const qkg_gate_up_config_v1 & config) {
    GGML_LOG_INFO("[quactlize-paired-plan] tensor=%s op=%s q=%d tokens=%d n=512 k=2048 experts=%d"
        " backend=%s split=%d tile_m=%d warps=%d activation=%s layout=0x%016" PRIx64 "\n",
        tensor,op,q,tokens,experts,config.backend==QKG_GATE_UP_SIMT?"simt":"tc",
        config.split,config.tile_m,config.warps,compute_name(compute),QKG_GATE_UP_N4_V1);
}

PairedWeights * paired_weights(Execution & owner, const ggml_quactlize_artifact & gate,
                              const ggml_quactlize_artifact * up, cudaStream_t stream,
                              const qkg_gate_up_call_v1 & call, const qkg_gate_up_config_v1 & config) {
    const int n = up ? gate.n : gate.n/2;
    if (gate.high || (up && (up->high || up->qtype!=gate.qtype || up->n!=gate.n ||
        up->k!=gate.k || up->experts!=gate.experts ||
        memcmp(&gate.arrangement,&up->arrangement,sizeof(gate.arrangement))))) return nullptr;
    std::array<uint64_t,9> key{uint64_t(uintptr_t(gate.low)),uint64_t(uintptr_t(gate.units)),
        uint64_t(uintptr_t(gate.ready)),uint64_t(uintptr_t(up?up->low:nullptr)),
        uint64_t(uintptr_t(up?up->units:nullptr)),uint64_t(uintptr_t(up?up->ready:nullptr)),
        uint64_t(n),uint64_t(gate.k),uint64_t(gate.experts)};
    auto found=owner.paired_weights.find(key);
    if (found!=owner.paired_weights.end()) return &found->second;
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    GGML_ASSERT(capture==cudaStreamCaptureStatusNone);
    PairedWeights result;
    if (owner.api->paired_layout(gate.qtype,&result.layout)!=QKG_OK ||
        memcmp(&gate.arrangement,&result.layout.packing,sizeof(gate.arrangement))) return nullptr;
    qkg_sizes_v1 sizes{};
    if (owner.api->paired_query(&call,&config,&result.layout,&sizes)!=QKG_OK || sizes.high_bytes)
        GGML_ABORT("[quactlize] paired plane size query failed");
    result.low=owner.private_storage(sizes.low_bytes);
    result.units=owner.private_storage(sizes.units_bytes);
    ggml_quactlize_wait_ready(gate,stream);
    if (up) ggml_quactlize_wait_ready(*up,stream);
    qkg_gate_up_repack_v1 repack{1,sizeof(repack),gate.qtype,n,int(gate.k),int(gate.experts),int(!up),
        gate.low,gate.units,up?up->low:nullptr,up?up->units:nullptr,result.low,result.units};
    if (owner.api->paired_repack(&repack,&result.layout,stream)!=QKG_OK)
        GGML_ABORT("[quactlize] canonical-to-paired GPU repack failed");
    // One-time setup, before capture. All later graphs may read the immutable
    // auxiliary planes without an event dependency or host wait per token.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    owner.scratch.try_emplace(stream);
    GGML_LOG_INFO("[quactlize-paired-weights] q=%d n=%d k=%" PRId64 " experts=%" PRId64
        " bytes=%" PRIu64 " canonical_retained=1 cache_format_unchanged=1\n",
        gate.qtype,n,gate.k,gate.experts,sizes.low_bytes+sizes.units_bytes);
    return &owner.paired_weights.emplace(key,result).first->second;
}

qkg_gate_up_call_v1 paired_call(int q, int experts, int tokens, int compute, bool indexed) {
    qkg_gate_up_call_v1 d{};d.version=1;d.size=sizeof(d);
    d.input.version=2;d.input.size=sizeof(d.input);d.input.compute_type=compute;
    d.output_type=QKG_F32;d.round_projection=indexed;
    auto & c=d.input.call;c.version=1;c.size=sizeof(c);c.qtype=q;
    c.n=512;c.k=2048;c.experts=experts;c.rows=tokens*(indexed?8:1);
    c.mode=indexed?QKG_INDEXED:QKG_DENSE;c.input_type=QKG_F32;c.channels=1;c.topk=indexed?8:1;
    c.a_row_stride=c.k;c.a_token_stride=c.k;c.ids_stride=c.topk;c.out_row_stride=c.n;
    return d;
}

qzd_call_v1 expansion_call(ggml_quactlize_artifact const & art, cudaStream_t stream) {
    qzd_call_v1 c{};
    c.version=1; c.size=sizeof(c); c.qtype=art.qtype; c.n=art.n; c.k=art.k; c.experts=art.experts;
    c.low=art.low; c.high=art.high; c.units=art.units; c.stream=stream;
    qkg_call_v1 shape{};
    shape.version=1; shape.size=sizeof(shape); shape.qtype=art.qtype;
    shape.n=art.n; shape.k=art.k; shape.experts=art.experts; shape.rows=1;
    shape.mode=art.experts==1 ? QKG_DENSE:QKG_GROUPED;
    shape.input_type=QKG_F16; shape.channels=1; shape.topk=1; shape.a_row_stride=art.k; shape.out_row_stride=art.n;
    qkg_config_v1 recipe{1,sizeof(recipe),16,4,1}; qkg_sizes_v1 sizes{};
    auto api=ggml_quactlize_execution_library();
    if (api->gemv_query(&shape,&recipe,&art.arrangement,&sizes)!=QKG_OK)
        GGML_ABORT("[quactlize] expansion size query failed");
    c.low_bytes=sizes.low_bytes; c.high_bytes=sizes.high_bytes; c.unit_bytes=sizes.units_bytes;
    return c;
}

unsigned prefill_mask(ggml_quactlize_execution_api const * api, bool grouped) {
    if (!api->dequant) return 1;
    const char * sdk=getenv("PPU_SDK");
    if (!api->full_prepare || !sdk) return 3;
    if (grouped) return getenv("QUACTLIZE_KPACK_DEEPGEMM_HELPER") && getenv("QUACTLIZE_KPACK_JIT_PYTHON") ? 7:3;
    std::string path=std::string(sdk)+"/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so";
    return access(path.c_str(),R_OK)==0 ? 7:3;
}

Key make_key(const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * ids,
             const ggml_tensor * output, const ggml_quactlize_artifact & art, cudaStream_t stream, int slot) {
    return {uint64_t(uintptr_t(weight)), uint64_t(uintptr_t(art.low)), uint64_t(uintptr_t(art.ready)),
        uint64_t(uintptr_t(input->data)), uint64_t(uintptr_t(output->data)),
        uint64_t(uintptr_t(ids ? ids->data : nullptr)), uint64_t(uintptr_t(stream)),
        uint64_t(art.qtype), uint64_t(art.n), uint64_t(art.k), uint64_t(art.experts),
        uint64_t(input->ne[1]), uint64_t(input->ne[2]), uint64_t(input->ne[3]),
        input->nb[1], input->nb[2], output->nb[1], output->nb[2],
        uint64_t(ids ? ids->ne[0] : 0), uint64_t(ids ? ids->ne[1] : 0), ids ? ids->nb[1] : 0,
        uint64_t(input->type), uint64_t(output->type), uint64_t(route_mode()),
        uint64_t(output->ne[1]), uint64_t(output->ne[2]), uint64_t(slot), uint64_t(compute_type(ids))};
}

Plan & prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
               const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output, int slot = 0) {
    auto & owner = *execution(ctx);
    ggml_quactlize_artifact art{};
    GGML_ASSERT(ggml_quactlize_artifact_for(weight, &art));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(output));
    GGML_ASSERT(input->ne[0] == art.k && output->ne[0] == art.n);
    cudaStream_t stream = ctx.stream();
    auto key = make_key(weight, input, ids, output, art, stream, slot);
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
    p->compute = compute_type(ids);
    if (p->compute == QK_COMPUTE_BF16 && !owner.api->query_compute)
        GGML_ABORT("[quactlize] BF16 requested but the runtime has no explicit BF16 compute interface");
    const RouteMode mode = route_mode();
    // Auto uses an exact measured SIMT recipe when present; otherwise retain
    // the selected tensor-core path. Forced FQ/SF never consult this policy.
    if (art.qtype == GGML_TYPE_Q8_0 && mode == RouteMode::Fq)
        GGML_ABORT("[quactlize] Q8_0 requires W8A16 with resident FP16 scale");
    if (mode == RouteMode::Auto || mode == RouteMode::Gemv) {
        auto & c = p->gemv;
        c.version = 1; c.size = sizeof(c); c.qtype = art.qtype; c.n = art.n; c.k = art.k;
        c.experts = art.experts; c.rows = p->rows; c.mode = ids ? QKG_INDEXED : QKG_DENSE;
        c.input_type = QKG_F32; c.channels = ids ? input->ne[1] : 1; c.topk = topk;
        c.a_row_stride = input->nb[1] / sizeof(float); c.a_token_stride = ids ? input->nb[2] / sizeof(float) : 0;
        c.ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0; c.out_row_stride = art.n;
        c.a = input->data; c.low = art.low; c.high = art.high; c.units = art.units;
        c.ids = ids ? (const int32_t *) ids->data : nullptr; c.output = (float *) output->data; c.stream = stream;
        if (mode == RouteMode::Auto && tokens <= 8 && c.input_type == QKG_F32 && owner.api->query_smallm_matched) {
            qkg_simt_call_v2 typed{2, sizeof(typed), c, p->compute};
            qks_smallm_choice_v2 selected{};
            int rc = owner.api->query_smallm_matched(owner.runtime, &typed, &art.arrangement, &selected);
            if (rc != QKS_OK && rc != QKS_MISS)
                GGML_ABORT("[quactlize] %s: matched decode query: %s", weight->name, owner.api->error());
            if (rc == QKS_OK) {
                if (!apply_matched(*p, selected))
                    GGML_ABORT("[quactlize] %s: invalid matched decode identity", weight->name);
                GGML_LOG_INFO("[quactlize-decode] tensor=%s table=%s kind=%d source=%dx%dx%d activation=%s\n",
                    weight->name, matched_name(p->smallm.policy), p->smallm.kind,
                    p->smallm.source_tokens, p->smallm.source_n, p->smallm.source_k, compute_name(p->compute));
            }
        }
        if (!p->matched) {
            if (p->compute == QK_COMPUTE_BF16 && tokens <= 8) {
                qkg_simt_call_v2 typed{2, sizeof(typed), c, p->compute};
                int rc = owner.api->query_smallm_compute(owner.runtime, &typed, &art.arrangement, &p->smallm);
                if (rc != QKS_OK && rc != QKS_MISS)
                    GGML_ABORT("[quactlize] %s: BF16 decode query: %s", weight->name, owner.api->error());
                if (rc == QKS_OK) {
                    p->direct = p->reuse = p->smallm.kind == QKS_SMALLM_SIMT;
                    p->table_tc = !p->direct;
                    if (p->table_tc) p->choice = p->smallm.tc;
                }
                if (p->reuse && p->smallm.policy == QKS_COMPUTE_INITIAL && mode == RouteMode::Auto &&
                    art.qtype == GGML_TYPE_Q4_K && owner.api->q4_select_compute) {
                    qkg_sizes_v1 sizes{};
                    int q4_status = owner.api->q4_select_compute(&typed, &art.arrangement, &p->q4_config, &sizes);
                    if (q4_status != QKG_OK && q4_status != QKG_SHAPE)
                        GGML_ABORT("[quactlize] %s: BF16 Q4 decode selection failed rc=%d", weight->name, q4_status);
                    if (q4_status == QKG_OK) {
                        p->q4_decode = true;
                        p->reuse = false;
                    }
                }
            } else if (p->compute == QK_COMPUTE_BF16) {
                // Prefill uses the grouped TC compute interface below.
            } else if (mode == RouteMode::Auto && art.qtype == GGML_TYPE_Q4_K && owner.api->q4_select) {
                qkg_sizes_v1 sizes{};
                int rc = owner.api->q4_select(&c, &art.arrangement, &p->q4_config, &sizes);
                if (rc != QKG_OK && rc != QKG_SHAPE)
                    GGML_ABORT("[quactlize] %s: Q4 decode selection failed rc=%d", weight->name, rc);
                p->direct = p->q4_decode = rc == QKG_OK;
            } else if (mode == RouteMode::Auto && owner.api->query_smallm) {
                int rc = owner.api->query_smallm(owner.runtime, &c, &art.arrangement, &p->smallm);
                if (rc != QKS_OK && rc != QKS_MISS)
                    GGML_ABORT("[quactlize] %s: decode table query failed: %s", weight->name, owner.api->error());
                if (rc == QKS_OK) {
                    p->direct = p->reuse = p->smallm.kind == QKS_SMALLM_SIMT;
                    p->table_tc = !p->direct;
                    if (p->table_tc) p->choice = p->smallm.tc;
                    GGML_LOG_INFO("[quactlize-decode] tensor=%s table=%s kind=%s source=%dx%dx%d\n",
                        weight->name, p->smallm.policy == QKS_SMALLM_EXACT ? "EXACT" : "BUCKET_PREDICTED",
                        p->direct ? "SIMT" : "TC", p->smallm.source_tokens, p->smallm.source_n, p->smallm.source_k);
                } else {
                    GGML_LOG_DEBUG("[quactlize-decode] tensor=%s table=MISS retain_existing=1\n", weight->name);
                }
            } else {
                p->direct = ggml_quactlize_gemv_config(c, &p->gemv_config);
            }
        }
        if (!p->direct && mode == RouteMode::Gemv)
            GGML_ABORT("[quactlize] %s: forced GEMV has no measured recipe for this request", weight->name);
    }
    if (p->direct) {
        auto & c = p->gemv;
        qkg_sizes_v1 sizes{};
        qkg_simt_call_v2 typed{2, sizeof(typed), c, p->compute};
        int rc;
        if (p->reuse) {
            rc = p->compute == QK_COMPUTE_BF16 ?
                owner.api->simt_query_compute(&typed, &p->smallm.simt, &art.arrangement, &sizes) :
                owner.api->simt_query(&c, &p->smallm.simt, &art.arrangement, &sizes);
        } else if (p->q4_decode) {
            if (p->matched) {
                sizes = p->smallm.sizes;
                rc = QKG_OK;
            } else {
                rc = p->compute == QK_COMPUTE_BF16 ?
                    owner.api->q4_select_compute(&typed, &art.arrangement, &p->q4_config, &sizes) :
                    owner.api->q4_select(&c, &art.arrangement, &p->q4_config, &sizes);
            }
        } else {
            GGML_ASSERT(p->compute == QK_COMPUTE_F16);
            rc = owner.api->gemv_query(&c, &p->gemv_config, &art.arrangement, &sizes);
        }
        if (rc != QKG_OK) GGML_ABORT("[quactlize] %s: GEMV query failed rc=%d", weight->name, rc);
        c.workspace = slot ? owner.private_storage(sizes.workspace_bytes) : owner.storage(stream, sizes.workspace_bytes);
        c.workspace_bytes = sizes.workspace_bytes;
        if (p->reuse) {
            auto f = p->smallm.simt;
            GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=gemv reader=%s q=%d rows=%d n=%" PRId64 " k=%" PRId64
                " variant=%d columns=%d warps=%d values=%d split=%d policy=%d activation=%s scale_resident=%d experts=%d channels=%d topk=%d\n",
                weight->name, ids ? "grouped" : "dense", f.variant >= 4 ? "simt-q8-vector" : "simt-reuse", art.qtype, p->rows, art.n, art.k,
                f.variant, f.columns, f.warps, f.values, f.split, p->smallm.policy, compute_name(p->compute), int(art.qtype == GGML_TYPE_Q8_0), c.experts, c.channels, c.topk);
        } else if (p->q4_decode) {
            auto f = p->q4_config;
            GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=gemv-q4-s1 q=%d rows=%d n=%" PRId64 " k=%" PRId64
                " reader=%d variant=%d warps=%d values=%d columns=%d split=1 selection=%s policy=%d activation=%s experts=%d channels=%d topk=%d\n",
                weight->name, ids ? "grouped" : "dense", art.qtype, p->rows, art.n, art.k,
                f.reader, f.variant, f.warps, f.values, f.columns,
                p->matched ? matched_name(p->smallm.policy) : p->compute == QK_COMPUTE_BF16 ? "INITIAL_COMPUTE" : "MEASURED_DECODE",
                p->matched ? p->smallm.policy : p->compute == QK_COMPUTE_BF16 ? QKS_COMPUTE_INITIAL : QKS_DECODE_MEASURED, compute_name(p->compute), c.experts, c.channels, c.topk);
        } else GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=gemv q=%d rows=%d n=%" PRId64 " k=%" PRId64
            " columns=%d warps=%d split=%d selection=MEASURED_GEMV_POOL activation=FP16 scale_resident=%d\n", weight->name, ids ? "grouped" : "dense",
            art.qtype, p->rows, art.n, art.k, p->gemv_config.columns, p->gemv_config.warps, p->gemv_config.split,
            int(art.qtype == GGML_TYPE_Q8_0));
    } else {
        qks_request_v1 r{1, sizeof(r), art.qtype, ids ? QK_GROUPED_FQ : QK_DENSE_FQ,
            p->rows, int(art.n), int(art.k), int(art.experts), int(tokens), art.arrangement.mapping_id};
        p->dense_io = !ids && tokens <= 8 && owner.api->query_dense_io && owner.api->prepare_dense_io;
        auto query_tc = [&](bool decode = false) {
            if (p->compute == QK_COMPUTE_BF16)
                return owner.api->query_compute(owner.runtime, &r, p->compute, p->dense_io ? QKD_F32 : 0,
                    int(decode), &p->choice);
            if (p->dense_io)
                return owner.api->query_dense_io(owner.runtime, &r, QKD_F32, int(decode), &p->choice);
            return decode ? owner.api->query_decode(owner.runtime, &r, &p->choice) :
                            owner.api->query(owner.runtime, &r, &p->choice);
        };
        int status = p->table_tc ? QKS_OK : QKS_MISS;
        if (!p->table_tc && p->compute == QK_COMPUTE_F16 && mode == RouteMode::Auto && art.qtype == GGML_TYPE_Q4_K && owner.api->query_decode) {
            status = query_tc(true);
            if (status != QKS_OK && status != QKS_MISS)
                GGML_ABORT("[quactlize] %s: decode TC selection: %s", weight->name, owner.api->error());
            p->q4_tc = status == QKS_OK;
        }
        int prefill_choice=-1;
        qks_prefill_choice_v1 prefill{};
        if (!p->table_tc && !p->q4_tc && mode==RouteMode::Auto && tokens>1 && art.qtype!=GGML_TYPE_Q8_0) {
            if (owner.api->prefill_choice) {
                unsigned mask=prefill_mask(owner.api,ids!=nullptr);
                if (p->compute==QK_COMPUTE_BF16 && !owner.api->dequant_compute) mask=1;
                int rc=p->compute==QK_COMPUTE_BF16 ? owner.api->prefill_compute(&r,mask,p->compute,&prefill) :
                    owner.api->prefill_choice(&r,mask,&prefill);
                if (rc!=QKS_OK && rc!=QKS_MISS) GGML_ABORT("[quactlize] prefill policy query failed rc=%d",rc);
                if (rc==QKS_OK) prefill_choice=prefill.route;
            } else if (p->compute==QK_COMPUTE_F16) prefill_choice=ggml_quactlize_prefill_route(r);
        }
        if (prefill_choice==2) {
            qkp_call_v1 call{};
            call.version=1; call.size=sizeof(call); call.weight=expansion_call(art,stream);
            call.weight.operation=1; call.weight.config=prefill.dequant_config; call.m=p->rows; call.device=ctx.device;
            GGML_ASSERT(ggml_nelements(input)/art.k<=INT32_MAX);
            call.a_rows=ggml_nelements(input)/art.k;
            call.a=static_cast<float const*>(input->data); call.output=static_cast<float*>(output->data);
            call.a_stride=input->nb[1]/sizeof(float); call.output_stride=output->nb[1]/sizeof(float);
            uint64_t bytes=0;
            if (owner.api->full_query(&call,&art.arrangement,&bytes)!=QKG_OK)
                GGML_ABORT("[quactlize] %s: full-BF16 workspace query failed",weight->name);
            size_t indices=ids ? align256(size_t(p->rows)*sizeof(int32_t)):0;
            size_t bounds=ids ? align256(size_t(art.experts+1)*sizeof(int32_t)):0;
            size_t head=2*indices+bounds;
            GGML_ASSERT(bytes<=SIZE_MAX-head);
            uint8_t * storage=owner.storage(stream,head+bytes);
            if (ids) {
                p->ids_src=reinterpret_cast<int32_t*>(storage);
                p->ids_dst=reinterpret_cast<int32_t*>(storage+indices);
                p->bounds=reinterpret_cast<int32_t*>(storage+2*indices);
                call.src_rows=p->ids_src;call.dst_rows=p->ids_dst;call.offsets=p->bounds;
            }
            call.workspace=storage+head;call.workspace_bytes=bytes;
            qkp_options_v1 options{1,sizeof(options),getenv("PPU_SDK"),getenv("QUACTLIZE_KPACK_JIT_PYTHON"),
                getenv("QUACTLIZE_KPACK_DEEPGEMM_HELPER")};
            if (owner.api->full_prepare(&call,&art.arrangement,&options,&p->full_handle)!=QKG_OK)
                GGML_ABORT("[quactlize] %s: full-BF16 prepare: %s",weight->name,owner.api->full_error());
            const char * image=owner.api->full_image(p->full_handle);
            GGML_ASSERT(image && !strchr(image,'\n') && !strchr(image,'"'));
            GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=full-bf16 q=%d rows=%d n=%" PRId64 " k=%" PRId64 " experts=%" PRId64 " dequant=%d knot_tokens=%d predicted=%d workspace=%" PRIu64 " cost_scope=ISOLATED_COMPONENT_SUM\n",
                weight->name,ids ? "grouped":"dense",art.qtype,p->rows,art.n,art.k,art.experts,prefill.dequant_config,prefill.measured_tokens,prefill.predicted,bytes);
            GGML_LOG_INFO("[quactlize-prefill-image] tensor=%s rows=%d provider=%s image=\"%s\"\n",
                weight->name,p->rows,ids ? "deepgemm":"cublas",image);
            auto * result=p.get();owner.plans.emplace(key,std::move(p));return *result;
        }
        if (prefill_choice==1 && owner.api->prefill_choice) p->sf_config=prefill.dequant_config;
        if (p->compute==QK_COMPUTE_BF16 && p->sf_config>=0 && p->sf_config!=4 && p->sf_config!=5) p->sf_config=0;
        if (p->compute==QK_COMPUTE_BF16 && prefill_choice>=0)
            GGML_LOG_INFO("[quactlize-prefill] tensor=%s route=%d proposal=F16_COMPONENT_TRANSFER bf16_measured=0\n",weight->name,prefill_choice);
        if (art.qtype == GGML_TYPE_Q8_0) {
            r.route = ids ? QK_GROUPED_SF : QK_DENSE_SF;
            if (!p->table_tc) status = query_tc();
            if (status != QKS_OK)
                GGML_ABORT("[quactlize] %s: Q8 W8A16 selection: %s", weight->name, owner.api->error());
            p->sf = p->scale_resident = true;
        } else if (mode == RouteMode::Sf || prefill_choice == 1 ||
                   (p->table_tc && !strncmp(p->choice.parent, "sf", 2))) {
            r.route = ids ? QK_GROUPED_SF : QK_DENSE_SF;
            if (!p->table_tc) status = query_tc();
            if (status == QKS_OK) {
                // Size query only. Values are expanded on the compute stream
                // before every GEMM; no per-weight scale allocation or event.
                qkg_call_v1 meta{};
                meta.version=1; meta.size=sizeof(meta); meta.qtype=art.qtype;
                meta.n=art.n; meta.k=art.k; meta.experts=art.experts; meta.rows=1;
                meta.mode=art.experts == 1 ? QKG_DENSE : QKG_GROUPED;
                meta.input_type=QKG_F16; meta.channels=1; meta.topk=1;
                meta.a_row_stride=art.k; meta.out_row_stride=art.n;
                qkg_config_v1 config{1, sizeof(config), 16, 4, 1};
                qkg_sizes_v1 sizes{};
                if (owner.api->gemv_query(&meta, &config, &art.arrangement, &sizes) != QKG_OK ||
                    !sizes.sf_plane_bytes || sizes.sf_plane_bytes > SIZE_MAX / 4)
                    GGML_ABORT("[quactlize] %s: SF scratch size query failed", weight->name);
                p->sf = true;
                p->sf_plane_bytes = sizes.sf_plane_bytes;
                p->units_bytes = sizes.units_bytes;
            } else if (status != QKS_OK && status != QKS_MISS) {
                GGML_ABORT("[quactlize] %s: SF selection failed: %s", weight->name, owner.api->error());
            }
        }
        if (!p->sf && !p->q4_tc && !p->table_tc) {
            if (mode == RouteMode::Sf) GGML_ABORT("[quactlize] %s: forced SF has no admitted resources/module", weight->name);
            r.route = ids ? QK_GROUPED_FQ : QK_DENSE_FQ;
            status = query_tc();
        }
        if (status == QKS_MISS) {
            if (p->compute == QK_COMPUTE_BF16)
                GGML_ABORT("[quactlize] %s: BF16 compute unavailable; refusing FP16 fallback: %s", weight->name, owner.api->error());
            p->legacy = true;
            GGML_LOG_INFO("[quactlize] %s: native policy miss, retain legacy K-pack FQ (%s)\n", weight->name, owner.api->error());
        } else {
            if (status != QKS_OK) GGML_ABORT("[quactlize] %s: native selection: %s", weight->name, owner.api->error());
            size_t a_bytes = p->dense_io ? 0 : align256(size_t(p->rows) * art.k * sizeof(half));
            size_t out_bytes = p->dense_io ? 0 : align256(size_t(p->rows) * art.n * sizeof(half));
            size_t index_bytes = ids ? align256(size_t(p->rows) * sizeof(int32_t)) : 0;
            size_t bound_bytes = ids ? align256(size_t(art.experts + 1) * sizeof(int32_t)) : 0;
            size_t head = a_bytes + out_bytes + 2 * index_bytes + bound_bytes;
            size_t plane_bytes = align256(p->sf_plane_bytes);
            GGML_ASSERT(2 * plane_bytes <= SIZE_MAX - head);
            size_t scale_offset = head;
            head += 2 * plane_bytes;
            GGML_ASSERT(p->choice.workspace_bytes <= SIZE_MAX - head);
            uint8_t * storage = slot ? owner.private_storage(head + p->choice.workspace_bytes) :
                owner.storage(stream, head + p->choice.workspace_bytes);
            if (!p->dense_io) {
                p->a = storage; p->out = storage + a_bytes;
            }
            if (p->sf && !p->scale_resident) {
                p->scale = (uint16_t *) (storage + scale_offset);
                p->zero = (uint16_t *) (storage + scale_offset + plane_bytes);
                if (p->sf_config>=0) {
                    p->expansion=expansion_call(art,stream);p->expansion.operation=0;p->expansion.config=p->sf_config;
                    p->expansion.output=p->scale;p->expansion.zero=p->zero;p->expansion.output_bytes=p->sf_plane_bytes;
                }
            }
            if (ids) {
                p->ids_src = (int32_t *) (storage + a_bytes + out_bytes);
                p->ids_dst = (int32_t *) ((uint8_t *) p->ids_src + index_bytes);
                p->bounds = (int32_t *) ((uint8_t *) p->ids_dst + index_bytes);
            }
            qk_call_v1 c{};
            c.version = 1; c.size = sizeof(c); c.m = p->rows; c.n = art.n; c.k = art.k; c.experts = art.experts;
            c.group_size = art.arrangement.group_size; c.device = p->choice.device; c.compute_units = p->choice.compute_units;
            GGML_ASSERT(c.device == ctx.device);
            c.mapping_id = art.arrangement.mapping_id; c.a = p->dense_io ? input->data : p->a;
            c.low = art.low; c.high = art.high;
            c.metadata = p->sf && !p->scale_resident ? (const void *) p->scale : (const void *) art.units;
            c.zero = p->zero; c.output = p->dense_io ? output->data : p->out; c.offsets_device = p->bounds;
            c.workspace = p->choice.workspace_bytes ? storage + head : nullptr;
            c.workspace_bytes = p->choice.workspace_bytes; c.stream = stream;
            qkd_dense_call_v1 typed{1,sizeof(typed),c,QKD_F32,QKD_F32};
            int prepare_rc;
            if (p->compute == QK_COMPUTE_BF16) {
                qkd_dense_call_v2 dense{2,sizeof(dense),typed,p->compute};
                qk_compute_device_call_v4 grouped{4,sizeof(grouped),{2,sizeof(qk_device_call_v2),c,int(tokens),0},p->compute,
                    art.qtype==GGML_TYPE_Q8_0 ? QK_METADATA_F16 : QK_METADATA_BF16};
                prepare_rc = p->dense_io ? owner.api->prepare_dense_compute(owner.runtime, &p->choice, &dense, &p->handle) :
                    owner.api->prepare_compute(owner.runtime, &p->choice, &grouped, &p->handle);
            } else prepare_rc = p->dense_io ? owner.api->prepare_dense_io(owner.runtime, &p->choice, &typed, &p->handle) :
                    owner.api->prepare(owner.runtime, &p->choice, &c, &p->handle);
            if (prepare_rc != QKS_OK)
                GGML_ABORT("[quactlize] %s: native prepare: %s", weight->name, owner.api->error());
            if (ids && owner.api->bind_indexed) {
                qk_llama_indexed_v1 io{1,sizeof(io),p->tokens,p->topk,int(input->ne[1]),0,
                    int64_t(ids->nb[1]/sizeof(int32_t)),int64_t(input->nb[1]/sizeof(float)),
                    int64_t(input->nb[2]/sizeof(float)),int64_t(output->nb[1]/sizeof(float)),
                    static_cast<int32_t const*>(ids->data),static_cast<float const*>(input->data),
                    static_cast<float*>(output->data),p->ids_dst};
                int rc=owner.api->bind_indexed(p->handle,&io);
                if (rc!=QKS_OK && rc!=QKS_MISS)
                    GGML_ABORT("[quactlize] %s: fused indexed binding failed rc=%d",weight->name,rc);
                p->indexed=rc==QKS_OK;
                GGML_LOG_INFO("[quactlize-adapter] tensor=%s indexed_prepare=%d reduce_scatter=%d\n",
                    weight->name,int(p->indexed),int(p->indexed));
            }
            GGML_LOG_INFO("[quactlize-plan] tensor=%s op=%s route=%s q=%d rows=%d n=%" PRId64 " k=%" PRId64
                " parent=%s build=%s algorithm=%d split=%d grid=%d policy=%d prefill_choice=%d activation=%s scale_resident=%d\n", weight->name, ids ? "grouped" : "dense",
                p->sf ? "sf" : "fq", art.qtype, p->rows, art.n, art.k, p->choice.parent, p->choice.build_key,
                p->choice.algorithm, p->choice.split, p->choice.grid, p->choice.policy, prefill_choice, compute_name(p->compute), int(p->scale_resident));
            if (p->dense_io)
                GGML_LOG_INFO("[quactlize-adapter] tensor=%s dense_io=FP32 standalone_adapters=0\n", weight->name);
        }
    }
    auto * result = p.get();
    owner.plans.emplace(key, std::move(p));
    return *result;
}
SharedPlan * prepare_shared(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int start, bool create) {
    auto * owner=execution(ctx);
    if (!owner || !owner->api->paired_select || route_mode()!=RouteMode::Auto ||
        !ctx.stream_context().concurrent_events.empty()) return nullptr;
    auto match=quactlize::llama::match_shared_gate_up(graph,start);
    if (!match.count) return nullptr;
    ggml_quactlize_artifact gate{},up{};
    if (!ggml_quactlize_artifact_for(match.gate->src[0],&gate) ||
        !ggml_quactlize_artifact_for(match.up->src[0],&up)) return nullptr;
    auto * input=match.gate->src[1];auto stream=ctx.stream();
    auto key=std::make_pair(make_key(match.gate->src[0],input,nullptr,match.output,gate,stream,0),
        make_key(match.up->src[0],input,nullptr,match.output,up,stream,0));
    auto found=owner->shared_plans.find(key);
    if (found!=owner->shared_plans.end()) return &found->second;
    if (!create) return nullptr;
    SharedPlan result;
    int tokens=int(input->ne[1]);
    if (owner->api->paired_select(gate.qtype,gate.n,gate.k,gate.experts,tokens,QK_COMPUTE_F16,&result.config)!=QKG_OK)
        return nullptr;
    result.call=paired_call(gate.qtype,gate.experts,tokens,QK_COMPUTE_F16,false);
    result.weights=paired_weights(*owner,gate,&up,stream,result.call,result.config);
    if (!result.weights) return nullptr;
    auto & c=result.call.input.call;
    c.a=input->data;c.output=static_cast<float*>(match.output->data);
    c.low=result.weights->low;c.units=result.weights->units;c.stream=stream;
    qkg_sizes_v1 sizes{};
    if (owner->api->paired_query(&result.call,&result.config,&result.weights->layout,&sizes)!=QKG_OK)
        GGML_ABORT("[quactlize] shared paired query failed");
    if (sizes.workspace_bytes) { c.workspace=owner->private_storage(sizes.workspace_bytes);c.workspace_bytes=sizes.workspace_bytes; }
    paired_log(match.gate->src[0]->name,"dense",gate.qtype,tokens,gate.experts,QK_COMPUTE_F16,result.config);
    return &owner->shared_plans.emplace(key,result).first->second;
}

MoePlan * prepare_moe(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int start, bool create) {
    auto * owner = execution(ctx);
    if (!owner || !owner->api->moe_create || !owner->api->moe_run || !owner->api->moe_destroy ||
        route_mode() == RouteMode::Gemv || !ctx.stream_context().concurrent_events.empty()) return nullptr;
    auto match = quactlize::llama::match_moe(graph,start);
    if (!match.count) return nullptr;
    if (match.finish && !owner->api->moe_bind_finish) return nullptr;
    Plan * plans[3]{};
    ggml_tensor * nodes[3] = {match.gate,match.up,match.down};
    for (int j=0;j<3;++j) {
        auto * node = nodes[j];
        if (!node) continue;
        if (node->src[1]->ne[2]>8) return nullptr;
        ggml_quactlize_artifact art{};
        if (!ggml_quactlize_artifact_for(node->src[0],&art)) return nullptr;
        auto key=make_key(node->src[0],node->src[1],node->src[2],node,art,ctx.stream(),j+1);
        auto found=owner->plans.find(key);
        if (found==owner->plans.end() && !create) return nullptr;
        plans[j]=found==owner->plans.end() ? &prepare(ctx,node->src[0],node->src[1],node->src[2],node,j+1) : found->second.get();
        if (plans[j]->legacy || (!plans[j]->indexed && !plans[j]->direct)) return nullptr;
        if (plans[j]->direct && !owner->api->moe_create_mixed) return nullptr;
        if (plans[j]->reuse && !owner->api->moe_create_reuse) return nullptr;
    }
    auto key=std::make_tuple(plans[0],plans[1],plans[2],match.weights,match.finish);
    auto found=owner->moe_plans.find(key);
    if (found!=owner->moe_plans.end()) return found->second.get();
    if (!create) return nullptr;
    auto result=std::make_unique<MoePlan>();
    result->api=owner->api; result->gate=plans[0]; result->up=plans[1]; result->down=plans[2];
    qks_moe_endpoint_v3 endpoints[3]{};
    for (int j=0;j<3;++j) {
        auto * p=plans[j];if (!p) continue;
        auto & e=endpoints[j];e.version=3;e.size=sizeof(e);
        if (!p->direct) { e.tc_handle=p->handle;continue; }
        result->simt_mask|=1u<<j;
        int status=owner->api->moe_simt_scratch(owner->runtime,&p->gemv,&e.scratch_bytes);
        if (status==QKS_MISS) return nullptr;
        if (status!=QKS_OK) GGML_ABORT("[quactlize] mixed MoE scratch: %s",owner->api->error());
        e.scratch=owner->private_storage(e.scratch_bytes);e.simt_call=&p->gemv;e.arrangement=&p->art.arrangement;
        if (p->reuse) e.reuse_config=&p->smallm.simt;
        else if (p->q4_decode) e.q4_config=&p->q4_config;
        else e.simt_config=&p->gemv_config;
    }
    auto create_mixed = [&] {
        if (plans[0]->compute == QK_COMPUTE_BF16) {
            qks_moe_endpoint_v4 typed[3]{};
            for (int j=0;j<3;++j) if (plans[j]) {
                if (plans[j]->compute != plans[0]->compute)
                    GGML_ABORT("[quactlize] mixed compute types in one MoE chain");
                typed[j]={4,sizeof(typed[j]),endpoints[j],plans[j]->compute};
            }
            return owner->api->moe_create_compute(owner->runtime,&typed[0],
                plans[1]?&typed[1]:nullptr,&typed[2],&result->handle);
        }
        if (owner->api->moe_create_reuse)
            return owner->api->moe_create_reuse(owner->runtime,&endpoints[0],
                plans[1]?&endpoints[1]:nullptr,&endpoints[2],&result->handle);
        qks_moe_endpoint_v2 legacy[3]{};
        for (int j=0;j<3;++j) {
            auto const & e=endpoints[j];
            legacy[j]={2,sizeof(legacy[j]),e.tc_handle,e.simt_call,e.q4_config,e.simt_config,e.arrangement,e.scratch,e.scratch_bytes};
        }
        return owner->api->moe_create_mixed(owner->runtime,&legacy[0],
            plans[1]?&legacy[1]:nullptr,&legacy[2],&result->handle);
    };
    int rc=(result->simt_mask || plans[0]->compute == QK_COMPUTE_BF16) ? create_mixed() :
        owner->api->moe_create(plans[0]->handle,plans[1]?plans[1]->handle:nullptr,plans[2]->handle,&result->handle);
    if (rc==QKS_MISS) return nullptr;
    if (rc!=QKS_OK) GGML_ABORT("[quactlize] MoE chain preparation: %s",owner->api->error());
    if (match.finish) {
        qk_llama_moe_finish_v1 finish{1,sizeof(finish),int64_t(match.weights->nb[2]/sizeof(float)),
            int64_t(match.finish->nb[1]/sizeof(float)),static_cast<float const*>(match.weights->data),
            static_cast<float*>(match.finish->data)};
        rc=owner->api->moe_bind_finish(owner->runtime,result->handle,&finish);
        if (rc==QKS_MISS) return nullptr;
        if (rc!=QKS_OK) GGML_ABORT("[quactlize] MoE finish binding: %s",owner->api->error());
    }
    auto & gate=plans[0]->art;
    if (owner->api->paired_select && route_mode()==RouteMode::Auto && !match.up &&
        plans[0]->compute==QK_COMPUTE_BF16 && gate.qtype==12 && gate.n==1024 && gate.k==2048 &&
        gate.experts==256 && plans[0]->topk==8 && match.gate->src[1]->ne[1]==1) {
        qks_moe_gate_up_v1 binding{};binding.version=1;binding.size=sizeof(binding);
        if (owner->api->paired_select(gate.qtype,gate.n/2,gate.k,gate.experts,plans[0]->tokens,
                plans[0]->compute,&binding.config)!=QKG_OK)
            GGML_ABORT("[quactlize] measured routed paired selection failed");
        auto call=paired_call(gate.qtype,gate.experts,plans[0]->tokens,plans[0]->compute,true);
        result->paired=paired_weights(*owner,gate,nullptr,ctx.stream(),call,binding.config);
        if (!result->paired) GGML_ABORT("[quactlize] routed paired arrangement differs");
        binding.low=result->paired->low;binding.units=result->paired->units;binding.layout=result->paired->layout;
        qkg_sizes_v1 sizes{};
        if (owner->api->paired_query(&call,&binding.config,&binding.layout,&sizes)!=QKG_OK)
            GGML_ABORT("[quactlize] routed paired workspace query failed");
        if (sizes.workspace_bytes) { binding.workspace=owner->private_storage(sizes.workspace_bytes);binding.workspace_bytes=sizes.workspace_bytes; }
        const int paired_rc=owner->api->moe_bind_gate_up(owner->runtime,result->handle,&binding);
        if (paired_rc!=QKS_OK)
            GGML_ABORT("[quactlize] routed paired binding rc=%d: %s",paired_rc,owner->api->error());
        paired_log(match.gate->src[0]->name,"grouped",gate.qtype,plans[0]->tokens,gate.experts,plans[0]->compute,binding.config);
    }
    GGML_LOG_INFO("[quactlize-moe] gate=%s down=%s merged=%d rows=%d simt_mask=%u shared_prepare=1 swiglu_to_down=1 reduce_scatter=%d weighted_sum_finish=%d activation=%s\n",
        match.gate->src[0]->name,match.down->src[0]->name,int(!match.up),plans[0]->rows,result->simt_mask,
        int(!(result->simt_mask&4)),int(match.finish!=nullptr),compute_name(plans[0]->compute));
    auto * pointer=result.get(); owner->moe_plans.emplace(key,std::move(result)); return pointer;
}
} // namespace

int ggml_quactlize_execution_shared_nodes(const ggml_cgraph * graph, int start) {
    return quactlize::llama::match_shared_gate_up(graph,start).count;
}
bool ggml_quactlize_execution_shared_run(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int start) {
    auto * plan=prepare_shared(ctx,graph,start,false);
    if (!plan) return false;
    auto call=plan->call;call.input.call.stream=ctx.stream();
    const int rc=execution(ctx)->api->paired_run(&call,&plan->config,&plan->weights->layout);
    if (rc!=QKG_OK) GGML_ABORT("[quactlize] shared paired launch failed rc=%d",rc);
    ggml_ncp_route_log(graph->nodes[start+2],"so-quactlize-kpack-shared-paired",nullptr);
    return true;
}

int ggml_quactlize_execution_moe_nodes(const ggml_cgraph * graph, int start) {
    return quactlize::llama::match_moe(graph,start).count;
}

static int expand_sf(const Plan & p, cudaStream_t stream) {
    if (!p.sf || p.scale_resident) return QKG_OK;
    if (p.sf_config>=0) {
        if (p.compute==QK_COMPUTE_BF16) {
            qzd_call_v2 typed{2,sizeof(typed),p.expansion,QK_METADATA_BF16};
            return p.api->dequant_compute(&typed,&p.art.arrangement);
        }
        return p.api->dequant(&p.expansion,&p.art.arrangement);
    }
    if (p.compute==QK_COMPUTE_BF16)
        return p.api->sf_prepare_compute(p.art.qtype,p.art.n,p.art.k,p.art.experts,p.art.units,
            p.units_bytes,p.scale,p.zero,p.sf_plane_bytes,&p.art.arrangement,QK_METADATA_BF16,stream);
    return p.api->sf_prepare(p.art.qtype,p.art.n,p.art.k,p.art.experts,p.art.units,
        p.units_bytes,p.scale,p.zero,p.sf_plane_bytes,&p.art.arrangement,stream);
}

static bool run_moe(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int start,
    qk_llama_router_v1 const * router) {
    auto * chain=prepare_moe(ctx,graph,start,false);
    if (!chain) return false;
    if (router && (!chain->api->moe_run_router || chain->gate->art.experts!=256)) return false;
    auto stream=ctx.stream();
    for (auto * p:{chain->gate,chain->up,chain->down}) {
        if (!p || (chain->paired && p!=chain->down)) continue;
        ggml_quactlize_wait_ready(p->art,stream);
        if (expand_sf(*p,stream)!=QKG_OK)
            GGML_ABORT("[quactlize] per-call MoE SF prepass failed");
    }
    int rc=router ? chain->api->moe_run_router(chain->handle,router,stream) : chain->api->moe_run(chain->handle,stream);
    if (rc==QKS_MISS) return false;
    if (rc!=QKS_OK)
        GGML_ABORT("[quactlize] fused MoE launch failed");
    ggml_ncp_route_log(graph->nodes[start+ggml_quactlize_execution_moe_nodes(graph,start)-1],
        router ? "so-quactlize-kpack-moe-router-fused" : "so-quactlize-kpack-moe-fused",nullptr);
    return true;
}
bool ggml_quactlize_execution_moe_run(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int start) {
    return run_moe(ctx,graph,start,nullptr);
}
bool ggml_quactlize_execution_moe_router_run(ggml_backend_cuda_context & ctx, ggml_cgraph * graph,
    int start, const qk_llama_router_v1 & router, const ggml_tensor * ids) {
    auto match=quactlize::llama::match_moe(graph,start);
    if (!match.count || match.gate->src[2]!=ids) return false;
    return run_moe(ctx,graph,start,&router);
}

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
    ctx.curr_stream_no = 0;
    if (ctx.stream_context().concurrent_events.empty())
        for (int i=0;i<graph->n_nodes;++i) {
            prepare_shared(ctx,graph,i,true);
            prepare_moe(ctx,graph,i,true);
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
    ggml_quactlize_wait_ready(p.art, stream);
    if (p.full_handle) {
        if (ids) {
            ggml_cuda_launch_mm_ids_helper(static_cast<int32_t const*>(ids->data),p.ids_src,p.ids_dst,p.bounds,
                p.art.experts,p.tokens,p.topk,input->ne[1],ids->nb[1]/sizeof(int32_t),input->nb[2]/input->nb[1],false,stream);
            CUDA_CHECK(cudaGetLastError());
        }
        if (p.api->full_run(p.full_handle,stream)!=QKG_OK)
            GGML_ABORT("[quactlize] %s: full-BF16 run: %s",weight->name,p.api->full_error());
        ggml_ncp_route_log(output,"so-quactlize-kpack-full-bf16",nullptr);
        return true;
    }
    if (expand_sf(p,stream)!=QKG_OK)
        GGML_ABORT("[quactlize] %s: per-call SF prepass failed", weight->name);
    if (p.direct) {
        qkg_simt_call_v2 typed{2,sizeof(typed),p.gemv,p.compute};
        int rc;
        if (p.reuse) {
            rc = p.compute == QK_COMPUTE_BF16 ?
                p.api->simt_run_compute(&typed, &p.smallm.simt, &p.art.arrangement) :
                p.api->simt_run(&p.gemv, &p.smallm.simt, &p.art.arrangement);
        } else if (p.q4_decode) {
            rc = p.compute == QK_COMPUTE_BF16 ?
                p.api->q4_run_compute(&typed, &p.q4_config, &p.art.arrangement) :
                p.api->q4_run(&p.gemv, &p.q4_config, &p.art.arrangement);
        } else {
            GGML_ASSERT(p.compute == QK_COMPUTE_F16);
            rc = p.api->gemv_run(&p.gemv, &p.gemv_config, &p.art.arrangement);
        }
        if (rc != QKG_OK)
            GGML_ABORT("[quactlize] %s: GEMV launch failed", weight->name);
        ggml_ncp_route_log(output, "so-quactlize-kpack-gemv", nullptr);
        return true;
    }
    if (p.indexed || p.dense_io) {
        if (p.api->run(p.handle,stream)!=QKS_OK)
            GGML_ABORT("[quactlize] %s: fused indexed GEMM launch failed",weight->name);
        ggml_ncp_route_log(output,p.sf ? "so-quactlize-kpack-sf" : "so-quactlize-kpack-fq-selected",nullptr);
        return true;
    }
    if (p.q4_tc) {
        p.gemv.stream = stream;
        int rc = ids ? p.api->q4_prepare(&p.gemv, p.a, p.bounds, p.ids_dst) :
            p.api->q4_cast(1, input->data, p.a, p.rows, p.art.k, p.gemv.a_row_stride, stream);
        if (rc != QKG_OK) GGML_ABORT("[quactlize] %s: decode TC preparation failed rc=%d", weight->name, rc);
        if (p.api->run(p.handle, stream) != QKS_OK)
            GGML_ABORT("[quactlize] %s: selected decode TC failed", weight->name);
        rc = ids ? p.api->q4_finish(&p.gemv, p.out, p.ids_dst) :
            p.api->q4_cast(0, p.out, output->data, p.rows, p.art.n, p.gemv.out_row_stride, stream);
        if (rc != QKG_OK) GGML_ABORT("[quactlize] %s: decode TC finish failed rc=%d", weight->name, rc);
        ggml_ncp_route_log(output, "so-quactlize-kpack-fq-selected", nullptr);
        return true;
    }
    if (ids) {
        const size_t smem = ggml_cuda_info().devices[ctx.device].smpbo;
        GGML_ASSERT(size_t(p.tokens) * sizeof(int32_t) <= smem && p.tokens < (1 << 22) && p.topk < (1 << 10));
        GGML_ASSERT(ids->nb[1] / sizeof(int32_t) <= INT32_MAX && input->nb[2] / input->nb[1] <= INT32_MAX);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, p.ids_src, p.ids_dst, p.bounds,
            p.art.experts, p.tokens, p.topk, input->ne[1], ids->nb[1] / sizeof(int32_t), input->nb[2] / input->nb[1], false, stream);
        CUDA_CHECK(cudaGetLastError());
    }
    if (p.compute == QK_COMPUTE_BF16)
        gather<<<p.rows, 256, 0, stream>>>((const float *) input->data, (__nv_bfloat16 *) p.a, p.ids_src, p.art.k, input->nb[1] / sizeof(float));
    else gather<<<p.rows, 256, 0, stream>>>((const float *) input->data, (half *) p.a, p.ids_src, p.art.k, input->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    if (p.api->run(p.handle, stream) != QKS_OK) GGML_ABORT("[quactlize] %s: native GEMM launch failed", weight->name);
    if (p.compute == QK_COMPUTE_BF16)
        scatter<<<p.rows, 256, 0, stream>>>((const __nv_bfloat16 *) p.out, (float *) output->data, p.ids_dst, p.art.n, output->nb[1] / sizeof(float));
    else scatter<<<p.rows, 256, 0, stream>>>((const half *) p.out, (float *) output->data, p.ids_dst, p.art.n, output->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    ggml_ncp_route_log(output, p.sf ? "so-quactlize-kpack-sf" : "so-quactlize-kpack-fq-selected", nullptr);
    return true;
}
#else
void ggml_quactlize_execution_prepare_graph(ggml_backend_cuda_context &, ggml_cgraph *) {}
int ggml_quactlize_execution_shared_nodes(const ggml_cgraph *, int) { return 0; }
bool ggml_quactlize_execution_shared_run(ggml_backend_cuda_context &, ggml_cgraph *, int) { return false; }
int ggml_quactlize_execution_moe_nodes(const ggml_cgraph *, int) { return 0; }
bool ggml_quactlize_execution_moe_run(ggml_backend_cuda_context &, ggml_cgraph *, int) { return false; }
bool ggml_quactlize_execution_moe_router_run(ggml_backend_cuda_context &, ggml_cgraph *, int,
    const qk_llama_router_v1 &, const ggml_tensor *) { return false; }
bool ggml_quactlize_execution_run(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
                                  const ggml_tensor *, ggml_tensor *) { return false; }
#endif
