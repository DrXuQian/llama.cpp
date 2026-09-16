#pragma once
#include "kpack_module.h"
#include "kpack_indexed.h"
#include "kpack_decode_io.h"
#include "kpack_q4_decode.h"
#include "kpack_simt.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKS_OK = 0, QKS_MISS = 1, QKS_INVALID = 2, QKS_BINDING = 3,
       QKS_RUNTIME = 4 };
enum { QKS_RECENT = 1, QKS_HISTORICAL = 2, QKS_PREDICTED = 3,
       QKS_DEVICE_BOUNDS = 4, QKS_MEASURED_GROUPED = 5, QKS_Q8_INITIAL = 6,
       QKS_DECODE_MEASURED = 7, QKS_COMPONENT_MEASURED = 8,
       QKS_SMALLM_EXACT = 9, QKS_SMALLM_BUCKET = 10,
       QKS_COMPUTE_INITIAL = 11, QKS_MATCHED_EXACT = 12,
       QKS_MATCHED_BUCKET = 13, QKS_MATCHED_ROUTER = 14,
       QKS_Q8_VECTOR_MEASURED = 15 };

// Additive Q8_0/W8A16 intake capability, without a device/context or JIT.
// Returns 1 for supported weight geometry and SF route (1=dense,3=grouped).
// These parents accept every positive M within the module's integer/resource
// limits; no M-dependent holes. Runtime allocation can still fail. Deployment
// must enable selected-parent JIT before admitting Q8 weights (no legacy Q8
// fallback exists). The Q8 choices are initial heuristics, NOT measured optima.
int quactlize_kpack_dispatch_q8_weight_supported_v1(
    int n, int k, int experts, int route, uint64_t mapping_id);

typedef struct {
    uint32_t version, size;
    int32_t qtype, route, m, n, k, experts, max_rows;
    uint64_t mapping_id;
} qks_request_v1;

typedef struct {
    uint32_t version, size;
    uint64_t ticket, workspace_bytes, shared_bytes;
    int32_t policy, algorithm, split, grid, device, compute_units;
    char parent[192], build_key[65];
} qks_choice_v1;

enum { QKS_SMALLM_TC = 0, QKS_SMALLM_SIMT = 1, QKS_SMALLM_Q4 = 2 };
typedef struct {
    uint32_t version,size;
    int32_t kind,policy,source_n,source_k,source_tokens;
    qkg_simt_config_v1 simt;
    qkg_sizes_v1 sizes;
    qks_choice_v1 tc;
} qks_smallm_choice_v1;
// Auto decode only: exact table, then same-format/operator logarithmic bucket.
// Dense F32 endpoints M1..8, or indexed E256/top8 tokens1..8, channels1 or8.
// No tuning or device ID readback. A TC choice may JIT/load its selected parent;
// prepare remains outside graph capture. SIMT includes its real F32 reducer.
// EXACT describes the request key, not a common-cohort performance guarantee.
// BUCKET is predicted; source_* identify the donor. Q4's joint board is unchanged
// and returns MISS here. A miss retains the caller's existing legal K-pack route.
int quactlize_kpack_dispatch_query_smallm_v1(void* runtime,qkg_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*,qks_smallm_choice_v1*);
// Explicit compute: BF16 geometry is an initial proposal, not an FP16 timing
// relabeled as BF16. F16 delegates to the existing policy unchanged.
int quactlize_kpack_dispatch_query_smallm_v2(void* runtime,qkg_simt_call_v2 const*,
    quactlize_ppu_placed_arrangement_v2 const*,qks_smallm_choice_v1*);
// Additive matched-pool override: explicit F16/BF16 measurements, complete
// producer+real reducer+indexed endpoint costs. Grouped choices minimize worst
// measured router-profile regret without reading IDs on the host. ROUTER
// denotes a measured compromise exceeding 5%, not an optimality guarantee.
// A bounded bucket is predicted; source_* name the donor. MISS means retain
// the prior legal selector. This entry is decode-only and never runs a tuner.
typedef struct {
    uint32_t version,size;
    qks_smallm_choice_v1 base;
    qkg_q4_decode_config_v1 q4;
    int32_t compute_type;
} qks_smallm_choice_v2;
int quactlize_kpack_dispatch_query_smallm_v3(void* runtime,qkg_simt_call_v2 const*,
    quactlize_ppu_placed_arrangement_v2 const*,qks_smallm_choice_v2*);

typedef struct {
    uint32_t version, size;
    // route: 0=FQ, 1=per-call SF, 2=per-call full BF16 + external provider.
    int32_t route, dequant_config, measured_tokens, predicted;
    double gemm_us, dequant_us;
} qks_prefill_choice_v1;
// Pure host query: no JIT/device work. Mask 1=FQ,3=FQ+measured SF,7=all.
// Only advertise SF/full when the measured expansion/provider is available.
// Dense M128..4096; grouped top8 E256 with max_rows=tokens128..4096.
// Costs are isolated component sums, excluding caller adapters and cache
// interaction. TC cost already includes its reducer. No small-M full path.
// Interior knot transfer sets predicted=1; no unmeasured 5% guarantee.
int quactlize_kpack_dispatch_prefill_v1(qks_request_v1 const*, uint32_t mask,
                                      qks_prefill_choice_v1*);

// One runtime per device/context. root contains the exact modules packaged
// with this host library. open/query/prepare belong outside graph capture.
// Missing selected modules return QKS_MISS, never another compiled tactic.
int quactlize_kpack_dispatch_open_v1(char const* root, void** runtime);
void quactlize_kpack_dispatch_close_v1(void* runtime);

typedef struct {
    uint32_t version, size;
    char const *python, *helper, *sdk, *cache;
} qks_jit_options_v1;

// Optional, explicit opt-in BEFORE any queries. Paths are copied. python and
// helper are absolute executable/script paths; helper is tools/kpack_jit.py.
// A query missing its selected module can compile that ONE parent. No tuning,
// device execution or alternative-config selection occurs in the compiler.
// Query/prepare must remain outside capture. Prepared run never invokes JIT.
// The cache and helper must be trusted local files, like the packaged DSOs.
int quactlize_kpack_dispatch_enable_jit_v1(void* runtime, qks_jit_options_v1 const*);
int quactlize_kpack_dispatch_query_v1(void* runtime, qks_request_v1 const*, qks_choice_v1*);
// Additive TC selection for the measured Q4 F32-endpoint decode path. Call
// only after the Q4 SIMT selector declines. Dense M<=8; grouped E256/top8,
// shared or slot-specific A, max_rows=tokens=m/8 (an upper bound, not observed
// routing). No device-ID readback. Unlisted requests return QKS_MISS; retain
// the ordinary selector. Forced FQ/SF and the v1 query are unchanged.
int quactlize_kpack_dispatch_query_decode_v1(void* runtime, qks_request_v1 const*, qks_choice_v1*);
// Dense M1..8 with explicit F32 or BF16 endpoints. Uses the SAME selector as
// query_v1 (decode_policy=0) or query_decode_v1 (decode_policy=1, after SIMT
// declines). Compiles only the selected typed parent, not an unused FP16 one.
// Uses packaged typed parents or selected-parent JIT/cache. Tickets and resources are distinct;
// old query/prepare retain FP16 semantics. No tuning or new fallback policy.
int quactlize_kpack_dispatch_query_dense_io_v1(void* runtime,qks_request_v1 const*,
                                            int32_t endpoint_type,int32_t decode_policy,qks_choice_v1*);
int quactlize_kpack_dispatch_prepare_dense_io_v1(void* runtime,qks_choice_v1 const*,
                                               qkd_dense_call_v1 const*,void** handle);
// compute_type is QK_COMPUTE_F16/BF16. endpoint_type is 0 for grouped native
// compute storage, or QKD_F32/BF16 for dense M1..8. BF16 group support includes
// prefill. Tickets, modules and JIT keys include compute precision.
int quactlize_kpack_dispatch_query_compute_v1(void* runtime,qks_request_v1 const*,
    int32_t compute_type,int32_t endpoint_type,int32_t decode_policy,qks_choice_v1*);
int quactlize_kpack_dispatch_prepare_compute_v1(void* runtime,qks_choice_v1 const*,
    qk_compute_device_call_v3 const*,void** handle);
int quactlize_kpack_dispatch_prepare_dense_io_v2(void* runtime,qks_choice_v1 const*,
    qkd_dense_call_v2 const*,void** handle);
int quactlize_kpack_dispatch_prepare_v1(void* runtime, qks_choice_v1 const*,
                                     qk_call_v1 const*, void** handle);
int quactlize_kpack_dispatch_run_v1(void* handle, void* stream);
// Optional, once-per-handle adapter binding, outside capture. A miss leaves
// the handle unchanged and requires the unfused caller's preparation path.
int quactlize_kpack_dispatch_bind_llama_indexed_v1(void*,qk_llama_indexed_v1 const*);
// Exact small-decode composition: shared preparation -> gate/up -> ordered
// FP16 completion + SwiGLU -> down -> ordered reduction/scatter. Passing NULL
// for up means gate is already merged [E,2N,K], gate first within each expert.
// Handles must be indexed-bound, with DISJOINT scratch and the same ID input.
// Create outside capture; retain handles/buffers until chain destruction and
// device completion. Run has no allocation, tuning, host read or device wait.
// SF expansion, if needed, is caller-owned and precedes this chain each time.
int quactlize_kpack_dispatch_moe_create_v1(void* gate,void* up,void* down,void** chain);
// Additive mixed chain. Exactly one of tc_handle/simt_call is present.
// SIMT config is either the automatic Q4 recipe or the caller's measured
// generic recipe. Scratch is disjoint per projection and retained by caller.
// Create copies descriptors, never GPU IDs. Existing handles remain usable.
typedef struct {
    uint32_t version,size;
    void* tc_handle;
    qkg_call_v1 const* simt_call;
    qkg_q4_decode_config_v1 const* q4_config;
    qkg_config_v1 const* simt_config;
    quactlize_ppu_placed_arrangement_v2 const* arrangement;
    void* scratch;
    uint64_t scratch_bytes;
} qks_moe_endpoint_v2;
int quactlize_kpack_dispatch_moe_simt_scratch_v1(void* runtime,qkg_call_v1 const*,uint64_t*);
int quactlize_kpack_dispatch_moe_create_v2(void* runtime,qks_moe_endpoint_v2 const* gate,
    qks_moe_endpoint_v2 const* up,qks_moe_endpoint_v2 const* down,void** chain);
// Additive all-format register-reuse reader. Exactly one of q4_config,
// simt_config (legacy generic) and reuse_config is present for a SIMT endpoint.
// Recipes are explicit, not selected here. Token 1..8; canonical bytes and
// the existing F16-activation/F32-output arithmetic contract are unchanged.
// A Split-K SIMT call retains its own disjoint workspace from simt_query_v1,
// in addition to the projection scratch from moe_simt_scratch_v1. Both remain
// live until device completion. v2 callers and automatic Q4 policy are unchanged.
typedef struct {
    uint32_t version,size;
    void* tc_handle;
    qkg_call_v1 const* simt_call;
    qkg_q4_decode_config_v1 const* q4_config;
    qkg_config_v1 const* simt_config;
    quactlize_ppu_placed_arrangement_v2 const* arrangement;
    void* scratch;
    uint64_t scratch_bytes;
    qkg_simt_config_v1 const* reuse_config;
} qks_moe_endpoint_v3;
int quactlize_kpack_dispatch_moe_create_v3(void* runtime,qks_moe_endpoint_v3 const* gate,
    qks_moe_endpoint_v3 const* up,qks_moe_endpoint_v3 const* down,void** chain);
typedef struct {
    uint32_t version,size;
    qks_moe_endpoint_v3 endpoint;
    int32_t compute_type;
} qks_moe_endpoint_v4;
// All linked projections must have the same compute type. BF16 SIMT uses
// reuse_config and the explicit v2 reader; legacy FP16-only readers decline.
int quactlize_kpack_dispatch_moe_create_v4(void* runtime,qks_moe_endpoint_v4 const* gate,
    qks_moe_endpoint_v4 const* up,qks_moe_endpoint_v4 const* down,void** chain);
// Optional once-only binding outside capture. Loads the small finish module,
// validates disjoint live inputs, and copies the immutable output contract.
// On success run writes finish.output, not the intermediate down tensor.
// A miss leaves the chain unchanged. No qtype/layout/config selection changes.
int quactlize_kpack_dispatch_moe_bind_finish_v1(void* runtime,void* chain,
    qk_llama_moe_finish_v1 const*);
// All chain versions use the same run/router/destroy entries. Mixed chains
// retain SIMT F32 results and TC FP16 completion semantics through SwiGLU.
int quactlize_kpack_dispatch_moe_run_v1(void* chain,void* stream);
// Same chain, but router+preparation are one kernel. The caller must match the
// complete top-k graph, preserve weights/IDs as outputs and prove input ready.
int quactlize_kpack_dispatch_moe_run_router_v1(void* chain,qk_llama_router_v1 const*,void* stream);
void quactlize_kpack_dispatch_moe_destroy_v1(void* chain);
// Finish all work using a handle before closing it. Runtime and module
// lifetimes are retained by live handles; closing performs no device waits.
void quactlize_kpack_dispatch_destroy_v1(void* handle);
char const* quactlize_kpack_dispatch_error_v1(void);

#ifdef __cplusplus
}
#endif
