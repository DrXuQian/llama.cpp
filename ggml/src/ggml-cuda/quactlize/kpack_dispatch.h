#pragma once
#include "kpack_module.h"
#include "kpack_indexed.h"
#include "kpack_decode_io.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKS_OK = 0, QKS_MISS = 1, QKS_INVALID = 2, QKS_BINDING = 3,
       QKS_RUNTIME = 4 };
enum { QKS_RECENT = 1, QKS_HISTORICAL = 2, QKS_PREDICTED = 3,
       QKS_DEVICE_BOUNDS = 4, QKS_MEASURED_GROUPED = 5, QKS_Q8_INITIAL = 6,
       QKS_DECODE_MEASURED = 7, QKS_COMPONENT_MEASURED = 8 };

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
