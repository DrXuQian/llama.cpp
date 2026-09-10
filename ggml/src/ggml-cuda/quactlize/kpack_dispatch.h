#pragma once
#include "kpack_module.h"
#include "kpack_indexed.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKS_OK = 0, QKS_MISS = 1, QKS_INVALID = 2, QKS_BINDING = 3,
       QKS_RUNTIME = 4 };
enum { QKS_RECENT = 1, QKS_HISTORICAL = 2, QKS_PREDICTED = 3,
       QKS_DEVICE_BOUNDS = 4, QKS_MEASURED_GROUPED = 5, QKS_Q8_INITIAL = 6 };

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
int quactlize_kpack_dispatch_prepare_v1(void* runtime, qks_choice_v1 const*,
                                     qk_call_v1 const*, void** handle);
int quactlize_kpack_dispatch_run_v1(void* handle, void* stream);
// Optional, once-per-handle adapter binding, outside capture. A miss leaves
// the handle unchanged and requires the unfused caller's preparation path.
int quactlize_kpack_dispatch_bind_llama_indexed_v1(void*,qk_llama_indexed_v1 const*);
// Small indexed MoE chain. NULL up selects a merged gate/up weight.
// Handles and disjoint scratch must outlive the chain and device completion.
int quactlize_kpack_dispatch_moe_create_v1(void* gate,void* up,void* down,void** chain);
int quactlize_kpack_dispatch_moe_run_v1(void* chain,void* stream);
int quactlize_kpack_dispatch_moe_run_router_v1(void* chain,qk_llama_router_v1 const*,void* stream);
void quactlize_kpack_dispatch_moe_destroy_v1(void* chain);
// Finish all work using a handle before closing it. Runtime and module
// lifetimes are retained by live handles; closing performs no device waits.
void quactlize_kpack_dispatch_destroy_v1(void* handle);
char const* quactlize_kpack_dispatch_error_v1(void);

#ifdef __cplusplus
}
#endif
