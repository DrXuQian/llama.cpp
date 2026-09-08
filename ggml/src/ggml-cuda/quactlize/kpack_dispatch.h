#pragma once
#include "kpack_module.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKS_OK = 0, QKS_MISS = 1, QKS_INVALID = 2, QKS_BINDING = 3,
       QKS_RUNTIME = 4 };
enum { QKS_RECENT = 1, QKS_HISTORICAL = 2, QKS_PREDICTED = 3,
       QKS_DEVICE_BOUNDS = 4 };

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
int quactlize_kpack_dispatch_query_v1(void* runtime, qks_request_v1 const*, qks_choice_v1*);
int quactlize_kpack_dispatch_prepare_v1(void* runtime, qks_choice_v1 const*,
                                     qk_call_v1 const*, void** handle);
int quactlize_kpack_dispatch_run_v1(void* handle, void* stream);
// Finish all work using a handle before closing it. Runtime and module
// lifetimes are retained by live handles; closing performs no device waits.
void quactlize_kpack_dispatch_destroy_v1(void* handle);
char const* quactlize_kpack_dispatch_error_v1(void);

#ifdef __cplusplus
}
#endif
