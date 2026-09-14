#pragma once
#include "kpack_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-call full BF16 expansion. No expanded-weight cache. Dense M>=128;
// grouped top8 E256 with total M>=1024. F32 caller endpoints are preserved.
// Grouped row maps and offsets are current GPU inputs, rebuilt by the caller
// on every invocation. src_rows indexes A rows; dst_rows indexes output rows.
// offsets[E+1] is nondecreasing, starts at 0, ends at m. Each source/dest row
// must fit the supplied caller tensors. dst_rows is a permutation of [0,m).
// A has a_rows addressable rows (tokens*channels for indexed input). Dense
// has a_rows=m and passes all three maps as NULL. Invalid offsets/indices
// set device status without accessing outside these buffers. Inspect that
// status after completion when validating caller-supplied maps.
typedef struct {
    uint32_t version, size;
    qzd_call_v1 weight;
    int32_t m, device, a_rows;
    float const* a;
    float* output;
    int64_t a_stride, output_stride;
    int32_t const *src_rows, *dst_rows, *offsets;
    void* workspace;
    uint64_t workspace_bytes;
} qkp_call_v1;

typedef struct {
    uint32_t version, size;
    char const *sdk, *python, *deepgemm_helper;
} qkp_options_v1;

// Query never dereferences device data. Workspace includes expanded weights,
// BF16 activations/output, GPU expert selection and provider scratch. Reuse a
// per-stream allocation across weights; do not allocate it per model tensor.
int quactlize_kpack_prefill_query_v1(qkp_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, uint64_t* workspace_bytes);
// Resolve installed cuBLAS or invoke the installed DeepGEMM Python entry's
// compile-only path ONCE, outside capture. Only that selected provider module
// is retained. There is no Python, JIT, allocation or host wait in run.
// Execute one eager warmup before graph capture/timing: the installed provider
// may lazily initialize its own internal resources on its first GEMM.
int quactlize_kpack_prefill_prepare_v1(qkp_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qkp_options_v1 const*, void**);
int quactlize_kpack_prefill_run_v1(void*, void* stream);
// Resident status, reset on each run. Reading it on the host is optional and
// requires caller synchronization; this accessor performs neither operation.
int quactlize_kpack_prefill_device_status_v1(void*, int32_t const**);
// Resolved provider image, valid for the handle's lifetime. For diagnostics
// outside the timed path; this does not identify a GEMM by its demangled name.
char const* quactlize_kpack_prefill_provider_image_v1(void*);
// Caller completes outstanding work before destroy. No synchronization here.
void quactlize_kpack_prefill_destroy_v1(void*);
char const* quactlize_kpack_prefill_error_v1(void);

#ifdef __cplusplus
}
#endif
