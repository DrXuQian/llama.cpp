#pragma once
#include "kpack_execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register-reuse reader for canonical Q2/3/4/5/6_K and Q8_0. Existing
// scalar/pair/Q4 APIs and their admitted recipes remain unchanged.
// variant bit0 shares A within Columns lanes; bit1 shares packed metadata
// within a warp. Values is adjacent output columns owned by one thread.
// Dot and output are F32. A is rounded to F16 in registers; dequantization
// uses F32 group-affine arithmetic, not per-weight F16 reconstruction.
typedef struct {
    uint32_t version, size;
    int32_t variant, columns, warps, values, split;
} qkg_simt_config_v1;

enum { QKG_COMPUTE_F16=0, QKG_COMPUTE_BF16=1 };
// BF16 storage is additive and valid only in the explicit v2 call. v1
// continues to reject it. This is not an alias for QKG_F16.
enum { QKG_SIMT_BF16=2 };
typedef struct {
    uint32_t version, size;
    qkg_call_v1 call;
    int32_t compute_type;
} qkg_simt_call_v2;

// Same canonical planes, row order and F32 output as v1. Compute=BF16
// rounds A into BF16 registers (or reads BF16 bits unchanged), including
// finite values above 65504. Group-affine B arithmetic and accumulation
// remain F32. No clipping, activation prepass or standalone conversion.
int quactlize_kpack_simt_query_v2(qkg_simt_call_v2 const*, qkg_simt_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qkg_sizes_v1*);
int quactlize_kpack_simt_run_v2(qkg_simt_call_v2 const*, qkg_simt_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*);

int quactlize_kpack_simt_query_v1(qkg_call_v1 const*, qkg_simt_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qkg_sizes_v1*);
// Enqueue only, including the real F32 reducer for split>1. No allocation,
// metadata prepass, A/output conversion kernel or host routing readback.
int quactlize_kpack_simt_run_v1(qkg_call_v1 const*, qkg_simt_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*);

#ifdef __cplusplus
}
#endif
