#pragma once
#include "kpack_execution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t version, size;
    int32_t reader, variant, warps, values, columns;
} qkg_q4_decode_config_v1;

// Measured PPU-ZW810 cold-weight policy, F32 endpoints with FP16 A rounding.
// Dense M=1..8; indexed E256/top8, tokens=1..8, channels=1 or 8.
// Query reads NO device data. Indexed top8 IDs must be distinct per token.
// QKG_SHAPE means retain selected TC (including unmeasured input geometries).
// Other errors describe invalid ABI/format/arrangement. No online tuning.
// Only SIMT choices populate config/sizes; no workspace or scale prepass.
int quactlize_kpack_q4_decode_select_v1(qkg_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qkg_q4_decode_config_v1*, qkg_sizes_v1*);

// The config must be the exact selected recipe for this public request.
// All dot/reduction work is one S1 CTA per output row/tile, F32 output in the
// caller's order. No gather/scatter, allocation, host readback or device wait.
// Callers retain buffers until completion and use one stream for dependencies.
int quactlize_kpack_q4_decode_run_v1(qkg_call_v1 const*,
    qkg_q4_decode_config_v1 const*, quactlize_ppu_placed_arrangement_v2 const*);

// F32-endpoint TC adapters used by the measured decode path. Dense cast:
// input=1 maps strided F32 -> compact F16; input=0 maps compact F16 -> F32.
int quactlize_kpack_q4_decode_cast_v1(int input, void const* source, void* destination,
    int rows, int columns, int stride, void* stream);
// Indexed tokens5..8/E256/top8 only; compact_a has rows*K F16 elements,
// offsets has E+1 I32 elements, row_ids has rows I32 elements. All caller-owned.
// The TC module still owns metadata, directory, GEMM and Split-K reduction.
int quactlize_kpack_q4_decode_indexed_prepare_v1(qkg_call_v1 const*,
    void* compact_a, int32_t* offsets, int32_t* row_ids);
int quactlize_kpack_q4_decode_indexed_finish_v1(qkg_call_v1 const*,
    void const* compact_output, int32_t const* row_ids);

#ifdef __cplusplus
}
#endif
