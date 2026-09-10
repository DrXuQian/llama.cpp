#pragma once
#include <stdint.h>
#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKG_OK = 0, QKG_INVALID = 20, QKG_FORMAT = 22,
       QKG_SHAPE = 24, QKG_OVERFLOW = 26, QKG_CAPACITY = 37,
       QKG_ARRANGEMENT = 38, QKG_RUNTIME = 41 };
enum { QKG_DENSE = 0, QKG_GROUPED = 1, QKG_INDEXED = 2 };
enum { QKG_F16 = 0, QKG_F32 = 1 };

// All strides count elements. Output is F32, with one N-vector per row.
// DENSE: one weight, rows activation vectors.
// GROUPED: compact activations, offsets[experts+1] on device; empty experts
// are legal. Offsets must be nondecreasing, start at 0 and end at rows.
// INDEXED: rows=tokens*topk; ids[token*ids_stride+slot] selects an expert,
// A[token*a_token_stride+(slot%channels)*a_row_stride] selects activation.
// Output row token*topk+slot retains the caller's order (no gather/scatter).
// Device IDs/bounds are caller-validated inputs, never read back by the API.
// All weight planes have contiguous, same-index expert slices. high=NULL
// is required when high_bits=0. The offline arrangement is unchanged.
// Q8_0 uses its canonical K-pack2 low plane and original FP16 d as units;
// input F32 is rounded to FP16 in registers, without an activation prepass.
// Q8 weights are dequantized to FP16 and accumulated into F32 output.
typedef struct {
    uint32_t version, size;
    int32_t qtype, n, k, experts, rows, mode, input_type, channels, topk;
    int64_t a_row_stride, a_token_stride, ids_stride, out_row_stride;
    void const * a;
    uint8_t const * low;
    uint8_t const * high;
    uint8_t const * units;
    int32_t const * offsets;
    int32_t const * ids;
    float * output;
    void * workspace;
    uint64_t workspace_bytes;
    void * stream;
} qkg_call_v1;

// Small explicit candidate space; no online measurement or implicit choice.
typedef struct {
    uint32_t version, size;
    int32_t columns, warps, split;
} qkg_config_v1;

typedef struct {
    uint64_t low_bytes, high_bytes, units_bytes, sf_plane_bytes, workspace_bytes;
} qkg_sizes_v1;

// Query is host-only and does not dereference any device pointer.
int quactlize_kpack_gemv_query_v1(qkg_call_v1 const *, qkg_config_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *, qkg_sizes_v1 *);
// Enqueue only; no allocation, host copies or synchronization. Split>1
// includes the reducer. Caller retains buffers until stream completion.
int quactlize_kpack_gemv_run_v1(qkg_call_v1 const *, qkg_config_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *);

// Explicit experimental reader, never selected by the v1 entry above.
// Same call/output/format ABI. Columns=16/32, warps=2/4/8, split=1/2/4/8.
// Pair affine uses FP16 FMA (one rounding), not the scalar two-rounding
// oracle. Both accumulate in FP32; independent numerical admission is needed.
int quactlize_kpack_gemv_pair_query_v1(qkg_call_v1 const *, qkg_config_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *, qkg_sizes_v1 *);
int quactlize_kpack_gemv_pair_run_v1(qkg_call_v1 const *, qkg_config_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *);

// Output planes are [E,K/group_size,N], FP16, independently 16-byte aligned.
// Enqueue before EACH ScaleFirst GEMM on its stream (including graph replay).
// Reuse scratch allocation, not expanded values across calls. The caller owns
// allocation and lifetime; this API only enqueues and performs no host wait.
int quactlize_kpack_sf_prepare_v1(int qtype, int n, int k, int experts,
    uint8_t const * units, uint64_t units_bytes,
    uint16_t * scale, uint16_t * zero, uint64_t plane_bytes,
    quactlize_ppu_placed_arrangement_v2 const *, void * stream);

#ifdef __cplusplus
}
#endif
