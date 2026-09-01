#pragma once
// Asynchronous device-pointer consumers exported by libquactlize_ppu.so.

#include <stdint.h>

#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete, versioned profile identity for the one shipping dense W4 fixed
// Split-K route.  The launch symbol itself fixes ScaleOnly fp16 metadata and a
// resident xplane-TK64 artifact; retaining those fields in the profile is what
// lets a stale row fail closed instead of inheriting the current binary's
// interpretation.  Boolean fields accept only 0 or 1.
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_PROFILE_SCHEMA_V1 1
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY 0
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ZERO 1
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_FP16_PLANES 0
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_PACKED_UNITS 1
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_RESIDENT_XPLANE 0
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_OTHER 1

typedef struct quactlize_ppu_dense_w4_splitk_key_v1 {
  int32_t rows;
  int32_t columns;
  int32_t inner;
  int32_t low_bits;
  int32_t high_bits;
  int32_t group_size;
  int32_t quant_semantics;
  int32_t metadata_storage;
  int32_t has_zero_plane;
  int32_t artifact_layout;
  int32_t artifact_tile_k;
  int32_t artifact_low_fold;
  int32_t artifact_high_fold;
  int32_t artifact_b_chunk;
  int32_t tactic_tile_m;
  int32_t tactic_tile_n;
  int32_t tactic_tile_k;
  int32_t tactic_warp_m;
  int32_t tactic_warp_n;
  int32_t tactic_stages;
  int32_t packed_a_rows;
  int32_t aiu_interleaved;
} quactlize_ppu_dense_w4_splitk_key_v1;

typedef struct quactlize_ppu_dense_w4_splitk_profile_v1 {
  uint32_t schema_version;
  quactlize_ppu_dense_w4_splitk_key_v1 key;
  int32_t selected_s;
} quactlize_ppu_dense_w4_splitk_profile_v1;

// All data pointers name device memory. stream is the caller's native hggc/CUDA stream handle cast to void*;
// nullptr selects the runtime default stream. A zero return means the kernel was enqueued, not completed.
// The caller retains every allocation and must enforce stream lifetime and dependency ordering.
int quactlize_ppu_vecdot_dense_dev_v1(uint8_t const* blocks, int64_t block_bytes,
                                      uint16_t const* x, float* out,
                                      int rows, int blocks_per_row, int qtype, void* stream);

// Scale-first CUDA-core tactic over the same fp16 affine planes as dense_lowbit. This entry owns no workspace and
// only enqueues; config_name must come from the dense inventory's enable_cuda_kernel record (or be null for it).
int quactlize_ppu_gemv_lowbit_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, void* stream, char const* config_name);
// Persistent ScaleFirst prefill over the same canonical K-pack4 bytes as the fully-quantized decode entry.  It owns
// no workspace; scale/zero are caller-owned fp16 planes derived from the resident packed units.  K-pack4 is admitted
// only for Q4_K and M>=64; decode continues through the fully-quantized v2 entry.
int quactlize_ppu_dense_lowbit_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, void* stream,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const* config_name);

// Dense symmetric W4A16 tensor-core route over the resident int4 xplane-TK64
// artifact and native fp16 gs128 scales.  It is deliberately separate from
// dense_lowbit's GGUF Q4_K ScaleZero/gs32 contract and from the packed-unit
// fully-quantized entries below.
//
// A null, stale, malformed, mismatched or explicit-S1 profile selects the
// historical Shipping::Gemm S1 path.  S={2,4,8} is admitted only when the
// complete profile key matches the measured exact-warm
// TM8/TN64/TK128-WM8/WN16/s2 production type (over ArtifactTK64) and the launch
// workspace is large enough and 128-byte aligned.  The query returns the FP32
// partial bytes for an admitted parallel profile, zero for an S1 fallback, and
// -1 for a problem outside the fixed M1/N256/K256 ABI or on size overflow.
// A successful device entry only enqueues work on stream.
int64_t quactlize_ppu_dense_w4_splitk_workspace_bytes_v1(
    int m, int n, int k,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile);
int quactlize_ppu_dense_w4_splitk_dev_v1(
    uint16_t const* act, uint8_t const* weight_xplane,
    uint16_t const* scales, uint16_t* out,
    int m, int n, int k, void* workspace, int64_t workspace_bytes,
    void* stream,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile);

// The placed low/high planes and packed units are the same artifact consumed by quactlize_ppu_bc_gemv.
// experts==0 selects one-launch dense SIMT decode and requires 1<=total_rows<8;
// grid.y owns the activation/output row. Grouped offsets are cumulative int[experts+1].
// For Q4_K (qtype=12), x, low, and units must each be 16-byte aligned because the shipping reader uses vector
// global loads. Both device entries below return 25 before enqueue when that contract is not met.
int quactlize_ppu_bc_gemv_dev_v1(uint16_t const* x,
                                 uint8_t const* low, uint8_t const* high, uint8_t const* units,
                                 int const* offsets, float* out,
                                 int total_rows, int n, int k, int experts, int max_rows, int qtype,
                                 void* stream);
// Arrangement-aware successor.  The artifact supplies the descriptor; a null/unknown/mismatched descriptor is an
// error and never falls back to the default reader map.
int quactlize_ppu_bc_gemv_for_arrangement_dev_v1(
    uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, void* stream);

// Fully-quantized tensor-core GEMM uses caller-owned device workspace. The size queries return -1 when the
// dimensions or qtype do not match this format-selected library. A successful device entry only enqueues work.
int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(
    int m, int n, int k, int qtype);
// Arrangement-aware query. It validates the exact v2 byte map before returning a bound shared by every compiled
// K-pack4 profile (SplitKSerial semaphore or S4 FP32 partials); unsupported descriptors return -1 and can never
// inherit the registry-default Xplane interpretation.
int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2(
    int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int quactlize_ppu_dense_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream);
// Config-selecting legacy reader-default successor. The v1 entry remains ABI-compatible and delegates with a null
// name: M<8 selects the compiled TM8 decode default, while M>=8 keeps the legacy default. Both keep the historical
// registry fully_quantized_tile_k tactic and artifact arrangement; a stale non-empty name keeps the legacy fallback.
int quactlize_ppu_dense_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name);
// Arrangement-aware successor.  Unlike v1/v2, this entry never infers the resident fold from qtype or tactic.
// A null/unknown/mismatched descriptor fails before launch; an unknown non-empty config name returns 39 rather than
// falling back. Null/empty config still selects the M-aware shipping default. The old entries keep their
// registry-default behavior.
int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name, quactlize_ppu_placed_arrangement_v1 const* arrangement);
int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name, quactlize_ppu_placed_arrangement_v2 const* arrangement);

// Grouped activations/output are concatenated in expert order. offsets is a cumulative device int[experts+1]
// with offsets[0]=0 and offsets[experts]=total_rows; max_rows is an upper bound on every expert row count.
// grouped_lowbit consumes the scale-first artifact selected offline: low/high code planes plus fp16 scales and no
// zero plane (FinegrainedScaleOnly). Its workspace holds the same ptr/stride arrays and m-tile prefix as the packed
// grouped route; neither device entry allocates or synchronizes.
int64_t quactlize_ppu_grouped_lowbit_workspace_bytes_v1(
    int max_rows, int n, int k, int group_size, int experts, int qtype);
int quactlize_ppu_grouped_lowbit_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream);
int quactlize_ppu_grouped_lowbit_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name);

int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1(
    int max_rows, int n, int k, int experts, int qtype);
int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
    int total_rows, int max_rows, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int quactlize_ppu_grouped_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream);
// Config-selecting successor. The v1 entry remains ABI-compatible and delegates with a null/default name.
int quactlize_ppu_grouped_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name);
// Arrangement-aware grouped successor. The descriptor is validated before
// metadata setup or GEMM enqueue; layout=1 selects the canonical K-pack4
// mainloop while retaining the existing ragged scheduler and ptr-array
// epilogue. Unknown config names return 39 rather than falling back.
int quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);

#ifdef __cplusplus
}
#endif
