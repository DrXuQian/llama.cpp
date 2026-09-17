#pragma once
#include "kpack_simt.h"

#define QKG_GATE_UP_N4_V1 UINT64_C(0x47554e3400000001)

#ifdef __cplusplus
extern "C" {
#endif

// A distinct artifact contract, not an alternative interpretation of the old
// gate-then-up tensor. Within each expert, physical N is G4,U4,G4,U4,... .
// The nested arrangement defines bit packing only. Both sources must have
// that qtype. Permute every code plane and its metadata together.
typedef struct {
    uint32_t version, size;
    uint64_t layout_id;
    quactlize_ppu_placed_arrangement_v2 packing;
} qkg_gate_up_layout_v1;

// call.n is ONE projection's N; stored weights have 2*N physical columns.
// A/ids/offsets follow qkg_call_v1. Output has N columns in original order.
// round_projection=1 rounds each complete dot to compute_type BEFORE SiLU;
// zero keeps F32 projection results (the existing dense F32-output path).
// Output type is QKG_F32, QKG_F16, or QKG_SIMT_BF16. Never clip overflow.
typedef struct {
    uint32_t version, size;
    qkg_simt_call_v2 input;
    int32_t output_type, round_projection;
} qkg_gate_up_call_v1;

enum { QKG_GATE_UP_SIMT=0, QKG_GATE_UP_TC=1 };
// Explicit inventory, not a selector: SIMT C4/P8, W4 or W8; TC TM8/TM16,
// TN64/WN16/S2; TK=64 for Q8/Q4, 128 for Q2/Q6, 256 for Q3/Q5.
typedef struct {
    uint32_t version, size;
    int32_t backend, split, tile_m, warps;
} qkg_gate_up_config_v1;

int quactlize_gate_up_layout_v1(int qtype, qkg_gate_up_layout_v1*);
int quactlize_gate_up_pack_v1(uint8_t const* gate, uint8_t const* up,
    uint8_t* low, uint8_t* high, uint8_t* units, int n, int k, int experts,
    int qtype, qkg_gate_up_layout_v1 const*, void* stream);
int quactlize_gate_up_query_v1(qkg_gate_up_call_v1 const*, qkg_gate_up_config_v1 const*,
    qkg_gate_up_layout_v1 const*, qkg_sizes_v1*);
// Enqueue only, no allocations, copies to host or synchronization. S1 writes
// only SwiGLU output. Split>1 writes [row,split,2N] FP32 partials and fuses
// ordered reduction + projection rounding + SwiGLU + typed output store.
int quactlize_gate_up_run_v1(qkg_gate_up_call_v1 const*, qkg_gate_up_config_v1 const*,
    qkg_gate_up_layout_v1 const*);

// Indexed rows may be emitted in an existing compact order. input_rows maps
// each output row to its original token/slot row; NULL keeps original order.
// The caller provides a permutation of [0,rows), ready on the same stream.
// A nonzero device status poisons output instead of consuming invalid routing.
typedef struct {
    uint32_t version, size;
    qkg_gate_up_call_v1 call;
    int32_t const * input_rows;
    int32_t const * status;
} qkg_gate_up_call_v2;
int quactlize_gate_up_run_v2(qkg_gate_up_call_v2 const*, qkg_gate_up_config_v1 const*,
    qkg_gate_up_layout_v1 const*);

// Create a separate paired runtime artifact from canonical Q4/Q8 planes.
// merged=1 means gate holds [E,2N,K], up pointers are NULL. Otherwise gate
// and up each hold [E,N,K]. Input and output spans must be disjoint.
typedef struct {
    uint32_t version, size;
    int32_t qtype, n, k, experts, merged;
    uint8_t const * gate_low, * gate_units, * up_low, * up_units;
    uint8_t * low, * units;
} qkg_gate_up_repack_v1;
int quactlize_gate_up_repack_v1(qkg_gate_up_repack_v1 const*,
    qkg_gate_up_layout_v1 const*, void* stream);

// Measured N512/K2048 cohort only. Returns QKG_SHAPE outside its exact
// precision/operator/token scope. No timing or device work in selection.
int quactlize_gate_up_select_v1(int qtype, int n, int k, int experts,
    int tokens, int compute_type, qkg_gate_up_config_v1*);

#ifdef __cplusplus
}
#endif
