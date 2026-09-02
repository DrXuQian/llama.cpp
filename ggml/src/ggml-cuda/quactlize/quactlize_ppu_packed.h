#pragma once
// Host-pointer producer half of libquactlize_ppu's fully-quantized artifact ABI.

#include <stdint.h>

#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bytes written by quactlize_ppu_prepare_units for one dense tensor. Returns -1 for an invalid shape or qtype.
// Grouped output requires experts * this many bytes.
int64_t quactlize_ppu_units_bytes(int n, int k, int qtype);

// blocks are official GGUF [N,K/256,raw-byte] bytes; units are [K-unit,N,unit-byte].
int quactlize_ppu_prepare_units(uint8_t const* blocks, uint8_t* units, int n, int k, int qtype);

// blocks are [E,N,K/256,raw-byte]; units are [E,K-unit,N,unit-byte].
int quactlize_ppu_prepare_units_grouped(uint8_t const* blocks, uint8_t* units,
                                        int n, int k, int experts, int qtype);

// Legacy complete host-only Xplane artifact seam. blocks/recovered are official GGUF
// [experts,N,K/256,raw-byte] records. experts must be positive (use 1 for a dense tensor). New K-pack loaders must
// use the arrangement-v2 producer below and the matching arrangement-v2 dense/grouped device consumer; passing
// those bytes to a legacy/default-reader entry can silently reinterpret the physical layout.
//
// The inverse is deliberately separate from dequantization: a loader can first require a byte-exact round trip of
// the shuffled representation, then pass the recovered official blocks to quactlize_ppu_dequantize. The `_v1`
// suffix prevents an older library with a different pointer contract from accepting either call.
int quactlize_ppu_prepare_fully_quantized_v1(uint8_t const* blocks,
                                             uint8_t* low, uint8_t* high, uint8_t* units,
                                             int n, int k, int experts, int qtype);
int quactlize_ppu_recover_fully_quantized_v1(uint8_t const* low, uint8_t const* high,
                                             uint8_t const* units, uint8_t* recovered,
                                             int n, int k, int experts, int qtype);

// Physical-layout-aware complete producer/inverse for online loaders. Unlike
// the legacy v1 seam, these entries never infer a TileK/Xplane arrangement.
// The descriptor is validated by the same format-selected library that places
// the code planes. Metadata remains byte-neutral and shares the v1 unit ABI.
//
// The raw records and all three resident planes are expert-major.  Expert e is one independent contiguous byte slice
// in each allocation, and the same e names the same expert in every allocation:
//
//   blocks_e/recovered_e = base + e * (N*(K/256)*GGUF-block-bytes)
//   low_e                 = low  + e * (N*K*bits/8)
//   high_e (when present) = high + e * (N*K*high_bits/8)
//   units_e               = units + e * quactlize_ppu_units_bytes(N,K,qtype)
//
// Thus low has experts*N*K*bits/8 bytes, high has experts*N*K*high_bits/8 bytes, units has
// experts*quactlize_ppu_units_bytes(N,K,qtype) bytes, and blocks/recovered have
// experts*N*(K/256)*GGUF-block-bytes bytes.  Experts are not interleaved inside any allocation.  Preparing or
// recovering one such slice with experts=1 is byte-for-byte equivalent to that slice of one experts>1 call.  Calls on
// mutually disjoint slices may execute concurrently when each call's immutable arrangement descriptor is stored
// outside every participating tensor range.
//
// high must be null exactly when arrangement->high_bits==0 and non-null otherwise.  N and K must be positive
// multiples of 256; because their byte-neutral metadata transport pairs adjacent K superblocks, Q3_K and Q6_K
// additionally require K to be a multiple of 512.
//
// All nonempty tensor byte ranges must be pairwise disjoint; in-place conversion and partial aliasing return 30
// before any tensor write. The arrangement descriptor is copied before tensor writes and need not remain live for
// the duration of the call. Unrepresentable paired-unit geometry returns 24 and byte/address-size overflow returns
// 26, both before range validation or tensor writes.
//
// The resulting low/high/units are consumed only by the arrangement-v2 dense/grouped device APIs declared in
// quactlize_ppu_device.h. A format-selected library accepts only its own qtype; passing a valid descriptor for
// another format returns 29 before reading or writing tensor bytes.
int quactlize_ppu_prepare_fully_quantized_for_arrangement_v2(
    uint8_t const* blocks, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int quactlize_ppu_recover_fully_quantized_for_arrangement_v2(
    uint8_t const* low, uint8_t const* high, uint8_t const* units,
    uint8_t* recovered, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);

#ifdef __cplusplus
}
#endif
