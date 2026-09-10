#pragma once

#include <stdint.h>
#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t raw_bytes, low_bytes, high_bytes, units_bytes;
} quactlize_ppu_kpack_sizes_v1;

// Canonical producer identity. Q8_0 (qtype=8) uses a distinct K-pack2 map:
// biased int8 low plane + resident FP16 d [E,K/32,N]; no high/zero plane.
// Its metadata is NOT a K-quant packed unit. Consumers must explicitly admit
// this mapping. An older consumer must decline before taking the buffer.
int quactlize_ppu_kpack_canonical_arrangement_v1(
    int qtype, quactlize_ppu_placed_arrangement_v2* arrangement);

// Dense is experts=1. Each plane has contiguous, equal-sized expert slices.
// Same canonical bytes as prepare_fully_quantized_for_arrangement_v2.
int quactlize_ppu_kpack_sizes_for_arrangement_v1(
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    quactlize_ppu_kpack_sizes_v1* sizes);

// All data pointers are on the current device. low/high are 2-byte aligned;
// high is NULL when high_bytes=0. Input and output spans must not overlap.
// No allocation, transfer, dequantization, synchronization or CPU conversion.
// Success means enqueued, not completed. Keep buffers alive until stream work
// completes; other compute/copy streams must wait on a caller-recorded event.
// Return codes: 20 argument, 22 qtype, 24 shape, 26 overflow, 30 overlap,
// 38 descriptor mismatch, 41 immediate runtime/launch error.
int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
    uint8_t const* blocks, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream);

// Same-qtype gate/up raw GGUF [E,N,K] -> canonical K-pack [E,2N,K].
// The order matches llama.cpp --fuse-gate-up-exps: gate then up for EACH
// expert. Output sizes are queried with 2*n. Both sources have the single
// tensor sizes queried with n; n must itself satisfy the format constraints.
// No arithmetic/requantization, temporary concatenation, or new mapping ID.
// The asynchronous lifetime/alignment/nonoverlap contract above also applies.
int quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1(
    uint8_t const* gate, uint8_t const* up,
    uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream);

#ifdef __cplusplus
}
#endif
