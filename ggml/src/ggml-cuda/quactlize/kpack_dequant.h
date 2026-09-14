#pragma once
#include <stdint.h>
#include "kpack_execution.h"

// Independent resident-weight expansion. No GEMM, allocation, transfer,
// synchronization or scale-ready cache. Caller owns the current stream.
// operation=0: FP16 scale/zero, each [E,K/group,N], canonical SF arithmetic.
// operation=1: BF16 weights [E,N,K], FP32 raw-GGUF arithmetic, one final RNE.
// All experts in the supplied contiguous slice are expanded. Sparse selection
// and the cost of constructing a selected expert slice are not hidden here.
// Candidate launches require E<=65535 and K/32<=65535 (three-dimensional grid).
typedef struct qzd_call_v1 {
    uint32_t version, size;
    int32_t qtype, n, k, experts, operation, config;
    void const *low, *high, *units;
    uint64_t low_bytes, high_bytes, unit_bytes;
    void *output, *zero;
    uint64_t output_bytes;
    void *stream;
} qzd_call_v1;

#ifdef __cplusplus
extern "C" {
#endif
// SF config0=unchanged production kernel/block256; 1=N16/block256;
// 2=N32/block256; 3=N32/block128. Full config0=direct K-major/block256;
// 1=N32K32 shared transpose/block256; 2=same tile/block128.
// Full3/4/5=N32K128 scalar/pair, scalar/uint4, uint4/uint4;
// Q4/Q5-only Full6..9=expanded shared experiments;
// Full10/11=packed-code exchange K128/K256, Full12=K256 K-fast CTA order.
// Config IDs are experimental candidates, not production heuristic choices.
int quactlize_kpack_dequant_v1(qzd_call_v1 const*, quactlize_ppu_placed_arrangement_v2 const*);
int quactlize_kpack_dequant_probe_v1(int* l2_bytes, int* sm_count, int* warp_size);
#ifdef __cplusplus
}
#endif
