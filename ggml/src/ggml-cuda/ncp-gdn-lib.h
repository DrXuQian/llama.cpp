#pragma once

#include <stddef.h>

// C ABI of libncp_gdn.so (actlizeLA PPU1.0 residual/full-chunk). See ncp_flash_lib/gdn/.
// Mirrored by ncp-lib.h, which is what ggml code actually calls.
//
// Layout: q,k[n_seqs,T,H,S]  v[n_seqs,T,HV,S]  gate/beta[n_seqs,T,HV]
//         h0,ht[n_seqs,HV,S,S]  o[n_seqs,T,HV,S].
// q/k/v/beta/o are BF16. gate is the raw natural-log decay in FP32. h0/ht are FP32 [k][v].
// H = key/query heads, HV = value heads. The PPU1.0 backend requires S=128 and applies 1/sqrt(128) internally.
// The same entry serves prefill and decode, including sequence tails shorter than the 64-token chunk.
//
// Returns: 0 success, -1 unsupported or invalid contract, 3 workspace missing, misaligned, or too small.
#ifdef __cplusplus
extern "C" {
#endif

int ncp_gdn_actlize_ppu10_residual_bf16_v1(const void *  q,
                                           const void *  k,
                                           const void *  v,
                                           const float * gate,
                                           const void *  beta,
                                           const float * h0,
                                           void *        o,
                                           float *       ht,
                                           int           n_seqs,
                                           int           T,
                                           int           H,
                                           int           HV,
                                           int           S,
                                           float         scale,
                                           void *        ws,
                                           size_t        ws_bytes,
                                           void *        stream);

size_t ncp_gdn_actlize_ppu10_residual_bf16_workspace_size_v1(int n_seqs, int T, int H, int HV, int S);

#ifdef __cplusplus
}
#endif
