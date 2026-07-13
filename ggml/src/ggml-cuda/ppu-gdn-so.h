#pragma once
// Thin C ABI for the external gated-delta-net .so (libppu_gdn.so), which wraps FLA's Triton
// fused_recurrent_gated_delta_rule kernel, AOT-compiled to embedded cubin (torch/python/triton-free; links only the
// CUDA driver). llama.cpp sees only this declaration -- the same dlopen "cuBLAS model" as ppu-fa-so.h / ppu-moe-so.h.
//
// Covers the RECURRENT path of the ggml GDN op (GGML_OP_GATED_DELTA_NET), for the subset where ggml's math+layout map
// 1:1 to FLA (verified): non-KDA (scalar gate), K_snapshot==1 (final state only), all tensors F32 and contiguous.
// The kernel is specialized per (H, HV, S) shape (Triton constexprs baked at AOT); ppu_gdn_recurrent dispatches by
// shape and returns -1 for any (H,HV,S) it wasn't built for -> caller falls back to the inline ggml kernel.
//
// Layouts (contiguous, matching ggml's native GDN tensors, element order):
//   q,k  : [n_seqs, T, H,  S]   (ggml ne {S,H,T,seqs})
//   v    : [n_seqs, T, HV, S]   (ggml ne {S,HV,T,seqs}); HV = H * v_repeat (GVA)
//   g    : [n_seqs, T, HV]      (log-space gate; ggml ne {1,HV,T,seqs})
//   beta : [n_seqs, T, HV]      (headwise scalar)
//   h0   : [n_seqs, HV, S, S]   initial recurrent state (ggml src_state)
//   o    : [n_seqs, T, HV, S]   attention output (-> dst attention-score region)
//   ht   : [n_seqs, HV, S, S]   final recurrent state (-> dst state tail / fused cache slot)
// scale = 1/sqrt(S). Returns 0 on success, -1 unsupported shape, -2 launch error (both -> inline fallback).
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int ppu_gdn_recurrent(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale, void * stream);

// Chunked prefill variant (WY tensor-core path; needs L2-normalized k for numerical stability, i.e. real models).
// h0/ht are in FLA [k][v] layout (NOT ggml's [v][k]); the caller transposes. Returns -1 for unsupported (H,HV,S).
// Bytes of device scratch ppu_gdn_chunked needs. A pure formula, so the caller can size the workspace before it
// knows whether the shape is even compiled in.
size_t ppu_gdn_chunked_workspace_size(int n_seqs, int T, int H, int HV, int S);

// The .so allocates NOTHING: `ws` is device scratch supplied by the caller (llama.cpp hands it a ggml_cuda_pool_alloc
// buffer). Doing this inside the .so meant a cudaMalloc/cudaFree of ~180 MB on every one of the model's 30 GDN
// layers, outside ggml's pool -- slow, prone to OOM once llama.cpp had filled the device, and enough on its own to
// keep the path out of CUDA-graph capture.
int ppu_gdn_chunked(
    const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale,
    void * ws, size_t ws_bytes, void * stream);

#ifdef __cplusplus
}
#endif
