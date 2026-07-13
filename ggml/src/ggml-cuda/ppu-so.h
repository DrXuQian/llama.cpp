#pragma once
// dlopen wrapper around the external FlashAttention kernel .so (libppu_fa.so).
//
// Why dlopen and not link-time: llama.cpp must build with ZERO cutlass / flash-attention / torch headers or link
// deps. The heavy kernels are built separately from the thirdparty/flash-attention submodule into a standalone .so
// that hides cutlass inside it; only the .so binary (+ the thin ABI header ppu-fa-so.h) is shipped, never the
// submodule source. At runtime we dlopen the .so and dlsym the C ABI; if it is absent or has no kernel for the
// requested shape, the caller transparently falls back to the inline ggml path.
//
// The .so is resolved (first hit wins):
//   1. env  GGML_PPU_FA_SO   (absolute path to the .so)
//   2. bare soname on the loader search path:  libppu_fa.so
//
// Everything here is a no-op (available()==false) unless the build defines GGML_PPU_SO.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// ---- FlashAttention (mirror of ppu-fa-so.h's ppu_flash_attn_fwd) ----
bool ggml_ppu_so_fa_available(void);
int  ggml_ppu_so_flash_attn_fwd(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float logit_softcap, int is_causal, int dtype, void * stream);

#ifdef __cplusplus
}
#endif
