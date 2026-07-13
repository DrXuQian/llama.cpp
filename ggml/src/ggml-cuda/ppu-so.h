#pragma once
// dlopen wrapper around the external MoE grouped-GEMM kernel .so (libppu_moe.so, wraps DeepGEMM).
//
// Why dlopen and not link-time: llama.cpp must build with ZERO cutlass / DeepGEMM / torch headers or link deps.
// The heavy kernel is built separately from the thirdparty/DeepGEMM submodule into a standalone .so that hides
// cutlass (and DeepGEMM's NVRTC JIT) inside it; only the .so binary (+ ppu-moe-so.h) is shipped to users, never
// the submodule source. At runtime we dlopen the .so and dlsym the C ABI; if the .so is absent, or has no kernel
// for the requested shape/dtype, the caller transparently falls back to the inline ggml path.
//
// .so location is resolved (first hit wins):
//   1. env  GGML_PPU_MOE_SO   (absolute path to the .so)
//   2. bare soname on the loader search path:  libppu_moe.so
//
// Everything here is a no-op (available()==false) unless the build defines GGML_PPU_SO.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// ---- MoE grouped-GEMM (mirrors ppu-moe-so.h; see it for the layout contracts) ----
// Which layout the hook uses is decided by WHICH SYMBOL THE .so EXPORTS. Preference: nopad > masked.

// NoPad: dense A = [total_rows, K], no padding at all. Only exported by a kernel that truly honours it.
bool ggml_ppu_so_moe_nopad_available(void);
int  ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// Masked: A = [n_experts, max_m, K], out = [n_experts, max_m, N], masked_m[g] unaligned.
bool ggml_ppu_so_moe_masked_available(void);
int  ggml_ppu_so_moe_grouped_gemm_bf16_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int max_m, int N, int K, int n_experts, int expected_m, void * stream);

// Diagnostics only: the padded-contiguous alignment the .so's own kernel needs; 0 if the .so is absent.
int  ggml_ppu_so_moe_row_alignment(void);

#ifdef __cplusplus
}
#endif
