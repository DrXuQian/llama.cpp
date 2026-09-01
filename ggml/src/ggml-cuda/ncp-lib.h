#pragma once
// dlopen wrapper around the external PPU kernel .so files (libncp_fa.so, libncp_moe.so, libncp_gdn.so).
//
// Why dlopen and not link-time: llama.cpp must build with ZERO cutlass / flash-attention / flash-linear-
// attention / torch headers or link deps. The heavy kernels are built separately from the private ncp_flash_lib
// repo into standalone .so files that hide their internals; only the .so binaries (+ ABI headers) are shipped
// to users, never the kernel source. At runtime we dlopen the .so and dlsym the C ABI; if the .so is absent or
// has no kernel for the requested shape, the caller transparently falls back to the inline ggml path.
//
// WHICH libraries are loaded is a BUILD decision, one flag per library:
//   cmake -DGGML_NCP_FA=ON   -> libncp_fa.so   (FlashAttention-3)
//   cmake -DGGML_NCP_MOE=ON  -> libncp_moe.so  (DeepGemm grouped-GEMM)
//   cmake -DGGML_NCP_GDN=ON  -> libncp_gdn.so  (flash-linear-attention gated delta net)
// A flag that is off compiles that feature away entirely: no dlopen, no symbols, no hook.
//
// WHERE each .so is found: next to the binaries that load them (build/bin, beside libggml-*.so). ggml-cuda carries
// an $ORIGIN runpath, so the bare soname resolves there -- no path option, no env var (an env var is dropped
// silently by any wrapper in the launch chain, and a lost .so is a silent slowdown, not an error). Point
// LD_LIBRARY_PATH somewhere else to load a different copy.
//
// THE FILE NAMES AND THE SYMBOL NAMES ARE NOT THE SAME QUESTION. The llama.cpp side -- these files, the wrappers
// below, the build flag -- is ncp_lib. The symbols dlsym'd are whatever the library actually exports, which for MoE is
// DeepGemm's own names prefixed with ncp_ (ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad and friends, see ncp-moe-lib.h).
// Renaming one side only resolves to null rather than failing loudly: that is what FA did until ncp_flash_lib renamed
// its export to ncp_flash_attn_fwd_v3 and its .so to libncp_fa.so, matching the dlsym below. Both sides must move
// together, and a stale .so on the loader path still reads as "kernel unavailable" -- correct output, inline speed.
//
// Everything here is a no-op (available()==false) unless the build enables that feature's flag.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

// ---- FlashAttention-3 (mirror of ncp-fa-lib.h's ncp_flash_attn_fwd_v3) ----
// head_dim_v may differ from head_dim (FA2 could not express that); window (-1,-1) = full, (-1,0) = causal.
bool ggml_ncp_lib_fa_available(void);
int  ggml_ncp_lib_flash_attn_fwd(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv,
    int head_dim, int head_dim_v,
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float softcap,
    int is_causal, int window_size_left, int window_size_right,
    int dtype, void * stream);

// ---- MoE grouped-GEMM (mirror of ncp-moe-lib.h; the .so exports these as ncp_m_grouped_gemm_*) ----

// Gated on the REAL GroupedNoPad entry, not the padded one: the hook only ever calls that, and a library exporting
// the others without it cannot serve a single shape.
bool ggml_ncp_lib_moe_nopad_available(void);

// Per-expert row alignment the PADDED compact layout would have to satisfy; 0 if the .so is absent. The NoPad path
// does not need it -- it is here so a caller that ever wants the padded entry does not have to guess.
int  ggml_ncp_lib_get_m_alignment_for_contiguous_layout(void);

int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// GroupedMasked: masked_m[e] = per-expert row count.
int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int total_rows, int N, int K, int n_experts, int expected_m, int max_block_n, void * stream);

// Real GroupedNoPad -- what the hook calls. m_rows[e] = per-expert row count; m_indices[row] = per-row expert id.
// The GEMM ignores m_indices; the batched-GEMV decode path inside the .so indexes B by it, and refuses rather than
// computing against expert 0 when it is null.
int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
    const void * A, const void * B, void * out, const int * m_indices, const int * m_rows,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// ---- GDN (mirror of ncp-gdn-lib.h) ----
// Two arms: recurrent (decode, all-f32, STATE_V_FIRST=1) and chunked (prefill, bf16 q/k/v/o, STATE_V_FIRST=0).
// g is the RAW log-space gate (pre-cumsum); each arm does its own cumsum. Returns 0 ok, -1 unsupported shape,
// 2 module load failed, 3 workspace too small (chunked only). rc != 0 -> inline fallback.
bool    ggml_ncp_lib_gdn_available(void);
bool    ggml_ncp_lib_gdn_chunked_available(void);
int     ggml_ncp_lib_gdn_recurrent(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta, const float * h0,
    float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S,
    float scale, void * stream);
int     ggml_ncp_lib_gdn_chunked_bf16(
    const void * q, const void * k, const void * v,
    const float * g_raw, const float * beta, const float * h0,
    void * o, float * ht,
    int n_seqs, int T, int H, int HV, int S,
    float scale, void * ws, size_t ws_bytes, void * stream);
size_t  ggml_ncp_lib_gdn_chunked_bf16_workspace_size(int n_seqs, int T, int H, int HV, int S);

// GDN chunked-prefill arm (FLA's WY tensor-core chain). ON by default; set GGML_NCP_GDN_CHUNKED=0 (or false/off)
// to fall back to the recurrent arm for every shape. Says nothing about whether the .so has the kernel -- pair it
// with ggml_ncp_lib_gdn_chunked_available().
bool ggml_ncp_gdn_chunked_enabled(void);

#ifdef __cplusplus
}
#endif
