#pragma once
// C ABI of the external PPU MoE grouped-GEMM .so (wraps DeepGemm; cutlass stays inside the .so, llama.cpp sees
// only these declarations -- the cuBLAS model). Mirrored by ncp-lib.h, which is what ggml code actually calls,
// exactly as ncp-fa-lib.h is mirrored there. libncp_moe.so is built in the ncp_flash_lib repo.
//
// EVERY NAME BELOW IS DeepGemm's OWN, PREFIXED WITH ncp_ AND OTHERWISE UNCHANGED. Strip the prefix and you have the
// upstream entry this re-implements: ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad <-> DeepGemm's
// m_grouped_gemm_bf16_bf16_bf16_nt_nopad. The trailing word is the scheduler (GroupedContiguous / GroupedMasked /
// GroupedNoPad) and carries no other meaning: what actually separates the three -- padding, whether shape_m must be
// exact, whether a caller can stay CUDA-graph capturable -- is written out per entry below, deliberately not encoded
// in the identifiers.
//
// These must stay byte-identical to what ncp_flash_lib exports; the loader dlsym's these exact strings. Renaming one
// side only resolves to null rather than failing loudly, so the symptom is silence: correct output at inline speed.
// FA sat that way for a while -- ncp-lib.cu asked for ncp_flash_attn_fwd_v3 while the library still exported
// ppu_flash_attn_fwd_v3 -- until ncp_flash_lib renamed its side to match. Both sides move together or not at all.
//
// out/A/B are bf16; m_indices[row] = expert id of compact row `row`; expected_m = total_rows / n_experts.
// A = [total_rows, K] bf16 (gathered/permuted), B = [n_experts, N, K] bf16, out = [total_rows, N] bf16.
// Returns 0 on success, non-zero if the .so has no kernel for this (N,K) (caller falls back to the inline AIU path).
#ifdef __cplusplus
extern "C" {
#endif

// Required per-expert row alignment of the compact layout (DeepGemm pins BLOCK_M to it and reads one expert id per
// BLOCK_M block). Each expert's row segment must start and end on a multiple of this, or the GEMM silently computes
// a straddling block against the wrong expert's weights. Typically 128.
int ncp_get_m_alignment_for_contiguous_layout(void);

// GroupedContiguous: the PADDED layout. Every expert's row segment must start and end on a multiple of the alignment
// above, and the scheduler derives its block count from an exact shape_m -- so the caller owes it
// sum(align(m_e, 128)), which is data-dependent and can only be had by reading the per-expert counts back from the
// device. That read is a stream sync, so a caller on this entry cannot be CUDA-graph capturable. ggml does not use it.
int ncp_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// GroupedMasked: masked_m[e] = number of rows for expert e (int32_t[n_experts]). No alignment or padding needed --
// empty blocks are skipped, so an upper bound on shape_m is enough. max_block_n limits the tile search space for N
// (0 = default 256). ggml does not use it.
int ncp_m_grouped_gemm_bf16_bf16_bf16_nt_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int total_rows, int N, int K, int n_experts, int expected_m, int max_block_n, void * stream);

// GroupedNoPad -- THE ENTRY THE ggml HOOK USES, and the only one its availability is gated on. A is expert-grouped
// and expert-ordered but not padded at all, so total_rows is the identity n_tokens * n_expert_used: nothing about it
// is data-dependent, which is what lets the whole MoE path stay free of a device-to-host read and therefore
// CUDA-graph capturable. That, not performance, is why ggml is on this entry and not the contiguous one.
//   m_rows[e]      = per-expert row count
//   m_indices[row] = per-row expert id. The GEMM ignores it; the batched-GEMV decode path indexes B by it, and must
//                    refuse rather than compute against expert 0 if it is null.
// A library that exports the two entries above but not this one can serve nothing: the hook checks for this symbol.
int ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
    const void * A, const void * B, void * out, const int * m_indices, const int * m_rows,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

#ifdef __cplusplus
}
#endif
