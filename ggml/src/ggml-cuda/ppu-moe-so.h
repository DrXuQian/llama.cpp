#pragma once
// Thin C ABI for the external MoE grouped-GEMM .so (wraps DeepGEMM; cutlass/nvrtc stay inside the .so, llama.cpp
// sees only these declarations -- the cuBLAS model). Build libppu_moe.so from DeepGEMM's JIT runtime + this entry.
//
// Two layouts are exposed; ppu_moe_row_alignment() below says which one the kernel can actually take.
// Both use bf16 A/B/out with B = [n_experts, N, K] bf16.
// Returns 0 on success, non-zero if the .so has no kernel for this shape/arch (caller falls back to the inline path).
#ifdef __cplusplus
extern "C" {
#endif

// The hook picks its layout by WHICH SYMBOL THE .so EXPORTS -- presence is the capability query. Preference order:
// nopad > masked. A .so that can only do padded contiguous exports neither, and the hook falls back to inline ggml.

// ---- NoPad: DENSE contiguous (preferred) ----
// A = [total_rows, K] bf16, expert-grouped and expert-ordered, with NO padding of any kind.
// out = [total_rows, N] bf16. m_indices[r] = the expert owning compact row r.
//
// This is the contract of the PPU kernel team's `bf16_grouped_deep_gemm_NoPad`
// (deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices); it runs computeBlockInfoKernel on the
// device to build the block->group map when n_experts >= 128). Because nothing is padded, the row count the kernel
// needs is just total_rows = n_tokens * n_expert_used -- an identity. No host value depends on the routing, so the
// whole llama.cpp path is D2H/H2D/sync free, and the scratch is the compact total_rows*K.
//
// ONLY export this symbol if you really honour that. Public DeepGEMM does NOT: its bf16 grouped kernel is
// `bf16_grouped_deep_gemm_contiguous`, which demands ppu_moe_row_alignment()-row segments.
int ppu_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// ---- MASKED (what a padded-contiguous-only kernel, e.g. public DeepGEMM, can still offer sync-free) ----
// A = [n_experts, max_m, K] bf16, out = [n_experts, max_m, N] bf16, masked_m[g] = real row count of expert g.
// masked_m needs NO alignment; the kernel skips whole row-blocks past masked_m[g], and its BLOCK_M is 64 or 128.
// So an expert holding 32 rows costs ceil(32/64)*64 = 64 padded rows, vs 128 for a 128-aligned contiguous
// layout. Crucially, max_m only has to be an UPPER BOUND on the per-expert row count, and a host-known one exists
// (an expert holds at most one row per token) -- that is what keeps this path free of any D2H.
// Rows [masked_m[g], max_m) of A may be uninitialized -- garbage in row r only reaches out[g][r], which you drop.
// `expected_m` = average real rows per expert (steers the tile heuristic).
int ppu_moe_grouped_gemm_bf16_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int max_m, int N, int K, int n_experts, int expected_m, void * stream);

// ---- PADDED CONTIGUOUS (diagnostics / standalone tests only; the hook never calls it) ----
// Requires each expert's row segment to start and end on a multiple of ppu_moe_row_alignment(). The padded total is
// therefore data-dependent -- a host value that depends on the routing -- which is precisely why the hook cannot use
// this layout without a D2H + stream sync. Kept because it is what public DeepGEMM natively provides.
int ppu_moe_row_alignment(void);
int ppu_moe_grouped_gemm_bf16_contiguous(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

#ifdef __cplusplus
}
#endif
