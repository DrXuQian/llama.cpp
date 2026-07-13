#pragma once
// Thin C ABI for the external MoE grouped-GEMM .so (wraps DeepGEMM; cutlass/nvrtc stay inside the .so, llama.cpp
// sees only these declarations -- the cuBLAS model). Build libppu_moe.so from DeepGEMM's JIT runtime + this entry.
//
// Two layouts are exposed. Both take bf16 A/B/out with B = [n_experts, N, K] bf16.
// Returns 0 on success, non-zero if the .so has no kernel for this shape/arch (caller falls back to the inline path).
#ifdef __cplusplus
extern "C" {
#endif

// ---- MASKED layout (preferred) ----
// A = [n_experts, max_m, K] bf16, out = [n_experts, max_m, N] bf16, masked_m[g] = real row count of expert g.
// masked_m needs NO alignment; the kernel skips whole row-blocks past masked_m[g], and its BLOCK_M is 64 or 128.
// So an expert holding 32 rows costs ceil(32/64)*64 = 64 padded rows, vs 128 in the contiguous layout below.
// Rows [masked_m[g], max_m) of A may be uninitialized -- garbage in row r only reaches out[g][r], which you drop.
// `expected_m` = average real rows per expert (steers the tile heuristic).
int ppu_moe_grouped_gemm_bf16_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int max_m, int N, int K, int n_experts, int expected_m, void * stream);

// ---- CONTIGUOUS layout (fallback: fewer buffer rows when expert loads are badly imbalanced) ----
// Required per-expert row alignment of the compact layout. DeepGEMM pins BLOCK_M to exactly this value
// (heuristics/sm90.hpp:33) and the kernel reads ONE expert id per BLOCK_M row block, from that block's first row.
// So each expert's row segment must start (and therefore end) on a multiple of it -- otherwise a block straddles
// two experts and is silently computed against the wrong weights. Typically 128.
int ppu_moe_row_alignment(void);

// A = [total_rows, K] bf16 (gathered, expert-grouped and expert-ordered), out = [total_rows, N] bf16,
// m_indices[row] = expert id of compact row `row`.
// "_nopad" = this entry does NOT pad for you; `total_rows` must ALREADY include the per-expert padding required by
// ppu_moe_row_alignment(). It does NOT mean the layout is padding-free -- it isn't. `expected_m` is ignored here.
int ppu_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

#ifdef __cplusplus
}
#endif
