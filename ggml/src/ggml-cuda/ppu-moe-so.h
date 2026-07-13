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

// ---- Capability query: what per-expert row alignment does the kernel need? ----
// == 1  the kernel takes each expert's rows as they are ("NoPad"). The caller then hands it a DENSE compact A with
//       total_rows = n_tokens * n_expert_used -- an identity, so no host value depends on the routing and the whole
//       path is D2H/H2D/sync free. This is what the PPU kernel team's bf16_grouped_deep_gemm_NoPad provides, and it
//       is the path the hook prefers: zero padding, zero wasted flops, compact scratch.
// >  1  each expert's row segment must start and end on a multiple of it (upstream/public DeepGEMM pins BLOCK_M to
//       exactly this value). The padded total is then data-dependent, i.e. it would cost a D2H + stream sync, so the
//       hook uses the masked entry below instead. Typically 128.
// == 0  the .so is absent.
int ppu_moe_row_alignment(void);

// ---- DENSE CONTIGUOUS / "NoPad" (preferred; requires ppu_moe_row_alignment() == 1) ----
// A = [total_rows, K] bf16, expert-grouped and expert-ordered, NO padding of any kind.
// out = [total_rows, N] bf16. m_indices[r] = the expert owning compact row r.
// `expected_m` = average rows per expert (steers the tile heuristic).
//
// Historical note on the name: "nopad" originally meant "this entry does not pad FOR you". With a kernel that
// reports alignment 1 it means what it says -- there is nothing to pad.
int ppu_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// ---- MASKED layout (fallback for a kernel that reports alignment > 1) ----
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

#ifdef __cplusplus
}
#endif
