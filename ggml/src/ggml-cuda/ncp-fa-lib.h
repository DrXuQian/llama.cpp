#pragma once
// C ABI of libncp_fa.so (FlashAttention-3, hopper/). See ncp_flash_lib/fa/flash_api_c.cpp.
// Mirrored by ncp-lib.h, which is what ggml code actually calls.
//
// v3 vs v2, in the signature:
//   head_dim_v                 FA3 allows V's head dim to differ from Q/K's (params.dv). FA2 could not express it.
//   window_size_left/right     FA3 does local attention. Pass (-1,-1) for full, (-1,0) for causal. A real window
//                              would let SWA models use the .so instead of falling back -- not wired up yet.
//   softcap                    passed RAW. v2 folded it into the scale; v3 keeps them separate.
//   seqused_k                  device [batch] i32, or null. The number of leading K/V entries that actually hold
//                              tokens. ggml rounds its KV cache view up so the graph shape stays reusable, so
//                              seqlen_k is only an upper bound; FA has no mask input and reads the causal offset off
//                              seqlen_k - seqlen_q, which without this lets a query row reach into the later tokens
//                              of its own ubatch. Null keeps seqlen_k at face value. It MUST be a device pointer, not
//                              a scalar: the value changes every ubatch while a captured CUDA graph is replayed with
//                              the same launch arguments.
//
// The symbol carries _v3 deliberately: C does not mangle, so a v2 .so would accept this call and read the trailing
// arguments as garbage. A rename cannot be silently mis-linked -- the dlsym fails and the inline path runs. NOTE: the
// seqused_k parameter was added WITHOUT renaming the symbol, so a mismatched .so/llama.cpp pair will misread the
// argument list -- the two repos must be updated together.
//
// Returns 0 on success; non-zero means "not supported / failed, use the inline path".
#ifdef __cplusplus
extern "C" {
#endif
// clang-format off
int ncp_flash_attn_fwd_v3(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv,
    int head_dim, int head_dim_v,
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float softcap,
    int is_causal, int window_size_left, int window_size_right,
    const int * seqused_k,          // device [batch] i32, or null
    int dtype,                      // 0 = fp16, 1 = bf16
    void * stream);
// clang-format on
#ifdef __cplusplus
}
#endif
