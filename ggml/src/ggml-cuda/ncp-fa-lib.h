#pragma once
// C ABI of libncp_fa.so (FlashAttention-3, hopper/). See ncp_flash_lib/fa/flash_api_c.cpp.
// Mirrored by ncp-lib.h, which is what ggml code actually calls.
//
// v3 vs v2, in the signature:
//   head_dim_v                 FA3 allows V's head dim to differ from Q/K's (params.dv). FA2 could not express it.
//   window_size_left/right     FA3 does local attention. Pass (-1,-1) for full, (-1,0) for causal. A real window
//                              would let SWA models use the .so instead of falling back -- not wired up yet.
//   softcap                    passed RAW. v2 folded it into the scale; v3 keeps them separate.
//
// The symbol carries _v3 deliberately: C does not mangle, so a v2 .so would accept this call and read the trailing
// arguments as garbage. A rename cannot be silently mis-linked -- the dlsym fails and the inline path runs.
//
// Returns 0 on success; non-zero means "not supported / failed, use the inline path".
#ifdef __cplusplus
extern "C" {
#endif
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
    int dtype,                      // 0 = fp16, 1 = bf16
    void * stream);
#ifdef __cplusplus
}
#endif
