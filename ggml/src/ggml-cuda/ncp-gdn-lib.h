#pragma once
// C ABI of libncp_gdn.so (flash-linear-attention gated delta net). See ncp_flash_lib/gdn/.
// Mirrored by ncp-lib.h, which is what ggml code actually calls.
//
// Two forward arms, with different signatures because they serve different paths:
//
//   ncp_gdn_recurrent       decode (any T).  All-f32.  STATE_V_FIRST=1 (ggml's [v][k] state layout).
//   ncp_gdn_chunked_bf16    prefill (T >= 128, multiple of 64).  q/k/v/o are bf16; g_raw, beta, h0, ht stay f32.
//                           STATE_V_FIRST=0 (FLA's [k][v]) -- the ggml hook transposes h0 in and ht out.
//
// Layout (both arms): q,k[n_seqs,T,H,S]  v[n_seqs,T,HV,S]  g/beta[n_seqs,T,HV]  h0,ht[n_seqs,HV,S,S]
//                     o[n_seqs,T,HV,S].  g is the RAW log-space gate (pre-cumsum); each arm does its own cumsum
//                     (the recurrent kernel includes it; the chunked .so runs chunk_local_cumsum as kernel 0).
//
// H = key/query heads, HV = value heads (H <= HV for GVA; H==HV for MHA). S = head dim = K = V.
//
// Returns:  0 success.  -1 unsupported (H,HV,S) -- the .so was not built for this shape; caller falls back to
//           inline.  2 module load failed (cubin/arch mismatch).  3 workspace too small (chunked only).
#ifdef __cplusplus
extern "C" {
#endif

// Recurrent forward (decode path).  All tensors are f32.  No workspace needed.
// h0/ht are [v][k] (STATE_V_FIRST=1) -- pass ggml's state straight through, no transpose.
int ncp_gdn_recurrent(
    const float * q, const float * k, const float * v,
    const float * g,                       // RAW gate (pre-cumsum)
    const float * beta, const float * h0,
    float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S,
    float scale, void * stream);

// Chunked forward (prefill path).  q/k/v/o are bf16; g_raw, beta, h0, ht are f32.
// h0/ht are [k][v] (STATE_V_FIRST=0) -- the ggml hook transposes h0 in and ht out.
// ws is caller-supplied device scratch, sized by ncp_gdn_chunked_bf16_workspace_size().
int ncp_gdn_chunked_bf16(
    const void * q, const void * k, const void * v,       // bf16
    const float * g_raw,                                  // f32, RAW gate (pre-cumsum)
    const float * beta, const float * h0,                  // f32
    void * o, float * ht,                                  // o bf16, ht f32
    int n_seqs, int T, int H, int HV, int S,
    float scale,
    void * ws, size_t ws_bytes, void * stream);

// Bytes of device scratch ncp_gdn_chunked_bf16 needs.  Valid for any shape; the caller can size the
// buffer before it knows whether the shape is compiled in.
size_t ncp_gdn_chunked_bf16_workspace_size(int n_seqs, int T, int H, int HV, int S);

#ifdef __cplusplus
}
#endif
