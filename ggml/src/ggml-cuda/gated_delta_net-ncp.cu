#include "gated_delta_net-ncp.cuh"

#ifdef GGML_NCP_GDN

#include "ncp-lib.h"
#include "common.cuh"
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Transpose each [S,S] state block between ggml's [v][k] and FLA's [k][v]. grid = n_seqs*HV blocks.
static __global__ void ncp_gdn_state_transpose(const float * __restrict__ in, float * __restrict__ out, int S) {
    const int blk = blockIdx.x;
    const float * bi = in  + (size_t) blk * S * S;
    float *       bo = out + (size_t) blk * S * S;
    for (int idx = threadIdx.x; idx < S * S; idx += blockDim.x) {
        const int i = idx / S, j = idx % S;   // out[i][j] = in[j][i]
        bo[i * S + j] = bi[j * S + i];
    }
}
// Copy a possibly-strided src (nb in elements) into a contiguous [n_seqs,T,HV,S] buffer. grid = n_seqs*T*HV rows.
static __global__ void ncp_gdn_make_contig(const float * __restrict__ src, float * __restrict__ dst, int S,
        long long s1, long long s2, long long s3, int HV, int T) {
    const int row = blockIdx.x;                 // flat (seq*T + t)*HV + hv
    const int hv  = row % HV;
    const int t   = (row / HV) % T;
    const int seq = row / (HV * T);
    const float * sp = src + (long long) seq * s3 + (long long) t * s2 + (long long) hv * s1;
    float * dp = dst + (long long) row * S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) dp[i] = sp[i];
}
// Same, but casting to bf16 -- the chunked .so wants bf16 q/k/v. The write side halves, so this is CHEAPER than the
// f32->f32 copy it replaces for v; q and k pay a new cast they did not before.
static __global__ void ncp_gdn_make_contig_bf16(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst, int S,
        long long s1, long long s2, long long s3, int HV, int T) {
    const int row = blockIdx.x;
    const int hv  = row % HV;
    const int t   = (row / HV) % T;
    const int seq = row / (HV * T);
    const float * sp = src + (long long) seq * s3 + (long long) t * s2 + (long long) hv * s1;
    nv_bfloat16 * dp = dst + (long long) row * S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) dp[i] = __float2bfloat16(sp[i]);
}

// THE GVA HEAD PAIRING IS NOT THE SAME ON BOTH SIDES OF THIS SEAM.
//
//   FLA   (fused_recurrent.py:68, and the chunked chain likewise)   i_h = i_hv // (HV / H)   -- repeat_INTERLEAVE
//   ggml  (the inline kernel above, and ggml.c's CPU reference)     iq1 = h_idx % H          -- repeat / TILE
//
// At H=16, HV=32 those are entirely different maps: FLA pairs V heads {0,1} with QK head 0, ggml pairs V head h with
// QK head h % 16, so its V heads 16..31 reuse QK heads 0..15. Feeding ggml's tensors to FLA's kernel pairs almost
// every head with the wrong one, and the output is garbage that still looks like language.
//
// It hid because it is INVISIBLE whenever H == HV: R = 1 makes h // 1 and h % H the same function. Every
// test-backend-ops GDN case had v_repeat = 1, and the .so's own golden harness compares FLA against FLA, so both
// sides used FLA's convention and agreed bit-exactly. The one combination nobody tested -- GVA, through this hook --
// is the only one the model runs.
//
// The fix has to be on the q/k side: no permutation of the QK heads can reproduce ggml's map, because hv/R is
// constant across a group of R V heads while hv % H varies within it. So TILE q/k up to HV heads --
// q'[hv] = q[hv % H] -- and call the .so with H' = HV, which makes FLA's R = 1 and its i_h = i_hv. The .so must
// therefore be built for (HV, HV, S), i.e. ./build.sh "32,32,128", not "16,32,128".
static __global__ void ncp_gdn_tile_qk_heads(const float * __restrict__ src, float * __restrict__ dst,
        int S, int H, int HV, long long s1, long long s2, long long s3, int T) {
    const int row = blockIdx.x;                 // flat (seq*T + t)*HV + hv
    const int hv  = row % HV;
    const int t   = (row / HV) % T;
    const int seq = row / (HV * T);
    const float * sp = src + (long long) seq * s3 + (long long) t * s2 + (long long) (hv % H) * s1;
    float * dp = dst + (long long) row * S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { dp[i] = sp[i]; }
}
static __global__ void ncp_gdn_tile_qk_heads_bf16(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
        int S, int H, int HV, long long s1, long long s2, long long s3, int T) {
    const int row = blockIdx.x;
    const int hv  = row % HV;
    const int t   = (row / HV) % T;
    const int seq = row / (HV * T);
    const float * sp = src + (long long) seq * s3 + (long long) t * s2 + (long long) (hv % H) * s1;
    nv_bfloat16 * dp = dst + (long long) row * S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { dp[i] = __float2bfloat16(sp[i]); }
}

static __global__ void ncp_gdn_f32_to_bf16(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
static __global__ void ncp_gdn_bf16_to_f32(const nv_bfloat16 * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Route the recurrent/chunked GDN op to the external FLA .so (libncp_gdn.so) where ggml's math+layout map 1:1 to
// FLA (verified): non-KDA scalar gate, K_snapshot==1 (final state only), F32 contiguous. The op's fused output
// holds attention scores in [0, attn_score_elems) and the final state in the tail, so ht writes there directly.
// Anything else -> return false so the caller falls through to the inline kernels.
bool ggml_cuda_op_gated_delta_net_ncp_so(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;    // value heads (HV)
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    const int64_t neqk1 = neq1;   // key/query heads (H)

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // strides in floats (q strides used for both q and k, since ggml_are_same_stride(src_q, src_k))
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    float * const gdn_state_out = dst_d + S_v * H * n_tokens * n_seqs;
    // g (src_g) is the RAW per-token gate; both entries cumsum it internally.
    {
        const bool common_ok = !kda && !keep_rs &&
            src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(src_q) && ggml_is_contiguous(src_k) &&
            ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta) && ggml_is_contiguous(src_state) &&
            ggml_is_contiguous(dst) && src_v->nb[0] == sizeof(float);

        // Chunked prefill (FLA's WY tensor-core chain) is ON by default, real prefills only (>= 2 chunks). It needs
        // L2-normalized k to stay numerically stable -- real GDN models do that upstream, random test inputs do not,
        // so GGML_NCP_GDN_CHUNKED=0 forces every shape onto the recurrent arm.
        static const bool chunk_on = ggml_ncp_gdn_chunked_enabled();
        const bool want_chunked = chunk_on && (int) n_tokens >= 128 && ggml_ncp_lib_gdn_chunked_available();
        const bool want_recur   = ggml_ncp_lib_gdn_available();

        if (common_ok && (want_chunked || want_recur)) {
            // q/k carry H heads; the .so's kernels pair V head hv with QK head hv/(HV/H), while ggml pairs it with
            // hv % H. Tile q/k up to HV heads so FLA's divisor becomes 1 and its map collapses onto ggml's. The .so
            // is therefore called with H' = HV, and must be BUILT for (HV, HV, S) -- build "32,32,128".
            const int64_t nqk_tiled = n_seqs * n_tokens * H * S_v;      // H here is HV (v heads)
            const int     nrow_qk   = (int) (n_seqs * n_tokens * H);

            // Both entries return non-zero when the (H, HV, S) shape was not compiled into the .so. That is the usual
            // reason a model silently stays on the inline path.

            // ggml's state is [v][k]; the chunked entry wants [k][v], so transpose h0 in and ht out. The recurrent
            // kernel is AOT'd with STATE_V_FIRST=1 and needs no transpose.
            if (want_chunked) {
                const int nblk = (int) (n_seqs * H);           // H here == HV (v heads)
                ggml_cuda_pool_alloc<float> h0t(ctx.pool(), (size_t) nblk * S_v * S_v);
                ggml_cuda_pool_alloc<float> htt(ctx.pool(), (size_t) nblk * S_v * S_v);
                // The .so allocates nothing: its intermediates (g, A, w, u, h, v_new) come out of ggml's CUDA pool,
                // like every other scratch buffer in the backend.
                const size_t ws_bytes = ggml_ncp_lib_gdn_chunked_bf16_workspace_size(
                    (int) n_seqs, (int) n_tokens, (int) H /*H' = HV*/, (int) H, (int) S_v);
                ggml_cuda_pool_alloc<char> ws(ctx.pool(), ws_bytes);

                // THE CHAIN RUNS IN BF16. FLA's kernels are dtype-generic, so the dtype of the tensors handed to the
                // JIT decides what gets compiled -- and with f32 operands the PPU profiles fwd_h at 256 registers per
                // thread (the ceiling) with 460 B/thread spilled to local memory, 2.95x slower than vLLM's bf16 at an
                // IDENTICAL grid and block. bf16 halves the register pressure of every tl.dot tile, so the operands
                // stage through the MMA's shared-memory path and nothing spills. The dtype is not a precision knob
                // here; it decides whether the tensor cores get used at all.
                //
                // ggml's GDN op is F32, so the conversion lands here. v's cast is FUSED into the make-contiguous copy
                // it already needed (strided f32 -> contiguous bf16, one pass, and the write side halves), so only q,
                // k and o are traffic we did not pay before.
                const int64_t nq = nqk_tiled;   // HV heads after tiling, not neqk1
                const int64_t nv = (int64_t) n_seqs * n_tokens * H * S_v;
                ggml_cuda_pool_alloc<nv_bfloat16> qb(ctx.pool(), (size_t) nq);
                ggml_cuda_pool_alloc<nv_bfloat16> kb(ctx.pool(), (size_t) nq);
                ggml_cuda_pool_alloc<nv_bfloat16> vb(ctx.pool(), (size_t) nv);
                ggml_cuda_pool_alloc<nv_bfloat16> ob(ctx.pool(), (size_t) nv);
                // tile + cast in one pass; q/k come out with HV heads, bf16
                ncp_gdn_tile_qk_heads_bf16<<<nrow_qk, 128, 0, stream>>>(
                    q_d, qb.ptr, (int) S_v, (int) neqk1, (int) H, sq1, sq2, sq3, (int) n_tokens);
                ncp_gdn_tile_qk_heads_bf16<<<nrow_qk, 128, 0, stream>>>(
                    k_d, kb.ptr, (int) S_v, (int) neqk1, (int) H, sq1, sq2, sq3, (int) n_tokens);
                ncp_gdn_make_contig_bf16<<<(int) (n_seqs * n_tokens * H), 128, 0, stream>>>(
                    v_d, vb.ptr, (int) S_v, sv1, sv2, sv3, (int) H, (int) n_tokens);

                ncp_gdn_state_transpose<<<nblk, 256, 0, stream>>>(s_d, h0t.ptr, (int) S_v);   // [v][k] -> [k][v]
                const int rc = ggml_ncp_lib_gdn_chunked_bf16(
                    qb.ptr, kb.ptr, vb.ptr, g_d, b_d, h0t.ptr, ob.ptr, htt.ptr,
                    (int) n_seqs, (int) n_tokens, (int) H /*H' = HV*/, (int) H, (int) S_v, scale,
                    ws.ptr, ws_bytes, stream);
                if (rc == 0) {
                    ncp_gdn_bf16_to_f32<<<(nv + 255) / 256, 256, 0, stream>>>(ob.ptr, dst_d, nv);
                    ncp_gdn_state_transpose<<<nblk, 256, 0, stream>>>(htt.ptr, gdn_state_out, (int) S_v);  // [k][v]->[v][k]
                    return true;
                }
            }

            // The recurrent entry is F32 and wants v PACKED. v is routinely a strided view on real models. This copy
            // used to be hoisted above the chunked arm because that arm's gate demanded ggml_is_contiguous(src_v) --
            // it no longer does: ncp_gdn_make_contig_bf16 reads the strides directly, so the chunked path never
            // materialises an F32 contiguous v at all.
            const float * v_use = v_d;
            ggml_cuda_pool_alloc<float> v_contig(ctx.pool());
            if (!ggml_is_contiguous(src_v)) {
                v_contig.alloc((size_t) n_seqs * n_tokens * H * S_v);
                const int nrow = (int) (n_seqs * n_tokens * H);
                ncp_gdn_make_contig<<<nrow, 128, 0, stream>>>(v_d, v_contig.ptr, (int) S_v,
                    sv1, sv2, sv3, (int) H, (int) n_tokens);
                v_use = v_contig.ptr;
            }

            if (want_recur) {
                ggml_cuda_pool_alloc<float> qt(ctx.pool(), (size_t) nqk_tiled);
                ggml_cuda_pool_alloc<float> kt(ctx.pool(), (size_t) nqk_tiled);
                ncp_gdn_tile_qk_heads<<<nrow_qk, 128, 0, stream>>>(
                    q_d, qt.ptr, (int) S_v, (int) neqk1, (int) H, sq1, sq2, sq3, (int) n_tokens);
                ncp_gdn_tile_qk_heads<<<nrow_qk, 128, 0, stream>>>(
                    k_d, kt.ptr, (int) S_v, (int) neqk1, (int) H, sq1, sq2, sq3, (int) n_tokens);

                const int rc = ggml_ncp_lib_gdn_recurrent(
                    qt.ptr, kt.ptr, v_use, g_d, b_d, s_d, dst_d, gdn_state_out,
                    (int) n_seqs, (int) n_tokens, (int) H /*H' = HV*/, (int) H /*HV*/, (int) S_v, scale, stream);
                if (rc == 0) {
                    return true;
                }
            }
        }
    }

    return false;
}

#endif // GGML_NCP_GDN
