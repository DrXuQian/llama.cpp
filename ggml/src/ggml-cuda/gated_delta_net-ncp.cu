#include "gated_delta_net-ncp.cuh"

#ifdef GGML_NCP_GDN

#include "ncp-lib.h"
#include "common.cuh"
#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Transpose each [S,S] state block between ggml's [v][k] and actlizeLA's [k][v].
static __global__ void ncp_gdn_state_transpose(const float * __restrict__ in, float * __restrict__ out, int S) {
    const int blk = blockIdx.x;
    const float * bi = in  + (size_t) blk * S * S;
    float *       bo = out + (size_t) blk * S * S;
    for (int idx = threadIdx.x; idx < S * S; idx += blockDim.x) {
        const int i = idx / S, j = idx % S;   // out[i][j] = in[j][i]
        bo[i * S + j] = bi[j * S + i];
    }
}
// Pack Q/K/V/Beta for actlizeLA in one launch. Q/K use ggml's GVA pairing qh = hv % Hq and are tiled
// to HV heads so the library can be called with q_heads == value_heads. V may be strided; all outputs are contiguous.
static __global__ void ncp_gdn_pack_inputs_bf16(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ beta, nv_bfloat16 * __restrict__ qo, nv_bfloat16 * __restrict__ ko,
        nv_bfloat16 * __restrict__ vo, nv_bfloat16 * __restrict__ betao, int S, int Hq, int HV,
        long long q1, long long q2, long long q3, long long v1, long long v2, long long v3, int T) {
    const int row = blockIdx.x;
    const int hv  = row % HV;
    const int t   = (row / HV) % T;
    const int seq = row / (HV * T);
    const int qh  = hv % Hq;

    const float * qp = q + (long long) seq * q3 + (long long) t * q2 + (long long) qh * q1;
    const float * kp = k + (long long) seq * q3 + (long long) t * q2 + (long long) qh * q1;
    const float * vp = v + (long long) seq * v3 + (long long) t * v2 + (long long) hv * v1;
    nv_bfloat16 * qop = qo + (long long) row * S;
    nv_bfloat16 * kop = ko + (long long) row * S;
    nv_bfloat16 * vop = vo + (long long) row * S;

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        qop[i] = __float2bfloat16(qp[i]);
        kop[i] = __float2bfloat16(kp[i]);
        vop[i] = __float2bfloat16(vp[i]);
    }
    if (threadIdx.x == 0) {
        betao[row] = __float2bfloat16(beta[row]);
    }
}

// pack two 32-bit nv_bfloat162 into one float2 via union (same size/align:
// 4+4 == 8 == sizeof(float2) on little-endian GPU). Used for a single 64-bit
// store.
__device__ __forceinline__ float2 pack_b2(nv_bfloat162 a, nv_bfloat162 b) {
    union { nv_bfloat162 p[2]; float2 f; } u;
    u.p[0] = a;
    u.p[1] = b;
    return u.f;
}
 
static __global__ void ncp_gdn_pack_inputs_bf16_opt(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ beta, nv_bfloat16 * __restrict__ qo, nv_bfloat16 * __restrict__ ko,
        nv_bfloat16 * __restrict__ vo, nv_bfloat16 * __restrict__ betao, int S, int Hq, int HV,
        long long q1, long long q2, long long q3, long long v1, long long v2, long long v3, int T) {
    const int nvh = HV / Hq;
 
    const int c    = threadIdx.x;   // 0..S/4-1 == 0..31 (S == 128)
    const int hq   = threadIdx.y;
    const int flat = blockIdx.x;
    const int t    = flat % T;
    const int seq  = flat / T;
 
    const long long qo4  = (long long)seq * q3 + (long long)t * q2;
    const long long vv4  = (long long)seq * v3 + (long long)t * v2;
    const long long rowb = (long long)seq * (HV * T) + (long long)t * HV;
    const long long qsrc = qo4 + (long long)hq * q1 + (long long)c * 4;
 
    const float4 q4 = *(const float4 *)(q + qsrc);
    const float4 k4 = *(const float4 *)(k + qsrc);
    const float2 fq  = pack_b2(__float22bfloat162_rn(make_float2(q4.x, q4.y)),
                                __float22bfloat162_rn(make_float2(q4.z, q4.w)));
    const float2 fk  = pack_b2(__float22bfloat162_rn(make_float2(k4.x, k4.y)),
                                __float22bfloat162_rn(make_float2(k4.z, k4.w)));
    #pragma unroll 2
    for (int j = 0; j < nvh; ++j) {
        const long long ob = (rowb + hq + j * Hq) * S + (long long)c * 4;
        reinterpret_cast<float2 *>(qo + ob)[0] = fq;
        reinterpret_cast<float2 *>(ko + ob)[0] = fk;
        const int hv = hq + j * Hq;
        const float4 v4 = *(const float4 *)(v + vv4 + (long long)hv * v1 + (long long)c * 4);
        reinterpret_cast<float2 *>(vo + ob)[0] =
            pack_b2(__float22bfloat162_rn(make_float2(v4.x, v4.y)),
                    __float22bfloat162_rn(make_float2(v4.z, v4.w)));
    }
 
    const int id = threadIdx.y * blockDim.x + threadIdx.x;
    const int n  = blockDim.x * blockDim.y;
    for (int r = id; r < HV; r += n) {
        betao[rowb + r] = __float2bfloat16(beta[rowb + r]);
    }
}

static __global__ void ncp_gdn_bf16_to_f32(const nv_bfloat16 * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Route supported prefill requests to actlizeLA's PPU1.0 residual/full-chunk implementation.
// Decode has n_tokens == 1 per sequence and stays on the inline recurrent kernel.
// Attention output always goes to dst; final state goes to state_out when cache fusion supplies one, otherwise to
// dst's snapshot tail. Unsupported contracts return false.
bool ggml_cuda_op_gated_delta_net_ncp_so(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_out) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;
    const int64_t neqk1    = neq1;

    const bool kda = src_g->ne[0] == S_v;
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;
    const bool extents_fit = n_seqs > 0 && n_tokens > 0 && H > 0 && neqk1 > 0 &&
        n_seqs <= INT_MAX && n_tokens <= INT_MAX && H <= INT_MAX && neqk1 <= INT_MAX &&
        n_seqs <= INT_MAX / H && n_tokens <= INT_MAX / (n_seqs * H);
    const bool prefill = n_tokens > 1;
    const bool supported = prefill && !kda && !keep_rs && extents_fit && S_v == 128 && H % neqk1 == 0 &&
        src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32 &&
        src_g->type == GGML_TYPE_F32 && src_beta->type == GGML_TYPE_F32 &&
        src_state->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src_q) && ggml_is_contiguous(src_k) &&
        ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta) && ggml_is_contiguous(src_state) &&
        ggml_is_contiguous(dst) && src_v->nb[0] == sizeof(float);
    if (!supported || !ggml_ncp_lib_gdn_available()) {
        return false;
    }

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;
    float * dst_d = (float *) dst->data;

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);

    const int nrow = (int) (n_seqs * n_tokens * H);
    const int nblk = (int) (n_seqs * H);
    const int64_t nv = (int64_t) nrow * S_v;
    const float scale = 1.0f / sqrtf((float) S_v);
    cudaStream_t stream = ctx.stream();

    const size_t ws_bytes = ggml_ncp_lib_gdn_actlize_ppu10_residual_bf16_workspace_size_v1(
        (int) n_seqs, (int) n_tokens, (int) H, (int) H, (int) S_v);
    if (ws_bytes == 0) {
        return false;
    }

    ggml_cuda_pool_alloc<nv_bfloat16> qb(ctx.pool(), (size_t) nv);
    ggml_cuda_pool_alloc<nv_bfloat16> kb(ctx.pool(), (size_t) nv);
    ggml_cuda_pool_alloc<nv_bfloat16> vb(ctx.pool(), (size_t) nv);
    ggml_cuda_pool_alloc<nv_bfloat16> bb(ctx.pool(), (size_t) nrow);
    ggml_cuda_pool_alloc<nv_bfloat16> ob(ctx.pool(), (size_t) nv);
    ggml_cuda_pool_alloc<float> h0t(ctx.pool(), (size_t) nblk * S_v * S_v);
    ggml_cuda_pool_alloc<float> htt(ctx.pool(), (size_t) nblk * S_v * S_v);
    ggml_cuda_pool_alloc<char> ws(ctx.pool(), ws_bytes + 255);
    void * const ws_aligned = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(ws.ptr) + 255u) & ~uintptr_t(255u));

    if (neqk1 <= 32 && sv1 % 4 == 0) {
        const int g = n_seqs * n_tokens;
        const dim3 b(S_v / 4, neqk1);
        ncp_gdn_pack_inputs_bf16_opt<<<g, b, 0, stream>>>(
        q_d, k_d, v_d, b_d, qb.ptr, kb.ptr, vb.ptr, bb.ptr, (int) S_v, (int) neqk1, (int) H,
        sq1, sq2, sq3, sv1, sv2, sv3, (int) n_tokens);
    }
    else {
        // One row is one (sequence, token, value head). The fused launch preserves ggml's hv % Hq pairing.
        ncp_gdn_pack_inputs_bf16<<<nrow, 128, 0, stream>>>(
        q_d, k_d, v_d, b_d, qb.ptr, kb.ptr, vb.ptr, bb.ptr, (int) S_v, (int) neqk1, (int) H,
        sq1, sq2, sq3, sv1, sv2, sv3, (int) n_tokens);
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    ncp_gdn_state_transpose<<<nblk, 256, 0, stream>>>(s_d, h0t.ptr, (int) S_v);

    const int rc = ggml_ncp_lib_gdn_actlize_ppu10_residual_bf16_v1(
        qb.ptr, kb.ptr, vb.ptr, g_d, bb.ptr, h0t.ptr, ob.ptr, htt.ptr,
        (int) n_seqs, (int) n_tokens, (int) H, (int) H, (int) S_v, scale,
        ws_aligned, ws_bytes, stream);
    if (rc != 0) {
        return false;
    }

    ncp_gdn_bf16_to_f32<<<(nv + 255) / 256, 256, 0, stream>>>(ob.ptr, dst_d, nv);
    if (state_out == nullptr) {
        state_out = dst_d + nv;
    }
    ncp_gdn_state_transpose<<<nblk, 256, 0, stream>>>(htt.ptr, state_out, (int) S_v);
    return true;
}

#endif // GGML_NCP_GDN
