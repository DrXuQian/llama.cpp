#include "mmid-ncp.cuh"

#ifdef GGML_NCP_MOE

#include "mmid.cuh"
#include "ncp-lib.h"
#include "ncp-route.cuh"

// NOT reached through common.cuh -- ggml-cuda.cu only had nv_bfloat16 because mmvf-ppu.cuh pulls this in for it.
#include <cuda_bf16.h>

// src1 (F32, ragged) -> A (bf16, [total_rows, K]) dense: compact row r pulls src1 row ids_src1[r] (the permutation
// mm_ids_helper produced) and casts F32->bf16. No padding -- row r is expert-grouped/ordered exactly as routed.
static __global__ void ncp_moe_gather_f32_to_bf16_dense(
        const float * __restrict__ src1, nv_bfloat16 * __restrict__ A,
        const int32_t * __restrict__ ids_src1, const int64_t K, const int64_t src1_row_stride) {
    const int64_t r = blockIdx.x;
    const float * s = src1 + (int64_t) ids_src1[r] * src1_row_stride;
    nv_bfloat16 * d = A + r * K;
    // Stride from blockDim, not a constant the launch has to agree with: a stride above the real block size leaves
    // elements written by nobody, which is silent garbage in A rather than a failure.
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        d[k] = __float2bfloat16(s[k]);
    }
}

// out (bf16, [total_rows, N]) -> dst (F32): inverse permutation, compact row r lands at dst row ids_dst[r].
static __global__ void ncp_moe_scatter_bf16_to_f32_dense(
        const nv_bfloat16 * __restrict__ out, float * __restrict__ dst,
        const int32_t * __restrict__ ids_dst, const int64_t N, const int64_t dst_row_stride) {
    const int64_t r = blockIdx.x;
    const nv_bfloat16 * s = out + r * N;
    float * d = dst + (int64_t) ids_dst[r] * dst_row_stride;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        d[j] = __bfloat162float(s[j]);
    }
}

// m_rows[e] = expert e's real row count = expert_bounds[e+1]-expert_bounds[e]. The NoPad kernel consumes this
// (directly for n_experts<128, via its device-side computeBlockInfoKernel for >=128).
static __global__ void ncp_moe_bounds_to_m_rows(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ m_rows, const int n_experts) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e < n_experts) {
        m_rows[e] = expert_bounds[e + 1] - expert_bounds[e];
    }
}

// m_indices[r] = expert id of grouped row r (the e with expert_bounds[e] <= r < expert_bounds[e+1]). The NoPad
// batched-GEMV path indexes the expert weights by this per-row id; the GEMM path ignores it.
static __global__ void ncp_moe_bounds_to_m_indices(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ m_indices,
        const int total_rows, const int n_experts) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= total_rows) return;
    int lo = 0, hi = n_experts;                    // largest e with expert_bounds[e] <= r; bounds is sorted
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (expert_bounds[mid] <= r) lo = mid; else hi = mid - 1;
    }
    m_indices[r] = lo;
}

// Route a bf16-weight MoE mul_mat_id through the external DeepGemm GroupedNoPad grouped-GEMM .so (libncp_moe.so).
//
// Device-side permutation via ggml's own ggml_cuda_launch_mm_ids_helper -- the same kernel mmq/mmf use -- which
// emits ids_src1 / ids_dst / expert_bounds with NO host round-trip; m_rows is a trivial bounds-diff. So there is no
// D2H, no H2D and no cudaStreamSynchronize, and the whole path stays CUDA-graph capturable, unlike ggml's own
// sorted-cuBLAS fallback below which sorts on the CPU.
//
// NoPad = dense contiguous: the kernel takes each expert's rows exactly as routed, so the row count it needs is
// just the identity total_rows = n_tokens*n_expert_used -- no host value depends on the routing. Zero padding, zero
// wasted flops. Returns false (-> inline fallback) if the .so exports no NoPad kernel or the shape is unsupported.
static bool ggml_cuda_mul_mat_id_ncp_lib_impl(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (!ggml_ncp_lib_moe_nopad_available()) {
        return false;
    }
    // DeepGemm entry is bf16 A x bf16 B -> bf16 out. Require bf16 expert weights, F32 activations/out.
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;                              // rhs [n_experts, N, K] must be contiguous; so must src1/dst
    }
    if (ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t K             = src0->ne[0];
    const int64_t N             = src0->ne[1];
    const int64_t n_experts     = src0->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = src1->ne[2];
    const int64_t total_rows    = n_tokens * n_expert_used;
    if (src1->ne[0] != K || dst->ne[0] != N || total_rows == 0) {
        return false;
    }

    const size_t ts = ggml_type_size(GGML_TYPE_BF16);
    if ((size_t) total_rows * (K + N) * ts > (size_t) 1024*1024*1024) {
        return false;                              // >1 GiB of bf16 scratch: not worth it, use inline
    }

    // mm_ids_helper's scan path stages one 4-byte entry per token in shared memory and hard-ASSERTS it fits
    // (mmid.cu) -- it aborts, it does not fall back. Guard so an oversized ubatch degrades to inline instead of
    // killing the process. (The counting-sort path it may pick instead has a looser bound; this is the strict one.)
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    if (n_tokens*sizeof(int32_t) > smpbo || n_tokens >= (1 << 22) || n_expert_used >= (1 << 10)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    // --- device-side permutation: ggml's own mul_mat_id helper (the one mmq/mmf use) ---
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), total_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), total_rows);
    ggml_cuda_pool_alloc<int32_t> bounds  (ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> m_rows  (ctx.pool(), n_experts);
    ggml_cuda_pool_alloc<int32_t> m_indices(ctx.pool(), total_rows);

    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));   // per-token slot stride in ids
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);      // per-token row stride in src1 (== ne11 when contiguous)
    GGML_ASSERT(sis1 > 0);

    // nchannels_y MUST be the RAW src1->ne[1]: mmf overwrites its local to ids->ne[0] when ne11==1 but passes the raw
    // value here; with ne11==1, ids_src1 = it*sis1 + iex%1 = it, the correct broadcast gather.
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.ptr, ids_dst.ptr, bounds.ptr,
        (int) n_experts, (int) n_tokens, (int) n_expert_used, (int) src1->ne[1], si1, sis1, stream);
    CUDA_CHECK(cudaGetLastError());

    constexpr int block_dim = 256;   // every kernel below reads blockDim.x, so this is a launch choice only

    ncp_moe_bounds_to_m_rows<<<(n_experts + block_dim - 1)/block_dim, block_dim, 0, stream>>>(
        bounds.ptr, m_rows.ptr, (int) n_experts);
    CUDA_CHECK(cudaGetLastError());

    ncp_moe_bounds_to_m_indices<<<(total_rows + block_dim - 1)/block_dim, block_dim, 0, stream>>>(
        bounds.ptr, m_indices.ptr, (int) total_rows, (int) n_experts);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<char> A_bf16(ctx.pool(), total_rows*K*ts);
    ncp_moe_gather_f32_to_bf16_dense<<<total_rows, block_dim, 0, stream>>>(
        (const float *) src1->data, (nv_bfloat16 *) A_bf16.ptr, ids_src1.ptr, K, src1->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<char> out_bf16(ctx.pool(), total_rows*N*ts);
    const int rc = ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        A_bf16.ptr, src0->data, out_bf16.ptr, m_indices.ptr, m_rows.ptr,
        (int) total_rows, (int) N, (int) K, (int) n_experts, (int) (total_rows / n_experts), stream);
    if (rc != 0) {
        GGML_LOG_WARN("[ncp-moe] NoPad miss -> inline: N=%d K=%d total_rows=%d n_exp=%d expected_m=%d\n",
                      (int) N, (int) K, (int) total_rows, (int) n_experts, (int) (total_rows / n_experts));
        return false;                              // .so has no NoPad kernel for this shape/arch -> inline fallback
    }

    ncp_moe_scatter_bf16_to_f32_dense<<<total_rows, block_dim, 0, stream>>>(
        (const nv_bfloat16 *) out_bf16.ptr, (float *) dst->data, ids_dst.ptr, N, dst->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mul_mat_id_ncp_lib_supported(const ggml_tensor * tensor) {
    // Mirrors the gates ggml_cuda_mul_mat_id_ncp_lib_impl applies before it commits to anything, minus the shape and
    // scratch-size limits: a caller asking this only needs to know whether the route is plausible, and answering
    // "yes" when the hook later declines costs one dropped fusion, not correctness.
    return tensor->op == GGML_OP_MUL_MAT_ID && ggml_ncp_lib_moe_nopad_available() &&
           tensor->src[0]->type == GGML_TYPE_BF16 &&
           tensor->src[1]->type == GGML_TYPE_F32 &&
           tensor->type == GGML_TYPE_F32 &&
           tensor->src[2] && tensor->src[2]->type == GGML_TYPE_I32;
}

bool ggml_cuda_mul_mat_id_ncp_lib(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    // The trace lives here rather than at the call site because a route that declines and one that was never built
    // are indistinguishable from the outside -- ggml computes the same answer either way.
    if (ggml_cuda_mul_mat_id_ncp_lib_impl(ctx, src0, src1, ids, dst)) {
        ggml_ncp_route_log(dst, "so-deepgemm-nopad", nullptr);
        return true;
    }
    ggml_ncp_route_log(dst, "so-deepgemm-nopad", "the .so hook declined -- see its gates");
    return false;
}

#endif // GGML_NCP_MOE
