#include "mmid-quactlize.cuh"

#ifdef GGML_NCP_QUACTLIZE

#include "mmid.cuh"
#include "ncp-route.cuh"
#include "quactlize-buft.cuh"

#include <cinttypes>
#include <cuda_fp16.h>

// src1 (F32, ragged) -> A (fp16, [total_rows, K]) dense: compact row r pulls src1 row ids_src1[r], the permutation
// mm_ids_helper produced, and casts F32->fp16. No padding -- row r is expert-grouped exactly as routed, which is the
// "activations concatenated in expert order" the grouped entry documents.
static __global__ void qz_moe_gather_f32_to_f16_dense(
        const float * __restrict__ src1, half * __restrict__ A,
        const int32_t * __restrict__ ids_src1, const int64_t K, const int64_t src1_row_stride) {
    const int64_t r = blockIdx.x;
    const float * s = src1 + (int64_t) ids_src1[r] * src1_row_stride;
    half * d = A + r * K;
    // Stride from blockDim rather than a constant the launch has to agree with: a stride above the real block size
    // leaves elements written by nobody, which is garbage in A rather than a failure.
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        d[k] = __float2half(s[k]);
    }
}

// out (fp16, [total_rows, N]) -> dst (F32): inverse permutation, compact row r lands at dst row ids_dst[r].
static __global__ void qz_moe_scatter_f16_to_f32_dense(
        const half * __restrict__ out, float * __restrict__ dst,
        const int32_t * __restrict__ ids_dst, const int64_t N, const int64_t dst_row_stride) {
    const int64_t r = blockIdx.x;
    const half * s = out + r * N;
    float * d = dst + (int64_t) ids_dst[r] * dst_row_stride;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        d[j] = __half2float(s[j]);
    }
}

bool ggml_cuda_mul_mat_id_is_quactlize(const ggml_tensor * src0) {
    ggml_quactlize_artifact art;
    return ggml_quactlize_artifact_for(src0, &art);
}

void ggml_cuda_mul_mat_id_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    ggml_quactlize_artifact art;
    if (!ggml_quactlize_artifact_for(src0, &art)) {
        GGML_ABORT("[quactlize] %s has no K-pack artifact but reached the K-pack mul_mat_id", src0->name);
    }

    // Everything below is a hard requirement, not a gate: there is no second path for this tensor, so a shape this
    // hook cannot express is a bug in supports_op and has to say so rather than quietly produce a wrong answer.
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t K             = art.k;
    const int64_t N             = art.n;
    const int64_t n_experts     = art.experts;
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = src1->ne[2];
    const int64_t total_rows    = n_tokens * n_expert_used;

    GGML_ASSERT(src1->ne[0] == K);
    GGML_ASSERT(dst->ne[0]  == N);

    if (total_rows == 0) {
        return;
    }

    // mm_ids_helper's scan path stages one 4-byte entry per token in shared memory and ASSERTS it fits -- it aborts,
    // it does not fall back. With no inline path to degrade to, that assert would fire anyway; checking here just
    // names the reason.
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    GGML_ASSERT((size_t) n_tokens*sizeof(int32_t) <= smpbo && n_tokens < (1 << 22) && n_expert_used < (1 << 10));

    cudaStream_t stream = ctx.stream();

    // Device-side permutation, ggml's own helper (the one mmq/mmf use): no D2H, no H2D, no stream sync, so the path
    // stays CUDA-graph capturable. expert_bounds is exactly what the grouped entry calls `offsets` -- cumulative
    // int[experts+1], bounds[0]=0, bounds[experts]=total_rows -- so it is handed over unchanged.
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), total_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), total_rows);
    ggml_cuda_pool_alloc<int32_t> bounds  (ctx.pool(), n_experts + 1);

    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));   // per-token slot stride in ids
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);      // per-token row stride in src1
    GGML_ASSERT(sis1 > 0);

    // nchannels_y MUST be the RAW src1->ne[1]: with ne11==1, ids_src1 = it*sis1 + iex%1 = it, the correct broadcast.
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.ptr, ids_dst.ptr, bounds.ptr,
        (int) n_experts, (int) n_tokens, (int) n_expert_used, (int) src1->ne[1], si1, sis1, stream);
    CUDA_CHECK(cudaGetLastError());

    constexpr int block_dim = 256;   // every kernel here reads blockDim.x, so this is a launch choice only

    ggml_cuda_pool_alloc<half> A(ctx.pool(), total_rows*K);
    qz_moe_gather_f32_to_f16_dense<<<total_rows, block_dim, 0, stream>>>(
        (const float *) src1->data, A.ptr, ids_src1.ptr, K, src1->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    // max_rows is an upper bound on any single expert's row count. n_tokens, not total_rows: a token contributes at
    // most one row to a given expert, so n_tokens is exact as a bound and n_expert_used times tighter -- and it is a
    // host value already, so nothing has to read the routing back from the device to size the workspace.
    const int max_rows = (int) n_tokens;

    const int64_t ws_bytes = ggml_quactlize_grouped_workspace_bytes(
        art.qtype, (int) total_rows, max_rows, (int) N, (int) K, (int) n_experts, &art.arrangement);
    if (ws_bytes < 0) {
        GGML_ABORT("[quactlize] %s: no workspace size for total_rows=%d N=%" PRId64 " K=%" PRId64 " experts=%" PRId64
                   " -- supports_op admitted a shape the library will not serve",
                   src0->name, (int) total_rows, N, K, n_experts);
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), ws_bytes > 0 ? (size_t) ws_bytes : 1);

    ggml_cuda_pool_alloc<half> out(ctx.pool(), total_rows*N);

    // config_name null = let the library pick by shape. Naming one here would be a second source for a decision its
    // own inventory already owns, and an unknown name is a hard decline rather than a fallback.
    const int rc = ggml_quactlize_grouped_dev(
        art.qtype, (const uint16_t *) A.ptr, art.low, art.high, art.units,
        bounds.ptr, (uint16_t *) out.ptr,
        (int) total_rows, (int) N, (int) K, (int) n_experts, max_rows,
        workspace.ptr, ws_bytes, stream, /*config_name=*/nullptr, &art.arrangement);
    if (rc != 0) {
        GGML_ABORT("[quactlize] %s: grouped launch returned %d (total_rows=%d N=%" PRId64 " K=%" PRId64
                   " experts=%" PRId64 " max_rows=%d). The GGUF bytes were replaced by the K-pack artifact at load, "
                   "so there is nothing to fall back to -- supports_op's ladder admitted a shape the library "
                   "declines at run time",
                   src0->name, rc, (int) total_rows, N, K, n_experts, max_rows);
    }

    qz_moe_scatter_f16_to_f32_dense<<<total_rows, block_dim, 0, stream>>>(
        out.ptr, (float *) dst->data, ids_dst.ptr, N, dst->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    ggml_ncp_route_log(dst, "so-quactlize-kpack", nullptr);
}

#else

bool ggml_cuda_mul_mat_id_is_quactlize(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
    return false;
}

void ggml_cuda_mul_mat_id_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(ids); GGML_UNUSED(dst);
    GGML_ABORT("[quactlize] built without GGML_NCP_QUACTLIZE");
}

#endif // GGML_NCP_QUACTLIZE
