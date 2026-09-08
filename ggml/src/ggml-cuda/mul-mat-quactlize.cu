#include "mul-mat-quactlize.cuh"

#ifdef GGML_NCP_QUACTLIZE

#include "ncp-route.cuh"
#include "quactlize-buft.cuh"

#include <cinttypes>
#include <cuda_fp16.h>

// No permutation here, unlike the MoE path: ggml's dense MUL_MAT already hands over src1 as [k, m] with ne[0]
// contiguous, which is row-major [m, k] -- exactly the layout the dense entry documents for act -- and dst as
// [n, m], row-major [m, n], exactly what it writes. So these two are casts and nothing else.
static __global__ void qz_cast_f32_to_f16(const float * __restrict__ src, half * __restrict__ dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

static __global__ void qz_cast_f16_to_f32(const half * __restrict__ src, float * __restrict__ dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

bool ggml_cuda_mul_mat_is_quactlize(const ggml_tensor * src0) {
    ggml_quactlize_artifact art;
    return ggml_quactlize_artifact_for(src0, &art);
}

void ggml_cuda_mul_mat_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_quactlize_artifact art;
    if (!ggml_quactlize_artifact_for(src0, &art)) {
        GGML_ABORT("[quactlize] %s has no K-pack artifact but reached the K-pack mul_mat", src0->name);
    }

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(art.experts == 1);

    const int64_t K = art.k;
    const int64_t N = art.n;
    GGML_ASSERT(src1->ne[0] == K);
    GGML_ASSERT(dst->ne[0]  == N);

    // The weight has no batch dimension (experts == 1), so every activation batch multiplies the same matrix: with
    // both sides contiguous, [k, m, b2, b3] is just m*b2*b3 rows of k and the batch collapses into M.
    const int64_t M = src1->ne[1] * src1->ne[2] * src1->ne[3];
    GGML_ASSERT(dst->ne[1] * dst->ne[2] * dst->ne[3] == M);
    if (M == 0) {
        return;
    }
    GGML_ASSERT(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX);

    cudaStream_t stream = ctx.stream();
    ggml_quactlize_wait_ready(art, stream);

    constexpr int block_dim = 256;

    ggml_cuda_pool_alloc<half> act(ctx.pool(), M*K);
    qz_cast_f32_to_f16<<<(M*K + block_dim - 1)/block_dim, block_dim, 0, stream>>>(
        (const float *) src1->data, act.ptr, M*K);
    CUDA_CHECK(cudaGetLastError());

    const int64_t ws_bytes = ggml_quactlize_dense_workspace_bytes(
        art.qtype, (int) M, (int) N, (int) K, &art.arrangement);
    if (ws_bytes < 0) {
        GGML_ABORT("[quactlize] %s: no workspace size for m=%" PRId64 " n=%" PRId64 " k=%" PRId64
                   " -- supports_op admitted a shape the library will not serve", src0->name, M, N, K);
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), ws_bytes > 0 ? (size_t) ws_bytes : 1);

    ggml_cuda_pool_alloc<half> out(ctx.pool(), M*N);

    // config_name null = shape-selected by the library, whose inventory already owns that decision.
    const int rc = ggml_quactlize_dense_dev(
        art.qtype, (const uint16_t *) act.ptr, art.low, art.high, art.units, (uint16_t *) out.ptr,
        (int) M, (int) N, (int) K, workspace.ptr, ws_bytes, stream, /*config_name=*/nullptr, &art.arrangement);
    if (rc != 0) {
        GGML_ABORT("[quactlize] %s: dense launch returned %d (m=%" PRId64 " n=%" PRId64 " k=%" PRId64 "). The GGUF "
                   "bytes were replaced by the K-pack artifact at load, so there is nothing to fall back to -- "
                   "supports_op's ladder admitted a shape the library declines at run time",
                   src0->name, rc, M, N, K);
    }

    qz_cast_f16_to_f32<<<(M*N + block_dim - 1)/block_dim, block_dim, 0, stream>>>(
        out.ptr, (float *) dst->data, M*N);
    CUDA_CHECK(cudaGetLastError());

    ggml_ncp_route_log(dst, "so-quactlize-kpack-dense", nullptr);
}

#else

bool ggml_cuda_mul_mat_is_quactlize(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
    return false;
}

void ggml_cuda_mul_mat_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("[quactlize] built without GGML_NCP_QUACTLIZE");
}

#endif // GGML_NCP_QUACTLIZE
