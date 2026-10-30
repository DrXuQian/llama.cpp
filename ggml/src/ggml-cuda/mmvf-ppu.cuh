#pragma once

// Dedicated PPU decode GEMV -- out[M] = W[M,K](bf16) @ x[K](f32) -> f32, the bandwidth-bound mul_mat_vec that the
// generic mmvf does for a single decode token. This is the winner of a long bench sweep in ppu_tests/gemv_ppu.cu
// (v8d): copy cuBLAS's structure exactly -- ONE output row per block, 128 threads splitting K, plain register LDG
// + block-reduce, NO shared / cp.async / split-K / multi-row. Reaches ~82% of HBM peak and ties/beats cuBLAS-fp16
// across every model shape (Qwen-27B / Llama-7B / Llama-70B + corners) with ONE config -> no per-model tuning.
//
// Key findings from the sweep: (1) GEMV is pure bandwidth, so any "smart" tiling (shared/cp.async/split-K) only adds
// overhead; (2) 1-row/block gives N blocks = always enough parallelism (N is thousands in real models); (3) the f32
// activation is what was slow (32B/8 elems, re-read N times) -- NOT the bf16 weight unpack -- so we round x to bf16
// (the model's native dtype: 16B, faithful, no fp16 range risk) for a ~17% win over reading f32 x. Stays OFF the
// generic mmvf so the PPU path is self-contained; dispatch falls back to ggml_cuda_mul_mat_vec_f for anything else.

#include "common.cuh"
#include <cuda_bf16.h>

#if defined(GGML_USE_PPU)

static __device__ __forceinline__ float ppu_gemv_warp_reduce(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Round the f32 activation to bf16 (the model's native dtype) so the GEMV reads 16B/8-elems instead of 32B.
static __global__ void ppu_gemv_f32_to_bf16(const float * __restrict__ src, __nv_bfloat16 * __restrict__ dst, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// out[row] = sum_k W[row,k] * x[k].  W bf16 [M,K] row-major, x bf16 [K]. 1 row per block, T threads split K, U-deep
// unroll issues U uint4 (8 bf16 each) together for load ILP, bf162 -> float2 unpack, one block-reduce. Any K % 8 == 0.
template <int T, int U>
__global__ __launch_bounds__(T) void mul_mat_vec_f_ppu_kernel(
        const __nv_bfloat16 * __restrict__ W, const __nv_bfloat16 * __restrict__ x, float * __restrict__ out, const int K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const __nv_bfloat16 * w = W + (int64_t) row * K;
    float acc = 0.0f;
    int k = tid * 8;
    for (; k + (U-1)*T*8 < K; k += T*8*U) {                // main: U uint4 issued together -> ILP
        uint4 wv[U], xv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) { wv[u] = *reinterpret_cast<const uint4*>(w + k + u*T*8); xv[u] = *reinterpret_cast<const uint4*>(x + k + u*T*8); }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const __nv_bfloat162 * wb = reinterpret_cast<const __nv_bfloat162*>(&wv[u]);
            const __nv_bfloat162 * xb = reinterpret_cast<const __nv_bfloat162*>(&xv[u]);
#pragma unroll
            for (int e = 0; e < 4; ++e) { const float2 wf = __bfloat1622float2(wb[e]), xf = __bfloat1622float2(xb[e]); acc += wf.x*xf.x + wf.y*xf.y; }
        }
    }
    for (; k < K; k += T*8) {                              // tail (< U uint4 per thread; handles K not a multiple of T*8)
        const uint4 wv = *reinterpret_cast<const uint4*>(w + k);
        const uint4 xv = *reinterpret_cast<const uint4*>(x + k);
        const __nv_bfloat162 * wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const __nv_bfloat162 * xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 wf = __bfloat1622float2(wb[e]), xf = __bfloat1622float2(xb[e]); acc += wf.x*xf.x + wf.y*xf.y; }
    }
    acc = ppu_gemv_warp_reduce(acc);                       // block reduce over T threads -> out[row]
    __shared__ float sm[T/32];
    if (lane == 0) sm[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < (T>>5)) ? sm[lane] : 0.0f;
        acc = ppu_gemv_warp_reduce(acc);
        if (lane == 0) out[row] = acc;
    }
}

// The simple decode GEMV the PPU kernel handles: bf16 weight [K,M] contiguous, single-token f32 activation, f32 out,
// no fusion / ids / batch / channels, K a multiple of 8 (16B-aligned uint4 loads). Everything else -> generic mmvf.
static bool ggml_cuda_mul_mat_vec_f_ppu_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_BF16
        && src1->type == GGML_TYPE_F32
        && dst->type  == GGML_TYPE_F32
        && src1->ne[1] == 1                          // single decode token
        && src0->ne[2] == 1 && src0->ne[3] == 1      // 2D weight (no batch / MoE)
        && src1->ne[2] == 1 && src1->ne[3] == 1
        && (src0->ne[0] % 8) == 0                    // K % 8 == 0 -> aligned 128-bit loads
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
}

static void ggml_cuda_mul_mat_vec_f_ppu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = src0->ne[1];   // output rows
    const int K = src0->ne[0];   // reduction
    // round the f32 activation to bf16 once (model-native dtype) so the GEMV streams 16B/8-elems instead of 32B.
    ggml_cuda_pool_alloc<__nv_bfloat16> x_bf16(ctx.pool(), K);
    constexpr int CVT = 256;
    ppu_gemv_f32_to_bf16<<<(K + CVT - 1) / CVT, CVT, 0, ctx.stream()>>>(
        (const float *) src1->data, x_bf16.get(), K);
    constexpr int T = 128, U = 4;   // parameter-free: 1 row/block (grid = M), fixed T/U robust across all shapes
    mul_mat_vec_f_ppu_kernel<T, U><<<M, T, 0, ctx.stream()>>>(
        (const __nv_bfloat16 *) src0->data, x_bf16.get(), (float *) dst->data, K);
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_USE_PPU
