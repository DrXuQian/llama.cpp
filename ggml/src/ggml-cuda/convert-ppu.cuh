#pragma once

// Dedicated PPU vectorized contiguous convert between {f32, f16, bf16} (any direction). ggml's generic
// convert_unary is SCALAR (1 elem/thread) -> latency-bound on the PPU (the trace: f32->bf16 ~52us, bf16->f32
// ~116us, ~10% of total). This does 8 elems/thread (vectorized 16B/32B loads+stores via a float[8] staging) ->
// HBM-bound. Convert is pure streaming, so MAX OCCUPANCY (1 chunk/thread, no per-thread unroll) saturates HBM;
// benched in ppu_tests/convert_ppu.cu (U1 best, U-unroll only cuts the block count and hurts -- opposite of the
// reduction-bound GEMV). Bit-for-bit identical to the scalar cast (same round-to-nearest-even intrinsics).
// Non-invasive (flash-attn style): self-contained here; convert_unary_cont_cuda routes the supported case here and
// keeps the generic scalar as the fallback for unsupported dtypes / non-contiguous / k % 8 != 0.

#include "common.cuh"
#include "convert.cuh"   // ggml_cuda_cast (used by strided fallback)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

#if defined(GGML_USE_PPU)

// ---- load 8 contiguous src elems -> float[8] (one wide vector load) ----
template <typename S> static __device__ __forceinline__ void ppu_load8(const S * x, int c, float * f);
template <> __device__ __forceinline__ void ppu_load8<float>(const float * x, int c, float * f) {
    const float4 a = reinterpret_cast<const float4 *>(x)[2*c + 0];   // 2 float4 = 32B
    const float4 b = reinterpret_cast<const float4 *>(x)[2*c + 1];
    f[0]=a.x; f[1]=a.y; f[2]=a.z; f[3]=a.w; f[4]=b.x; f[5]=b.y; f[6]=b.z; f[7]=b.w;
}
template <> __device__ __forceinline__ void ppu_load8<half>(const half * x, int c, float * f) {
    const uint4 v = reinterpret_cast<const uint4 *>(x)[c];           // 1 uint4 = 8 halfs = 16B
    const half2 * h = reinterpret_cast<const half2 *>(&v);
    float2 p;
    p = __half22float2(h[0]); f[0]=p.x; f[1]=p.y;  p = __half22float2(h[1]); f[2]=p.x; f[3]=p.y;
    p = __half22float2(h[2]); f[4]=p.x; f[5]=p.y;  p = __half22float2(h[3]); f[6]=p.x; f[7]=p.y;
}
template <> __device__ __forceinline__ void ppu_load8<nv_bfloat16>(const nv_bfloat16 * x, int c, float * f) {
    const uint4 v = reinterpret_cast<const uint4 *>(x)[c];
    const nv_bfloat162 * h = reinterpret_cast<const nv_bfloat162 *>(&v);
    float2 p;
    p = __bfloat1622float2(h[0]); f[0]=p.x; f[1]=p.y;  p = __bfloat1622float2(h[1]); f[2]=p.x; f[3]=p.y;
    p = __bfloat1622float2(h[2]); f[4]=p.x; f[5]=p.y;  p = __bfloat1622float2(h[3]); f[6]=p.x; f[7]=p.y;
}

// ---- store float[8] -> 8 contiguous dst elems (one wide vector store; round-to-nearest-even == ggml's cast) ----
template <typename D> static __device__ __forceinline__ void ppu_store8(D * y, int c, const float * f);
template <> __device__ __forceinline__ void ppu_store8<float>(float * y, int c, const float * f) {
    reinterpret_cast<float4 *>(y)[2*c + 0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4 *>(y)[2*c + 1] = make_float4(f[4], f[5], f[6], f[7]);
}
template <> __device__ __forceinline__ void ppu_store8<half>(half * y, int c, const float * f) {
    half2 h[4] = { __float22half2_rn(make_float2(f[0],f[1])), __float22half2_rn(make_float2(f[2],f[3])),
                   __float22half2_rn(make_float2(f[4],f[5])), __float22half2_rn(make_float2(f[6],f[7])) };
    reinterpret_cast<uint4 *>(y)[c] = *reinterpret_cast<const uint4 *>(h);
}
template <> __device__ __forceinline__ void ppu_store8<nv_bfloat16>(nv_bfloat16 * y, int c, const float * f) {
    nv_bfloat162 b[4] = { __float22bfloat162_rn(make_float2(f[0],f[1])), __float22bfloat162_rn(make_float2(f[2],f[3])),
                          __float22bfloat162_rn(make_float2(f[4],f[5])), __float22bfloat162_rn(make_float2(f[6],f[7])) };
    reinterpret_cast<uint4 *>(y)[c] = *reinterpret_cast<const uint4 *>(b);
}

// 8 contiguous src -> 8 dst per thread. n8 = nelements/8 (k % 8 == 0 -> no tail, 16B-aligned vector ld/st).
template <typename src_t, typename dst_t>
static __global__ void convert_cont_ppu_kernel(const src_t * __restrict__ x, dst_t * __restrict__ y, const int n8) {
    const int c = blockIdx.x * (int) blockDim.x + threadIdx.x;
    if (c >= n8) return;
    float f[8];
    ppu_load8 <src_t>(x, c, f);
    ppu_store8<dst_t>(y, c, f);
}

// Compile-time: the PPU vec path handles any cross-convert among {f32, f16, bf16}. if constexpr on this in
// convert_unary_cont_cuda keeps the kernel out of unsupported instantiations (quant src, same-type, etc.).
template <typename src_t, typename dst_t>
static constexpr bool ppu_convert_cont_supported() {
    constexpr bool s_ok = std::is_same<src_t, float>::value || std::is_same<src_t, half>::value || std::is_same<src_t, nv_bfloat16>::value;
    constexpr bool d_ok = std::is_same<dst_t, float>::value || std::is_same<dst_t, half>::value || std::is_same<dst_t, nv_bfloat16>::value;
    return s_ok && d_ok && !std::is_same<src_t, dst_t>::value;
}

template <typename src_t, typename dst_t>
static void ppu_convert_unary_cont(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int n8 = (int) (k >> 3);                  // k % 8 == 0 guaranteed by the caller
    const int BS = 256;
    convert_cont_ppu_kernel<src_t, dst_t><<<(n8 + BS - 1) / BS, BS, 0, stream>>>((const src_t *) vx, y, n8);
}



// ---- strided (non-contiguous) convert: 4 elems/thread + float4 + fastdiv + 差异化 cache ----
// Routes non-contiguous converts through the vectorized kernel with:
//   - float4 vectorized load/store (4 elems/thread)
//   - fastdiv for i02/i03 decomposition (avoids integer division)
//   - __launch_bounds__ for max occupancy
//   - cache: float4 (.cv read / .cs write), bf162 (default cache, L1 reuse)
//   - contiguous fast path when ne0203==1 && ne01==1 (skip fastdiv)
// Scalar fallback for unsupported dtypes (same routing as cont path).

#ifndef PPU_STRIDED_BLOCK_SIZE
#define PPU_STRIDED_BLOCK_SIZE 256
#endif
#define PPU_STRIDED_MIN_BLOCKS (2048 / PPU_STRIDED_BLOCK_SIZE)

// .cv load: bypass L1+L2 cache logic
static __device__ __forceinline__ float4 ppu_ld_cv_f4(const float * ptr) {
    float4 r;
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
        : "l"(ptr));
    return r;
}

// .cs store: bypass L1, L2 evict-first — float4 write(no reuse)
static __device__ __forceinline__ void ppu_st_cs_f4(float * ptr, float4 val) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}


template <typename src_t, typename dst_t>
__device__ __forceinline__ void ppu_convert_4elem_body(
        const src_t * x, dst_t * y,
        const int64_t ix, const int64_t iy,
        const int64_t ne00, const int64_t i00_base,
        const int64_t ne00_vec4) {
    if (i00_base < ne00_vec4) {
        // ---- 4 elements ----
        if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, nv_bfloat16>) {
            // F32→BF16: float4 read .cv (bypass L1+L2), bf162 write.wb (L1 write-combine)
            float4 vin = ppu_ld_cv_f4(&x[ix]);
            nv_bfloat162 lo = __float22bfloat162_rn(make_float2(vin.x, vin.y));
            nv_bfloat162 hi = __float22bfloat162_rn(make_float2(vin.z, vin.w));
            *reinterpret_cast<nv_bfloat162*>(&y[iy])     = lo;
            *reinterpret_cast<nv_bfloat162*>(&y[iy + 2]) = hi;
        } else if constexpr (std::is_same_v<src_t, nv_bfloat16> && std::is_same_v<dst_t, float>) {
            // BF16→F32: bf162 read .ca (b1 hits b0's L1 line), float4 write .cs (bypass L1)
            nv_bfloat162 b0 = *reinterpret_cast<const nv_bfloat162*>(&x[ix]);
            nv_bfloat162 b1 = *reinterpret_cast<const nv_bfloat162*>(&x[ix + 2]);
            float2 f0 = __bfloat1622float2(b0);
            float2 f1 = __bfloat1622float2(b1);
            ppu_st_cs_f4(&y[iy], make_float4(f0.x, f0.y, f1.x, f1.y));
        } else {
            // Scalar fallback for other type pairs (e.g. f16↔f32, f16↔bf16)
            for (int i = 0; i < 4; i++)
                y[iy + i] = ggml_cuda_cast<dst_t>(x[ix + i]);
        }
    } else {
        for (int64_t i = 0; i < ne00 - i00_base; i++)
            y[iy + i] = ggml_cuda_cast<dst_t>(x[ix + i]);
    }
}

template <typename src_t, typename dst_t>
__launch_bounds__(PPU_STRIDED_BLOCK_SIZE, PPU_STRIDED_MIN_BLOCKS)
__global__ void convert_strided_ppu_kernel(
        const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00_base = ((int64_t)blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (i00_base >= ne00) return;

    const src_t * x = (const src_t *) vx;
    const int64_t ne00_vec4 = ne00 & ~3;

    if (ne0203 == 1 && ne01 == 1) {
        // ── contiguous fast: skip fastdiv ──
        ppu_convert_4elem_body<src_t, dst_t>(x, y, i00_base, i00_base,
                ne00, i00_base, ne00_vec4);
    } else {
        // ── strided: fastdiv + grid-stride loop ──
        #pragma unroll 2
        for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
            #pragma unroll 2
            for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
                const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
                const int64_t i02 = dm.y;
                const int64_t i03 = dm.x;

                const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00_base;
                const int64_t iy = (i0203*ne01 + i01)*ne00 + i00_base;

                ppu_convert_4elem_body<src_t, dst_t>(x, y, ix, iy,
                        ne00, i00_base, ne00_vec4);
            }
        }
    }
}

template <typename src_t, typename dst_t>
static void ppu_convert_unary_strided(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02 * ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const int64_t n_groups = (ne00 + 3) / 4;
    const dim3 num_blocks(
        (n_groups + PPU_STRIDED_BLOCK_SIZE - 1) / PPU_STRIDED_BLOCK_SIZE,
        (int)std::min(ne01, (int64_t)65535),
        (int)std::min(ne0203, (int64_t)65535));
    convert_strided_ppu_kernel<src_t, dst_t><<<num_blocks, PPU_STRIDED_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

#endif // GGML_USE_PPU