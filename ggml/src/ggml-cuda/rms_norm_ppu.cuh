#pragma once

// rms_norm_ppu.cuh — optimized vectorized RMS_NORM(+MUL[+ADD]) fused kernel (opt_* symbols).
// Non-invasive integration mirroring convert_ppu.cuh: self-contained, whole body guarded by #if defined(GGML_USE_PPU).
// norm.cu's rms_norm_mul_f32_cuda routes here when the gate (large ncols + large batch + 4-aligned) passes,
//   otherwise it falls through to the native kernel.
// Key ideas: float4 vectorization + block 640 even split + compile-time fastmodulo elimination (mul_direct) +
//   pass1 loads each thread's float4 into registers and squares them (also provides MLP), pass2 reuses the
//   registers directly (no second x read).
// All shared helpers come from common.cuh: warp_reduce_sum / block_reduce / block_reduce_method /
//   fastmodulo / init_fastdiv_values / ggml_cuda_pdl_sync / ggml_cuda_pdl_lc / WARP_SIZE.

#include "common.cuh"

#if defined(GGML_USE_PPU)

// ── Routing gate ──
// Use this kernel only for large shape + large batch + float4-aligned (the regime validated to win);
//   otherwise norm.cu keeps the native kernel.
// total_rows = nrows*nchannels*nsamples (number of rows to normalize): large in prefill, small in decode.
// 512 is a fixed threshold (not numSM/hardware dependent): 512 rows -> 512 concurrent blocks -> fills any
//   GPU for one wave, so the throughput advantage is realized.
// Alignment: the kernel loads x via float4 (16B), so each row base (base + row*stride_row +
//   channel*stride_channel + sample*stride_sample) must be 16B-aligned. ncols%4==0 keeps the contiguous case
//   aligned (dst is always contiguous, so it is covered too); additionally requiring the x strides to be
//   4-aligned covers non-contiguous inputs. The scalar tail only fixes the element count, not base alignment,
//   so we gate here instead of relying on it (a misaligned float4 load faults, it does not just compute wrong).
#define OPT_RMS_MIN_NCOLS 1024
#define OPT_RMS_MIN_ROWS  512
static inline bool opt_rms_norm_should_use(int ncols, int64_t total_rows,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample) {
    return ncols >= OPT_RMS_MIN_NCOLS && (ncols & 3) == 0 && total_rows >= OPT_RMS_MIN_ROWS
        && (stride_row & 3) == 0 && (stride_channel & 3) == 0 && (stride_sample & 3) == 0;
}

// large ncols(>=1024) path block_size: 640 = 1280 float4 / 640 threads = an even 2 float4 per thread.
static constexpr int OPT_RMS_BLOCK_LARGE = 640;

// Column broadcast index: direct=true (mul_ncols>=ncols, col always < N) -> col%N==col, return col at compile
//   time and drop fastmodulo; direct=false (true broadcast, N<ncols) keeps fastmodulo (from common.cuh).
template <bool direct>
static __device__ __forceinline__ uint32_t opt_bcast_col(int col, const uint3 & ncols_packed) {
    if constexpr (direct) {
        return (uint32_t) col;
    } else {
        return fastmodulo((uint32_t) col, ncols_packed);
    }
}

// pass2: apply scale (*mul[+add]) to one float4; shared by both the cached-reuse and the reread paths to
//   avoid duplicating the if constexpr chain.
template <bool do_multiply, bool do_add, bool mul_direct, bool add_direct>
static __device__ __forceinline__ float4 opt_rms_apply_f4(
        float4 v, float scale, int col,
        const float * mul, const uint3 & mul_ncols_packed,
        const float * add, const uint3 & add_ncols_packed) {
    if constexpr (do_multiply && do_add) {
        v.x = scale*v.x*mul[opt_bcast_col<mul_direct>(col+0, mul_ncols_packed)] + add[opt_bcast_col<add_direct>(col+0, add_ncols_packed)];
        v.y = scale*v.y*mul[opt_bcast_col<mul_direct>(col+1, mul_ncols_packed)] + add[opt_bcast_col<add_direct>(col+1, add_ncols_packed)];
        v.z = scale*v.z*mul[opt_bcast_col<mul_direct>(col+2, mul_ncols_packed)] + add[opt_bcast_col<add_direct>(col+2, add_ncols_packed)];
        v.w = scale*v.w*mul[opt_bcast_col<mul_direct>(col+3, mul_ncols_packed)] + add[opt_bcast_col<add_direct>(col+3, add_ncols_packed)];
    } else if constexpr (do_multiply) {
        v.x = scale*v.x*mul[opt_bcast_col<mul_direct>(col+0, mul_ncols_packed)];
        v.y = scale*v.y*mul[opt_bcast_col<mul_direct>(col+1, mul_ncols_packed)];
        v.z = scale*v.z*mul[opt_bcast_col<mul_direct>(col+2, mul_ncols_packed)];
        v.w = scale*v.w*mul[opt_bcast_col<mul_direct>(col+3, mul_ncols_packed)];
    } else {
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    }
    return v;
}

// ---- rms_norm PPU kernel (float4 + register cache: pass1 caches x into registers, pass2 reuses, no reread) ----
// do_multiply=true: dst = scale * x * mul[broadcast]; do_add=true: then + add[broadcast].
// mul/add column broadcast: mul_direct/add_direct=true uses direct col index (fastmodulo removed at compile
//   time), false uses fastmodulo.
template <int block_size, bool do_multiply = false, bool do_add = false, bool mul_direct = false, bool add_direct = false>
static __global__ void rms_norm_opt_kernel(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    // float4 vectorization + register cache: pass1 loads this thread's float4 into registers and accumulates
    //   the sum of squares, pass2 reuses those registers (no second x read) -> kills the pass2 x load and its
    //   memory-dependency stall.
    //   REG_CACHE = max float4 cached per thread. At 640 each thread has exactly 2 (all cached, no reread);
    //   pass1 issues the REG_CACHE loads together then squares them (also provides MLP).
    //   The part beyond REG_CACHE*block_size (larger ncols) falls back to a grid-stride reread; ncols%4 uses
    //   the scalar tail. Stays generally correct.
    const int      ncols4 = ncols / 4;
    const float4 * x4     = reinterpret_cast<const float4 *>(x);
    constexpr int  REG_CACHE = 2;

    float4 xv[REG_CACHE];      // cache this thread's first REG_CACHE float4 (kept alive across the barrier)
    bool   valid[REG_CACHE];   // whether this slot is in bounds

    ggml_cuda_pdl_sync();
    // pass1: issue all REG_CACHE independent loads first (MLP), then square them together
#pragma unroll
    for (int u = 0; u < REG_CACHE; ++u) {
        const int c4 = tid + u * block_size;
        valid[u] = (c4 < ncols4);
        if (valid[u]) {
            xv[u] = x4[c4];
        }
    }
#pragma unroll
    for (int u = 0; u < REG_CACHE; ++u) {
        if (valid[u]) {
            tmp += xv[u].x*xv[u].x + xv[u].y*xv[u].y + xv[u].z*xv[u].z + xv[u].w*xv[u].w;
        }
    }
    // float4 beyond the cache capacity (only when ncols4 > REG_CACHE*block_size): reread and square
    for (int col4 = tid + REG_CACHE * block_size; col4 < ncols4; col4 += block_size) {
        const float4 v = x4[col4];
        tmp += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // ncols%4 scalar tail
    for (int col = ncols4*4 + tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    float4 * dst4 = reinterpret_cast<float4 *>(dst);
    // pass2: reuse the xv cached in pass1 (no x reread)
#pragma unroll
    for (int u = 0; u < REG_CACHE; ++u) {
        if (valid[u]) {
            const int col4 = tid + u * block_size;
            dst4[col4] = opt_rms_apply_f4<do_multiply, do_add, mul_direct, add_direct>(
                xv[u], scale, col4 * 4, mul, mul_ncols_packed, add, add_ncols_packed);
        }
    }
    // float4 beyond the cache: reread + apply
    for (int col4 = tid + REG_CACHE * block_size; col4 < ncols4; col4 += block_size) {
        dst4[col4] = opt_rms_apply_f4<do_multiply, do_add, mul_direct, add_direct>(
            x4[col4], scale, col4 * 4, mul, mul_ncols_packed, add, add_ncols_packed);
    }
    // ncols%4 scalar tail
    for (int col = ncols4*4 + tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            dst[col] = scale * x[col] * mul[opt_bcast_col<mul_direct>(col, mul_ncols_packed)] + add[opt_bcast_col<add_direct>(col, add_ncols_packed)];
        } else if constexpr (do_multiply) {
            dst[col] = scale * x[col] * mul[opt_bcast_col<mul_direct>(col, mul_ncols_packed)];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

// ---- host launch (same signature as norm.cu rms_norm_mul_f32_cuda, plain <<<>>>) ----
// mul==nullptr: plain rms_norm; add==nullptr: rms_norm*mul; otherwise rms_norm*mul+add.
static void opt_rms_norm_mul_f32(const float *  x,
                                 const float *  mul,
                                 const float *  add,
                                 float *        dst,
                                 const int      ncols,
                                 const int      nrows,
                                 const int      nchannels,
                                 const int      nsamples,
                                 const int64_t  stride_row,
                                 const int64_t  stride_channel,
                                 const int64_t  stride_sample,
                                 const int64_t  mul_stride_row,
                                 const int64_t  mul_stride_channel,
                                 const int64_t  mul_stride_sample,
                                 const uint32_t mul_ncols,
                                 const uint32_t mul_nrows,
                                 const uint32_t mul_nchannels,
                                 const uint32_t mul_nsamples,
                                 const int64_t  add_stride_row,
                                 const int64_t  add_stride_channel,
                                 const int64_t  add_stride_sample,
                                 const uint32_t add_ncols,
                                 const uint32_t add_nrows,
                                 const uint32_t add_nchannels,
                                 const uint32_t add_nsamples,
                                 const float    eps,
                                 cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    const size_t shmem = 32 * sizeof(float);  // block_size > WARP_SIZE, needs block_reduce buffer

    if (mul == nullptr) {
        // plain rms_norm path
        if (ncols < 1024) {
            rms_norm_opt_kernel<256, false><<<blocks_num, dim3(256,1,1), shmem, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        } else {
            rms_norm_opt_kernel<OPT_RMS_BLOCK_LARGE, false><<<blocks_num, dim3(OPT_RMS_BLOCK_LARGE,1,1), shmem, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        }
        return;
    }

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (add == nullptr) {
        // rms_norm * mul (Qwen main norm path)
        // when mul_ncols >= ncols the column-broadcast modulo is identity (col%N==col); mul_direct drops
        // fastmodulo at compile time, otherwise (true broadcast) keep it.
        const bool mul_direct = (mul_ncols >= (uint32_t) ncols);
        if (ncols < 1024) {
            if (mul_direct) {
                rms_norm_opt_kernel<256, true, false, true, false><<<blocks_num, dim3(256,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            } else {
                rms_norm_opt_kernel<256, true, false, false, false><<<blocks_num, dim3(256,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            }
        } else {
            if (mul_direct) {
                rms_norm_opt_kernel<OPT_RMS_BLOCK_LARGE, true, false, true, false><<<blocks_num, dim3(OPT_RMS_BLOCK_LARGE,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            } else {
                rms_norm_opt_kernel<OPT_RMS_BLOCK_LARGE, true, false, false, false><<<blocks_num, dim3(OPT_RMS_BLOCK_LARGE,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            }
        }
    } else {
        // rms_norm * mul + add
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        // only go direct when both mul and add column broadcasts are identity (drop both fastmodulos); mixed
        // cases fall back to keeping fastmodulo (still correct).
        const bool both_direct = (mul_ncols >= (uint32_t) ncols) && (add_ncols >= (uint32_t) ncols);
        if (ncols < 1024) {
            if (both_direct) {
                rms_norm_opt_kernel<256, true, true, true, true><<<blocks_num, dim3(256,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add, add_stride_row, add_stride_channel, add_stride_sample,
                    add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);
            } else {
                rms_norm_opt_kernel<256, true, true, false, false><<<blocks_num, dim3(256,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add, add_stride_row, add_stride_channel, add_stride_sample,
                    add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);
            }
        } else {
            if (both_direct) {
                rms_norm_opt_kernel<OPT_RMS_BLOCK_LARGE, true, true, true, true><<<blocks_num, dim3(OPT_RMS_BLOCK_LARGE,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add, add_stride_row, add_stride_channel, add_stride_sample,
                    add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);
            } else {
                rms_norm_opt_kernel<OPT_RMS_BLOCK_LARGE, true, true, false, false><<<blocks_num, dim3(OPT_RMS_BLOCK_LARGE,1,1), shmem, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add, add_stride_row, add_stride_channel, add_stride_sample,
                    add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);
            }
        }
    }
}

#endif // GGML_USE_PPU
