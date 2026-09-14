#pragma once

// RMS_NORM with an independent dtype for each operand: x (input), dst (output), mul and add.
// Every operand may be F32, F16 or BF16, giving 3^4 combinations for the mul+add variant.
//
// The kernel deliberately mirrors rms_norm_f32 in norm.cu: scalar grid-stride access (no
// vectorization), fp32 accumulation through block_reduce, fastmodulo broadcasting for mul/add and
// the same 256/1024 block size split. The only additions are the load/store conversions.
//
// Numerics:
//   - operands are converted to fp32 on load, never multiplied in the low precision domain
//   - the sum of squares, mean, and rsqrt are all fp32
//   - the result is rounded to the destination type exactly once, on store

// half / nv_bfloat16 and the conversion intrinsics come from common.cuh via vendors/*.h, which keeps
// this header usable in the HIP and MUSA builds as well.
#include "common.cuh"

#include <type_traits>

// Explicit conversions instead of ggml_cuda_cast: the half <-> float direction there relies on an
// implicit conversion, which is not available when __CUDA_NO_HALF_CONVERSIONS__ is set.
template <typename T> static __host__ __device__ inline float rmsc_to_f32(const T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(v);
    } else {
        return __bfloat162float(v);
    }
}

template <typename T> static __host__ __device__ inline T rmsc_from_f32(const float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

template <int block_size, typename src_t, typename dst_t, typename mul_t, typename add_t,
          bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_cast(const src_t * x,
                                     dst_t *       dst,
                                     const int     ncols,
                                     const int64_t stride_row,
                                     const int64_t stride_channel,
                                     const int64_t stride_sample,
                                     const float   eps,
                                     const mul_t * mul,
                                     const int64_t mul_stride_row,
                                     const int64_t mul_stride_channel,
                                     const int64_t mul_stride_sample,
                                     const uint3   mul_ncols_packed,
                                     const uint3   mul_nrows_packed,
                                     const uint3   mul_nchannels_packed,
                                     const uint3   mul_nsamples_packed,
                                     const add_t * add,
                                     const int64_t add_stride_row,
                                     const int64_t add_stride_channel,
                                     const int64_t add_stride_sample,
                                     const uint3   add_ncols_packed,
                                     const uint3   add_nrows_packed,
                                     const uint3   add_nchannels_packed,
                                     const uint3   add_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample*mul_stride_sample + mul_channel*mul_stride_channel + mul_row*mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample*add_stride_sample + add_channel*add_stride_channel + add_row*add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = rmsc_to_f32(x[col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = rmsc_to_f32(x[col]);
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col] = rmsc_from_f32<dst_t>(scale*xi*rmsc_to_f32(mul[mul_col]) + rmsc_to_f32(add[add_col]));
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col] = rmsc_from_f32<dst_t>(scale*xi*rmsc_to_f32(mul[mul_col]));
        } else {
            dst[col] = rmsc_from_f32<dst_t>(scale*xi);
        }
    }
}

// All arguments of one launch. dst is assumed contiguous with a row length of ncols, matching the
// dst indexing of rms_norm_f32 in norm.cu. mul == nullptr selects the plain variant, add == nullptr
// the multiply-only variant.
struct rms_norm_cast_args {
    const void * src = nullptr;
    void *       dst = nullptr;
    const void * mul = nullptr;
    const void * add = nullptr;

    ggml_type src_type = GGML_TYPE_F32;
    ggml_type dst_type = GGML_TYPE_F32;
    ggml_type mul_type = GGML_TYPE_F32;
    ggml_type add_type = GGML_TYPE_F32;

    int ncols     = 0;
    int nrows     = 1;
    int nchannels = 1;
    int nsamples  = 1;

    // strides in elements, not bytes
    int64_t stride_row     = 0;
    int64_t stride_channel = 0;
    int64_t stride_sample  = 0;

    int64_t  mul_stride_row     = 0;
    int64_t  mul_stride_channel = 0;
    int64_t  mul_stride_sample  = 0;
    uint32_t mul_ncols          = 0;
    uint32_t mul_nrows          = 0;
    uint32_t mul_nchannels      = 0;
    uint32_t mul_nsamples       = 0;

    int64_t  add_stride_row     = 0;
    int64_t  add_stride_channel = 0;
    int64_t  add_stride_sample  = 0;
    uint32_t add_ncols          = 0;
    uint32_t add_nrows          = 0;
    uint32_t add_nchannels      = 0;
    uint32_t add_nsamples       = 0;

    float eps = 0.0f;
};

#define RMSC_LAUNCH(BS)                                                                            \
    do {                                                                                           \
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params{                   \
            blocks_num, dim3(BS, 1, 1), (BS) > WARP_SIZE ? 32*sizeof(float) : 0, stream };          \
        ggml_cuda_kernel_launch(                                                                    \
            rms_norm_cast<BS, src_t, dst_t, mul_t, add_t, do_multiply, do_add>, lp,                 \
            (const src_t *) a.src, (dst_t *) a.dst, a.ncols,                                        \
            a.stride_row, a.stride_channel, a.stride_sample, a.eps,                                 \
            (const mul_t *) a.mul, a.mul_stride_row, a.mul_stride_channel, a.mul_stride_sample,     \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,           \
            (const add_t *) a.add, a.add_stride_row, a.add_stride_channel, a.add_stride_sample,     \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);          \
    } while (0)

// Same branch structure as rms_norm_mul_f32_cuda in norm.cu, except that the fuse mode is a
// template parameter so that unused mul/add types are not instantiated.
template <bool do_multiply, bool do_add, typename src_t, typename dst_t, typename mul_t, typename add_t>
static void rms_norm_cast_cuda(const rms_norm_cast_args & a, cudaStream_t stream) {
    const dim3 blocks_num(a.nrows, a.nchannels, a.nsamples);

    uint3 mul_ncols_packed     = make_uint3(0, 0, 0);
    uint3 mul_nrows_packed     = make_uint3(0, 0, 0);
    uint3 mul_nchannels_packed = make_uint3(0, 0, 0);
    uint3 mul_nsamples_packed  = make_uint3(0, 0, 0);

    uint3 add_ncols_packed     = make_uint3(0, 0, 0);
    uint3 add_nrows_packed     = make_uint3(0, 0, 0);
    uint3 add_nchannels_packed = make_uint3(0, 0, 0);
    uint3 add_nsamples_packed  = make_uint3(0, 0, 0);

    if constexpr (do_multiply) {
        mul_ncols_packed     = init_fastdiv_values(a.mul_ncols);
        mul_nrows_packed     = init_fastdiv_values(a.mul_nrows);
        mul_nchannels_packed = init_fastdiv_values(a.mul_nchannels);
        mul_nsamples_packed  = init_fastdiv_values(a.mul_nsamples);
    }

    if constexpr (do_add) {
        add_ncols_packed     = init_fastdiv_values(a.add_ncols);
        add_nrows_packed     = init_fastdiv_values(a.add_nrows);
        add_nchannels_packed = init_fastdiv_values(a.add_nchannels);
        add_nsamples_packed  = init_fastdiv_values(a.add_nsamples);
    }

    if (a.ncols < 1024) {
        RMSC_LAUNCH(256);
    } else {
        RMSC_LAUNCH(1024);
    }
}

#undef RMSC_LAUNCH

template <typename T> struct rmsc_tag { using type = T; };

template <typename F> static void rmsc_dispatch_type(ggml_type t, F && f) {
    switch (t) {
        case GGML_TYPE_F32:  f(rmsc_tag<float>{});       break;
        case GGML_TYPE_F16:  f(rmsc_tag<half>{});        break;
        case GGML_TYPE_BF16: f(rmsc_tag<nv_bfloat16>{}); break;
        default: GGML_ABORT("rms_norm_cast: unsupported type %s", ggml_type_name(t));
    }
}

// Runtime dtype dispatch. The fuse mode is resolved before the mul/add types so that the plain and
// multiply-only variants do not drag in the unused type combinations.
static void rms_norm_cast_launch(const rms_norm_cast_args & a, cudaStream_t stream) {
    GGML_ASSERT(a.src != nullptr && a.dst != nullptr);
    GGML_ASSERT(a.ncols > 0 && a.nrows > 0 && a.nchannels > 0 && a.nsamples > 0);
    GGML_ASSERT(a.eps >= 0.0f);
    GGML_ASSERT(a.mul != nullptr || a.add == nullptr); // add without mul is not supported

    rmsc_dispatch_type(a.src_type, [&](auto ts) {
        rmsc_dispatch_type(a.dst_type, [&](auto td) {
            using src_t = typename decltype(ts)::type;
            using dst_t = typename decltype(td)::type;

            if (a.mul == nullptr) {
                rms_norm_cast_cuda<false, false, src_t, dst_t, float, float>(a, stream);
                return;
            }

            rmsc_dispatch_type(a.mul_type, [&](auto tm) {
                using mul_t = typename decltype(tm)::type;

                if (a.add == nullptr) {
                    rms_norm_cast_cuda<true, false, src_t, dst_t, mul_t, float>(a, stream);
                    return;
                }

                rmsc_dispatch_type(a.add_type, [&](auto ta) {
                    using add_t = typename decltype(ta)::type;
                    rms_norm_cast_cuda<true, true, src_t, dst_t, mul_t, add_t>(a, stream);
                });
            });
        });
    });
}

// ggml_tensor level entry points, mirroring the three existing ones in norm.cuh. Nothing calls
// these yet; they exist so that wiring this op into the graph does not need to touch this file.
void ggml_cuda_op_rms_norm_cast(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_cast_fused(ggml_backend_cuda_context & ctx,
                                      ggml_tensor *               dst,
                                      ggml_tensor *               mul_tensor);

void ggml_cuda_op_rms_norm_cast_fused_add(ggml_backend_cuda_context & ctx,
                                          ggml_tensor *               dst,
                                          ggml_tensor *               mul_tensor,
                                          ggml_tensor *               add_tensor);
