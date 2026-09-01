#include "common.cuh"

// Vectorized NEOX RoPE (header-only): 4 (x0,x1) pairs per thread via float4
// loads/stores, block.x sized from head_dim. Kept in a single header included
// by rope.cu so the generic implementation there stays untouched.
//
// ggml_cuda_rope_neox_vec_f32 returns false when the fast path does not apply
// (non-f32, partial rope, unaligned strides, SET_ROWS fusion); the caller then
// runs the generic kernel.

namespace rope_neox_vec_detail {

struct corr_dims_vec {
    float v[2];
};

__device__ __forceinline__ float rope_yarn_ramp_vec(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

template <bool forward>
__device__ __forceinline__ void rope_yarn_vec(const float         theta_extrap,
                                              const float         freq_scale,
                                              const corr_dims_vec corr_dims,
                                              const int           i0,
                                              const float         ext_factor,
                                              float               mscale,
                                              float &             cos_theta,
                                              float &             sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_vec(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// block.x adapts to head_dim (ne00/8 threads cover all pairs), block.y fills
// the remaining budget for row parallelism.
constexpr int ROPE_NEOX_VEC_BLOCK_SIZE = 256;
constexpr int ROPE_NEOX_VEC_MAX_X      = 64;

template <bool forward, bool has_ff>
__global__ void rope_neox_vec(const float *       x,
                              float *             dst,
                              const int           ne00,
                              const int           ne01,
                              const int           ne02,
                              const int           s01,
                              const int           s02,
                              const int           s03,
                              const int           s1,
                              const int           s2,
                              const int           s3,
                              const int           n_dims,
                              const int           nr,
                              const int32_t *     pos,
                              const float         freq_scale,
                              const float         ext_factor,
                              const float         attn_factor,
                              const corr_dims_vec corr_dims,
                              const float         theta_scale,
                              const float *       freq_factors) {
    const int i0_half = 4 * (blockDim.x * blockIdx.y + threadIdx.x);
    const int i0      = 2 * i0_half;
    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_dst >= nr) {
        return;
    }

    const int n_dims_half = n_dims / 2;
    const int ne01_ne02   = ne01 * ne02;

    const uint32_t i3  = row_dst / ne01_ne02;
    const uint32_t rem = row_dst - i3 * ne01_ne02;
    const uint32_t i2  = rem / ne01;
    const uint32_t i1  = rem - i2 * ne01;

    const int ix   = i0_half + i1 * s01 + i2 * s02 + i3 * s03;
    const int idst = i0_half + i1 * s1 + i2 * s2 + i3 * s3;

    const float4 x0    = __ldg(reinterpret_cast<const float4 *>(x + ix));
    const float4 x1    = __ldg(reinterpret_cast<const float4 *>(x + ix + n_dims_half));
    const float  pos_f = (float) pos[i2];

    const float * x0f = reinterpret_cast<const float *>(&x0);
    const float * x1f = reinterpret_cast<const float *>(&x1);

    float4 ff4 = has_ff ? __ldg(reinterpret_cast<const float4 *>(freq_factors + i0_half))
                        : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const float * fff = reinterpret_cast<const float *>(&ff4);

    float theta_inv = powf(theta_scale, (float) i0_half);

    float out0[4];
    float out1[4];

#pragma unroll
    for (int k = 0; k < 4; k++) {
        const float theta = pos_f * theta_inv / fff[k];
        float       cos_theta;
        float       sin_theta;
        rope_yarn_vec<forward>(theta, freq_scale, corr_dims, i0 + 2 * k, ext_factor, attn_factor, cos_theta,
                               sin_theta);
        out0[k] = x0f[k] * cos_theta - x1f[k] * sin_theta;
        out1[k] = x0f[k] * sin_theta + x1f[k] * cos_theta;
        theta_inv *= theta_scale;
    }

    *reinterpret_cast<float4 *>(dst + idst)               = *reinterpret_cast<float4 *>(out0);
    *reinterpret_cast<float4 *>(dst + idst + n_dims_half) = *reinterpret_cast<float4 *>(out1);
}

template <bool forward>
void launch_rope_neox_vec(const float *       x,
                          float *             dst,
                          int                 ne00,
                          int                 ne01,
                          int                 ne02,
                          int                 s01,
                          int                 s02,
                          int                 s03,
                          int                 s1,
                          int                 s2,
                          int                 s3,
                          int                 n_dims,
                          int                 nr,
                          const int32_t *     pos,
                          float               freq_scale,
                          float               ext_factor,
                          float               attn_factor,
                          const corr_dims_vec corr_dims,
                          float               theta_scale,
                          const float *       freq_factors,
                          cudaStream_t        stream) {
    // threads_x_full = (ne00/2)/4 = ne00/8 groups of 4 pairs.
    const int threads_x_full = ne00 / 8;
    int       block_x        = threads_x_full < ROPE_NEOX_VEC_MAX_X ? threads_x_full : ROPE_NEOX_VEC_MAX_X;
    int       block_y        = ROPE_NEOX_VEC_BLOCK_SIZE / block_x;
    if (block_y < 1) {
        block_y = 1;
    }

    const dim3 block_dims(block_x, block_y, 1);
    const dim3 block_nums((nr + block_y - 1) / block_y, (threads_x_full + block_x - 1) / block_x, 1);

    if (freq_factors == nullptr) {
        rope_neox_vec<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    } else {
        rope_neox_vec<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

}  // namespace rope_neox_vec_detail

// Returns true when the vectorized fast path handled the op, false otherwise.
static inline bool ggml_cuda_rope_neox_vec_f32(const float *   x,
                                               float *         dst,
                                               int             ne00,
                                               int             ne01,
                                               int             ne02,
                                               int             s01,
                                               int             s02,
                                               int             s03,
                                               int             s1,
                                               int             s2,
                                               int             s3,
                                               int             n_dims,
                                               int             nr,
                                               const int32_t * pos,
                                               float           freq_scale,
                                               float           freq_base,
                                               float           ext_factor,
                                               float           attn_factor,
                                               const float *   corr_dims,
                                               const float *   freq_factors,
                                               bool            forward,
                                               int             set_rows_stride,
                                               cudaStream_t    stream) {
    // float4 access needs full rope, 8-element granularity and 4-aligned strides.
    const bool eligible = set_rows_stride == 0 && n_dims == ne00 && (ne00 % 8 == 0) && (s01 % 4 == 0) &&
                          (s02 % 4 == 0) && (s03 % 4 == 0) && (s1 % 4 == 0) && (s2 % 4 == 0) && (s3 % 4 == 0);
    if (!eligible) {
        return false;
    }

    rope_neox_vec_detail::corr_dims_vec cd;
    cd.v[0] = corr_dims[0];
    cd.v[1] = corr_dims[1];

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (forward) {
        rope_neox_vec_detail::launch_rope_neox_vec<true>(x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                                                         n_dims, nr, pos, freq_scale, ext_factor, attn_factor, cd,
                                                         theta_scale, freq_factors, stream);
    } else {
        rope_neox_vec_detail::launch_rope_neox_vec<false>(x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                                                          n_dims, nr, pos, freq_scale, ext_factor, attn_factor, cd,
                                                          theta_scale, freq_factors, stream);
    }

    return true;
}
