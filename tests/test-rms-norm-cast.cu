// Standalone correctness + performance test for the rms_norm_cast kernels.
//
// Build (adjust -arch and the ggml library path to your tree):
//
//   nvcc -std=c++17 -O3 -arch=sm_90 -lineinfo \
//        -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//        tests/test-rms-norm-cast.cu \
//        -L build/bin -lggml-cuda -lggml-base -o test-rms-norm-cast
//
// Linking ggml is required because ggml_cuda_kernel_launch (used by both the kernel under test and
// the transcribed baseline) reaches into ggml_cuda_get_device / ggml_cuda_error for the PDL path.
// Using the same launch path for both sides is what makes the timing comparison fair.
//
// Reference values are computed on the host in double precision from the exact same input bits the
// kernel reads, so the reported error is the conversion error of the kernel and nothing else.

#include "rms_norm_cast.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ============================================================================================
// baseline: transcription of rms_norm_f32 / rms_norm_mul_f32_cuda from ggml-cuda/norm.cu
// ============================================================================================

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void baseline_rms_norm_f32(const float * x,
                                             float *       dst,
                                             const int     ncols,
                                             const int64_t stride_row,
                                             const int64_t stride_channel,
                                             const int64_t stride_sample,
                                             const float   eps,
                                             const float * mul,
                                             const int64_t mul_stride_row,
                                             const int64_t mul_stride_channel,
                                             const int64_t mul_stride_sample,
                                             const uint3   mul_ncols_packed,
                                             const uint3   mul_nrows_packed,
                                             const uint3   mul_nchannels_packed,
                                             const uint3   mul_nsamples_packed,
                                             const float * add,
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

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

#define BASELINE_LAUNCH(BS, DO_MUL, DO_ADD)                                                        \
    do {                                                                                           \
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params{                  \
            blocks_num, dim3(BS, 1, 1), (BS) > WARP_SIZE ? 32*sizeof(float) : 0, stream };         \
        ggml_cuda_kernel_launch(                                                                   \
            baseline_rms_norm_f32<BS, DO_MUL, DO_ADD>, lp,                                         \
            (const float *) a.src, (float *) a.dst, a.ncols,                                       \
            a.stride_row, a.stride_channel, a.stride_sample, a.eps,                                \
            (const float *) a.mul, a.mul_stride_row, a.mul_stride_channel, a.mul_stride_sample,    \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,          \
            (const float *) a.add, a.add_stride_row, a.add_stride_channel, a.add_stride_sample,    \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed);         \
    } while (0)

// F32-only, same dispatch structure as rms_norm_mul_f32_cuda with the PPU-specific branch removed.
static void baseline_rms_norm_cuda(const rms_norm_cast_args & a, cudaStream_t stream) {
    GGML_ASSERT(a.src_type == GGML_TYPE_F32 && a.dst_type == GGML_TYPE_F32);

    const dim3 blocks_num(a.nrows, a.nchannels, a.nsamples);

    uint3 mul_ncols_packed     = make_uint3(0, 0, 0);
    uint3 mul_nrows_packed     = make_uint3(0, 0, 0);
    uint3 mul_nchannels_packed = make_uint3(0, 0, 0);
    uint3 mul_nsamples_packed  = make_uint3(0, 0, 0);

    uint3 add_ncols_packed     = make_uint3(0, 0, 0);
    uint3 add_nrows_packed     = make_uint3(0, 0, 0);
    uint3 add_nchannels_packed = make_uint3(0, 0, 0);
    uint3 add_nsamples_packed  = make_uint3(0, 0, 0);

    if (a.mul != nullptr) {
        GGML_ASSERT(a.mul_type == GGML_TYPE_F32);
        mul_ncols_packed     = init_fastdiv_values(a.mul_ncols);
        mul_nrows_packed     = init_fastdiv_values(a.mul_nrows);
        mul_nchannels_packed = init_fastdiv_values(a.mul_nchannels);
        mul_nsamples_packed  = init_fastdiv_values(a.mul_nsamples);
    }

    if (a.add != nullptr) {
        GGML_ASSERT(a.add_type == GGML_TYPE_F32);
        add_ncols_packed     = init_fastdiv_values(a.add_ncols);
        add_nrows_packed     = init_fastdiv_values(a.add_nrows);
        add_nchannels_packed = init_fastdiv_values(a.add_nchannels);
        add_nsamples_packed  = init_fastdiv_values(a.add_nsamples);
    }

    if (a.mul == nullptr) {
        if (a.ncols < 1024) { BASELINE_LAUNCH(256, false, false); } else { BASELINE_LAUNCH(1024, false, false); }
    } else if (a.add == nullptr) {
        if (a.ncols < 1024) { BASELINE_LAUNCH(256, true, false); } else { BASELINE_LAUNCH(1024, true, false); }
    } else {
        if (a.ncols < 1024) { BASELINE_LAUNCH(256, true, true); } else { BASELINE_LAUNCH(1024, true, true); }
    }
}

#undef BASELINE_LAUNCH

// ============================================================================================
// host side dtype plumbing
// ============================================================================================

#define CU_CHECK(expr)                                                                             \
    do {                                                                                           \
        const cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));\
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

static const char * type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        default:             return "?";
    }
}

static ggml_type type_from_name(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    fprintf(stderr, "unknown type '%s' (expected f32, f16 or bf16)\n", s.c_str());
    exit(1);
}

static size_t type_bytes(ggml_type t) {
    return t == GGML_TYPE_F32 ? 4 : 2;
}

// Round a float to the storage type and read it back, i.e. the value the kernel actually sees.
static float quantize_roundtrip(ggml_type t, float v) {
    switch (t) {
        case GGML_TYPE_F32:  return v;
        case GGML_TYPE_F16:  return __half2float(__float2half(v));
        case GGML_TYPE_BF16: return __bfloat162float(__float2bfloat16(v));
        default:             GGML_ABORT("bad type");
    }
}

static void store_as(ggml_type t, void * base, size_t i, float v) {
    switch (t) {
        case GGML_TYPE_F32:  ((float *)        base)[i] = v;                    break;
        case GGML_TYPE_F16:  ((half *)         base)[i] = __float2half(v);      break;
        case GGML_TYPE_BF16: ((nv_bfloat16 *)  base)[i] = __float2bfloat16(v);  break;
        default:             GGML_ABORT("bad type");
    }
}

static float load_as(ggml_type t, const void * base, size_t i) {
    switch (t) {
        case GGML_TYPE_F32:  return ((const float *)       base)[i];
        case GGML_TYPE_F16:  return __half2float(((const half *) base)[i]);
        case GGML_TYPE_BF16: return __bfloat162float(((const nv_bfloat16 *) base)[i]);
        default:             GGML_ABORT("bad type");
    }
}

// A host buffer plus its device mirror, typed at runtime.
struct buffer {
    ggml_type            type = GGML_TYPE_F32;
    size_t               n    = 0;      // element count
    std::vector<uint8_t> host;
    void *               dev  = nullptr;

    void alloc(ggml_type t, size_t count) {
        type = t;
        n    = count;
        host.assign(count * type_bytes(t), 0);
        CU_CHECK(cudaMalloc(&dev, host.size()));
    }

    void free_all() {
        if (dev) {
            CU_CHECK(cudaFree(dev));
            dev = nullptr;
        }
        host.clear();
        host.shrink_to_fit();
    }

    void upload()   { CU_CHECK(cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice)); }
    void download() { CU_CHECK(cudaMemcpy(host.data(), dev, host.size(), cudaMemcpyDeviceToHost)); }

    void set(size_t i, float v)      { store_as(type, host.data(), i, v); }
    float get(size_t i) const        { return load_as(type, host.data(), i); }
};

static void fill_uniform(buffer & b, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < b.n; ++i) {
        b.set(i, dist(rng));
    }
}

// Perf runs only need finite, normalized data, so generate one small random tile and replicate it.
// A per-element fill would dominate the wall clock for the multi-million element shapes.
static void fill_fast(buffer & b, std::mt19937 & rng, float lo, float hi) {
    const size_t tile = std::min<size_t>(b.n, 4096);

    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < tile; ++i) {
        b.set(i, dist(rng));
    }

    const size_t ts = type_bytes(b.type);
    for (size_t off = tile; off < b.n; off += tile) {
        const size_t cnt = std::min(tile, b.n - off);
        memcpy(b.host.data() + off*ts, b.host.data(), cnt*ts);
    }
}

// ============================================================================================
// case description
// ============================================================================================

enum fuse_mode { FUSE_NONE = 0, FUSE_MUL = 1, FUSE_MUL_ADD = 2 };

static const char * fuse_name(int f) {
    switch (f) {
        case FUSE_NONE:    return "plain";
        case FUSE_MUL:     return "mul";
        case FUSE_MUL_ADD: return "mul+add";
        default:           return "?";
    }
}

struct test_case {
    int64_t ne[4]     = {0, 1, 1, 1};
    int64_t row_pad   = 0;           // extra elements between src rows (non-contiguous src)
    float   eps       = 1e-6f;
    int     fuse      = FUSE_NONE;
    int64_t mul_ne[4] = {0, 1, 1, 1};
    int64_t add_ne[4] = {0, 1, 1, 1};

    ggml_type ts = GGML_TYPE_F32;
    ggml_type td = GGML_TYPE_F32;
    ggml_type tm = GGML_TYPE_F32;
    ggml_type ta = GGML_TYPE_F32;

    float data_lo = -10.0f;
    float data_hi =  10.0f;

    std::string shape_str() const {
        char buf[96];
        snprintf(buf, sizeof(buf), "%lldx%lldx%lldx%lld",
                 (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        return buf;
    }

    std::string dtype_str() const {
        std::string s = std::string(type_name(ts)) + "/" + type_name(td);
        if (fuse >= FUSE_MUL)     { s += std::string("/") + type_name(tm); }
        if (fuse == FUSE_MUL_ADD) { s += std::string("/") + type_name(ta); }
        return s;
    }

    std::string bcast_str() const {
        if (fuse == FUSE_NONE) {
            return "-";
        }
        char buf[128];
        if (fuse == FUSE_MUL) {
            snprintf(buf, sizeof(buf), "mul[%lld,%lld,%lld,%lld]",
                     (long long) mul_ne[0], (long long) mul_ne[1],
                     (long long) mul_ne[2], (long long) mul_ne[3]);
        } else {
            snprintf(buf, sizeof(buf), "mul[%lld,%lld,%lld,%lld] add[%lld,%lld,%lld,%lld]",
                     (long long) mul_ne[0], (long long) mul_ne[1],
                     (long long) mul_ne[2], (long long) mul_ne[3],
                     (long long) add_ne[0], (long long) add_ne[1],
                     (long long) add_ne[2], (long long) add_ne[3]);
        }
        return buf;
    }
};

// ============================================================================================
// host reference in double precision
// ============================================================================================

static void reference(const test_case & c,
                      const buffer &    src,
                      const buffer &    mul,
                      const buffer &    add,
                      std::vector<float> & out) {
    const int64_t ncols = c.ne[0];
    const int64_t nrows = c.ne[1];
    const int64_t nch   = c.ne[2];
    const int64_t nsmp  = c.ne[3];

    const int64_t s_row = ncols + c.row_pad;
    const int64_t s_ch  = s_row * nrows;
    const int64_t s_smp = s_ch  * nch;

    out.assign((size_t) ncols*nrows*nch*nsmp, 0.0f);

    for (int64_t s = 0; s < nsmp; ++s) {
        for (int64_t ch = 0; ch < nch; ++ch) {
            for (int64_t r = 0; r < nrows; ++r) {
                const size_t x0 = (size_t) (s*s_smp + ch*s_ch + r*s_row);

                double sum = 0.0;
                for (int64_t col = 0; col < ncols; ++col) {
                    const double xi = src.get(x0 + col);
                    sum += xi * xi;
                }

                const double mean  = sum / (double) ncols;
                const double scale = 1.0 / std::sqrt(mean + (double) c.eps);

                size_t m0 = 0;
                if (c.fuse >= FUSE_MUL) {
                    const int64_t mr = r  % c.mul_ne[1];
                    const int64_t mc = ch % c.mul_ne[2];
                    const int64_t ms = s  % c.mul_ne[3];
                    m0 = (size_t) (((ms*c.mul_ne[2] + mc)*c.mul_ne[1] + mr) * c.mul_ne[0]);
                }

                size_t a0 = 0;
                if (c.fuse == FUSE_MUL_ADD) {
                    const int64_t ar = r  % c.add_ne[1];
                    const int64_t ac = ch % c.add_ne[2];
                    const int64_t as = s  % c.add_ne[3];
                    a0 = (size_t) (((as*c.add_ne[2] + ac)*c.add_ne[1] + ar) * c.add_ne[0]);
                }

                const size_t y0 = (size_t) (((s*nch + ch)*nrows + r) * ncols);

                for (int64_t col = 0; col < ncols; ++col) {
                    double v = scale * (double) src.get(x0 + col);
                    if (c.fuse >= FUSE_MUL) {
                        v *= (double) mul.get(m0 + (size_t) (col % c.mul_ne[0]));
                    }
                    if (c.fuse == FUSE_MUL_ADD) {
                        v += (double) add.get(a0 + (size_t) (col % c.add_ne[0]));
                    }
                    out[y0 + (size_t) col] = quantize_roundtrip(c.td, (float) v);
                }
            }
        }
    }
}

// ============================================================================================
// comparison
// ============================================================================================

struct cmp_result {
    double  nmse       = 0.0;
    double  rms        = 0.0;  // rms of the reference
    double  max_abs    = 0.0;
    double  amax       = 0.0;  // max_abs / rms, the gating metric
    double  max_rel    = 0.0;  // diagnostic only, see the comment on amax
    int64_t worst_idx  = -1;
    double  worst_got  = 0.0;
    double  worst_want = 0.0;
};

// Two error measures are collected:
//
//   amax    = max|got - want| / rms(want)
//   max_rel = max( |got - want| / |want| )
//
// amax is the one worth gating on. The kernel rounds relative to the magnitude of the intermediate
// terms (scale*x*mul and add), not to the magnitude of the final value, so when those terms cancel
// the result is near zero while the absolute error stays at one ULP of the intermediate. That makes
// max_rel blow up on perfectly correct output, which is why it is reported but not enforced.
static cmp_result compare(const buffer & got, const std::vector<float> & want) {
    cmp_result r;

    double sum_d2 = 0.0;
    double sum_r2 = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        sum_r2 += (double) want[i] * want[i];
    }

    r.rms = std::sqrt(sum_r2 / std::max<size_t>(1, want.size()));

    const double floor = std::max(1e-30, r.rms * 1e-3);

    for (size_t i = 0; i < want.size(); ++i) {
        const double g = got.get(i);
        const double w = want[i];
        const double d = std::fabs(g - w);

        sum_d2 += d * d;

        if (d > r.max_abs) {
            r.max_abs    = d;
            r.worst_idx  = (int64_t) i;
            r.worst_got  = g;
            r.worst_want = w;
        }

        r.max_rel = std::max(r.max_rel, d / std::max(std::fabs(w), floor));
    }

    r.nmse = sum_r2 > 0.0 ? sum_d2 / sum_r2 : sum_d2;
    r.amax = r.rms > 0.0 ? r.max_abs / r.rms : r.max_abs;
    return r;
}

static double nmse_tol(ggml_type dst_type) {
    switch (dst_type) {
        // F32: the only difference from the double reference is that the kernel reduces and computes
        // the scale in fp32, which lands around 1e-13 for the widest rows tested here.
        case GGML_TYPE_F32:  return 1e-11;
        case GGML_TYPE_F16:  return 1e-6;
        case GGML_TYPE_BF16: return 1e-5;
        default:             return 1e-5;
    }
}

static double amax_tol(ggml_type dst_type) {
    // Bound on max|err| / rms(ref). The rounding error of one element is at most one ULP of its own
    // magnitude, and max|ref| runs a small factor above rms(ref) for the data used here, so a few
    // times the ULP of the destination type is the right order.
    switch (dst_type) {
        case GGML_TYPE_F32:  return 1e-5;   // 2^-24 plus the fp32 reduction error in scale
        case GGML_TYPE_F16:  return 5e-3;   // 2^-11 = 4.9e-4
        case GGML_TYPE_BF16: return 3e-2;   // 2^-8  = 3.9e-3
        default:             return 3e-2;
    }
}

// ============================================================================================
// buffer setup
// ============================================================================================

static int64_t prod4(const int64_t ne[4]) {
    return ne[0]*ne[1]*ne[2]*ne[3];
}

struct case_state {
    buffer src, dst, mul, add;
    buffer dst_baseline;

    rms_norm_cast_args args;

    void free_all() {
        src.free_all();
        dst.free_all();
        mul.free_all();
        add.free_all();
        dst_baseline.free_all();
    }
};

// Allocates and fills all operands and builds the launch arguments. The padding between src rows and
// the initial dst contents are NaN so that an out-of-range read or a missing write is loud.
static void setup_case(const test_case & c, case_state & st, std::mt19937 & rng, bool want_baseline,
                       bool fast = false) {
    const int64_t ncols = c.ne[0];
    const int64_t s_row = ncols + c.row_pad;
    const int64_t s_ch  = s_row * c.ne[1];
    const int64_t s_smp = s_ch  * c.ne[2];

    const int64_t n_src = s_smp * c.ne[3];
    const int64_t n_dst = prod4(c.ne);

    st.src.alloc(c.ts, (size_t) n_src);
    st.dst.alloc(c.td, (size_t) n_dst);

    if (fast) {
        fill_fast(st.src, rng, c.data_lo, c.data_hi);
    } else {
        fill_uniform(st.src, rng, c.data_lo, c.data_hi);
    }

    if (!fast && c.row_pad > 0) {
        for (int64_t s = 0; s < c.ne[3]; ++s) {
            for (int64_t ch = 0; ch < c.ne[2]; ++ch) {
                for (int64_t r = 0; r < c.ne[1]; ++r) {
                    const size_t base = (size_t) (s*s_smp + ch*s_ch + r*s_row + ncols);
                    for (int64_t p = 0; p < c.row_pad; ++p) {
                        st.src.set(base + (size_t) p, NAN);
                    }
                }
            }
        }
    }

    if (!fast) {
        for (size_t i = 0; i < st.dst.n; ++i) {
            st.dst.set(i, NAN);
        }
    }

    st.src.upload();
    st.dst.upload();

    rms_norm_cast_args & a = st.args;
    a = rms_norm_cast_args{};

    a.src      = st.src.dev;
    a.src_type = c.ts;
    a.dst      = st.dst.dev;
    a.dst_type = c.td;

    a.ncols     = (int) ncols;
    a.nrows     = (int) c.ne[1];
    a.nchannels = (int) c.ne[2];
    a.nsamples  = (int) c.ne[3];

    a.stride_row     = s_row;
    a.stride_channel = s_ch;
    a.stride_sample  = s_smp;

    a.eps = c.eps;

    if (c.fuse >= FUSE_MUL) {
        st.mul.alloc(c.tm, (size_t) prod4(c.mul_ne));
        fill_uniform(st.mul, rng, -2.0f, 2.0f);
        st.mul.upload();

        a.mul                = st.mul.dev;
        a.mul_type           = c.tm;
        a.mul_stride_row     = c.mul_ne[0];
        a.mul_stride_channel = c.mul_ne[0]*c.mul_ne[1];
        a.mul_stride_sample  = c.mul_ne[0]*c.mul_ne[1]*c.mul_ne[2];
        a.mul_ncols          = (uint32_t) c.mul_ne[0];
        a.mul_nrows          = (uint32_t) c.mul_ne[1];
        a.mul_nchannels      = (uint32_t) c.mul_ne[2];
        a.mul_nsamples       = (uint32_t) c.mul_ne[3];
    }

    if (c.fuse == FUSE_MUL_ADD) {
        st.add.alloc(c.ta, (size_t) prod4(c.add_ne));
        fill_uniform(st.add, rng, -2.0f, 2.0f);
        st.add.upload();

        a.add                = st.add.dev;
        a.add_type           = c.ta;
        a.add_stride_row     = c.add_ne[0];
        a.add_stride_channel = c.add_ne[0]*c.add_ne[1];
        a.add_stride_sample  = c.add_ne[0]*c.add_ne[1]*c.add_ne[2];
        a.add_ncols          = (uint32_t) c.add_ne[0];
        a.add_nrows          = (uint32_t) c.add_ne[1];
        a.add_nchannels      = (uint32_t) c.add_ne[2];
        a.add_nsamples       = (uint32_t) c.add_ne[3];
    }

    if (want_baseline) {
        st.dst_baseline.alloc(GGML_TYPE_F32, (size_t) n_dst);
        for (size_t i = 0; i < st.dst_baseline.n; ++i) {
            st.dst_baseline.set(i, NAN);
        }
        st.dst_baseline.upload();
    }
}

// The baseline is F32-only, so it can only cross-check cases where every operand is F32.
static bool case_has_baseline(const test_case & c) {
    if (c.ts != GGML_TYPE_F32 || c.td != GGML_TYPE_F32) {
        return false;
    }
    if (c.fuse >= FUSE_MUL && c.tm != GGML_TYPE_F32) {
        return false;
    }
    if (c.fuse == FUSE_MUL_ADD && c.ta != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

// ============================================================================================
// correctness
// ============================================================================================

struct case_result {
    bool       passed       = false;
    bool       has_nan      = false;
    cmp_result cmp;
    bool       has_baseline = false;
    cmp_result cmp_baseline;
};

static case_result run_correctness(const test_case & c, std::mt19937 & rng, cudaStream_t stream) {
    case_result res;

    const bool want_baseline = case_has_baseline(c);

    case_state st;
    setup_case(c, st, rng, want_baseline);

    rms_norm_cast_launch(st.args, stream);
    CU_CHECK(cudaStreamSynchronize(stream));
    st.dst.download();

    if (want_baseline) {
        rms_norm_cast_args ab = st.args;
        ab.dst      = st.dst_baseline.dev;
        ab.dst_type = GGML_TYPE_F32;
        baseline_rms_norm_cuda(ab, stream);
        CU_CHECK(cudaStreamSynchronize(stream));
        st.dst_baseline.download();
    }

    std::vector<float> want;
    reference(c, st.src, st.mul, st.add, want);

    for (size_t i = 0; i < st.dst.n; ++i) {
        if (std::isnan(st.dst.get(i)) && !std::isnan(want[i])) {
            res.has_nan = true;
            break;
        }
    }

    res.cmp = compare(st.dst, want);

    if (want_baseline) {
        res.has_baseline = true;
        res.cmp_baseline = compare(st.dst_baseline, want);
    }

    res.passed = !res.has_nan
              && res.cmp.nmse <= nmse_tol(c.td)
              && res.cmp.amax <= amax_tol(c.td);

    st.free_all();
    return res;
}

// ============================================================================================
// case generation
// ============================================================================================

static const ggml_type ALL_TYPES[3] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };

struct shape4 { int64_t ne[4]; };

// Correctness shapes: chosen for branch coverage, small enough for a double precision host
// reference. The comment names the branch each one exercises.
static const shape4 CORRECTNESS_SHAPES[] = {
    { { 5120,  8, 1, 1 } },  // large ncols           -> block 1024
    { { 5120,  1, 1, 1 } },  // single row            -> block 1024, one CTA
    { { 1024,  5, 4, 3 } },  // ncols == 1024         -> block 1024, all four dims > 1
    { { 1023,  5, 4, 3 } },  // ncols <  1024         -> block  256, odd ncols
    { {  128, 40, 8, 1 } },  // Qwen3 q_norm head_dim -> block  256
    { {   33,  7, 3, 2 } },  // ncols < warp size     -> block  256, no shared reduction
    { { 2048,  3, 2, 2 } },  // mid size              -> block 1024
    { {  256, 16, 4, 2 } },  // mid size              -> block  256
};

// Perf shapes: real Qwen3 workloads plus a bandwidth saturating case.
static const shape4 PERF_SHAPES[] = {
    { { 5120,  512,   1, 1 } },  // Qwen3-32B attn_norm, prefill
    { { 5120, 4096,   1, 1 } },  // long prefill, bandwidth bound
    { { 5120,    1,   1, 1 } },  // decode, launch latency bound
    { {  128,   40, 512, 1 } },  // Qwen3 q_norm, prefill
    { {  128,    8, 512, 1 } },  // Qwen3 k_norm, prefill
    { { 1024, 2048,   1, 1 } },  // block 1024 boundary at scale
};

static void set_ne(int64_t dst[4], int64_t a, int64_t b, int64_t c, int64_t d) {
    dst[0] = a; dst[1] = b; dst[2] = c; dst[3] = d;
}

// The broadcast patterns applied to mul and add. Returns false when a pattern does not apply to this
// shape (the ncols/4 pattern needs ncols divisible by 4).
static bool bcast_ne(int variant, const int64_t ne[4], int64_t out[4]) {
    switch (variant) {
        case 0: set_ne(out, ne[0], 1, 1, 1);             return true;  // per-column weight
        case 1: set_ne(out, ne[0], ne[1], ne[2], ne[3]); return true;  // no broadcast
        case 2:
            if (ne[0] % 4 != 0) {
                return false;
            }
            set_ne(out, ne[0]/4, 1, 1, 1);               return true;  // real column broadcast
        case 3: set_ne(out, ne[0], 1, ne[2], 1);         return true;  // mixed dims
        default: return false;
    }
}

// Full 3^4 dtype coverage on one shape: 9 plain + 27 mul + 81 mul+add = 117 cases.
static void gen_dtype_sweep(std::vector<test_case> & out, const shape4 & s) {
    for (ggml_type ts : ALL_TYPES) {
        for (ggml_type td : ALL_TYPES) {
            test_case c;
            memcpy(c.ne, s.ne, sizeof(c.ne));
            c.fuse = FUSE_NONE;
            c.ts   = ts;
            c.td   = td;
            out.push_back(c);

            for (ggml_type tm : ALL_TYPES) {
                test_case c1 = c;
                c1.fuse = FUSE_MUL;
                c1.tm   = tm;
                bcast_ne(0, s.ne, c1.mul_ne);
                out.push_back(c1);

                for (ggml_type ta : ALL_TYPES) {
                    test_case c2 = c1;
                    c2.fuse = FUSE_MUL_ADD;
                    c2.ta   = ta;
                    bcast_ne(0, s.ne, c2.add_ne);
                    out.push_back(c2);
                }
            }
        }
    }
}

struct dtype_combo {
    ggml_type    ts, td, tm, ta;
    const char * label;
};

static const dtype_combo SHAPE_SWEEP_COMBOS[] = {
    { GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32, "f32/f32"   },
    { GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32, "bf16/bf16" },
    { GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16, "f16 all"   },
    { GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_BF16, GGML_TYPE_F32, "bf16>f32"  },
};

// Every shape x every fuse mode x representative dtype combos x contiguous/padded x eps.
static void gen_shape_sweep(std::vector<test_case> & out) {
    const int64_t pads[2] = { 0, 3 };
    const float   epss[2] = { 1e-6f, 0.0f };

    for (const shape4 & s : CORRECTNESS_SHAPES) {
        for (int fuse = FUSE_NONE; fuse <= FUSE_MUL_ADD; ++fuse) {
            for (const dtype_combo & dc : SHAPE_SWEEP_COMBOS) {
                for (int64_t pad : pads) {
                    for (float eps : epss) {
                        test_case c;
                        memcpy(c.ne, s.ne, sizeof(c.ne));
                        c.row_pad = pad;
                        c.eps     = eps;
                        c.fuse    = fuse;
                        c.ts = dc.ts;
                        c.td = dc.td;
                        c.tm = dc.tm;
                        c.ta = dc.ta;
                        bcast_ne(0, s.ne, c.mul_ne);
                        bcast_ne(0, s.ne, c.add_ne);
                        out.push_back(c);
                    }
                }
            }
        }
    }
}

static const dtype_combo BCAST_SWEEP_COMBOS[] = {
    { GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  "f32/f32"   },
    { GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32,  "bf16/bf16" },
    { GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, "bf16 all"  },
};

// Shapes with all four dims > 1 so that row / channel / sample broadcasting is observable.
static const shape4 BCAST_SHAPES[] = {
    { { 1024,  5, 4, 3 } },
    { {  256, 16, 4, 2 } },
    { {   33,  7, 3, 2 } },
};

static void gen_bcast_sweep(std::vector<test_case> & out) {
    for (const shape4 & s : BCAST_SHAPES) {
        for (const dtype_combo & dc : BCAST_SWEEP_COMBOS) {
            for (int mv = 0; mv < 4; ++mv) {
                int64_t mul_ne[4];
                if (!bcast_ne(mv, s.ne, mul_ne)) {
                    continue;
                }

                test_case c1;
                memcpy(c1.ne, s.ne, sizeof(c1.ne));
                c1.fuse = FUSE_MUL;
                c1.ts = dc.ts; c1.td = dc.td; c1.tm = dc.tm; c1.ta = dc.ta;
                memcpy(c1.mul_ne, mul_ne, sizeof(mul_ne));
                out.push_back(c1);

                for (int av = 0; av < 4; ++av) {
                    int64_t add_ne[4];
                    if (!bcast_ne(av, s.ne, add_ne)) {
                        continue;
                    }
                    test_case c2 = c1;
                    c2.fuse = FUSE_MUL_ADD;
                    memcpy(c2.add_ne, add_ne, sizeof(add_ne));
                    out.push_back(c2);
                }
            }
        }
    }
}

// Tiny magnitudes with eps == 0 stress the mean -> 0 path where scale becomes very large.
static void gen_small_magnitude_sweep(std::vector<test_case> & out) {
    const shape4 s = { { 1024, 5, 4, 3 } };

    std::vector<test_case> tmp;
    gen_dtype_sweep(tmp, s);

    for (test_case & c : tmp) {
        c.data_lo = -1e-3f;
        c.data_hi =  1e-3f;
        c.eps     = 0.0f;
        out.push_back(c);
    }
}

// ============================================================================================
// performance
// ============================================================================================

static const dtype_combo PERF_COMBOS[] = {
    { GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  "f32/f32"   },
    { GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32,  "bf16/bf16" },
    { GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, "bf16 all"  },
    { GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F32,  GGML_TYPE_F32,  "f16/f16"   },
    { GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16,  "f16 all"   },
    { GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  "bf16>f32"  },
    { GGML_TYPE_F32,  GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32,  "f32>bf16"  },
};

static const int N_PERF_COMBOS = (int) (sizeof(PERF_COMBOS)/sizeof(PERF_COMBOS[0]));

// Median of `reps` batches of `iters` launches. The median avoids the first-batch clock ramp and any
// stray interference showing up as the reported number.
template <typename Launch>
static double time_us(Launch && launch, cudaStream_t stream, int warmup, int reps, int iters) {
    reps  = std::max(1, reps);
    iters = std::max(1, iters);

    cudaEvent_t ev0, ev1;
    CU_CHECK(cudaEventCreate(&ev0));
    CU_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; ++i) {
        launch(stream);
    }
    CU_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(reps);

    for (int r = 0; r < reps; ++r) {
        CU_CHECK(cudaEventRecord(ev0, stream));
        for (int i = 0; i < iters; ++i) {
            launch(stream);
        }
        CU_CHECK(cudaEventRecord(ev1, stream));
        CU_CHECK(cudaEventSynchronize(ev1));

        float ms = 0.0f;
        CU_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples.push_back((double) ms * 1000.0 / (double) iters);
    }

    CU_CHECK(cudaEventDestroy(ev0));
    CU_CHECK(cudaEventDestroy(ev1));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size()/2];
}

// Minimum traffic: one read of every input plus one write of the output. The kernel reads x twice
// (sum of squares, then scale), just like the baseline, so the achieved DRAM traffic can be higher;
// counting the minimum keeps the number comparable to test-backend-ops op_size.
static double bytes_moved(const test_case & c) {
    double b = (double) prod4(c.ne) * (double) (type_bytes(c.ts) + type_bytes(c.td));
    if (c.fuse >= FUSE_MUL) {
        b += (double) prod4(c.mul_ne) * (double) type_bytes(c.tm);
    }
    if (c.fuse == FUSE_MUL_ADD) {
        b += (double) prod4(c.add_ne) * (double) type_bytes(c.ta);
    }
    return b;
}

struct perf_row {
    std::string         shape;
    int                 fuse = FUSE_NONE;
    double              baseline_us = 0.0;
    double              baseline_gb = 0.0;
    std::vector<double> us;
    std::vector<double> gb;
};

static test_case make_perf_case(const shape4 & s, int fuse, const dtype_combo & dc) {
    test_case c;
    memcpy(c.ne, s.ne, sizeof(c.ne));
    c.fuse = fuse;
    c.ts = dc.ts;
    c.td = dc.td;
    c.tm = dc.tm;
    c.ta = dc.ta;
    bcast_ne(0, s.ne, c.mul_ne);
    bcast_ne(0, s.ne, c.add_ne);
    return c;
}

static perf_row run_perf_row(const shape4 & s, int fuse, std::mt19937 & rng, cudaStream_t stream,
                             int warmup, int reps, int iters, int combo_filter) {
    perf_row row;
    row.fuse = fuse;
    row.us.assign(N_PERF_COMBOS, 0.0);
    row.gb.assign(N_PERF_COMBOS, 0.0);

    // baseline: all operands F32, same launch path
    {
        const dtype_combo f32_all = { GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, "f32" };
        const test_case   c       = make_perf_case(s, fuse, f32_all);
        row.shape = c.shape_str();

        case_state st;
        setup_case(c, st, rng, false, /*fast=*/true);
        row.baseline_us = time_us([&](cudaStream_t str) { baseline_rms_norm_cuda(st.args, str); },
                                  stream, warmup, reps, iters);
        row.baseline_gb = bytes_moved(c) / (row.baseline_us * 1e-6) / 1e9;
        st.free_all();
    }

    for (int i = 0; i < N_PERF_COMBOS; ++i) {
        if (combo_filter >= 0 && combo_filter != i) {
            continue;
        }

        const test_case c = make_perf_case(s, fuse, PERF_COMBOS[i]);

        case_state st;
        setup_case(c, st, rng, false, /*fast=*/true);
        row.us[i] = time_us([&](cudaStream_t str) { rms_norm_cast_launch(st.args, str); },
                            stream, warmup, reps, iters);
        row.gb[i] = bytes_moved(c) / (row.us[i] * 1e-6) / 1e9;
        st.free_all();
    }

    return row;
}

static void print_perf_tables(const std::vector<perf_row> & rows, int combo_filter) {
    auto header = [&](const char * title, const char * unit) {
        printf("\n=== %s (%s) ===\n", title, unit);
        printf("%-22s %-9s %10s", "shape", "fuse", "baseline");
        for (int i = 0; i < N_PERF_COMBOS; ++i) {
            if (combo_filter >= 0 && combo_filter != i) {
                continue;
            }
            printf(" %10s", PERF_COMBOS[i].label);
        }
        printf("\n");
    };

    header("TIME", "us, median");
    for (const perf_row & r : rows) {
        printf("%-22s %-9s %10.2f", r.shape.c_str(), fuse_name(r.fuse), r.baseline_us);
        for (int i = 0; i < N_PERF_COMBOS; ++i) {
            if (combo_filter >= 0 && combo_filter != i) {
                continue;
            }
            printf(" %10.2f", r.us[i]);
        }
        printf("\n");
    }

    header("RATIO vs baseline", "lower is better");
    for (const perf_row & r : rows) {
        printf("%-22s %-9s %10s", r.shape.c_str(), fuse_name(r.fuse), "1.00");
        for (int i = 0; i < N_PERF_COMBOS; ++i) {
            if (combo_filter >= 0 && combo_filter != i) {
                continue;
            }
            printf(" %10.2f", r.baseline_us > 0.0 ? r.us[i] / r.baseline_us : 0.0);
        }
        printf("\n");
    }

    header("BANDWIDTH", "GB/s, minimum traffic");
    for (const perf_row & r : rows) {
        printf("%-22s %-9s %10.1f", r.shape.c_str(), fuse_name(r.fuse), r.baseline_gb);
        for (int i = 0; i < N_PERF_COMBOS; ++i) {
            if (combo_filter >= 0 && combo_filter != i) {
                continue;
            }
            printf(" %10.1f", r.gb[i]);
        }
        printf("\n");
    }
}

static void write_perf_csv(const std::string & path, const std::vector<perf_row> & rows, int combo_filter) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "failed to open '%s' for writing\n", path.c_str());
        return;
    }

    fprintf(f, "shape,fuse,dtype,time_us,ratio,gb_s\n");
    for (const perf_row & r : rows) {
        fprintf(f, "%s,%s,baseline,%.3f,1.000,%.2f\n",
                r.shape.c_str(), fuse_name(r.fuse), r.baseline_us, r.baseline_gb);
        for (int i = 0; i < N_PERF_COMBOS; ++i) {
            if (combo_filter >= 0 && combo_filter != i) {
                continue;
            }
            fprintf(f, "%s,%s,%s,%.3f,%.3f,%.2f\n",
                    r.shape.c_str(), fuse_name(r.fuse), PERF_COMBOS[i].label,
                    r.us[i], r.baseline_us > 0.0 ? r.us[i]/r.baseline_us : 0.0, r.gb[i]);
        }
    }

    fclose(f);
    printf("\nwrote %s\n", path.c_str());
}

// ============================================================================================
// CLI
// ============================================================================================

struct options {
    bool do_correctness = true;
    bool do_perf        = true;
    bool quick          = false;
    bool verbose        = false;

    bool   have_shape = false;
    shape4 shape      = { { 0, 1, 1, 1 } };

    bool      have_dtype = false;
    ggml_type dt[4]      = { GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32 };

    std::string csv;

    int      warmup = 20;
    int      reps   = 11;
    int      iters  = 50;
    unsigned seed   = 1234;
};

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t                   start = 0;
    while (true) {
        const size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

static void usage(const char * prog) {
    printf("usage: %s [options]\n", prog);
    printf("  -c, --correctness   run correctness only\n");
    printf("  -p, --perf          run perf only\n");
    printf("      --quick         correctness: full dtype sweep on one shape only\n");
    printf("  -v, --verbose       print every case, not just failures\n");
    printf("      --shape a,b,c,d restrict to one shape\n");
    printf("      --dtype s,d,m,a restrict to one dtype quadruple (f32|f16|bf16)\n");
    printf("      --csv FILE      write the perf matrix as csv\n");
    printf("      --iters N       launches per timed batch (default 50)\n");
    printf("      --reps N        timed batches, median is reported (default 11)\n");
    printf("      --warmup N      untimed launches (default 20)\n");
    printf("      --seed N        rng seed (default 1234)\n");
}

static options parse_args(int argc, char ** argv) {
    options o;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires an argument\n", what);
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(0);
        } else if (arg == "-c" || arg == "--correctness") {
            o.do_perf = false;
        } else if (arg == "-p" || arg == "--perf") {
            o.do_correctness = false;
        } else if (arg == "--quick") {
            o.quick = true;
        } else if (arg == "-v" || arg == "--verbose") {
            o.verbose = true;
        } else if (arg == "--shape") {
            const std::vector<std::string> p = split(next("--shape"), ',');
            if (p.size() != 4) {
                fprintf(stderr, "--shape expects four comma separated values\n");
                exit(1);
            }
            for (int k = 0; k < 4; ++k) {
                o.shape.ne[k] = atoll(p[k].c_str());
            }
            o.have_shape = true;
        } else if (arg == "--dtype") {
            const std::vector<std::string> p = split(next("--dtype"), ',');
            if (p.size() != 4) {
                fprintf(stderr, "--dtype expects four comma separated types\n");
                exit(1);
            }
            for (int k = 0; k < 4; ++k) {
                o.dt[k] = type_from_name(p[k]);
            }
            o.have_dtype = true;
        } else if (arg == "--csv") {
            o.csv = next("--csv");
        } else if (arg == "--iters") {
            o.iters = atoi(next("--iters").c_str());
        } else if (arg == "--reps") {
            o.reps = atoi(next("--reps").c_str());
        } else if (arg == "--warmup") {
            o.warmup = atoi(next("--warmup").c_str());
        } else if (arg == "--seed") {
            o.seed = (unsigned) atoi(next("--seed").c_str());
        } else {
            fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
            usage(argv[0]);
            exit(1);
        }
    }

    return o;
}

static bool dtype_matches(const options & o, const test_case & c) {
    if (!o.have_dtype) {
        return true;
    }
    if (c.ts != o.dt[0] || c.td != o.dt[1]) {
        return false;
    }
    if (c.fuse >= FUSE_MUL && c.tm != o.dt[2]) {
        return false;
    }
    if (c.fuse == FUSE_MUL_ADD && c.ta != o.dt[3]) {
        return false;
    }
    return true;
}

static int perf_combo_filter(const options & o) {
    if (!o.have_dtype) {
        return -1;
    }
    for (int i = 0; i < N_PERF_COMBOS; ++i) {
        const dtype_combo & d = PERF_COMBOS[i];
        if (d.ts == o.dt[0] && d.td == o.dt[1] && d.tm == o.dt[2] && d.ta == o.dt[3]) {
            return i;
        }
    }
    fprintf(stderr, "warning: --dtype does not match any perf column, timing all of them\n");
    return -1;
}

static void print_case_line(int idx, int total, const test_case & c, const case_result & r) {
    printf("[%5d/%5d] %-18s pad=%lld eps=%-8g %-8s %-20s %-40s\n",
           idx, total, c.shape_str().c_str(), (long long) c.row_pad, c.eps,
           fuse_name(c.fuse), c.dtype_str().c_str(), c.bcast_str().c_str());
    printf("             nmse=%.3e amax=%.3e max_abs=%.3e (max_rel=%.3e)",
           r.cmp.nmse, r.cmp.amax, r.cmp.max_abs, r.cmp.max_rel);
    if (r.has_baseline) {
        printf("  |  baseline nmse=%.3e amax=%.3e", r.cmp_baseline.nmse, r.cmp_baseline.amax);
    }
    printf("   %s\n", r.passed ? "PASS" : "FAIL");
    if (!r.passed) {
        if (r.has_nan) {
            printf("             unexpected NaN in the output\n");
        }
        printf("             worst idx=%lld got=%.8g want=%.8g rms=%.4g (tol: nmse %.1e, amax %.1e)\n",
               (long long) r.cmp.worst_idx, r.cmp.worst_got, r.cmp.worst_want, r.cmp.rms,
               nmse_tol(c.td), amax_tol(c.td));
    }
}

int main(int argc, char ** argv) {
    const options o = parse_args(argc, argv);

    CU_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CU_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    cudaStream_t stream;
    CU_CHECK(cudaStreamCreate(&stream));

    int n_fail = 0;

    if (o.do_correctness) {
        std::vector<test_case> cases;

        if (o.have_shape) {
            gen_dtype_sweep(cases, o.shape);
        } else if (o.quick) {
            gen_dtype_sweep(cases, CORRECTNESS_SHAPES[0]);
        } else {
            gen_dtype_sweep(cases, CORRECTNESS_SHAPES[0]);  // large ncols, block 1024
            gen_dtype_sweep(cases, CORRECTNESS_SHAPES[4]);  // small ncols, block 256
            gen_shape_sweep(cases);
            gen_bcast_sweep(cases);
            gen_small_magnitude_sweep(cases);
        }

        std::vector<test_case> filtered;
        for (const test_case & c : cases) {
            if (dtype_matches(o, c)) {
                filtered.push_back(c);
            }
        }

        printf("\n=== CORRECTNESS === %zu cases, reference is a double precision host pass\n",
               filtered.size());
        printf("gated on nmse and amax=max_abs/rms(ref); max_rel is printed for information only\n");
        printf("tolerances by dst type: f32 nmse %.0e amax %.0e | f16 nmse %.0e amax %.0e | bf16 nmse %.0e amax %.0e\n",
               nmse_tol(GGML_TYPE_F32),  amax_tol(GGML_TYPE_F32),
               nmse_tol(GGML_TYPE_F16),  amax_tol(GGML_TYPE_F16),
               nmse_tol(GGML_TYPE_BF16), amax_tol(GGML_TYPE_BF16));

        std::mt19937 rng(o.seed);

        for (size_t i = 0; i < filtered.size(); ++i) {
            const case_result r = run_correctness(filtered[i], rng, stream);

            if (!r.passed) {
                ++n_fail;
            }
            if (o.verbose || !r.passed) {
                print_case_line((int) i + 1, (int) filtered.size(), filtered[i], r);
            } else if ((i + 1) % 50 == 0 || i + 1 == filtered.size()) {
                printf("\r  %zu/%zu ...", i + 1, filtered.size());
                fflush(stdout);
            }
        }

        printf("\r  %zu cases: %zu passed, %d failed\n",
               filtered.size(), filtered.size() - (size_t) n_fail, n_fail);
    }

    if (o.do_perf) {
        const int combo_filter = perf_combo_filter(o);

        std::vector<shape4> shapes;
        if (o.have_shape) {
            shapes.push_back(o.shape);
        } else {
            for (const shape4 & s : PERF_SHAPES) {
                shapes.push_back(s);
            }
        }

        printf("\nperf: %d warmup, %d batches of %d launches, median reported\n",
               o.warmup, o.reps, o.iters);

        std::mt19937          rng(o.seed);
        std::vector<perf_row> rows;

        for (const shape4 & s : shapes) {
            for (int fuse = FUSE_NONE; fuse <= FUSE_MUL_ADD; ++fuse) {
                rows.push_back(run_perf_row(s, fuse, rng, stream, o.warmup, o.reps, o.iters, combo_filter));
            }
        }

        print_perf_tables(rows, combo_filter);

        if (!o.csv.empty()) {
            write_perf_csv(o.csv, rows, combo_filter);
        }
    }

    CU_CHECK(cudaStreamDestroy(stream));

    if (n_fail > 0) {
        printf("\nFAILED\n");
        return 1;
    }

    printf("\nOK\n");
    return 0;
}
