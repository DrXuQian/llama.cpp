// NCP external kernel .so unit test. Calls the wrapper function in ncp-lib.cu, which dlopens libncp_moe.so
// internally and returns -1 on failure. The output is compared against a CPU reference computed with the same
// f32 accumulation and bf16 output quantization, so results must be bit-identical (NMSE == 0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../ggml/src/ggml-cuda/ncp-lib.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ---- metrics ----

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0, mse_a_0 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        mse_a_b += d * d;
        mse_a_0 += (double) a[i] * (double) a[i];
    }
    return mse_a_0 > 0.0 ? mse_a_b / mse_a_0 : mse_a_b;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// k/128 with k in [-128,127] is exactly representable in bf16, so f32->bf16 is lossless.
static void fill_bf16_exact(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_int_distribution<int> mant(-128, 127);
    for (auto & x : v) {
        x = (float) mant(rng) / 128.0f;
    }
}

// ---- MoE ----

struct moe_case {
    const char * name;
    int total_rows;
    int n_experts;
    int K;
    int N;
};

static const moe_case moe_cases[] = {
    { "decode-batch",   64,  8, 256, 512 },
    { "prefill-small", 128, 16, 256, 512 },
    { "prefill",      4096,  8, 512, 512 },
};

enum case_result { CASE_PASS, CASE_FAIL, CASE_SKIP };

static case_result run_moe_case(ggml_backend_t backend, const moe_case & c) {
    const int rows_per_expert = c.total_rows / c.n_experts;
    const int expected_m      = rows_per_expert;

    std::mt19937 rng(1234567 + (unsigned) c.total_rows);

    // B: [n_experts, N, K] bf16, A: [total_rows, K] bf16 -- both lossless from f32.
    std::vector<float> B_f32((size_t) c.n_experts * c.N * c.K);
    std::vector<float> A_f32((size_t) c.total_rows * c.K);
    fill_bf16_exact(B_f32, rng);
    fill_bf16_exact(A_f32, rng);
    std::vector<ggml_bf16_t> B_bf16(B_f32.size()), A_bf16(A_f32.size());
    for (size_t i = 0; i < B_f32.size(); i++) B_bf16[i] = ggml_fp32_to_bf16(B_f32[i]);
    for (size_t i = 0; i < A_f32.size(); i++) A_bf16[i] = ggml_fp32_to_bf16(A_f32[i]);

    // Contiguous routing: rows 0..rows_per_expert-1 -> expert 0, etc.
    std::vector<int32_t> m_indices(c.total_rows), m_rows(c.n_experts, rows_per_expert);
    for (int r = 0; r < c.total_rows; r++) m_indices[r] = r / rows_per_expert;

    // Allocate on GPU.
    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc   = true;
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) { printf("[SKIP] moe/%-14s no ggml context\n", c.name); return CASE_SKIP; }

    ggml_tensor * B_dev  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, c.K, c.N, c.n_experts);
    ggml_tensor * A_dev  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, c.K, c.total_rows);
    ggml_tensor * out_dev = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, c.N, c.total_rows);
    ggml_tensor * mi_dev = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.total_rows);
    ggml_tensor * mr_dev = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.n_experts);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) { printf("[SKIP] moe/%-14s GPU alloc failed\n", c.name); return CASE_SKIP; }

    ggml_backend_tensor_set(B_dev,  B_bf16.data(),  0, B_bf16.size()  * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(A_dev,  A_bf16.data(),  0, A_bf16.size()  * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(mi_dev, m_indices.data(), 0, m_indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mr_dev, m_rows.data(),    0, m_rows.size()    * sizeof(int32_t));

    // Call the wrapper. It dlopens libncp_moe.so internally; rc=-1 means load failed.
    // stream=nullptr: default stream; ggml_backend_tensor_get below is synchronous (cudaMemcpy D2H).
    const int rc = ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        A_dev->data, B_dev->data, out_dev->data,
        (const int *) mi_dev->data, (const int *) mr_dev->data,
        c.total_rows, c.N, c.K, c.n_experts, expected_m, /*stream=*/nullptr);

    if (rc != 0) {
        printf("[FAIL] moe/%-14s wrapper returned %d -- libncp_moe.so not loaded. In build/bin?\n", c.name, rc);
        return CASE_FAIL;
    }

    // Read back .so result (bf16 -> f32).
    std::vector<ggml_bf16_t> out_bf16((size_t) c.total_rows * c.N);
    ggml_backend_tensor_get(out_dev, out_bf16.data(), 0, out_bf16.size() * sizeof(ggml_bf16_t));
    std::vector<float> so_out(out_bf16.size());
    ggml_bf16_to_fp32_row(out_bf16.data(), so_out.data(), (int64_t) out_bf16.size());

    // CPU reference: f32 accumulation, output quantized to bf16 to match the .so.
    std::vector<float> ref_out((size_t) c.total_rows * c.N, 0.0f);
    for (int r = 0; r < c.total_rows; r++) {
        const int e = m_indices[r];
        const float * A_row = A_f32.data() + (size_t) r * c.K;
        const ggml_bf16_t * B_e = B_bf16.data() + (size_t) e * c.N * c.K;
        float * out_row = ref_out.data() + (size_t) r * c.N;
        for (int n = 0; n < c.N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < c.K; k++) {
                sum += ggml_bf16_to_fp32(B_e[n * c.K + k]) * A_row[k];
            }
            out_row[n] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(sum));
        }
    }

    const double err = nmse(ref_out, so_out);
    const float  mad = max_abs_diff(ref_out, so_out);
    const bool   bad = !(err == 0);

    printf("[%s] moe/%-14s nmse=%.3e max_abs=%.3e  (rows=%d E=%d K=%d N=%d)\n",
           bad ? "FAIL" : "PASS", c.name, err, mad,
           c.total_rows, c.n_experts, c.K, c.N);
    return bad ? CASE_FAIL : CASE_PASS;
}

// ---- main ----

int main(void) {
    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { printf("[SKIP] no GPU device\n"); return 0; }
    ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
    if (!gpu) { printf("[SKIP] GPU init failed\n"); return 0; }
    printf("gpu: %s\n\n", ggml_backend_dev_description(gpu_dev));

    printf("--- MoE (DeepGemm GroupedNoPad, BF16) ---\n");
    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (const auto & c : moe_cases) {
        switch (run_moe_case(gpu.get(), c)) {
            case CASE_PASS: n_pass++; break;
            case CASE_FAIL: n_fail++; break;
            case CASE_SKIP: n_skip++; break;
        }
    }
    printf("  moe: %d passed, %d failed, %d skipped\n\n", n_pass, n_fail, n_skip);

    printf("%s (%d failures)\n", n_fail == 0 ? "ALL PASSED" : "FAILURES", n_fail);
    return n_fail == 0 ? 0 : 1;
}
