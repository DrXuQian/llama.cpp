// NCP external kernel .so unit test. Calls the wrapper function in ncp-lib.cu, which dlopens libncp_moe.so
// internally and returns -1 on failure. The output is compared against a CPU reference computed with the same
// f32 accumulation and bf16 output quantization, so results must be bit-identical (NMSE == 0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../ggml/src/ggml-cuda/ncp-lib.h"
#include "../ggml/src/ggml-cuda/mxfp4-ncp-layout.h"
#include "../ggml/src/ggml-cuda/mxfp4-ncp.cuh"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// Independent Sail permutation oracle; do not use code_index() to generate expected packed words.
static std::vector<uint8_t> mxfp4_oracle(const std::vector<uint8_t> & raw, int N, int K, int E) {
    std::vector<uint8_t> packed(raw.size());
    const int            interleave[] = {0, 2, 4, 6, 1, 3, 5, 7};
    const int            scale_swap[] = {0, 2, 1, 3};
    uint64_t             output       = 0;
    for (int e = 0; e < E; ++e) {
        for (int kt = 0; kt < K; kt += 16) {
            for (int nt = 0; nt < N; nt += 64) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int rows[] = {2 * (lane % 4), 2 * (lane % 4) + 1, 2 * (lane % 4) + 8, 2 * (lane % 4) + 9};
                    for (int tile = 0; tile < 4; ++tile) {
                        for (int s : interleave) {
                            const int      k = kt + rows[s % 4], n = nt + tile * 16 + lane / 4 + 8 * (s / 4);
                            const size_t   block = (size_t(e) * N + n) * (K / 32) + k / 32;
                            const unsigned code  = (raw[block * 17 + 1 + k % 16] >> (k % 32 >= 16 ? 4 : 0)) & 15;
                            packed[output / 2] |= code << (4 * (output % 2));
                            GGML_ASSERT(ggml_ncp_mxfp4::code_index(N, K, e, n, k) == output);
                            ++output;
                        }
                    }
                }
            }
        }
    }
    size_t scale_out = 0;
    for (int e = 0; e < E; ++e) {
        for (int kb = 0; kb < K / 32; ++kb) {
            for (int base = 0; base < N; base += 64) {
                for (int group = 0; group < 16; ++group) {
                    for (int sw : scale_swap) {
                        const int i = 4 * group + sw, n = base + i / 8 + (i % 8) * 8;
                        packed[output / 2 + scale_out] = raw[((size_t(e) * N + n) * (K / 32) + kb) * 17];
                        GGML_ASSERT(ggml_ncp_mxfp4::scale_index(N, K, e, n, kb) == scale_out++);
                    }
                }
            }
        }
    }
    return packed;
}

static std::vector<uint8_t> mxfp4_data(int N, int K, int E, bool arbitrary_scales) {
    std::mt19937         rng(314159 + N + K + E);
    std::vector<uint8_t> raw(size_t(E) * N * (K / 32) * 17);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = i % 17 ? rng() & 255 : arbitrary_scales ? (i / 17) & 255 : 121 + rng() % 7;
    }
    return raw;
}

static void test_mxfp4_layout() {
    for (int M : {1, 2, 8, 128, 5120}) {
        ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context_ptr ctx(ggml_init(ip));
        auto *           logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, M);
        auto *           ids    = ggml_argsort_top_k(ctx.get(), logits, 6);
        GGML_ASSERT(ids->view_src && ids->nb[1] == 256 * sizeof(int32_t));
        GGML_ASSERT(ggml_is_contiguous(ids) == (M == 1));
        GGML_ASSERT(ggml_ncp_mxfp4_ids_layout_supported(ids));
        ggml_tensor bad = *ids;
        bad.nb[0]       = 2 * sizeof(int32_t);
        GGML_ASSERT(!ggml_ncp_mxfp4_ids_layout_supported(&bad));
        bad = *ids;
        ++bad.nb[1];
        GGML_ASSERT(!ggml_ncp_mxfp4_ids_layout_supported(&bad));
        bad.nb[1] = (size_t(INT_MAX) + 1) * sizeof(int32_t);
        GGML_ASSERT(!ggml_ncp_mxfp4_ids_layout_supported(&bad));
        printf("NCP_MXFP4_TOPK_LAYOUT PASS M=%d ids_stride=256 legacy_contiguous=%d\n", M, ggml_is_contiguous(ids));
    }
    for (int N : {64, 128, 192}) {
        for (int K : {64, 128, 256}) {
            const int             E        = 3;
            const auto            raw      = mxfp4_data(N, K, E, true);
            auto                  expected = mxfp4_oracle(raw, N, K, E);
            // Align word access independently of the byte-vector allocator.
            std::vector<uint32_t> words((expected.size() + 3) / 4);
            memcpy(words.data(), expected.data(), expected.size());
            for (size_t i = 0; i < raw.size(); ++i) {
                GGML_ASSERT(ggml_ncp_mxfp4::raw_byte(reinterpret_cast<const uint8_t *>(words.data()), N, K, E, i) ==
                            raw[i]);
            }
            for (int e = 0; e < E; ++e) {
                for (int n = 0; n < N; n += 64) {
                    const auto * tile = raw.data() + (size_t(e) * N + n) * (K / 32) * 17;
                    for (int j = 0; j < 64 * K / 8; ++j) {
                        const size_t dst = (size_t(e) * K / 16 + j / 128) * 2 * N + 2 * n + j % 128;
                        GGML_ASSERT(ggml_ncp_mxfp4::tile_word(tile, K, j) == words[dst]);
                    }
                }
            }
            words[7] ^= 1;
            bool rejected = false;
            for (size_t i = 0; i < raw.size(); ++i) {
                rejected |=
                    ggml_ncp_mxfp4::raw_byte(reinterpret_cast<const uint8_t *>(words.data()), N, K, E, i) != raw[i];
            }
            GGML_ASSERT(rejected);
        }
    }
    GGML_ASSERT(!ggml_ncp_mxfp4::shape(63, 64, 8));
    GGML_ASSERT(!ggml_ncp_mxfp4::shape(64, 96, 8));
    GGML_ASSERT(!ggml_ncp_mxfp4::shape(64, 64, 1025));
    GGML_ASSERT(!ggml_ncp_mxfp4::shape(65536, 65536, 1));
    printf("NCP_MXFP4_LAYOUT PASS shapes=9 bit_exact=1 planted_fault=DETECTED\n");
}

static void test_mxfp4_shards(ggml_backend_dev_t dev) {
    // Exercise Meta's real split/upload path on one physical GPU; no collectives are needed for loading.
    ggml_backend_dev_t                   devices[] = {dev, dev};
    static ggml_backend_meta_split_state states[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context_ptr ctx(ggml_init(ip));
        auto *           w     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_MXFP4, 256, 256, 8);
        auto &           split = states[axis];
        split.axis             = static_cast<ggml_backend_meta_split_axis>(axis);
        split.ne[0] = split.ne[1] = w->ne[axis] / 2;
        split.nr[0] = split.n_segments = 1;
        auto * meta                    = ggml_backend_meta_device(
            devices, 2,
            [](const ggml_tensor *, void * data) { return *static_cast<ggml_backend_meta_split_state *>(data); },
            &split);
        auto * buft = ggml_backend_meta_device_get_weight_buft(meta, w, GGML_OP_MUL_MAT_ID);
        GGML_ASSERT(buft && ggml_backend_meta_buffer_type_count(buft) == 2);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        GGML_ASSERT(buffer);
        const auto raw = mxfp4_data(256, 256, 8, true);
        ggml_backend_tensor_set(w, raw.data(), 0, raw.size());
        for (int rank = 0; rank < 2; ++rank) {
            auto * local = ggml_backend_meta_tensor_shard(w, rank, nullptr);
            GGML_ASSERT(local);
            std::vector<uint8_t> got(ggml_nbytes(local));
            ggml_backend_tensor_get(local, got.data(), 0, got.size());
            size_t offset = 0;
            for (int e = 0; e < local->ne[2]; ++e) {
                for (int n = 0; n < local->ne[1]; ++n) {
                    const int    ge     = e + (axis == 2 ? rank * local->ne[2] : 0);
                    const int    gn     = n + (axis == 1 ? rank * local->ne[1] : 0);
                    const int    kb     = axis == 0 ? rank * local->ne[0] / 32 : 0;
                    const auto * source = raw.data() + ((size_t(ge) * 256 + gn) * 8 + kb) * 17;
                    GGML_ASSERT(memcmp(got.data() + offset, source, local->nb[1]) == 0);
                    offset += local->nb[1];
                }
            }
        }
        printf("NCP_MXFP4_SHARD PASS axis=%d ranks=2 physical_devices=1\n", axis);
    }
}

static void test_mxfp4_case(ggml_backend_t backend, int M, int E, int channels, bool topk_view = false) {
    constexpr int    N = 128, K = 256, topk = 6;
    ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr weights(ggml_init(ip)), ctx(ggml_init(ip));
    GGML_ASSERT(weights && ctx);
    auto * w = ggml_new_tensor_3d(weights.get(), GGML_TYPE_MXFP4, K, N, E);
    ggml_set_name(w, "ncp-mxfp4-test");
    const auto dev = ggml_backend_get_device(backend);
    const auto reg = ggml_backend_dev_backend_reg(dev);
    const auto preferred =
        (ggml_backend_dev_get_weight_buft_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_weight_buft");
    GGML_ASSERT(preferred);
    auto buft = preferred(dev, w, GGML_OP_MUL_MAT_ID);
    GGML_ASSERT(buft && "libncp_moe.so must export W4FA16 and release_workspace v1");
    GGML_ASSERT(!preferred(dev, w, GGML_OP_MUL_MAT));
    ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
    GGML_ASSERT(wb);
    ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto raw = mxfp4_data(N, K, E, false);
    ggml_backend_tensor_set(w, raw.data(), 0, raw.size());
    // Partial byte updates cross metadata, block and word boundaries.
    raw[15] ^= 0x91;
    ggml_backend_tensor_set(w, raw.data() + 15, 15, 19);
    std::vector<uint8_t> back(raw.size());
    ggml_backend_tensor_get(w, back.data(), 0, back.size());
    GGML_ASSERT(back == raw);
    // A K-shard-style upload with gaps in the source, contiguous destination rows.
    const size_t         row = K / 32 * 17;
    std::vector<uint8_t> strided(2 * raw.size(), 0x95);
    for (int r = 0; r < N * E; ++r) {
        memcpy(strided.data() + r * 2 * row, raw.data() + r * row, row);
    }
    ggml_backend_tensor_set_2d(w, strided.data(), 0, row, N * E, row, 2 * row);
    ggml_backend_tensor_get(w, back.data(), 0, back.size());
    GGML_ASSERT(back == raw);

    auto * a      = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, channels, M);
    auto * logits = topk_view ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, E, M) : nullptr;
    auto * ids =
        topk_view ? ggml_argsort_top_k(ctx.get(), logits, topk) : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, topk, M);
    auto * output = ggml_mul_mat_id(ctx.get(), w, a, ids);
    auto * graph  = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buf && ggml_backend_supports_op(backend, output));
    auto * bad = ggml_view_3d(ctx.get(), w, K, N / 2, E, w->nb[1], w->nb[2], 0);
    GGML_ASSERT(!ggml_backend_supports_op(backend, bad));
    auto * alias = ggml_view_tensor(ctx.get(), w);
    GGML_ASSERT(ggml_backend_supports_op(backend, alias));
    const float fp4[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<float>   input(size_t(M) * channels * K, 0), got(size_t(M) * topk * N);
        std::vector<int32_t> routes(size_t(M) * topk);
        for (int t = 0; t < M; ++t) {
            for (int c = 0; c < channels; ++c) {
                if (M == 9) {
                    for (int k = 0; k < K; ++k) {
                        input[(size_t(t) * channels + c) * K + k] = ((k + t + c + replay) % 17 - 8) / 8.0f;
                    }
                } else {
                    input[(size_t(t) * channels + c) * K + (t * 7 + c * 13 + replay) % K] = 1;
                }
            }
            for (int j = 0; j < topk; ++j) {
                routes[t * topk + j] = replay == 1 ? j : (t * 11 + j + replay) % E;
            }
        }
        ggml_backend_tensor_set_async(backend, a, input.data(), 0, input.size() * sizeof(float));
        std::vector<float> scores;
        if (topk_view) {
            scores.resize(size_t(M) * E);
            for (int t = 0; t < M; ++t) {
                for (int e = 0; e < E; ++e) {
                    scores[size_t(t) * E + e] = -e;
                }
                for (int j = 0; j < topk; ++j) {
                    scores[size_t(t) * E + routes[t * topk + j]] = 2 * E - j;
                }
            }
            ggml_backend_tensor_set_async(backend, logits, scores.data(), 0, scores.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set_async(backend, ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(output, got.data(), 0, got.size() * sizeof(float));
        for (int t = 0; t < M; ++t) {
            for (int j = 0; j < topk; ++j) {
                const int k = (t * 7 + (j % channels) * 13 + replay) % K, e = routes[t * topk + j];
                for (int n = 0; n < N; ++n) {
                    const size_t b        = (size_t(e) * N + n) * (K / 32) + k / 32;
                    const int    code     = (raw[b * 17 + 1 + k % 16] >> (k % 32 >= 16 ? 4 : 0)) & 15;
                    float        expected = std::ldexp(fp4[code], int(raw[b * 17]) - 127);
                    if (M == 9) {
                        float sum = 0;
                        for (int col = 0; col < K; ++col) {
                            const size_t block = (size_t(e) * N + n) * (K / 32) + col / 32;
                            const int    value = (raw[block * 17 + 1 + col % 16] >> (col % 32 >= 16 ? 4 : 0)) & 15;
                            sum += input[(size_t(t) * channels + j % channels) * K + col] *
                                   std::ldexp(fp4[value], int(raw[block * 17]) - 127);
                        }
                        expected = ggml_bf16_to_fp32(ggml_fp32_to_bf16(sum));
                    }
                    const float actual = got[(size_t(t) * topk + j) * N + n];
                    if (actual != expected) {
                        GGML_ABORT(
                            "W4FA16 mismatch M=%d E=%d channels=%d replay=%d t=%d j=%d n=%d actual=%g expected=%g", M,
                            E, channels, replay, t, j, n, actual, expected);
                    }
                }
            }
        }
    }
    printf("NCP_MXFP4 PASS M=%d E=%d topk=%d channels=%d topk_view=%d replays=3 buffer=%s\n", M, E, topk, channels,
           topk_view, ggml_backend_buft_name(buft));
    fflush(stdout);
}

// ---- main ----

int main(int argc, char ** argv) {
    const bool layout = argc == 2 && strcmp(argv[1], "--mxfp4-layout") == 0;
    const bool mxfp4  = argc == 2 && strcmp(argv[1], "--mxfp4") == 0;
    if (argc != 1 && !layout && !mxfp4) {
        fprintf(stderr, "usage: %s [--mxfp4-layout|--mxfp4]\n", argv[0]);
        return 1;
    }
    if (layout || mxfp4) {
        test_mxfp4_layout();
        if (layout) {
            return 0;
        }
    }
    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) {
        printf("[SKIP] no GPU device\n");
        return mxfp4 ? 1 : 0;
    }
    ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
    if (!gpu) {
        printf("[SKIP] GPU init failed\n");
        return mxfp4 ? 1 : 0;
    }
    printf("gpu: %s\n\n", ggml_backend_dev_description(gpu_dev));

    if (mxfp4) {
        test_mxfp4_shards(gpu_dev);
        for (int M : {1, 2, 3, 4, 5, 6, 7, 8, 9, 128, 512, 1024, 2048, 4096, 5120, 32769}) {
            test_mxfp4_case(gpu.get(), M, 8, 1);
            test_mxfp4_case(gpu.get(), M, 8, 6);
        }
        for (int M : {1, 8, 128, 5120}) {
            test_mxfp4_case(gpu.get(), M, 256, 1, true);
            test_mxfp4_case(gpu.get(), M, 256, 6, true);
        }
        ggml_backend_ptr second(ggml_backend_dev_init(gpu_dev, nullptr));
        GGML_ASSERT(second);
        test_mxfp4_case(second.get(), 1, 256, 1);
        test_mxfp4_case(second.get(), 32, 256, 6);
        second.reset();
        test_mxfp4_case(gpu.get(), 1, 8, 1);
        printf("NCP_MXFP4 DONE cases=43 graph_capture=UNSUPPORTED_BY_PROVIDER\n");
        return 0;
    }

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
