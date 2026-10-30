// NCP external FlashAttention-3 .so unit test. Two independent parts, neither of which needs a model.
//
// Part 1 -- llama_kv_cells::is_prefix_ordered() against a naive full rescan. The .so derives the causal mask from
// positions instead of reading the mask tensor, so it is only correct while the used cells are exactly [0, n) and
// their positions are non-decreasing over that range. llama-graph asks that predicate before it publishes kv_used,
// and the predicate is answered from a memo maintained BY HAND: every mutator that writes pos[i] has to pull
// pos_ordered_upto back to i. Nothing enforces that -- set_input_kv_used asks the same memo, so it is fooled along
// with everyone else -- and the symptom of a missing pullback is silently degraded output, not a crash. A
// differential test is the only thing that catches it. No GPU needed, so it runs first and unconditionally.
//
// Part 2 -- ggml_ncp_lib_flash_attn_fwd() against a CPU reference, straight through the wrapper in ncp-lib.cu the
// way test-llama-ncp.cpp does for MoE. No llama_context, no synthesised model, no graph: tensors are placed by hand
// in the packed [batch][seq][head][dim] layout the hook feeds the .so, the wrapper is called, and the result is
// compared against a plain O(seqlen_q * seqlen_k) softmax attention.
//
// What Part 2 is really for is seqused_k. ggml pads its KV cache view up so the graph shape stays reusable, so
// seqlen_k is an upper bound, and FA -- having no mask input -- reads the bottom-right causal offset off
// seqlen_k - seqlen_q. seqused_k is what carries the live count. Feeding K/V whose padding tail holds live-looking
// values (which is what a real cache tail holds) makes that visible: with seqused_k the answer matches the
// reference, and the negative control at the end drops it to check the answer goes WRONG. Both directions matter --
// a test that only checks "the number is small" cannot tell a working seqused_k from a padding tail that happened
// to be harmless.

// clang-format off
// IncludeBlocks: Regroup would merge the groups below and reattach the TODO to the wrong include
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-kv-cells.h"

#include "../ggml/src/ggml-cuda/ncp-lib.h"
// clang-format on

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ============================================================================================================
// part 1: is_prefix_ordered() memo vs full rescan
// ============================================================================================================

// The definition, with no memo of any kind. Reads the cells through the public accessors, so it cannot drift out
// of sync with the object under test. pos_get on a used cell is safe here: get_used() == n means [0, n) are all
// used, and the loop does not go past n.
static bool naive_prefix_ordered(const llama_kv_cells & cells) {
    const uint32_t n = cells.used_max_p1();

    if (cells.get_used() != n) {
        return false;
    }

    for (uint32_t i = 1; i < n; ++i) {
        if (cells.pos_get(i) < cells.pos_get(i - 1)) {
            return false;
        }
    }

    return true;
}

static int n_reported = 0;

// memo first: what we want under test is the answer it gives after whatever history preceded this call
static bool agree(const llama_kv_cells & cells, const char * what, uint32_t step) {
    const bool memo  = cells.is_prefix_ordered();
    const bool naive = naive_prefix_ordered(cells);

    if (memo == naive) {
        return true;
    }

    if (n_reported++ < 8) {
        printf("    MISMATCH step %-5u after %-8s memo=%-5s naive=%-5s used=%u used_max_p1=%u\n", step, what,
               memo ? "true" : "false", naive ? "true" : "false", cells.get_used(), cells.used_max_p1());
    }

    return false;
}

// The exact sequence a server-side context shift walks through, driven on the cells directly. Worth scripting
// separately from the fuzz below because the last phase is the only state in which the O(1) hole test passes
// while the layout is still wrong -- a fuzz reaches it only by luck.
static bool test_context_shift_timeline() {
    const uint32_t N    = 32;  // prompt length
    const uint32_t KEEP = 8;   // n_keep
    const uint32_t DISC = 8;   // n_discard

    llama_kv_cells cells;
    cells.resize(64);

    bool     ok   = true;
    uint32_t step = 0;

    // prefill: find_slot hands out the lowest free indices in order, so index order is position order
    for (uint32_t i = 0; i < N; ++i) {
        cells.pos_set(i, i);
        cells.seq_add(i, 0);
        ok = agree(cells, "pos_set", step++) && ok;
    }
    if (!cells.is_prefix_ordered()) {
        printf("    FAIL: a plain prefill is prefix-ordered, but it was reported as not\n");
        ok = false;
    }

    // seq_rm(seq, KEEP, KEEP + DISC) -- frees the cells that held those positions
    for (uint32_t i = KEEP; i < KEEP + DISC; ++i) {
        cells.seq_rm(i, 0);
        ok = agree(cells, "seq_rm", step++) && ok;
    }
    if (cells.is_prefix_ordered()) {
        printf("    FAIL: an open hole must not be reported as prefix-ordered\n");
        ok = false;
    }

    // seq_add(seq, KEEP + DISC, -1, -DISC) -- the surviving suffix slides down, the cells do not move
    for (uint32_t i = KEEP + DISC; i < N; ++i) {
        cells.pos_add(i, -(llama_pos) DISC);
        ok = agree(cells, "pos_add", step++) && ok;
    }

    // seq_rm pulled head back to the hole, so the next DISC tokens land there -- newest positions, lowest indices
    for (uint32_t k = 0; k < DISC; ++k) {
        cells.pos_set(KEEP + k, N - DISC + k);
        cells.seq_add(KEEP + k, 0);
        ok = agree(cells, "refill", step++) && ok;
    }

    // The hole is closed, so used == used_max_p1 and the O(1) test is satisfied again. The positions now read
    // 0..KEEP-1, then the newest DISC of them, then the shifted older ones: only the scan can still see this.
    if (cells.get_used() != cells.used_max_p1()) {
        printf("    FAIL: the refill should have closed the hole exactly (used=%u used_max_p1=%u)\n", cells.get_used(),
               cells.used_max_p1());
        ok = false;
    }
    if (cells.is_prefix_ordered()) {
        printf("    FAIL: a refilled hole leaves the positions out of order and must be reported\n");
        ok = false;
    }

    return ok;
}

// query_every > 1 lets the memo go stale across several mutations before it is read, which is what a real decode
// does -- querying after every single mutation is the easy case and would hide a missing pullback that a later
// mutation happens to cover up.
static bool test_fuzz(uint32_t seed, uint32_t n_steps, uint32_t query_every) {
    const uint32_t n_cells = 24;

    std::mt19937   gen(seed);
    llama_kv_cells cells;
    cells.resize(n_cells);

    bool ok = true;

    std::vector<uint32_t> idx_empty;
    std::vector<uint32_t> idx_used;

    for (uint32_t step = 0; step < n_steps; ++step) {
        idx_empty.clear();
        idx_used.clear();
        for (uint32_t i = 0; i < n_cells; ++i) {
            (cells.is_empty(i) ? idx_empty : idx_used).push_back(i);
        }

        const auto pick = [&](const std::vector<uint32_t> & v) {
            return v[std::uniform_int_distribution<size_t>(0, v.size() - 1)(gen)];
        };

        // writes are the only thing that can invalidate the memo, so they carry most of the weight
        const int    op   = std::uniform_int_distribution<int>(0, 9)(gen);
        const char * what = "none";

        if (op <= 4 && !idx_empty.empty()) {
            const uint32_t i = pick(idx_empty);
            cells.pos_set(i, std::uniform_int_distribution<llama_pos>(0, 63)(gen));
            cells.seq_add(i, 0);
            what = "pos_set";
        } else if (op <= 6 && !idx_used.empty()) {
            cells.seq_rm(pick(idx_used), 0);
            what = "seq_rm";
        } else if (op <= 7 && !idx_used.empty()) {
            // may drive pos negative, in which case pos_add drops the cell -- that path needs covering too
            cells.pos_add(pick(idx_used), std::uniform_int_distribution<llama_pos>(-8, 8)(gen));
            what = "pos_add";
        } else if (op <= 8 && !idx_used.empty()) {
            cells.pos_div(pick(idx_used), std::uniform_int_distribution<int>(2, 3)(gen));
            what = "pos_div";
        } else {
            cells.reset();
            what = "reset";
        }

        if (step % query_every == 0) {
            ok = agree(cells, what, step) && ok;
        }
    }

    return ok;
}

// ============================================================================================================
// part 2: ggml_ncp_lib_flash_attn_fwd() vs a CPU reference
// ============================================================================================================

// normalized mean squared error = mse(a, b) / mse(a, 0); rel_rms is its square root
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

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

// k/1024 with k in [-1024, 1023] is exactly representable in f16, so the f32 -> f16 the .so consumes is lossless
// on the inputs. What is left in the error budget is the accumulation order and the f16 output, not the operands.
static void fill_f16_exact(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_int_distribution<int> mant(-1024, 1023);
    for (auto & x : v) {
        x = (float) mant(rng) / 1024.0f;
    }
}

struct fa_case {
    const char * name;
    int          batch;
    int          seqlen_q;
    int          seqlen_k;  // the PADDED cell count, i.e. what ggml's KV view reports
    int          n_live;    // how much of it holds tokens; seqused_k[b] = this
    int          n_heads_q;
    int          n_heads_kv;
    int          head_dim;
    int          is_causal;
};

// head_dim is restricted to what fattn-ncp.cu lets through and the .so instantiates: 64 / 96 / 128 / 192 / 256.
// n_live < seqlen_k everywhere causal, because that gap IS the thing under test -- with n_live == seqlen_k the
// padded bound and the real one coincide and seqused_k cannot be shown to do anything.
// clang-format off
static const fa_case fa_cases[] = {
    // name              b   sq   sk  live  hq  hkv    d  causal
    { "decode",          1,   1, 256,  100,  8,   8, 128,      1 },
    { "decode-gqa",      1,   1, 256,  100,  8,   2, 128,      1 },
    { "decode-d64",      1,   1, 128,   37,  4,   4,  64,      1 },
    { "prefill",         1,  32, 256,   32,  8,   2, 128,      1 },
    { "prefill-append",  1,  16, 256,  120,  4,   4, 128,      1 },  // 104 of history + this ubatch's 16
    { "prefill-d256",    1,   8, 256,   72,  2,   2, 256,      1 },
    { "batch2",          2,   8, 128,   96,  4,   2, 128,      1 },  // live becomes 96 / 88, so seqused_k[1] matters
    { "noncausal",       1,   8,  64,   64,  4,   4, 128,      0 },
};
// clang-format on

// Bottom-right causal attention, computed the obvious way. live[b] is the real history length of that batch entry:
// query row i sees keys [0, live - seqlen_q + i], which is what the .so has to arrive at from seqused_k.
// clang-format off
static std::vector<float> cpu_reference(const fa_case & c, const std::vector<float> & Q, const std::vector<float> & K,
                                        const std::vector<float> & V, float scale, const std::vector<int32_t> & live) {
    // clang-format on
    const int    d    = c.head_dim;
    const int    hq   = c.n_heads_q;
    const int    hkv  = c.n_heads_kv;
    const int    gqa  = hq / hkv;
    const int    sq   = c.seqlen_q;
    const int    sk   = c.seqlen_k;
    const size_t q_bs = (size_t) sq * hq * d;  // packed [batch][seq][head][dim]
    const size_t k_bs = (size_t) sk * hkv * d;

    std::vector<float>  O((size_t) c.batch * sq * hq * d, 0.0f);
    std::vector<double> scores(sk);

    for (int b = 0; b < c.batch; ++b) {
        const int lv = live[b];

        for (int h = 0; h < hq; ++h) {
            const int h_kv = h / gqa;

            for (int i = 0; i < sq; ++i) {
                // how far this query row may reach. non-causal sees the whole live range.
                const int n_max = c.is_causal ? std::min(lv, lv - sq + i + 1) : lv;
                if (n_max <= 0) {
                    continue;  // a row with nothing to attend to; the kernel writes zeros
                }

                const float * q_row = Q.data() + (size_t) b * q_bs + ((size_t) i * hq + h) * d;

                double m = -INFINITY;
                for (int j = 0; j < n_max; ++j) {
                    const float * k_row = K.data() + (size_t) b * k_bs + ((size_t) j * hkv + h_kv) * d;
                    double        s     = 0.0;
                    for (int e = 0; e < d; ++e) {
                        s += (double) q_row[e] * (double) k_row[e];
                    }
                    scores[j] = s * (double) scale;
                    m         = std::max(m, scores[j]);
                }

                double sum = 0.0;
                for (int j = 0; j < n_max; ++j) {
                    scores[j] = std::exp(scores[j] - m);
                    sum += scores[j];
                }

                float * o_row = O.data() + (size_t) b * q_bs + ((size_t) i * hq + h) * d;
                for (int j = 0; j < n_max; ++j) {
                    const double  p     = scores[j] / sum;
                    const float * v_row = V.data() + (size_t) b * k_bs + ((size_t) j * hkv + h_kv) * d;
                    for (int e = 0; e < d; ++e) {
                        o_row[e] += (float) (p * (double) v_row[e]);
                    }
                }
            }
        }
    }

    return O;
}

enum case_result { CASE_PASS, CASE_FAIL, CASE_SKIP };

// with_seqused = false calls the wrapper exactly as above but hands it a null seqused_k, which is what the kernel
// saw before the parameter existed. Returned through so the negative control can reuse the whole body.
// clang-format off
static case_result run_fa_case(ggml_backend_t backend, const fa_case & c, bool with_seqused, double nmse_max,
                               bool expect_fail) {
    // clang-format on
    const int    d     = c.head_dim;
    const size_t n_q   = (size_t) c.batch * c.seqlen_q * c.n_heads_q * d;
    const size_t n_kv  = (size_t) c.batch * c.seqlen_k * c.n_heads_kv * d;
    const float  scale = 1.0f / std::sqrt((float) d);

    std::mt19937 rng(998244353 + (unsigned) (c.seqlen_q * 1000 + c.seqlen_k));

    // K/V are filled over the WHOLE padded range, tail included. That tail is not scratch in a real run: it holds
    // whatever the cache held before, so a kernel that reads past n_live reads plausible numbers rather than zeros
    // or NaNs. Zero-filling it would make the bug it is here to catch nearly invisible.
    std::vector<float> Q_f32(n_q), K_f32(n_kv), V_f32(n_kv);
    fill_f16_exact(Q_f32, rng);
    fill_f16_exact(K_f32, rng);
    fill_f16_exact(V_f32, rng);

    std::vector<ggml_fp16_t> Q_f16(n_q), K_f16(n_kv), V_f16(n_kv);
    for (size_t i = 0; i < n_q; i++) {
        Q_f16[i] = ggml_fp32_to_fp16(Q_f32[i]);
    }
    for (size_t i = 0; i < n_kv; i++) {
        K_f16[i] = ggml_fp32_to_fp16(K_f32[i]);
        V_f16[i] = ggml_fp32_to_fp16(V_f32[i]);
    }

    // seqused_k is a per-batch array, not a scalar -- that is why kv_used is I32[n_stream] and not an op_param. Give
    // each entry a different length so a kernel that reads only seqused_k[0] and applies it to every batch is a FAIL
    // here rather than a coincidence. Cases with batch == 1 are unaffected.
    std::vector<int32_t> seqused(c.batch);
    for (int b = 0; b < c.batch; ++b) {
        seqused[b] = std::max(1, c.n_live - b * 8);
    }

    ggml_init_params ip = {};
    ip.mem_size         = ggml_tensor_overhead() * 8;
    ip.no_alloc         = true;
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) {
        printf("[SKIP] fa/%-16s no ggml context\n", c.name);
        return CASE_SKIP;
    }

    // Flat 1d tensors: the .so is told the layout through the stride arguments below, not through ne, so there is
    // nothing to gain from shaping these -- and a 1d buffer cannot be accidentally permuted.
    ggml_tensor * Q_dev  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, (int64_t) n_q);
    ggml_tensor * K_dev  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, (int64_t) n_kv);
    ggml_tensor * V_dev  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, (int64_t) n_kv);
    ggml_tensor * O_dev  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, (int64_t) n_q);
    ggml_tensor * su_dev = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.batch);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        printf("[SKIP] fa/%-16s GPU alloc failed\n", c.name);
        return CASE_SKIP;
    }

    ggml_backend_tensor_set(Q_dev, Q_f16.data(), 0, Q_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(K_dev, K_f16.data(), 0, K_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(V_dev, V_f16.data(), 0, V_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(su_dev, seqused.data(), 0, seqused.size() * sizeof(int32_t));

    // Element strides for the packed [batch][seq][head][dim] layout, which is what fattn-ncp.cu ends up passing:
    // ggml's KV view stores every head of a cell together, and dst is physically [batch][seq][head][dv].
    const long long q_row  = (long long) c.n_heads_q * d;
    const long long q_head = d;
    const long long q_bat  = (long long) c.seqlen_q * c.n_heads_q * d;
    const long long k_row  = (long long) c.n_heads_kv * d;
    const long long k_head = d;
    const long long k_bat  = (long long) c.seqlen_k * c.n_heads_kv * d;

    // window (-1,-1) is full attention, (-1,0) is causal -- the same pair the hook derives.
    // clang-format off
    const int rc = ggml_ncp_lib_flash_attn_fwd(
        Q_dev->data, K_dev->data, V_dev->data, O_dev->data,
        c.batch, c.seqlen_q, c.seqlen_k, c.n_heads_q, c.n_heads_kv, d, d,
        q_bat, q_head, q_row,
        k_bat, k_head, k_row,
        k_bat, k_head, k_row,
        q_bat, q_head, q_row,
        scale, /*softcap=*/0.0f,
        c.is_causal, /*window_left=*/-1, /*window_right=*/c.is_causal ? 0 : -1,
        with_seqused ? (const int *) su_dev->data : nullptr,
        /*dtype=*/0, /*stream=*/nullptr);
    // clang-format on

    if (rc != 0) {
        printf("[SKIP] fa/%-16s wrapper returned %d -- no kernel for this shape\n", c.name, rc);
        return CASE_SKIP;
    }

    std::vector<ggml_fp16_t> O_f16(n_q);
    ggml_backend_tensor_get(O_dev, O_f16.data(), 0, O_f16.size() * sizeof(ggml_fp16_t));
    std::vector<float> so_out(n_q);
    ggml_fp16_to_fp32_row(O_f16.data(), so_out.data(), (int64_t) n_q);

    // The reference always uses the TRUE lengths, including in the negative control -- the question there is whether
    // dropping seqused_k moves the kernel away from the right answer, so the right answer is what it is measured
    // against either way.
    const std::vector<float> ref = cpu_reference(c, Q_f32, K_f32, V_f32, scale, seqused);

    const double err = nmse(ref, so_out);
    const float  mad = max_abs_diff(ref, so_out);
    // expect_fail inverts the verdict: the negative control has to move the answer, and "moved" is the pass.
    const bool   bad = expect_fail ? (err <= nmse_max) : !(err <= nmse_max);

    // clang-format off
    printf("[%s] fa/%-16s nmse=%.3e rel_rms=%.3e max_abs=%.3e  (b=%d sq=%d sk=%d live=%d hq=%d hkv=%d d=%d%s)\n",
           bad ? "FAIL" : "PASS", c.name, err, std::sqrt(err), mad,
           c.batch, c.seqlen_q, c.seqlen_k, c.n_live, c.n_heads_q, c.n_heads_kv, d,
           c.is_causal ? " causal" : "");
    // clang-format on

    return bad ? CASE_FAIL : CASE_PASS;
}

// ============================================================================================================
// main
// ============================================================================================================

int main(void) {
    int n_fail = 0;

    printf("--- KV cells: is_prefix_ordered() memo vs full rescan ---\n");
    {
        struct {
            const char * name;
            bool         ok;
            // clang-format off
        } cases[] = {
            { "context shift timeline", test_context_shift_timeline()  },
            { "fuzz, query every step", test_fuzz(1234,  4000, 1)      },
            { "fuzz, query every 7th",  test_fuzz(5678,  4000, 7)      },
            { "fuzz, query every 31st", test_fuzz(9012, 12000, 31)     },
        };

        // clang-format on

        for (const auto & c : cases) {
            printf("  %-24s %s\n", c.name, c.ok ? "OK" : "\033[1;31mFAIL\033[0m");
            n_fail += c.ok ? 0 : 1;
        }
    }

    printf("\n--- FA3 .so vs CPU reference ---\n");

    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) {
        printf("  [SKIP] no GPU device\n");
    } else if (!ggml_ncp_lib_fa_available()) {
        printf("  [SKIP] libncp_fa.so not loaded -- nothing to test. In build/bin?\n");
    } else {
        ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
        if (!gpu) {
            printf("  [SKIP] GPU init failed\n");
        } else {
            printf("  gpu: %s\n", ggml_backend_dev_description(gpu_dev));

            // f16 operands are exact (fill_f16_exact), so what is left is the accumulation order and the f16
            // output: rel_rms lands around 1e-3, i.e. nmse around 1e-6. 1e-5 leaves room for that without
            // leaving room for the failure this is looking for -- attending to a stale cell at full weight, or
            // letting a query reach past itself, moves the logits by orders of magnitude, not by a few ulps.
            const double nmse_max = 1e-5;

            int n_pass = 0, n_skip = 0;
            for (const auto & c : fa_cases) {
                // clang-format off
                switch (run_fa_case(gpu.get(), c, /*with_seqused=*/true, nmse_max, /*expect_fail=*/false)) {
                    case CASE_PASS: n_pass++; break;
                    case CASE_FAIL: n_fail++; break;
                    case CASE_SKIP: n_skip++; break;
                }
                // clang-format on
            }
            printf("  %d passed, %d skipped\n", n_pass, n_skip);

            // Negative control. Same call with seqused_k = null, which is what the kernel got before the parameter
            // existed: the causal offset becomes seqlen_k - seqlen_q, so every query row reaches past its own
            // position into the padding tail. Without this, a seqused_k that silently stopped being honoured --
            // dropped by the graph, misread across a stale .so ABI, ignored inside the kernel -- would leave every
            // case above still passing, because the reference and the kernel would simply agree on the wrong thing
            // only when the tail happens not to matter. The case picked has live=32 against sk=256, so there is no
            // way for the difference to be small.
            printf("\n  negative control -- seqused_k = null must break the answer:\n");
            {
                const fa_case & c = fa_cases[3];  // "prefill": live 32 of 256
                // clang-format off
                switch (run_fa_case(gpu.get(), c, /*with_seqused=*/false, /*nmse_max=*/1e-3, /*expect_fail=*/true)) {
                    case CASE_FAIL:
                        printf("    ^ this FAIL means dropping seqused_k did NOT change the answer, which means it\n"
                               "      is not being honoured -- every PASS above is passing for the wrong reason\n");
                        n_fail++;
                        break;
                    case CASE_SKIP: break;
                    case CASE_PASS: break;
                }
                // clang-format on
            }
        }
    }

    printf("\n%s (%d failures)\n", n_fail == 0 ? "ALL PASSED" : "FAILURES", n_fail);
    return n_fail == 0 ? 0 : 1;
}
