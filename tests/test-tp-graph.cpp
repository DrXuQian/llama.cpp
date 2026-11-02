#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-cuda/tp-graph-launch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

struct launch_probe {
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::vector<std::thread::id> threads;
    std::vector<int> seen;
    std::vector<int> submitted;
    size_t initialized = 0, entered = 0;
    int iteration = 0;
    int collective = 0;
    bool segment = false;

    explicit launch_probe(size_t ranks) : threads(ranks), seen(ranks, -1), submitted(ranks, 0) {}

    static void initialize(void * data, size_t rank) {
        auto & probe = *static_cast<launch_probe *>(data);
        std::lock_guard<std::mutex> lock(probe.mutex);
        probe.threads[rank] = std::this_thread::get_id();
        ++probe.initialized;
    }

    static void submit(void * data, size_t rank) {
        auto & probe = *static_cast<launch_probe *>(data);
        std::unique_lock<std::mutex> lock(probe.mutex);
        GGML_ASSERT(probe.threads[rank] == std::this_thread::get_id());
        GGML_ASSERT(probe.seen[rank] != probe.iteration);
        probe.seen[rank] = probe.iteration;
        if (++probe.entered == probe.threads.size()) {
            probe.entered_cv.notify_all();
        }
        // A submission that waits for its peers must not block their submissions.
        GGML_ASSERT(probe.entered_cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return probe.entered == probe.threads.size();
        }));
        if (probe.segment) {
            GGML_ASSERT(probe.submitted[rank] == probe.collective);
            ++probe.submitted[rank];
        }
    }
};

static void test_host_launcher() {
    for (size_t ranks : {size_t(2), size_t(4)}) {
        launch_probe probe(ranks);
        {
            ggml_cuda_tp_graph_launcher launcher(ranks, &probe, launch_probe::initialize, launch_probe::submit);
            GGML_ASSERT(probe.initialized == ranks);
            for (size_t j = 0; j < ranks; ++j) {
                GGML_ASSERT(probe.threads[j] != std::this_thread::get_id());
                for (size_t k = 0; k < j; ++k) {
                    GGML_ASSERT(probe.threads[j] != probe.threads[k]);
                }
            }
            for (int iteration = 0; iteration < 256; ++iteration) {
                probe.entered = 0;
                probe.iteration = iteration;
                // Reuse the workers for graph replay and dependent prefill segments.
                probe.segment = iteration % 4 != 0;
                launcher.launch();
                GGML_ASSERT(probe.entered == ranks && probe.initialized == ranks);
                for (int seen : probe.seen) {
                    GGML_ASSERT(seen == iteration);
                }
                if (probe.segment) {
                    ++probe.collective;
                }
                for (int submitted : probe.submitted) {
                    GGML_ASSERT(submitted == probe.collective);
                }
            }
        }
    }
    fprintf(stderr, "PASS: persistent rank threads, concurrent submissions and segment/collective ordering\n");
}

struct rank_graph {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * input;
    ggml_tensor * partial;
    ggml_tensor * output;
    std::array<ggml_cgraph, 2> steps;

    rank_graph(ggml_backend_t backend, int64_t ne, bool moe = false) {
        ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
        GGML_ASSERT(ctx);
        ggml_tensor * weight = nullptr;
        ggml_tensor * ids = nullptr;
        if (moe) {
            input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, ne);
            weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ne);
            partial = ggml_mul_mat_id(ctx, weight, input, ids);
        } else {
            input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            partial = ggml_scale(ctx, input, 2.0f);
        }
        output = ggml_scale(ctx, partial, 3.0f);
        ggml_cgraph * full = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(full, output);
        GGML_ASSERT(full->n_nodes == 2);
        steps[0] = ggml_graph_view(full, 0, 1);
        steps[1] = ggml_graph_view(full, 1, 2);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        if (moe) {
            const float scale = 2.0f;
            const std::vector<int32_t> experts(ne, 0);
            ggml_backend_tensor_set(weight, &scale, 0, sizeof(scale));
            ggml_backend_tensor_set(ids, experts.data(), 0, ne*sizeof(int32_t));
        }
    }

    ~rank_graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

int main(int argc, char ** argv) {
    test_host_launcher();
    if (argc == 2 && strcmp(argv[1], "--host-launch") == 0) {
        return 0;
    }
    const bool eager = argc == 2 && strcmp(argv[1], "--eager") == 0;
    const bool no_capture = argc == 2 && strcmp(argv[1], "--no-capture") == 0;
    if (argc > 1 && !eager && !no_capture) {
        fprintf(stderr, "usage: %s [--host-launch|--eager|--no-capture]\n", argv[0]);
        return 1;
    }
    auto set_env = [](const char * name, const char * value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    };
    set_env("GGML_CUDA_TP_GRAPHS", eager ? "0" : "1");
    set_env("GGML_CUDA_TP_PARALLEL", "1");
    if (no_capture) {
        set_env("GGML_CUDA_DISABLE_GRAPHS", "1");
    }
    set_env("GGML_CUDA_ALLREDUCE", "nccl");
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2) {
        fprintf(stderr, "SKIP: requires two CUDA/PPU devices\n");
        return 77;
    }
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto free_comm = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto compute = (ggml_backend_comm_graph_compute_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_graph_compute");
    if (!init || !free_comm || !compute) {
        fprintf(stderr, "SKIP: requires NCCL/PCCL and the TP execution hook\n");
        return 77;
    }
    const size_t n_ranks = std::min(size_t(4), ggml_backend_reg_dev_count(reg));
    std::vector<ggml_backend_t> backends(n_ranks);
    for (size_t j = 0; j < backends.size(); ++j) {
        backends[j] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, j), nullptr);
        GGML_ASSERT(backends[j]);
    }
    void * comm = init(backends.data(), backends.size());
    GGML_ASSERT(comm);

    // Keep old buffers alive until their captured graphs have been invalidated.
    std::vector<std::unique_ptr<rank_graph>> owners;
    uint64_t uid = 1;
    // Switch from capture to an unsupported MoE graph and back on the same workers.
    const int64_t bf16_ne = n_ranks == 2 ? 32768 : n_ranks == 3 ? 131072 : 262144;
    for (const auto & test_case : {std::make_pair(int64_t(64), false), std::make_pair(bf16_ne, false),
                                   std::make_pair(int64_t(129), true), std::make_pair(int64_t(64), false)}) {
        const int64_t ne = test_case.first;
        std::vector<rank_graph *> ranks(n_ranks);
        std::vector<ggml_cgraph *> graphs(2*n_ranks);
        for (size_t j = 0; j < ranks.size(); ++j) {
            owners.emplace_back(std::make_unique<rank_graph>(backends[j], ne, test_case.second));
            ranks[j] = owners.back().get();
            graphs[j] = &ranks[j]->steps[0];
            graphs[n_ranks + j] = &ranks[j]->steps[1];
        }
        std::vector<float> input(ne), result(ne);
        for (int iteration = 0; iteration < 12; ++iteration) {
            if (iteration % 4 == 0) {
                ++uid;
            }
            const bool inactive = iteration >= 4 && iteration < 8;
            if (inactive) {
                ranks.back()->partial->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            } else {
                ranks.back()->partial->flags |= GGML_TENSOR_FLAG_COMPUTE;
            }
            for (size_t j = 0; j < ranks.size(); ++j) {
                for (int64_t k = 0; k < ne; ++k) {
                    input[k] = float(1 + j + iteration + k % 7);
                }
                ggml_backend_tensor_set(ranks[j]->input, input.data(), 0, ne*sizeof(float));
            }
            GGML_ASSERT(compute(comm, graphs.data(), 2, eager && iteration == 0 ? 0 : uid));
            for (ggml_backend_t backend : backends) {
                ggml_backend_synchronize(backend);
            }
            for (size_t j = 0; j < ranks.size(); ++j) {
                ggml_backend_tensor_get(ranks[j]->output, result.data(), 0, ne*sizeof(float));
                for (int64_t k = 0; k < ne; ++k) {
                    float expected = 0.0f;
                    for (size_t rank = 0; rank < n_ranks; ++rank) {
                        if (!inactive || rank + 1 != n_ranks) {
                            expected += 6.0f * float(1 + rank + iteration + k % 7);
                        }
                    }
                    if (!std::isfinite(result[k]) || std::fabs(result[k] - expected) > 1e-4f) {
                        fprintf(stderr, "rank=%zu ne=%lld iteration=%d index=%lld got=%g expected=%g\n",
                                j, (long long) ne, iteration, (long long) k, result[k], expected);
                        GGML_ABORT("TP graph result mismatch");
                    }
                }
            }
        }
    }
    // Invalid input must decline execution and invalidate cached replay.
    GGML_ASSERT(!compute(comm, nullptr, 0, 0));
    free_comm(comm);
    owners.clear();
    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    fprintf(stderr, "PASS: TP warmup, graph replay, eager MoE, inactive shards, BF16 and invalidation\n");
    return 0;
}
