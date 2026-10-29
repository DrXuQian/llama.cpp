#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

struct rank_graph {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * input;
    ggml_tensor * partial;
    ggml_tensor * output;
    std::array<ggml_cgraph, 2> steps;

    rank_graph(ggml_backend_t backend, int64_t ne) {
        ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
        GGML_ASSERT(ctx);
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
        partial = ggml_scale(ctx, input, 2.0f);
        output = ggml_scale(ctx, partial, 3.0f);
        ggml_cgraph * full = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(full, output);
        GGML_ASSERT(full->n_nodes == 2);
        steps[0] = ggml_graph_view(full, 0, 1);
        steps[1] = ggml_graph_view(full, 1, 2);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
    }

    ~rank_graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2 || getenv("GGML_CUDA_DISABLE_GRAPHS")) {
        fprintf(stderr, "SKIP: requires two CUDA/PPU devices with graphs enabled\n");
        return 77;
    }
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto free_comm = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto compute = (ggml_backend_comm_graph_compute_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_graph_compute");
    if (!init || !free_comm || !compute) {
        fprintf(stderr, "SKIP: requires NCCL/PCCL and graph support\n");
        return 77;
    }
#ifdef _WIN32
    _putenv_s("GGML_CUDA_TP_GRAPHS", "1");
    _putenv_s("GGML_CUDA_ALLREDUCE", "nccl");
#else
    setenv("GGML_CUDA_TP_GRAPHS", "1", 1);
    setenv("GGML_CUDA_ALLREDUCE", "nccl", 1);
#endif
    std::array<ggml_backend_t, 2> backends;
    for (size_t j = 0; j < backends.size(); ++j) {
        backends[j] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, j), nullptr);
        GGML_ASSERT(backends[j]);
    }
    void * comm = init(backends.data(), backends.size());
    GGML_ASSERT(comm);

    // Keep old buffers alive until their captured graphs have been invalidated.
    std::vector<std::unique_ptr<rank_graph>> owners;
    uint64_t uid = 1;
    // The larger case exercises BF16 workspace; repeating 64 uses different addresses.
    for (int64_t ne : {64, 32768, 64}) {
        std::array<rank_graph *, 2> ranks;
        std::array<ggml_cgraph *, 4> graphs;
        for (size_t j = 0; j < ranks.size(); ++j) {
            owners.emplace_back(std::make_unique<rank_graph>(backends[j], ne));
            ranks[j] = owners.back().get();
            graphs[j] = &ranks[j]->steps[0];
            graphs[2 + j] = &ranks[j]->steps[1];
        }
        std::vector<float> input(ne), result(ne);
        for (int iteration = 0; iteration < 12; ++iteration) {
            if (iteration % 4 == 0) {
                ++uid;
            }
            const bool inactive = iteration >= 4 && iteration < 8;
            if (inactive) {
                ranks[1]->partial->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            } else {
                ranks[1]->partial->flags |= GGML_TENSOR_FLAG_COMPUTE;
            }
            for (size_t j = 0; j < ranks.size(); ++j) {
                for (int64_t k = 0; k < ne; ++k) {
                    input[k] = float(1 + j + iteration + k % 7);
                }
                ggml_backend_tensor_set(ranks[j]->input, input.data(), 0, ne*sizeof(float));
            }
            GGML_ASSERT(compute(comm, graphs.data(), 2, uid));
            for (ggml_backend_t backend : backends) {
                ggml_backend_synchronize(backend);
            }
            for (size_t j = 0; j < ranks.size(); ++j) {
                ggml_backend_tensor_get(ranks[j]->output, result.data(), 0, ne*sizeof(float));
                for (int64_t k = 0; k < ne; ++k) {
                    const float x0 = float(1 + iteration + k % 7);
                    const float x1 = float(2 + iteration + k % 7);
                    const float expected = 6.0f*(x0 + (inactive ? 0.0f : x1));
                    if (!std::isfinite(result[k]) || std::fabs(result[k] - expected) > 1e-4f) {
                        fprintf(stderr, "rank=%zu ne=%lld iteration=%d index=%lld got=%g expected=%g\n",
                                j, (long long) ne, iteration, (long long) k, result[k], expected);
                        GGML_ABORT("TP graph result mismatch");
                    }
                }
            }
        }
    }
    // A graph without a stable UID must decline execution and invalidate cached replay.
    GGML_ASSERT(!compute(comm, nullptr, 0, 0));
    free_comm(comm);
    owners.clear();
    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    fprintf(stderr, "PASS: TP graph warmup, capture, replay, inactive shards, BF16 and invalidation\n");
    return 0;
}
