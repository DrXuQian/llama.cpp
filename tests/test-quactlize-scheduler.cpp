// Exercise the real backend and scheduler. Stub libraries answer inventory queries;
// no pack or GEMM entry is called, and weight contents are never read.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#define CHECK(expr) do { \
    if (!(expr)) { fprintf(stderr, "scheduler check failed: %s\n", #expr); exit(1); } \
} while (0)

static void run_case(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_buffer_type_t buft,
                     int qtype, int experts, int tokens) {
    ggml_init_params init = { 2 << 20, nullptr, true };
    ggml_context * weights = ggml_init(init);
    ggml_context * ctx = ggml_init(init);
    CHECK(weights && ctx);
    ggml_tensor * weight = ggml_new_tensor_3d(weights, (ggml_type) qtype, 512, 256, experts);
    ggml_set_name(weight, "blk.0.ffn_down_exps.weight");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
    CHECK(buffer && weight->data && weight->op == GGML_OP_NONE);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(ggml_backend_supports_op(gpu, weight));

    ggml_tensor * input = experts == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens)
                                     : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, tokens);
    ggml_tensor * out;
    if (experts == 1) {
        out = ggml_mul_mat(ctx, weight, input);
    } else {
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens);
        ggml_set_input(ids);
        out = ggml_mul_mat_id(ctx, weight, input, ids);
    }
    ggml_set_input(input);
    ggml_set_output(out);
    CHECK(ggml_backend_supports_op(gpu, out));

    ggml_tensor unsupported = *out;
    unsupported.op = GGML_OP_DUP;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));
    unsupported.op = GGML_OP_VIEW;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));
    unsupported = *out;
    unsupported.src[0] = input;
    unsupported.src[1] = weight;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);
    CHECK(sched);
    printf("KPACK_SCHEDULER_RESERVE q=%d experts=%d tokens=%d buffer=%s op=%s\n",
           qtype, experts, tokens, ggml_backend_buft_name(buft), ggml_op_name(weight->op));
    fflush(stdout);
    CHECK(ggml_backend_sched_reserve(sched, graph));
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    CHECK(ggml_backend_sched_get_tensor_backend(sched, weight) == gpu);
    CHECK(ggml_backend_sched_get_tensor_backend(sched, out) == gpu);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(weights);
}

int main(int argc, char ** argv) {
    const bool reserve_only = argc == 2 && strcmp(argv[1], "--reserve-only") == 0;
    CHECK(argc == 1 || reserve_only);
    if (ggml_backend_cuda_get_device_count() == 0) {
        fprintf(stderr, "KPACK_SCHEDULER SKIP: no CUDA/PPU device\n");
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(gpu && cpu);
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    auto extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    CHECK(extra_bufts);
    ggml_backend_buffer_type_t * extras = extra_bufts(dev);
    CHECK(extras && extras[0] && !extras[1]);
    ggml_backend_buffer_type_t kpack = extras[0];
    CHECK(kpack->device == dev);
    if (!reserve_only) {
        CHECK(ggml_backend_supports_buft(gpu, kpack));
        CHECK(!ggml_backend_supports_buft(cpu, kpack));
        CHECK(ggml_backend_supports_buft(gpu, ggml_backend_cuda_buffer_type(0)));
        CHECK(ggml_backend_supports_buft(gpu, ggml_backend_cuda_split_buffer_type(0, nullptr)));
        ggml_backend_buffer_type foreign = *kpack;
        foreign.device = ggml_backend_get_device(cpu);
        CHECK(!ggml_backend_supports_buft(gpu, &foreign));
        ggml_backend_buffer_type impostor = *kpack;
        impostor.iface.get_name = [](ggml_backend_buffer_type_t) { return "CUDA0_KPACK"; };
        CHECK(!ggml_backend_supports_buft(gpu, &impostor));
    }
    for (int qtype = 10; qtype <= 14; ++qtype) {
        for (int tokens : { 1, 16 }) {
            run_case(gpu, cpu, kpack, qtype, 1, tokens);
            run_case(gpu, cpu, kpack, qtype, 3, tokens);
        }
    }
    run_case(gpu, cpu, kpack, 12, 256, 4);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("KPACK_SCHEDULER PASS formats=5 cases=21 real_backend=1 real_scheduler=1 compute=NOT_RUN\n");
    return 0;
}
