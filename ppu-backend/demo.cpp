#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct model_graph {
    ggml_context * ctx;
    ggml_tensor * input;
    ggml_tensor * norm;
    ggml_tensor * output;
    ggml_cgraph * graph;

    model_graph(int64_t columns, int64_t rows) {
        ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true});
        GGML_ASSERT(ctx);
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        ggml_set_input(input);
        norm = ggml_rms_norm(ctx, input, 1e-5f);
        ggml_set_name(norm, "attn_norm-0");
        output = ggml_scale(ctx, norm, 0.5f);
        ggml_set_name(output, "cpu_scale");
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, output);
    }
    ~model_graph() { ggml_free(ctx); }
};

static void run_case(ggml_backend_t plugin, ggml_backend_t cpu, int64_t columns, int64_t rows) {
    model_graph actual(columns, rows), reference(columns, rows);
    GGML_ASSERT(ggml_backend_supports_op(plugin, actual.norm));
    GGML_ASSERT(!ggml_backend_supports_op(plugin, actual.output));

    ggml_backend_t backends[] = {plugin, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 32, false, true);
    GGML_ASSERT(sched);
    ggml_backend_sched_set_tensor_backend(sched, actual.input, cpu);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, actual.graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, actual.norm) == plugin);
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, actual.output) == cpu);
    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(reference.ctx, cpu);
    GGML_ASSERT(ref_buffer);

    const size_t count = size_t(columns*rows), bytes = count*sizeof(float);
    std::vector<float> input(count), result(count), expected(count);
    float max_error = 0;
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (size_t i = 0; i < count; ++i) {
            input[i] = iteration == 2 ? 0.0f : float(int(i % 67) - 33 + iteration)*0.125f;
        }
        ggml_backend_tensor_set(actual.input, input.data(), 0, bytes);
        ggml_backend_tensor_set(reference.input, input.data(), 0, bytes);
        GGML_ASSERT(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, actual.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(actual.output, result.data(), 0, bytes);
        ggml_backend_tensor_get(reference.output, expected.data(), 0, bytes);
        for (size_t i = 0; i < count; ++i) {
            const float error = std::fabs(result[i] - expected[i]);
            if (!std::isfinite(result[i]) || error > 5e-5f*std::max(1.0f, std::fabs(expected[i]))) {
                GGML_ABORT("RMS_NORM mismatch at %zu: got=%g expected=%g", i, result[i], expected[i]);
            }
            max_error = std::max(max_error, error);
        }
    }
    std::printf("PASS columns=%lld rows=%lld RMS_NORM=%s SCALE=CPU max_error=%g\n",
                (long long) columns, (long long) rows, ggml_backend_name(plugin), max_error);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(ref_buffer);
}

int main(int argc, char ** argv) {
    if (!std::getenv("GGML_BACKEND_PATH")) {
        std::fprintf(stderr, "Set GGML_BACKEND_PATH to the plugin .so file\n");
        return 1;
    }
    ggml_backend_load_all_from_path(GGML_RUNTIME_DIR);
    const char * name = argc > 1 ? argv[1] : "PPU0";
    ggml_backend_t plugin = ggml_backend_init_by_name(name, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!plugin || !cpu) {
        std::fprintf(stderr, "Need device %s and the unmodified CPU backend\n", name);
        if (plugin) ggml_backend_free(plugin);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }
    run_case(plugin, cpu, 1, 1);
    run_case(plugin, cpu, 257, 3);
    run_case(plugin, cpu, 4096, 2);
    ggml_backend_free(plugin);
    ggml_backend_free(cpu);
    return 0;
}
