// Real cached-plane uploads and ready-event consumers. Synthetic byte patterns
// isolate transfer/lifetime correctness; no CPU packer or GEMM oracle is used.
#include "quactlize-buft.cuh"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static void require(bool ok) {
    if (!ok) { fprintf(stderr, "KPACK_UPLOAD_DEVICE invariant failed\n"); exit(1); }
}

static __global__ void consume(const uint8_t * weights, uint8_t * out, size_t bytes) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += gridDim.x * blockDim.x) {
        out[i] = weights[i] ^ 0x33;
    }
}

static void run_case(int qtype, int experts) {
    ggml_init_params init = {1 << 20, nullptr, true};
    ggml_context * ctx = ggml_init(init);
    require(ctx != nullptr);
    ggml_tensor * weights[] = {
        ggml_new_tensor_3d(ctx, (ggml_type) qtype, 512, 256, experts),
        ggml_new_tensor_3d(ctx, (ggml_type) qtype, 512, 256, experts),
    };
    auto * buft = ggml_backend_cuda_quactlize_buffer_type(0);
    require(buft != nullptr);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    require(buffer != nullptr);
    const size_t bytes = ggml_nbytes(weights[0]);
    ggml_quactlize_planes p{};
    require(ggml_quactlize_plane_layout(weights[0], &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement));
    require(p.low_bytes + p.high_bytes + p.units_bytes == bytes);
    std::vector<uint8_t> low(p.low_bytes), high(p.high_bytes), units(p.units_bytes);
    p.low = low.data(); p.high = high.empty() ? nullptr : high.data(); p.units = units.data();
    uint8_t * output;
    CUDA_CHECK(cudaMalloc((void **) &output, bytes));
    std::vector<uint8_t> result(bytes);
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    for (int tensor = 0; tensor < 2; ++tensor) {
        const uint8_t tag = qtype * 7 + tensor;
        std::fill(low.begin(), low.end(), tag);
        std::fill(high.begin(), high.end(), tag + 1);
        std::fill(units.begin(), units.end(), tag + 2);
        ggml_quactlize_set_planes(weights[tensor], &p);
        // Callers may immediately release or overwrite these pageable inputs.
        std::fill(low.begin(), low.end(), 0xdf);
        std::fill(high.begin(), high.end(), 0xdf);
        std::fill(units.begin(), units.end(), 0xdf);
    }
    for (bool capture : {false, true}) {
        for (int tensor = 0; tensor < 2; ++tensor) {
            ggml_quactlize_artifact art{};
            require(ggml_quactlize_artifact_for(weights[tensor], &art));
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t instance = nullptr;
            if (capture) { CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeRelaxed)); }
            ggml_quactlize_wait_ready(art, compute);
            consume<<<128, 128, 0, compute>>>(art.low, output, bytes);
            CUDA_CHECK(cudaGetLastError());
            if (capture) {
                CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
                if (capture) { CUDA_CHECK(cudaGraphLaunch(instance, compute)); }
                CUDA_CHECK(cudaMemcpyAsync(result.data(), output, bytes, cudaMemcpyDeviceToHost, compute));
                CUDA_CHECK(cudaStreamSynchronize(compute));
                const uint8_t tag = qtype * 7 + tensor;
                size_t bad = 0;
                for (size_t i = 0; i < bytes; ++i) {
                    const uint8_t want = i < p.low_bytes ? tag : i < p.low_bytes + p.high_bytes ? tag + 1 : tag + 2;
                    bad += result[i] != (want ^ 0x33);
                }
                require(bad == 0);
            }
            if (capture) {
                CUDA_CHECK(cudaGraphExecDestroy(instance));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(output));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("KPACK_UPLOAD_DEVICE PASS qtype=%d experts=%d tensors=2 bytes=%zu eager=2 graph_replays=6\n",
           qtype, experts, bytes);
    fflush(stdout);
}

int main() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { return 77; }
    CUDA_CHECK(cudaSetDevice(0));
    for (int qtype = 10; qtype <= 14; ++qtype) {
        run_case(qtype, 1);
        run_case(qtype, 257);
    }
    printf("KPACK_UPLOAD_DEVICE_ALL PASS formats=5 cases=10 eager=20 graph_replays=60 source_reuse=PASS\n");
    return 0;
}
