// Real stream/capture/replay coverage for the production readiness helper.
// Synthetic bytes isolate synchronization; no PPU pack or GEMM is called.
#include "quactlize-buft.cuh"

#include <chrono>
#include <condition_variable>
#include <mutex>

static void require(bool ok) {
    if (!ok) { fprintf(stderr, "KPACK_READY invariant failed\n"); exit(1); }
}

void ggml_cuda_error(const char * stmt, const char * func, const char *, int line, const char * msg) {
    fprintf(stderr, "KPACK_READY CUDA_ERROR function=%s line=%d call=%s error=%s\n", func, line, stmt, msg);
    exit(2);
}

struct gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;

    void await_entry() {
        std::unique_lock<std::mutex> lock(mutex);
        require(cv.wait_for(lock, std::chrono::seconds(10), [&]() { return entered; }));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

static void CUDART_CB hold_stream(void * ptr) {
    auto * g = (gate *) ptr;
    std::unique_lock<std::mutex> lock(g->mutex);
    g->entered = true;
    g->cv.notify_all();
    require(g->cv.wait_for(lock, std::chrono::seconds(10), [&]() { return g->released; }));
}

static __global__ void read_weights(const uint8_t * weights, uint8_t * output, int count) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) { output[i] = weights[i] ^ 0x33; }
}

static int bad_bytes(const uint8_t * bytes, int count, uint8_t want) {
    int bad = 0;
    for (int i = 0; i < count; ++i) { bad += bytes[i] != want; }
    return bad;
}

static void require_pending(cudaEvent_t event) {
    require(cudaEventQuery(event) == cudaErrorNotReady);
    const cudaError_t last = cudaGetLastError();
    require(last == cudaSuccess || last == cudaErrorNotReady);
}

static void run_case(bool pending_pack, bool stalled_copy) {
    constexpr int bytes = 4096;
    constexpr uint8_t packed = 0x5a;
    constexpr uint8_t want = packed ^ 0x33;
    cudaStream_t pack, compute, copy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&pack, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    cudaEvent_t ready, done;
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    uint8_t * weights, * output, * result, * snapshot;
    CUDA_CHECK(cudaMalloc(&weights, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));
    CUDA_CHECK(cudaMallocHost(&result, bytes));
    CUDA_CHECK(cudaMallocHost(&snapshot, bytes));
    CUDA_CHECK(cudaMemsetAsync(weights, 0, bytes, pack));
    CUDA_CHECK(cudaStreamSynchronize(pack));

    gate pack_gate, copy_gate;
    if (pending_pack) {
        CUDA_CHECK(cudaLaunchHostFunc(pack, hold_stream, &pack_gate));
        pack_gate.await_entry();
    }
    CUDA_CHECK(cudaMemsetAsync(weights, packed, bytes, pack));
    CUDA_CHECK(cudaEventRecord(ready, pack));
    if (pending_pack) {
        require_pending(ready);
        // Omitting the wait reads the old bytes: the readiness edge is necessary.
        read_weights<<<1, 128, 0, compute>>>(weights, output, bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(result, output, bytes, cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        require(bad_bytes(result, bytes, want) == bytes);
        printf("KPACK_READY missing_wait=EXPECTED_RED bad=%d\n", bytes);
    } else {
        CUDA_CHECK(cudaEventSynchronize(ready));
    }
    if (stalled_copy) {
        CUDA_CHECK(cudaStreamWaitEvent(copy, ready, 0));
        CUDA_CHECK(cudaLaunchHostFunc(copy, hold_stream, &copy_gate));
        copy_gate.await_entry();
        CUDA_CHECK(cudaMemcpyAsync(snapshot, weights, bytes, cudaMemcpyDeviceToHost, copy));
        CUDA_CHECK(cudaEventRecord(done, copy));
    }

    ggml_quactlize_artifact art{};
    art.ready = ready;
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeRelaxed));
    ggml_quactlize_wait_ready(art, compute);
    read_weights<<<1, 128, 0, compute>>>(weights, output, bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
    cudaGraphNode_t nodes[8];
    size_t count = 8;
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &count));
    require(count <= 8);
    int waits = 0;
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        waits += type == cudaGraphNodeTypeWaitEvent;
    }
    require(waits == 1);
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 3; ++repeat) {
        CUDA_CHECK(cudaGraphLaunch(instance, compute));
        if (pending_pack && repeat == 0) {
            // Capture and submission return while the producer is still held.
            require_pending(ready);
            pack_gate.release();
        }
        CUDA_CHECK(cudaMemcpyAsync(result, output, bytes, cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        require(bad_bytes(result, bytes, want) == 0);
        if (stalled_copy) { require_pending(done); }
    }
    if (stalled_copy) {
        copy_gate.release();
        CUDA_CHECK(cudaStreamSynchronize(copy));
        require(bad_bytes(snapshot, bytes, packed) == 0);
    }
    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamSynchronize(pack));
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaStreamDestroy(pack));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(copy));
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFreeHost(result));
    CUDA_CHECK(cudaFreeHost(snapshot));
    printf("KPACK_READY PASS pending_pack=%d stalled_D2H=%d replays=3 wait_nodes=%d\n",
           pending_pack, stalled_copy, waits);
}

int main() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { return 77; }
    CUDA_CHECK(cudaSetDevice(0));
    run_case(false, false);
    run_case(true, false);
    run_case(false, true);
    printf("KPACK_READY_ALL PASS graph_replays=9 model_oracle=NOT_RUN\n");
    return 0;
}
