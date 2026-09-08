// Execute the production buffer against delayed host queues, not a PPU oracle.
#include "common.cuh"
#include "quactlize-lib.h"
#include <functional>
#include <set>

static void require(bool ok) {
    if (!ok) { fprintf(stderr, "quactlize buffer invariant failed\n"); exit(1); }
}

struct test_stream {
    std::vector<std::function<void()>> work;
    size_t cursor = 0;
};
struct test_event { test_stream * stream = nullptr; size_t end = 0; };
static std::set<void *> allocations, pinned;
static bool allow_d2h = false;
static bool plant_copy_wait = false;
static int host_waits = 0, copies = 0, pack_calls = 0, external_waits = 0;

static test_stream * ts(cudaStream_t stream) { return (test_stream *) stream; }
static test_event * te(cudaEvent_t event) { return (test_event *) event; }
static void drain(test_stream * stream, size_t end) {
    require(stream != nullptr && end <= stream->work.size());
    while (stream->cursor < end) stream->work[stream->cursor++]();
}
static cudaError_t test_malloc(void ** out, size_t bytes) {
    *out = malloc(bytes);
    require(*out != nullptr);
    allocations.insert(*out);
    return cudaSuccess;
}
static cudaError_t test_free(void * ptr) {
    require(allocations.erase(ptr) == 1);
    free(ptr);
    return cudaSuccess;
}
static cudaError_t test_stream_create(cudaStream_t * out, unsigned flags) {
    require(flags == cudaStreamNonBlocking);
    *out = (cudaStream_t) new test_stream;
    return cudaSuccess;
}
static cudaError_t test_stream_destroy(cudaStream_t stream) {
    require(ts(stream)->cursor == ts(stream)->work.size());
    delete ts(stream);
    return cudaSuccess;
}
static cudaError_t test_stream_sync(cudaStream_t stream) {
    ++host_waits;
    drain(ts(stream), ts(stream)->work.size());
    return cudaSuccess;
}
static cudaError_t test_event_create(cudaEvent_t * out, unsigned flags) {
    require(flags == cudaEventDisableTiming);
    *out = (cudaEvent_t) new test_event;
    return cudaSuccess;
}
static cudaError_t test_event_record(cudaEvent_t event, cudaStream_t stream) {
    te(event)->stream = ts(stream);
    te(event)->end = ts(stream)->work.size();
    return cudaSuccess;
}
static cudaError_t test_event_sync(cudaEvent_t event) {
    ++host_waits;
    drain(te(event)->stream, te(event)->end);
    return cudaSuccess;
}
static cudaError_t test_event_destroy(cudaEvent_t event) {
    delete te(event);
    return cudaSuccess;
}
static cudaError_t test_wait(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    require((flags == 0 || flags == cudaEventWaitExternal) && event != nullptr);
    external_waits += flags == cudaEventWaitExternal;
    test_stream * dependency = te(event)->stream;
    const size_t end = te(event)->end;
    ts(stream)->work.push_back([=]() { drain(dependency, end); });
    return cudaSuccess;
}
static cudaError_t test_copy(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    ts(stream)->work.push_back([=]() {
        if (kind == cudaMemcpyDeviceToHost) {
            if (!allow_d2h) fprintf(stderr, "D2H ran while compute must remain launchable\n");
            require(allow_d2h);
            ++copies;
        } else {
            require(kind == cudaMemcpyHostToDevice);
        }
        memcpy(dst, src, bytes);
    });
    if (kind == cudaMemcpyDeviceToHost && plant_copy_wait) {
        drain(ts(stream), ts(stream)->work.size());
    }
    return cudaSuccess;
}
static cudaError_t test_attributes(cudaPointerAttributes * attr, const void * ptr) {
    attr->type = pinned.count((void *) ptr) ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered;
    return cudaSuccess;
}
static cudaError_t test_memset(void * ptr, int value, size_t bytes) {
    memset(ptr, value, bytes);
    return cudaSuccess;
}
static cudaError_t test_last_error() { return cudaSuccess; }

static int test_prepare(int qtype, const uint8_t * raw, uint8_t * low, uint8_t * high, uint8_t * units,
                        int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arr, void * stream) {
    quactlize_ppu_kpack_sizes_v1 sizes;
    require(ggml_quactlize_device_pack_sizes(qtype, n, k, experts, arr, &sizes) == 0);
    const size_t raw_e = sizes.raw_bytes / experts;
    const size_t low_e = sizes.low_bytes / experts;
    const size_t high_e = sizes.high_bytes / experts;
    const size_t units_e = sizes.units_bytes / experts;
    ++pack_calls;
    ts((cudaStream_t) stream)->work.push_back([=]() {
        for (int e = 0; e < experts; ++e) {
            memcpy(low + e * low_e, raw + e * raw_e, low_e);
            if (high_e) memcpy(high + e * high_e, raw + e * raw_e + low_e, high_e);
            memcpy(units + e * units_e, raw + e * raw_e + low_e + high_e, units_e);
        }
    });
    return 0;
}

// Only the runtime boundary is substituted. The buffer, offsets, ready-event
// helper and asynchronous copy function below are the actual implementation.
#define cudaMalloc test_malloc
#define cudaFree test_free
#define cudaStreamCreateWithFlags test_stream_create
#define cudaStreamDestroy test_stream_destroy
#define cudaStreamSynchronize test_stream_sync
#define cudaEventCreateWithFlags test_event_create
#define cudaEventRecord test_event_record
#define cudaEventSynchronize test_event_sync
#define cudaEventDestroy test_event_destroy
#define cudaStreamWaitEvent test_wait
#define cudaMemcpyAsync test_copy
#define cudaPointerGetAttributes test_attributes
#define cudaMemset test_memset
#define cudaGetLastError test_last_error
#define ggml_quactlize_prepare_device test_prepare
#include "../ggml/src/ggml-cuda/quactlize-buft.cu"

void ggml_cuda_set_device(int device) { require(device == 0); }
void ggml_cuda_error(const char * stmt, const char *, const char *, int, const char * msg) {
    fprintf(stderr, "%s: %s\n", stmt, msg);
    abort();
}
ggml_backend_reg_t ggml_backend_cuda_reg(void) { return nullptr; }

static void run_case(int qtype, int experts) {
    ggml_init_params init = { 1 << 20, nullptr, true };
    ggml_context * gctx = ggml_init(init);
    ggml_tensor * weight = ggml_new_tensor_3d(gctx, (ggml_type) qtype, 512, 256, experts);
    ggml_tensor * second = ggml_new_tensor_3d(gctx, (ggml_type) qtype, 512, 256, experts);
    const size_t bytes = ggml_nbytes(weight);
    std::vector<uint8_t> raw(bytes), host(bytes, 0xA5);
    for (size_t i = 0; i < bytes; ++i) raw[i] = i % 251;
    const std::vector<uint8_t> expected = raw;
    qz_buft_context buft_ctx = { 0, "test-kpack" };
    ggml_backend_buffer_type buft = { qz_buft_interface, nullptr, &buft_ctx };
    ggml_backend_buffer_t buffer = qz_buft_alloc_buffer(&buft, bytes * 2);
    require(buffer != nullptr);
    weight->buffer = second->buffer = buffer;
    weight->data = qz_buffer_get_base(buffer);
    second->data = (uint8_t *) weight->data + bytes;
    const int prior_packs = pack_calls;
    qz_buffer_set_tensor(buffer, weight, raw.data(), 0, bytes);
    ggml_quactlize_artifact art;
    require(ggml_quactlize_artifact_for(weight, &art));
    require(te(art.ready)->stream->cursor < te(art.ready)->end);
    if (experts == 1000) require(pack_calls - prior_packs > 1);
    memset(raw.data(), 0xDF, bytes);

    cudaEvent_t done;
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    ggml_backend_event copy_event{nullptr, done};
    const int before = host_waits, copied = copies;
    require(!ggml_quactlize_copy_range_async(weight, host.data(), 0, bytes, &copy_event));
    pinned.insert(host.data());
    require(!ggml_quactlize_copy_range_async(weight, host.data(), 0, bytes + 1, &copy_event));
    require(ggml_quactlize_copy_range_async(weight, host.data(), 0, bytes, &copy_event));
    require(host_waits == before && copies == copied);

    // A second intake and compute can proceed while the first D2H is held.
    qz_buffer_set_tensor(buffer, second, raw.data(), 0, bytes);
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    const int before_launch = host_waits;
    const int before_external = external_waits;
    ggml_quactlize_wait_ready(art, compute);
    require(host_waits == before_launch);
    require(external_waits == before_external + 1);
    CUDA_CHECK(cudaStreamSynchronize(compute));
    require(copies == copied);

    const size_t raw_e = bytes / experts;
    const size_t low_e = (size_t) (art.high ? art.high - art.low : art.units - art.low) / experts;
    const size_t high_e = art.high ? (art.units - art.high) / experts : 0;
    const size_t unit_e = raw_e - low_e - high_e;
    auto verify = [&](const uint8_t * low, const uint8_t * high, const uint8_t * units) {
        for (int e = 0; e < experts; ++e) {
            require(memcmp(low + e * low_e, expected.data() + e * raw_e, low_e) == 0);
            if (high_e) require(memcmp(high + e * high_e, expected.data() + e * raw_e + low_e, high_e) == 0);
            require(memcmp(units + e * unit_e, expected.data() + e * raw_e + low_e + high_e, unit_e) == 0);
        }
    };
    verify(art.low, art.high, art.units);
    allow_d2h = true;
    CUDA_CHECK(cudaEventSynchronize(done));
    require(copies == copied + 1);
    verify(host.data(), high_e ? host.data() + low_e * experts : nullptr,
           host.data() + (low_e + high_e) * experts);
    require(!ggml_quactlize_copy_range_async(weight, host.data(), bytes, 1, &copy_event));
    require(!ggml_quactlize_copy_range_async(weight, host.data(), 0, 0, &copy_event));
    copy_event.device = (ggml_backend_dev_t) 1;
    require(!ggml_quactlize_copy_range_async(weight, host.data(), 0, 16, &copy_event));
    copy_event.device = nullptr;
    const int before_range = host_waits;
    require(ggml_quactlize_copy_range_async(weight, host.data(), bytes - 97, 97, &copy_event));
    require(host_waits == before_range);
    require(ggml_quactlize_copy_range_wait(&copy_event));
    require(memcmp(host.data(), art.low + bytes - 97, 97) == 0);
    pinned.erase(host.data());
    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaStreamDestroy(compute));
    ggml_backend_buffer_free(buffer);
    require(allocations.empty());
    ggml_free(gctx);
    allow_d2h = false;
}

int main(int argc, char ** argv) {
    plant_copy_wait = argc == 2 && strcmp(argv[1], "--plant-copy-wait") == 0;
    for (int qtype = 10; qtype <= 14; ++qtype) run_case(qtype, 3);
    run_case(14, 1000);
    printf("KPACK_BUFFER_HOST PASS formats=5 chunked_experts=1000 delayed_D2H_compute=PASS device_validation=0\n");
    return 0;
}
