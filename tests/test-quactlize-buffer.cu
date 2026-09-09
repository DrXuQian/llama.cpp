// Execute the production buffer against delayed host queues, not a PPU oracle.
#include "common.cuh"
#include "quactlize-lib.h"
#include <functional>
#include <map>
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
static std::map<const void *, size_t> upload_allocations;
static std::set<const void *> uploads_pending;
static bool allow_d2h = false;
static bool plant_copy_wait = false;
static bool plant_upload_wait = false, plant_upload_reuse = false, testing_upload = false;
static int host_waits = 0, copies = 0, pack_calls = 0, stream_waits = 0;
static int upload_submissions = 0;

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
static cudaError_t test_malloc_host(void ** out, size_t bytes) {
    require(bytes == 8 * 1024 * 1024 && upload_allocations.size() < 2);
    *out = malloc(bytes);
    require(*out != nullptr);
    upload_allocations[*out] = bytes;
    pinned.insert(*out);
    return cudaSuccess;
}
static cudaError_t test_free_host(void * ptr) {
    require(uploads_pending.count(ptr) == 0 && upload_allocations.erase(ptr) == 1);
    require(pinned.erase(ptr) == 1);
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
static cudaError_t test_stream_capture_status(cudaStream_t stream, cudaStreamCaptureStatus * status) {
    require(stream != nullptr && status != nullptr);
    *status = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
static cudaError_t test_event_create(cudaEvent_t * out, unsigned flags) {
    require(flags == cudaEventDisableTiming || flags == 0);
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
    if (testing_upload && plant_upload_reuse) { return cudaSuccess; }
    drain(te(event)->stream, te(event)->end);
    return cudaSuccess;
}
static cudaError_t test_event_destroy(cudaEvent_t event) {
    delete te(event);
    return cudaSuccess;
}
static cudaError_t test_wait(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    require(flags == 0 && event != nullptr);
    ++stream_waits;
    test_stream * dependency = te(event)->stream;
    const size_t end = te(event)->end;
    ts(stream)->work.push_back([=]() { drain(dependency, end); });
    return cudaSuccess;
}
static cudaError_t test_copy(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const bool staged_upload = kind == cudaMemcpyHostToDevice && upload_allocations.count(src);
    if (staged_upload) {
        require(bytes <= upload_allocations.at(src) && uploads_pending.insert(src).second);
        require(uploads_pending.size() <= 2);
        ++upload_submissions;
    }
    ts(stream)->work.push_back([=]() {
        if (staged_upload) { require(uploads_pending.erase(src) == 1); }
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
    if (staged_upload && plant_upload_wait) {
        ++host_waits;
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
#define cudaMallocHost test_malloc_host
#define cudaFreeHost test_free_host
#define cudaStreamCreateWithFlags test_stream_create
#define cudaStreamDestroy test_stream_destroy
#define cudaStreamSynchronize test_stream_sync
#define cudaStreamIsCapturing test_stream_capture_status
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
    const int before_stream_waits = stream_waits;
    ggml_quactlize_wait_ready(art, compute);
    require(host_waits == before_launch);
    require(stream_waits == before_stream_waits + 1);
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

static void run_upload_case(int qtype, int experts, bool teardown_pending) {
    ggml_init_params init = {1 << 20, nullptr, true};
    ggml_context * gctx = ggml_init(init);
    auto * first = ggml_new_tensor_3d(gctx, (ggml_type) qtype, 512, 256, experts);
    auto * second = ggml_dup_tensor(gctx, first);
    const size_t bytes = ggml_nbytes(first);
    const size_t chunks = (bytes + qz_upload_chunk_bytes - 1) / qz_upload_chunk_bytes;
    qz_buft_context buft_ctx = {0, "test-kpack-upload"};
    ggml_backend_buffer_type buft = {qz_buft_interface, nullptr, &buft_ctx};
    auto * buffer = qz_buft_alloc_buffer(&buft, bytes * 2);
    require(buffer != nullptr);
    first->buffer = second->buffer = buffer;
    first->data = qz_buffer_get_base(buffer);
    second->data = (uint8_t *) first->data + bytes;
    memset(first->data, 0xa5, bytes * 2);
    std::vector<uint8_t> expected(bytes);
    for (size_t i = 0; i < bytes; ++i) { expected[i] = (i * 13 + qtype) % 251; }
    ggml_quactlize_planes p{};
    require(ggml_quactlize_plane_layout(first, &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement));
    std::vector<uint8_t> low(expected.begin(), expected.begin() + p.low_bytes);
    std::vector<uint8_t> high(expected.begin() + p.low_bytes, expected.begin() + p.low_bytes + p.high_bytes);
    std::vector<uint8_t> units(expected.end() - p.units_bytes, expected.end());
    p.low = low.data(); p.high = p.high_bytes ? high.data() : nullptr; p.units = units.data();
    const int before = host_waits, submitted = upload_submissions, packs = pack_calls;
    testing_upload = true;
    ggml_quactlize_set_planes(first, &p);
    ggml_quactlize_set_planes(second, &p);
    testing_upload = false;
    // The final two chunks remain in flight even across tensor boundaries.
    require(upload_submissions - submitted == (int) (2 * chunks));
    require(host_waits - before == (int) (2 * chunks - 2));
    require(uploads_pending.size() == 2 && upload_allocations.size() == 2 && pack_calls == packs);
    memset(low.data(), 0xdf, low.size());
    if (!high.empty()) { memset(high.data(), 0xdf, high.size()); }
    memset(units.data(), 0xdf, units.size());
    if (!teardown_pending) {
        cudaStream_t compute;
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        const int waits = host_waits;
        for (auto * tensor : {first, second}) {
            ggml_quactlize_artifact art{};
            require(ggml_quactlize_artifact_for(tensor, &art));
            ggml_quactlize_wait_ready(art, compute);
            ts(compute)->work.push_back([&, tensor]() { require(memcmp(tensor->data, expected.data(), bytes) == 0); });
        }
        require(host_waits == waits);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamDestroy(compute));
        require(uploads_pending.empty());
    }
    ggml_backend_buffer_free(buffer);
    require(uploads_pending.empty() && upload_allocations.empty() && allocations.empty() && pinned.empty());
    ggml_free(gctx);
}

int main(int argc, char ** argv) {
    plant_copy_wait = argc == 2 && strcmp(argv[1], "--plant-copy-wait") == 0;
    plant_upload_wait = argc == 2 && strcmp(argv[1], "--plant-upload-wait") == 0;
    plant_upload_reuse = argc == 2 && strcmp(argv[1], "--plant-upload-reuse") == 0;
    for (int qtype = 10; qtype <= 14; ++qtype) run_case(qtype, 3);
    run_case(14, 1000);
    for (int qtype = 10; qtype <= 14; ++qtype) {
        run_upload_case(qtype, 1, false);
        run_upload_case(qtype, 257, false);
    }
    run_upload_case(12, 257, true);
    printf("KPACK_UPLOAD_HOST PASS formats=5 slots=2 slot_MiB=8 source_reuse=PASS tail_async=PASS teardown=PASS device_validation=0\n");
    printf("KPACK_BUFFER_HOST PASS formats=5 chunked_experts=1000 delayed_D2H_compute=PASS device_validation=0\n");
    return 0;
}
