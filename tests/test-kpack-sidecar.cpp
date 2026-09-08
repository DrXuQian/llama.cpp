// The persistent K-pack sidecar, exercised without a device: a GGUF is written with the gguf API, planes come from
// the stub quactlize libraries, the writer publishes a bundle, the reader validates it, and every guard is shown to
// fire on a bundle that is wrong in exactly one way. With KPACK_PY / KPACK_PY_REPO set, quactlize's own
// load_kpack_bundle is run on the same bundle -- the interoperability check that matters, since the format is theirs.

#include "llama-kpack-sidecar.h"
#include "llama-kpack-cache.h"
#include "llama-impl.h"
#include "quactlize-lib.h"
#include "quactlize-sidecar.h"
#include "ggml-backend-impl.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <cstdarg>
#include <thread>
#include <string>
#include <sys/stat.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static int g_failures = 0;

// libllama's logging seam, without loading a GPU backend in this host test.
void llama_log_internal(ggml_log_level, const char * format, ...) {
    va_list args;
    va_start(args, format); vfprintf(stderr, format, args); va_end(args);
}

namespace cache_mock {
struct event { const uint8_t * src = nullptr; void * dst = nullptr; size_t bytes = 0; bool pending = false; };
static ggml_backend_reg reg{};
static ggml_backend_device dev{};
static ggml_backend_buffer_type device_buft{}, host_buft{};
static std::thread::id owner;
static std::promise<void> entered, release;
static std::shared_future<void> released;
static std::atomic<int> waits{0};
static std::atomic<int> copies{0}, in_flight{0}, violations{0};
static int uploads = 0, host_allocations = 0;
static bool descriptor_mismatch = false;
static int reject_copy = 0, reject_wait = 0;
static bool hold = true;

static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t buft, size_t bytes) {
    if (buft == &host_buft) { ++host_allocations; }
    ggml_backend_buffer_i iface{};
    iface.get_base = [](ggml_backend_buffer_t b) { return b->context; };
    iface.free_buffer = [](ggml_backend_buffer_t b) { free(b->context); };
    return ggml_backend_buffer_init(buft, iface, malloc(bytes), bytes);
}
static bool is_kpack(const ggml_tensor * t) { return t->buffer && t->buffer->buft == &device_buft; }
static bool layout(const ggml_tensor * t, size_t * lo, size_t * hi, size_t * un,
                   quactlize_ppu_placed_arrangement_v2 * arr) {
    int64_t l, h, u;
    if (!ggml_quactlize_arrangement_for(t->type, arr) ||
        !ggml_quactlize_plane_sizes(t->type, t->ne[1], t->ne[0], t->ne[2], arr, &l, &h, &u)) { return false; }
    *lo = l; *hi = h; *un = u;
    if (descriptor_mismatch) { ++arr->mapping_id; }
    return true;
}
static void set(ggml_tensor * t, const ggml_quactlize_planes * p) {
    auto * dst = (uint8_t *) t->data;
    memcpy(dst, p->low, p->low_bytes);
    if (p->high_bytes) { memcpy(dst + p->low_bytes, p->high, p->high_bytes); }
    memcpy(dst + p->low_bytes + p->high_bytes, p->units, p->units_bytes);
    ++uploads;
}
static bool copy(const ggml_tensor * t, void * dst, size_t off, size_t bytes, ggml_backend_event_t ev) {
    auto * e = (event *) ev->context;
    if (std::this_thread::get_id() == owner || e->pending || bytes > 8 * 1024 * 1024 ||
        off > ggml_nbytes(t) || bytes > ggml_nbytes(t) - off) { ++violations; return false; }
    if (++copies == reject_copy) { return false; }
    *e = {(const uint8_t *) t->data + off, dst, bytes, true};
    if (++in_flight > 2) { ++violations; }
    return true;
}
static bool wait(ggml_backend_event_t ev) {
    if (std::this_thread::get_id() == owner) { ++violations; return false; }
    const int call = ++waits;
    if (call == 1 && hold) { entered.set_value(); released.wait(); }
    auto * e = (event *) ev->context;
    if (!e->pending) { ++violations; return false; }
    memcpy(e->dst, e->src, e->bytes);
    e->pending = false;
    --in_flight;
    return call != reject_wait;
}
static void init() {
    owner = std::this_thread::get_id();
    released = release.get_future().share();
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (!strcmp(name, "ggml_quactlize_tensor_is_kpack")) { return (void *) is_kpack; }
        if (!strcmp(name, "ggml_quactlize_plane_layout")) { return (void *) layout; }
        if (!strcmp(name, "ggml_quactlize_set_planes")) { return (void *) set; }
        if (!strcmp(name, "ggml_quactlize_copy_range_async")) { return (void *) copy; }
        if (!strcmp(name, "ggml_quactlize_copy_range_wait")) { return (void *) wait; }
        return nullptr;
    };
    dev.reg = &reg;
    dev.iface.get_host_buffer_type = [](ggml_backend_dev_t) { return &host_buft; };
    dev.iface.event_new = [](ggml_backend_dev_t d) { return new ggml_backend_event{d, new event}; };
    dev.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t ev) {
        if (((event *) ev->context)->pending) { ++violations; }
        delete (event *) ev->context; delete ev;
    };
    device_buft.device = host_buft.device = &dev;
    host_buft.iface.alloc_buffer = allocate;
}
} // namespace cache_mock

#define CHECK(cond, ...) do { \
    const bool ok_ = (cond); \
    printf("    %-70s %s\n", #cond, ok_ ? "ok" : "FAIL"); \
    if (!ok_) { g_failures++; printf("      "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static std::vector<uint8_t> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_file(const std::string & p, const std::vector<uint8_t> & d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write((const char *) d.data(), (std::streamsize) d.size());
}
static void copy_bundle(const std::string & src, const std::string & dst) {
    mkdir(dst.c_str(), 0755);
    write_file(dst + "/manifest.json", read_file(src + "/manifest.json"));
    write_file(dst + "/weights.bin",   read_file(src + "/weights.bin"));
}
static void rm_bundle(const std::string & d) {
    unlink((d + "/manifest.json").c_str()); unlink((d + "/weights.bin").c_str()); unlink((d + "/extra").c_str()); rmdir(d.c_str());
}

// ---- SHA-256 ----
static void test_sha256() {
    printf("  [sha256]\n");
    const std::string abc = "abc";
    std::vector<uint8_t> million(1000000, 'a');
    for (int portable = 0; portable < 2; ++portable) {
        llama_kpack_sha256_force_portable(portable);
        CHECK(llama_kpack_sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "portable=%d", portable);
        CHECK(llama_kpack_sha256_hex(abc.data(), 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "portable=%d", portable);
        CHECK(llama_kpack_sha256_hex(million.data(), million.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "portable=%d", portable);
    }
    // The two paths on random lengths: the only reason to trust the SHA-extension schedule.
    unsigned seed = 42; int mism = 0;
    for (int i = 0; i < 200; ++i) {
        seed = seed*1103515245u + 12345u;
        const size_t len = (seed >> 8) % 1500;
        std::vector<uint8_t> b(len);
        for (auto & x : b) { seed = seed*1103515245u + 12345u; x = (uint8_t) (seed >> 16); }
        llama_kpack_sha256_force_portable(true);  const std::string a = llama_kpack_sha256_hex(b.data(), len);
        llama_kpack_sha256_force_portable(false); const std::string c = llama_kpack_sha256_hex(b.data(), len);
        if (a != c) { mism++; }
    }
    CHECK(mism == 0, "portable vs accelerated mismatches: %d", mism);
    llama_kpack_sha256_force_portable(false);
}

struct made_tensor {
    std::string name; int qtype; int64_t n, k, experts; int rank;
    std::vector<uint8_t> data;            // GGUF bytes
    std::vector<uint8_t> low, high, units;
    llama_kpack_planes planes;
    llama_kpack_source_tensor src;
};

static bool make_planes(made_tensor & t) {
    quactlize_ppu_placed_arrangement_v2 arr;
    if (!ggml_quactlize_arrangement_for(t.qtype, &arr)) { return false; }
    const int64_t e1 = t.rank == 3 ? t.experts : 1;
    int64_t lo, hi, un;
    if (!ggml_quactlize_plane_sizes(t.qtype, t.n, t.k, e1, &arr, &lo, &hi, &un)) { return false; }
    t.low.resize((size_t) lo); t.high.resize((size_t) (hi ? hi : 1)); t.units.resize((size_t) un);
    if (ggml_quactlize_prepare(t.qtype, t.data.data(), t.low.data(), hi ? t.high.data() : nullptr, t.units.data(),
                               (int) t.n, (int) t.k, (int) e1, &arr) != 0) { return false; }
    auto & p = t.planes;
    p.low = t.low.data(); p.low_bytes = (size_t) lo;
    p.high = hi ? t.high.data() : nullptr; p.high_bytes = (size_t) hi;
    p.units = t.units.data(); p.units_bytes = (size_t) un;
    p.layout = arr.layout; p.bits = arr.bits; p.high_bits = arr.high_bits; p.artifact_tile_k = arr.artifact_tile_k;
    p.transport_tile_k = arr.transport_tile_k; p.group_size = arr.group_size; p.reserved = arr.reserved; p.mapping_id = arr.mapping_id;
    return true;
}

int main() {
    test_sha256();

    // ---- a GGUF with one dense and one grouped Q4_K tensor, plus one that is not packable ----
    printf("  [gguf]\n");
    std::string base = getenv("KPACK_TEST_DIR") ? getenv("KPACK_TEST_DIR") : "/tmp";
    char tmpl[4096]; snprintf(tmpl, sizeof(tmpl), "%s/kpack-test-XXXXXX", base.c_str());
    const std::string dir = mkdtemp(tmpl);
    const std::string gguf_path = dir + "/model.gguf";

    struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(ip);
    made_tensor dense  { "blk.0.attn_q.weight",        GGML_TYPE_Q4_K, 256, 512, 0, 2, {}, {}, {}, {}, {}, {} };
    made_tensor grouped{ "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 256, 512, 257, 3, {}, {}, {}, {}, {}, {} };
    made_tensor high   { "blk.0.attn_v.weight",        GGML_TYPE_Q5_K, 256, 512, 0, 2, {}, {}, {}, {}, {}, {} };
    ggml_tensor * t_norm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 256); ggml_set_name(t_norm, "blk.0.attn_norm.weight");
    ggml_tensor * t_dense = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_K, dense.k, dense.n); ggml_set_name(t_dense, dense.name.c_str());
    ggml_tensor * t_group = ggml_new_tensor_3d(gctx, GGML_TYPE_Q4_K, grouped.k, grouped.n, grouped.experts); ggml_set_name(t_group, grouped.name.c_str());
    ggml_tensor * t_high = ggml_new_tensor_2d(gctx, GGML_TYPE_Q5_K, high.k, high.n); ggml_set_name(t_high, high.name.c_str());
    unsigned seed = 7;
    for (ggml_tensor * t : { t_norm, t_dense, t_group, t_high }) {
        uint8_t * d = (uint8_t *) t->data;
        for (size_t i = 0; i < ggml_nbytes(t); ++i) { seed = seed*1103515245u + 12345u; d[i] = (uint8_t) (seed >> 16); }
    }
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_add_tensor(g, t_norm);     // index 0: not packable
    gguf_add_tensor(g, t_dense);    // index 1
    gguf_add_tensor(g, t_group);    // index 2
    gguf_add_tensor(g, t_high);     // index 3: a nonempty high plane
    gguf_write_to_file(g, gguf_path.c_str(), false);

    auto fill = [&](made_tensor & m, ggml_tensor * t, int index) {
        m.data.assign((uint8_t *) t->data, (uint8_t *) t->data + ggml_nbytes(t));
        m.src.name = m.name; m.src.gguf_index = index;
        // gguf_get_data_offset() is only meaningful for a context read FROM a file; for one being written, the data
        // section starts right after the metadata, which is what gguf_get_meta_size() reports.
        m.src.data_offset = gguf_get_meta_size(g) + gguf_get_tensor_offset(g, index);
        m.src.size_bytes = ggml_nbytes(t); m.src.ggml_type = m.qtype; m.src.rank = m.rank;
        m.src.n = m.n; m.src.k = m.k; m.src.experts = m.rank == 3 ? m.experts : 0;
    };
    fill(dense, t_dense, 1); fill(grouped, t_group, 2);
    fill(high, t_high, 3);
    CHECK(make_planes(dense),   "stub library must supply planes for the dense tensor");
    CHECK(make_planes(grouped), "stub library must supply planes for the grouped tensor");
    CHECK(make_planes(high), "stub library must supply a nonempty high plane");
    std::vector<llama_kpack_source_tensor> inventory = { dense.src, grouped.src, high.src };

    // ---- writer ----
    printf("  [writer]\n");
    const std::string bundle = gguf_path + ".kpack";
    std::string err;
    {
        llama_kpack_sidecar_writer w;
        CHECK(w.begin(bundle, err), "%s", err.c_str());
        w.skip("blk.0.attn_norm.weight", "F32", "not a k-quant this build packs");
        CHECK(w.add(dense.src, dense.data.data(), dense.planes, err), "%s", err.c_str());
        // out of order must be refused: a record with a lower index after a higher one
        llama_kpack_source_tensor bad = dense.src; bad.name = "x"; bad.gguf_index = 0;
        CHECK(!w.add(bad, dense.data.data(), dense.planes, err), "index order guard");
        CHECK(w.add(grouped.src, grouped.data.data(), grouped.planes, err), "%s", err.c_str());
        CHECK(w.add(high.src, high.data.data(), high.planes, err), "%s", err.c_str());
        CHECK(w.finish(gguf_path, gguf_path, err), "%s", err.c_str());
        CHECK(w.packed() == 3, "packed=%zu", w.packed());
    }
    struct stat st{};
    CHECK(stat((bundle + "/manifest.json").c_str(), &st) == 0 && stat((bundle + "/weights.bin").c_str(), &st) == 0, "bundle files exist");
    {
        llama_kpack_sidecar_writer w2;
        CHECK(!w2.begin(bundle, err), "a second writer must refuse the existing bundle");
    }
    {
        const std::string target = dir + "/in-progress";
        const std::string staging = target + ".partial." + std::to_string((long) getpid());
        const std::vector<uint8_t> original = { 1, 2, 3, 4 };
        CHECK(mkdir(staging.c_str(), 0755) == 0, "create another writer's staging directory");
        write_file(staging + "/weights.bin", original);
        {
            llama_kpack_sidecar_writer w;
            CHECK(!w.begin(target, err), "existing staging directory must be refused");
        }
        CHECK(read_file(staging + "/weights.bin") == original, "a rejected writer must not remove another writer's files");
        rm_bundle(staging);
    }
    {
        const std::string target = dir + "/active-writer";
        const std::string staging = target + ".partial." + std::to_string((long) getpid());
        llama_kpack_sidecar_writer w;
        CHECK(w.begin(target, err), "%s", err.c_str());
        CHECK(!w.begin(dir + "/second-output", err), "begin twice must preserve the active writer");
        CHECK(stat((staging + "/weights.bin").c_str(), &st) == 0, "active staging file must remain owned");
        w.abort();
        CHECK(stat(staging.c_str(), &st) != 0, "abort must clean up the owned staging directory");
    }
    {
        const std::string replacement = dir + "/same-bytes-other-inode.gguf";
        write_file(replacement, read_file(gguf_path));
        const int original_fd = open(gguf_path.c_str(), O_RDONLY);
        CHECK(original_fd >= 0, "open loader source");
        llama_kpack_sidecar_writer writer;
        CHECK(!writer.bind_source(replacement, err, original_fd), "bind must reject a different file even with identical bytes");
        CHECK(writer.bind_source(gguf_path, err, original_fd), "bind exact loader descriptor");
        close(original_fd);
        unlink(replacement.c_str());
    }

    // ---- reader ----
    printf("  [background streaming]\n");
    const std::string streamed = dir + "/streamed";
    {
        llama_kpack_background_writer writer;
        CHECK(writer.prepare(streamed, gguf_path, err), "%s", err.c_str());
        std::promise<void> entered, release;
        auto ready = entered.get_future();
        auto released = release.get_future().share();
        std::atomic<bool> first{true};
        std::atomic<size_t> calls{0};
        std::vector<uint8_t> staging(4093);
        const auto job_for = [&](const made_tensor & tensor) {
            llama_kpack_write_job job{tensor.src, tensor.planes, {}};
            job.planes.low = job.planes.high = job.planes.units = nullptr;
            job.read = [&, tensor_ptr = &tensor](size_t offset, size_t bytes, std::string & why) -> const uint8_t * {
                if (first.exchange(false)) { entered.set_value(); released.wait(); }
                if (bytes > staging.size()) { why = "unbounded request"; return nullptr; }
                const auto & p = tensor_ptr->planes;
                const uint8_t * ptrs[] = {p.low, p.high, p.units};
                const size_t sizes[] = {p.low_bytes, p.high_bytes, p.units_bytes};
                for (int j = 0; j < 3; ++j) {
                    if (offset < sizes[j]) {
                        if (bytes > sizes[j] - offset) { why = "cross-plane request"; return nullptr; }
                        memcpy(staging.data(), ptrs[j] + offset, bytes);
                        ++calls;
                        return staging.data();
                    }
                    offset -= sizes[j];
                }
                why = "invalid offset"; return nullptr;
            };
            return job;
        };
        // Loading order is not file order. Neither the jobs nor their reader
        // contain the raw GGUF data pointer.
        CHECK(writer.start({job_for(high), job_for(grouped), job_for(dense)}, inventory, staging.size(), err), "%s", err.c_str());
        CHECK(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "worker reached delayed copy");
        CHECK(stat((streamed + "/manifest.json").c_str(), &st) != 0, "no publication while a copy is pending");
        CHECK(calls == 0, "start returned before any D2H completion");
        release.set_value();
        CHECK(writer.wait(err), "%s", err.c_str());
        CHECK(calls > 3, "bounded stream uses multiple chunks");
        CHECK(read_file(streamed + "/weights.bin") == read_file(bundle + "/weights.bin"), "streamed bytes match synchronous writer");
        llama_kpack_sidecar_reader reader;
        CHECK(reader.open(streamed, err) && reader.load_unchecked(gguf_path, inventory, err), "%s", err.c_str());
        CHECK(!reader.verify_source(gguf_path, inventory, 2, err) && !reader.verify_storage(2, err),
              "a runtime cache must not claim verified-bundle provenance");
        const auto bytes = read_file(streamed + "/manifest.json");
        const std::string manifest(bytes.begin(), bytes.end());
        CHECK(manifest.find("llama.kpack-cache") != std::string::npos && manifest.find("sha256") == std::string::npos,
              "runtime manifest records local source identity, not checksums");
    }
    for (int variant = 0; variant < 3; ++variant) {
        const std::string target = dir + "/stream-fail-" + std::to_string(variant);
        const std::string source = dir + "/stream-source-" + std::to_string(variant);
        write_file(source, read_file(gguf_path));
        llama_kpack_background_writer writer;
        CHECK(writer.prepare(target, source, err), "%s", err.c_str());
        std::promise<void> entered, release;
        auto ready = entered.get_future();
        auto released = release.get_future().share();
        std::atomic<bool> first{true};
        llama_kpack_write_job job{dense.src, dense.planes, {}};
        job.read = [&](size_t offset, size_t, std::string & why) -> const uint8_t * {
            if (first.exchange(false)) { entered.set_value(); released.wait(); }
            if (variant == 0) { why = "planted D2H rejection"; return nullptr; }
            return offset < dense.planes.low_bytes ? dense.planes.low + offset :
                dense.planes.units + offset - dense.planes.low_bytes;
        };
        CHECK(writer.start({job}, inventory, 4093, err), "%s", err.c_str());
        CHECK(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "copy entered");
        if (variant == 1) {
            // Same content, different inode: a replaced source is not the file
            // whose identity was captured before loading.
            const std::string replaced = source + ".old";
            CHECK(rename(source.c_str(), replaced.c_str()) == 0, "replace source");
            write_file(source, read_file(gguf_path));
            unlink(replaced.c_str());
        }
        if (variant == 2) {
            auto cancelled = std::async(std::launch::async, [&]() { writer.cancel(); });
            CHECK(cancelled.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout,
                  "cancellation must not release an in-flight copy destination");
            release.set_value();
            cancelled.get();
        } else {
            release.set_value();
        }
        CHECK(!writer.wait(err), "copy failure, changed source and cancellation must not publish");
        CHECK(stat(target.c_str(), &st) != 0, "no final directory after failure");
        const std::string staging = target + ".partial." + std::to_string((long) getpid());
        CHECK(stat(staging.c_str(), &st) != 0, "no owned partial directory after failure");
        unlink(source.c_str());
    }

    // ---- reader ----
    printf("  [reader]\n");
    {
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err), "%s", err.c_str());
        CHECK(r.verify_source(gguf_path, inventory, 4, err), "%s", err.c_str());
        CHECK(r.verify_storage(4, err), "%s", err.c_str());
        const auto * rd = r.find(dense.name); const auto * rg = r.find(grouped.name);
        CHECK(rd && rg && !r.find("nope"), "records");
        if (rd && rg) {
            CHECK(rd->planes.low_bytes == dense.planes.low_bytes && memcmp(rd->planes.low, dense.low.data(), dense.planes.low_bytes) == 0, "dense low plane round-trips");
            CHECK(rd->planes.units_bytes == dense.planes.units_bytes && memcmp(rd->planes.units, dense.units.data(), dense.planes.units_bytes) == 0, "dense units plane round-trips");
            CHECK(rg->planes.low_bytes == grouped.planes.low_bytes && memcmp(rg->planes.low, grouped.low.data(), grouped.planes.low_bytes) == 0, "grouped low plane round-trips");
            CHECK(rg->planes.high == nullptr && rg->planes.high_bytes == 0, "Q4_K has no high plane");
            CHECK(rd->planes.mapping_id == dense.planes.mapping_id && rd->planes.bits == 4, "arrangement carried");
            CHECK(rg->units.shape == std::vector<int64_t>({ grouped.experts, 2, 256, 16 }), "grouped units shape [E, K/256, N, 16]");
            CHECK(rd->units.shape == std::vector<int64_t>({ 2, 256, 16 }), "dense units shape [K/256, N, 16]");
            CHECK(rd->low.shape == std::vector<int64_t>({ 1, 256, 256 }), "dense low shape [1, N, K/2]");
        }
    }

    printf("  [model cache integration]\n");
    const std::string model_cache = dir + "/model-cache";
    {
        cache_mock::init();
        auto ready = cache_mock::entered.get_future();
        auto * tensor = ggml_dup_tensor(gctx, t_group);
        ggml_set_name(tensor, grouped.name.c_str());
        tensor->buffer = cache_mock::allocate(&cache_mock::device_buft, ggml_nbytes(tensor));
        tensor->data = ggml_backend_buffer_get_base(tensor->buffer);
        // A finished GPU pack is represented by resident plane bytes here.
        auto * data = (uint8_t *) tensor->data;
        memcpy(data, grouped.low.data(), grouped.planes.low_bytes);
        memcpy(data + grouped.planes.low_bytes, grouped.units.data(), grouped.planes.units_bytes);
        auto * other = ggml_dup_tensor(gctx, t_high);
        ggml_set_name(other, high.name.c_str());
        other->buffer = cache_mock::allocate(&cache_mock::device_buft, ggml_nbytes(other));
        other->data = ggml_backend_buffer_get_base(other->buffer);
        auto * high_data = (uint8_t *) other->data;
        memcpy(high_data, high.low.data(), high.planes.low_bytes);
        memcpy(high_data + high.planes.low_bytes, high.high.data(), high.planes.high_bytes);
        memcpy(high_data + high.planes.low_bytes + high.planes.high_bytes, high.units.data(), high.planes.units_bytes);
        {
            llama_kpack_cache cache(model_cache, gguf_path, inventory);
            CHECK(!cache.load(tensor), "cache miss uses the normal GPU producer");
            cache.capture(other); // sorting must not mix up the shared staging slots
            cache.capture(tensor);
            cache.capture(tensor); // aliases must not create duplicate records
            cache.start();
            CHECK(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "writer reached async completion");
            CHECK(cache_mock::host_allocations == 2, "two bounded pinned slots per device");
            CHECK(cache_mock::copies == 2 && cache_mock::in_flight == 2,
                  "both slots submitted before waiting for the first D2H completion");
            CHECK(memcmp(tensor->data, grouped.low.data(), grouped.planes.low_bytes) == 0,
                  "compute can read resident weights while D2H is held");
            CHECK(stat(model_cache.c_str(), &st) != 0, "cache remains unpublished until copies finish");
            cache_mock::release.set_value();
        } // waits before tensor->buffer is freed
        CHECK(cache_mock::copies == 7 && cache_mock::waits == 7 && cache_mock::in_flight == 0,
              "multi-chunk low, partial tail, absent/present high and cross-tensor reuse drained exactly once");
        CHECK(stat((model_cache + "/manifest.json").c_str(), &st) == 0, "normal model teardown completes persistence");
        memset(tensor->data, 0xA5, ggml_nbytes(tensor));
        memset(other->data, 0xA5, ggml_nbytes(other));
        {
            llama_kpack_cache cache(model_cache, gguf_path, inventory);
            cache_mock::descriptor_mismatch = true;
            CHECK(!cache.load(tensor), "a changed registry cannot consume cached bytes");
            cache_mock::descriptor_mismatch = false;
            CHECK(cache.load(tensor), "second model load trusts cached planes without repacking or hashing");
            CHECK(cache_mock::uploads == 1, "one direct plane upload on cache hit");
            CHECK(memcmp(tensor->data, grouped.low.data(), grouped.planes.low_bytes) == 0, "cached low bytes match after slot reuse");
            CHECK(memcmp(data + grouped.planes.low_bytes, grouped.units.data(), grouped.planes.units_bytes) == 0, "cached unit bytes match");
            CHECK(cache.load(other), "second tensor reuses the same bounded staging pool correctly");
            CHECK(memcmp(high_data, high.low.data(), high.planes.low_bytes) == 0 &&
                  memcmp(high_data + high.planes.low_bytes, high.high.data(), high.planes.high_bytes) == 0 &&
                  memcmp(high_data + high.planes.low_bytes + high.planes.high_bytes, high.units.data(), high.planes.units_bytes) == 0,
                  "Q5 low/high/units bytes match after cross-tensor slot reuse");
            cache.start(); // a cache hit must not start a replacement writer
        }
        CHECK(cache_mock::copies == 7 && cache_mock::host_allocations == 2, "hit allocates no staging and starts no D2H");
        // Rejecting the prefetched submission or its predecessor's completion
        // leaves one other range in flight. It must be drained on the worker.
        cache_mock::hold = false;
        for (int variant = 0; variant < 3; ++variant) {
            const std::string target = dir + "/pipeline-fail-" + std::to_string(variant);
            cache_mock::reject_copy = variant == 0 ? cache_mock::copies + 2 : 0;
            cache_mock::reject_wait = variant == 1 ? cache_mock::waits + 1 : 0;
            struct rlimit original{}, limited{};
            const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
            CHECK(getrlimit(RLIMIT_FSIZE, &original) == 0, "get file-size limit");
            limited = original; limited.rlim_cur = 4096;
            if (variant == 2) { CHECK(setrlimit(RLIMIT_FSIZE, &limited) == 0, "plant a partial write followed by EFBIG"); }
            {
                llama_kpack_cache cache(target, gguf_path, inventory);
                cache.capture(tensor);
                cache.start();
            }
            if (variant == 2) { CHECK(setrlimit(RLIMIT_FSIZE, &original) == 0, "restore file-size limit"); }
            std::signal(SIGXFSZ, old_signal);
            CHECK(cache_mock::in_flight == 0, "failure drains prefetched DMA before releasing pinned slots");
            CHECK(stat(target.c_str(), &st) != 0, "failed pipeline must not publish");
            const std::string staging = target + ".partial." + std::to_string((long) getpid());
            CHECK(stat(staging.c_str(), &st) != 0, "failed pipeline removes its owned partial cache");
        }
        CHECK(cache_mock::violations == 0, "no early reuse/free, owner-thread wait or unbounded staging");
        ggml_backend_buffer_free(tensor->buffer);
        tensor->buffer = nullptr;
        ggml_backend_buffer_free(other->buffer);
        other->buffer = nullptr;
    }
    rm_bundle(model_cache);

    // ---- negatives: one wrong thing each ----
    printf("  [negatives]\n");
    {   // Unchecked runtime loading deliberately does NOT detect payload corruption.
        const std::string c = dir + "/trusted-corrupt"; copy_bundle(streamed, c);
        auto w = read_file(c + "/weights.bin"); w[1000] ^= 1; write_file(c + "/weights.bin", w);
        llama_kpack_sidecar_reader r;
        CHECK(r.open(c, err) && r.load_unchecked(gguf_path, inventory, err), "trusted cache must not secretly hash payloads");
        w.pop_back(); write_file(c + "/weights.bin", w);
        llama_kpack_sidecar_reader truncated;
        CHECK(!truncated.open(c, err), "truncation must still fail the structural size check");
        rm_bundle(c);
    }
    {   // Cheap local identity still rejects a different source inode.
        const std::string other = dir + "/local-other.gguf"; write_file(other, read_file(gguf_path));
        llama_kpack_sidecar_reader r;
        CHECK(r.open(streamed, err) && !r.load_unchecked(other, inventory, err), "local cache binds to its original source file");
        auto wrong = inventory; wrong[0].data_offset += 32;
        CHECK(!r.load_unchecked(gguf_path, wrong, err), "unchecked loading retains the tensor inventory contract");
        unlink(other.c_str());
    }
    {   // a flipped byte inside a span
        const std::string c = dir + "/c1"; copy_bundle(bundle, c);
        auto w = read_file(c + "/weights.bin"); w[1000] ^= 0x01; write_file(c + "/weights.bin", w);
        llama_kpack_sidecar_reader r;
        CHECK(r.open(c, err) && !r.verify_storage(4, err), "flipped weights byte must fail storage verification (%s)", err.c_str());
        CHECK(r.load_unchecked(gguf_path, inventory, err), "runtime can reuse old v3 bundles without hashing them");
        rm_bundle(c);
    }
    {   // an extra file in the root
        const std::string c = dir + "/c2"; copy_bundle(bundle, c); write_file(c + "/extra", { 1 });
        llama_kpack_sidecar_reader r;
        CHECK(!r.open(c, err), "extra root entry must be refused");
        rm_bundle(c);
    }
    {   // a duplicate key in the manifest
        const std::string c = dir + "/c3"; copy_bundle(bundle, c);
        auto m = read_file(c + "/manifest.json"); std::string s(m.begin(), m.end());
        s.insert(s.find("\"schema\""), "\"schema\": \"x\",\n  "); write_file(c + "/manifest.json", std::vector<uint8_t>(s.begin(), s.end()));
        llama_kpack_sidecar_reader r;
        CHECK(!r.open(c, err), "duplicate manifest key must be refused");
        rm_bundle(c);
    }
    {   // the wrong GGUF: one byte of tensor data changed
        auto gb = read_file(gguf_path); gb[dense.src.data_offset + 77] ^= 0x80;
        const std::string other = dir + "/other.gguf"; write_file(other, gb);
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err) && !r.verify_source(other, inventory, 4, err), "a modified source must fail (%s)", err.c_str());
        unlink(other.c_str());
    }
    {   // an inventory that disagrees on offset
        auto inv2 = inventory; inv2[0].data_offset += 32;
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err) && !r.verify_source(gguf_path, inv2, 4, err), "inventory/offset disagreement must fail");
    }

    // ---- quactlize's own validator on the same bundle ----
    printf("  [interop]\n");
    const char * py = getenv("KPACK_PY"); const char * repo = getenv("KPACK_PY_REPO");
    if (py && repo) {
        auto run = [&](const std::string & b, const std::string & src) {
            // No pipe: system() returns the LAST command's status, and a `| tail` would make every run look like 0.
            std::string cmd = "cd '" + std::string(repo) + "' && '" + py + "' -c \"from quactlize.pack_gguf import load_kpack_bundle; "
                "b=load_kpack_bundle('" + b + "', source='" + src + "'); print('PY-OK', len(b.artifacts))\" > /dev/null 2>&1";
            return system(cmd.c_str());
        };
        CHECK(run(bundle, gguf_path) == 0, "quactlize load_kpack_bundle must accept the bundle");
        CHECK(run(streamed, gguf_path) != 0, "runtime cache must not masquerade as a verified offline bundle");
        const std::string c = dir + "/c4"; copy_bundle(bundle, c);
        auto w = read_file(c + "/weights.bin"); w[2000] ^= 0x01; write_file(c + "/weights.bin", w);
        CHECK(run(c, gguf_path) != 0, "quactlize must reject the flipped-byte copy (the oracle can say no)");
        rm_bundle(c);
    } else {
        printf("    (KPACK_PY / KPACK_PY_REPO not set: quactlize cross-check skipped)\n");
    }

    rm_bundle(streamed);
    rm_bundle(bundle);
    unlink(gguf_path.c_str()); rmdir(dir.c_str());
    gguf_free(g); ggml_free(gctx);
    printf("test-kpack-sidecar: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
