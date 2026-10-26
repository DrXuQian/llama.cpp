#include "../src/llama-kpack-cache.h"
#include "../src/llama-mmap.h"
#include "../ggml/src/ggml-cuda/ncp-quactlize-lib.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"

#include <nlohmann/json.hpp>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace fs = std::filesystem;
using json   = nlohmann::json;

static void check(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

#ifdef LLAMA_KPACK_TEST_FSYNC
static bool                  trace_sync = false;
static size_t                fail_sync = 0, interrupt_sync = 0;
static std::vector<fs::path> sync_paths;
static std::vector<uint64_t> sync_sizes;

extern "C" int __real_fsync(int fd);
extern "C" int __wrap_fsync(int fd);

extern "C" int __wrap_fsync(int fd) {
    if (trace_sync) {
        const auto path = fs::read_symlink("/proc/self/fd/" + std::to_string(fd));
        sync_paths.push_back(path);
        sync_sizes.push_back(fs::is_regular_file(path) ? fs::file_size(path) : 0);
        if (sync_paths.size() == fail_sync || sync_paths.size() == interrupt_sync) {
            errno = sync_paths.size() == interrupt_sync ? EINTR : EIO;
            return -1;
        }
    }
    return __real_fsync(fd);
}
#endif

struct fixture {
    fs::path                                           root, source;
    std::vector<llama_kpack_source_tensor>             inventory;
    std::vector<llama_kpack_write_job>                 jobs;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> payloads;

    fixture() {
        root = fs::current_path() /
               ("kpack-sidecar." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(fs::create_directory(root), "create test directory");
        source          = root / "source.gguf";
        uint64_t offset = 128;
        for (int q : {8, 10, 11, 12, 13, 14}) {
            for (int experts : {0, 3}) {
                const int                 bits     = q == 8 ? 8 : q <= 11 ? 2 : 4;
                const int                 high     = q == 11 || q == 13 ? 1 : q == 14 ? 2 : 0;
                const size_t              elements = 256 * 512 * (experts ? experts : 1);
                llama_kpack_source_tensor src;
                src.name        = "q" + std::to_string(q) + ".e" + std::to_string(experts);
                src.gguf_index  = inventory.size();
                src.ggml_type   = q;
                src.n           = 256;
                src.k           = 512;
                src.experts     = experts;
                src.rank        = experts ? 3 : 2;
                src.data_offset = offset;
                src.size_bytes  = ggml_row_size((ggml_type) q, 512) * 256 * (experts ? experts : 1);
                offset += src.size_bytes;
                llama_kpack_planes planes;
                planes.bits        = bits;
                planes.high_bits   = high;
                planes.layout      = q == 8 ? 4 : q == 12 ? 1 : 2;
                planes.group_size  = q == 10 || q == 11 || q == 14 ? 16 : 32;
                planes.mapping_id  = q == 8  ? QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID :
                                     q == 12 ? QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID :
                                               QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID;
                planes.low_bytes   = elements * bits / 8;
                planes.high_bytes  = elements * high / 8;
                planes.units_bytes = src.size_bytes - planes.low_bytes - planes.high_bytes;
                auto data          = std::make_shared<std::vector<uint8_t>>(src.size_bytes);
                for (size_t i = 0; i < data->size(); ++i) {
                    (*data)[i] = (i * 17 + q) % 256;
                }
                jobs.push_back({src,
                                planes,
                                [data](size_t off, size_t bytes, std::string &) {
                                    check(off <= data->size() && bytes <= data->size() - off && bytes <= 4096,
                                          "chunk bounds");
                                    return data->data() + off;
                                },
                                {}});
                inventory.push_back(src);
                payloads.push_back(data);
            }
        }
        std::ofstream out(source, std::ios::binary);
        out.seekp(offset - 1);
        out.put(0);
        check(bool(out), "source fixture");
    }

    ~fixture() {
        // Only this test's newly created directory; remove_all does not follow symlinks.
        std::error_code error;
        fs::remove_all(root, error);
    }

    void write(const fs::path & dir) {
        std::string                   error;
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), source.u8string(), error), error);
        check(writer.start(jobs, inventory, 4096, error), error);
        check(writer.wait(error), error);
    }

    void verify(const fs::path & dir) {
        std::string                error;
        llama_kpack_sidecar_reader reader;
        check(reader.open(dir.u8string(), error), error);
        check(reader.load_unchecked(source.u8string(), inventory, error), error);
        check(reader.size() == jobs.size(), "incomplete inventory");
        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto * rec = reader.find(inventory[i].name);
            check(rec && rec->planes.low && (rec->planes.high != nullptr) == (jobs[i].planes.high_bytes != 0),
                  "planes");
            check(memcmp(rec->planes.low, payloads[i]->data(), payloads[i]->size()) == 0, "roundtrip");
        }
    }

    std::vector<fs::path> staging(const std::string & name) const {
        std::vector<fs::path> result;
        for (const auto & entry : fs::directory_iterator(root)) {
            if (entry.path().filename().u8string().find(name + ".partial.") == 0) {
                result.push_back(entry.path());
            }
        }
        return result;
    }
};

static int set_env(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "");
#else
    return value ? setenv(name, value, 1) : unsetenv(name);
#endif
}

struct scoped_env {
    const char *      name;
    const bool        present;
    const std::string value;

    explicit scoped_env(const char * name) :
        name(name),
        present(getenv(name) != nullptr),
        value(present ? getenv(name) : "") {}

    ~scoped_env() { set_env(name, present ? value.c_str() : nullptr); }
};

static void test_cache_directory(fixture & f) {
    scoped_env configured("LLAMA_KPACK_CACHE_DIR");
#ifdef _WIN32
    scoped_env tmp("TMP");
    const auto root = f.root / "llama-kpack";
#else
    scoped_env tmp("TMPDIR");
    const auto root = f.root / ("llama-kpack-" + std::to_string(geteuid()));
#endif
    check(set_env(configured.name, nullptr) == 0 && set_env(tmp.name, f.root.u8string().c_str()) == 0,
          "default environment");
    std::string    error;
    const fs::path automatic = fs::u8path(llama_kpack_cache_directory(f.source.u8string(), error));
    check(!automatic.empty() && automatic.parent_path() == root, "default cache directory: " + error);

#ifndef _WIN32
    struct stat st {};

    check(lstat(root.c_str(), &st) == 0 && st.st_uid == geteuid() && (st.st_mode & 0077) == 0,
          "default cache is not private");
#endif
    check(llama_kpack_cache_directory(f.source.u8string(), error) == automatic.u8string(), "unstable model key");
    f.write(automatic);
    f.verify(fs::u8path(llama_kpack_cache_directory(f.source.u8string(), error)));

    const auto custom = f.root / "custom";
    check(set_env(configured.name, (custom.u8string() + "/").c_str()) == 0, "custom environment");
    const fs::path overridden = fs::u8path(llama_kpack_cache_directory(f.source.u8string(), error));
    check(!overridden.empty() && overridden.parent_path() == custom && overridden.filename() == automatic.filename(),
          "cache root override: " + error);
    fs::create_hard_link(f.source, f.root / "alias.gguf");
    // Creating a hard link updates ctime, so compare both names after the link exists.
    check(llama_kpack_cache_directory((f.root / "alias.gguf").u8string(), error) ==
              llama_kpack_cache_directory(f.source.u8string(), error),
          "source alias changed key");
    fs::create_directory(f.root / "another");
    const auto other = f.root / "another/source.gguf";
    fs::copy_file(f.source, other);
    const auto other_key = llama_kpack_cache_directory(other.u8string(), error);
    check(!other_key.empty() && other_key != overridden.u8string(), "different models share a key");
    fs::resize_file(other, fs::file_size(other) + 1);
    const auto changed_key = llama_kpack_cache_directory(other.u8string(), error);
    check(!changed_key.empty() && changed_key != other_key, "changed source reused key");

#ifndef _WIN32
    const auto link = f.root / "cache-link";
    fs::create_directory_symlink(custom, link);
    check(set_env(configured.name, (link.u8string() + "/").c_str()) == 0, "symlink environment");
    check(llama_kpack_cache_directory(f.source.u8string(), error).empty(), "symlink cache root accepted");
    check(set_env(configured.name, custom.u8string().c_str()) == 0 && chmod(custom.c_str(), 0777) == 0,
          "shared cache root");
    check(llama_kpack_cache_directory(f.source.u8string(), error).empty(), "shared writable cache root accepted");
    check(chmod(custom.c_str(), 0700) == 0, "restore private cache root");
#endif
    check(set_env(configured.name, f.source.u8string().c_str()) == 0, "file environment");
    check(llama_kpack_cache_directory(f.source.u8string(), error).empty(), "cache root file accepted");
    check(set_env(configured.name, (f.root / "unused").u8string().c_str()) == 0, "missing source environment");
    check(llama_kpack_cache_directory((f.root / "missing.gguf").u8string(), error).empty() &&
              !fs::exists(f.root / "unused"),
          "missing source created cache directory");
}

static void test_prefetch_coverage() {
    const auto prefetch = [](uint64_t source, uint64_t cached) {
        return llama_kpack_cache_coverage{source, cached}.should_prefetch();
    };
    check(prefetch(0, 0), "empty load disabled prefetch");
    check(prefetch(1000, 0), "cache miss disabled prefetch");
    check(prefetch(1000, 1), "one cached byte disabled prefetch");
    check(prefetch(1000, 949), "partial cache disabled prefetch");
    check(!prefetch(1000, 950) && !prefetch(1000, 1000), "covered load kept whole-file prefetch");
    check(prefetch(19, 18) && !prefetch(20, 19), "fractional coverage rounded down");
    check(prefetch(2000, 1000), "uncached CPU or ordinary GPU bytes were ignored");
    check(prefetch(UINT64_MAX, 0) && !prefetch(UINT64_MAX, UINT64_MAX), "coverage overflow");
}

#ifdef LLAMA_KPACK_TEST_FSYNC
static void test_sync(fixture & f) {
    // Fail each barrier separately, then retry an interrupted file sync.
    for (size_t fault = 0; fault <= 6; ++fault) {
        const auto                    dir = f.root / ("sync-" + std::to_string(fault));
        std::string                   error;
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), f.source.u8string(), error), error);
        const auto staging = f.staging(dir.filename().u8string());
        check(staging.size() == 1, "sync fixture staging");
        auto expected = std::vector<fs::path>{staging[0] / "weights.bin", staging[0] / "manifest.json", staging[0], dir,
                                              dir.parent_path()};
        fail_sync     = fault <= 5 ? fault : 0;
        interrupt_sync = fault == 6 ? 1 : 0;
        sync_paths.clear();
        sync_sizes.clear();
        trace_sync = true;
        check(writer.start({f.jobs.front()}, f.inventory, 4096, error), error);
        const bool ok = writer.wait(error);
        trace_sync    = false;
        check(ok == (fail_sync == 0), "sync error was not propagated");
        if (fail_sync) {
            check(!error.empty(), "sync error lacks context");
            expected.resize(fail_sync);
        } else if (interrupt_sync) {
            expected.insert(expected.begin(), staging[0] / "weights.bin");
        }
        check(sync_paths == expected, "cache sync order differs");
        check(sync_sizes.front() == f.payloads.front()->size(), "weights were not flushed before sync");
        const size_t manifest_sync = interrupt_sync ? 2 : 1;
        if (sync_sizes.size() > manifest_sync) {
            check(sync_sizes[manifest_sync] > 0, "manifest was not flushed before sync");
        }
        check(f.staging(dir.filename().u8string()).empty(), "sync failure leaked staging");
        if (fail_sync && fail_sync <= 3) {
            check(!fs::exists(dir), "cache published before staging was synced");
        } else {
            // A post-publication sync error must not erase an already visible cache.
            const auto count = sync_paths.size();
            trace_sync       = true;
            llama_kpack_sidecar_reader reader;
            check(reader.open(dir.u8string(), error) && reader.load_unchecked(f.source.u8string(), f.inventory, error),
                  error);
            const auto * rec = reader.find(f.inventory.front().name);
            check(reader.size() == 1 && rec &&
                      memcmp(rec->planes.low, f.payloads.front()->data(), f.payloads.front()->size()) == 0,
                  "synced cache payload differs");
            trace_sync = false;
            check(sync_paths.size() == count, "cache loading performs sync");
        }
    }
    fail_sync = interrupt_sync = 0;
    puts("KPACK_SIDECAR_SYNC PASS order=weights+manifest+staging+target+parent failures=5 retry=EINTR load_syncs=0");
}
#endif

struct tp_weight_device {
    ggml_backend_device        device{};
    ggml_backend_buffer_type   normal{}, packed{}, other{};
    ggml_backend_buffer_type_t extras[3]{};
    ggml_tensor                observed{};
    bool                       enabled = true, alternate = false, supported = true;
    size_t                     calls = 0;
};

static ggml_backend_buffer_type_t tp_weight_preferred(ggml_backend_dev_t dev, const ggml_tensor * weight, ggml_op op) {
    auto & state = *static_cast<tp_weight_device *>(dev->context);
    ++state.calls;
    state.observed = *weight;
    if (!state.enabled || op != GGML_OP_MUL_MAT || weight->ne[2] != 1 || weight->ne[3] != 1 || weight->ne[0] % 512 ||
        weight->ne[1] % 256 ||
        (weight->type != GGML_TYPE_Q8_0 && (weight->type < GGML_TYPE_Q2_K || weight->type > GGML_TYPE_Q6_K))) {
        return nullptr;
    }
    check(weight->buffer == nullptr && weight->data == nullptr && ggml_is_contiguous(weight), "TP admission metadata");
    return state.alternate ? &state.other : &state.packed;
}

static void test_tp_weight_selection(
    ggml_backend_dev_get_weight_buft_t preferred = ggml_backend_meta_device_get_weight_buft) {
    static tp_weight_device              states[GGML_BACKEND_META_MAX_DEVICES];
    static ggml_backend_meta_split_state split{};
    static ggml_backend_reg              reg{};
    reg.api_version            = GGML_BACKEND_API_VERSION;
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (strcmp(name, "ggml_backend_dev_get_weight_buft") == 0) {
            return reinterpret_cast<void *>(tp_weight_preferred);
        }
        if (strcmp(name, "ggml_backend_dev_get_override_bufts") == 0) {
            return reinterpret_cast<void *>(+[](ggml_backend_dev_t dev) -> ggml_backend_buffer_type_t * {
                return static_cast<tp_weight_device *>(dev->context)->extras;
            });
        }
        return nullptr;
    };
    std::vector<ggml_backend_dev_t> devices;
    for (auto & state : states) {
        state.device.context        = &state;
        state.device.reg            = &reg;
        state.device.iface.get_name = [](ggml_backend_dev_t) {
            return "tp-weight-host";
        };
        state.device.iface.get_description = state.device.iface.get_name;
        state.device.iface.get_type        = [](ggml_backend_dev_t) {
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        };
        state.device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<tp_weight_device *>(dev->context)->normal;
        };
        state.device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
            return buft->device == dev && static_cast<tp_weight_device *>(dev->context)->supported;
        };
        state.device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) {
            return true;
        };
        for (auto * buft : {&state.normal, &state.packed, &state.other}) {
            *buft        = *ggml_backend_cpu_buffer_type();
            buft->device = &state.device;
        }
        state.packed.iface.get_name = [](ggml_backend_buffer_type_t) {
            return "preferred-host";
        };
        state.other.iface.get_name = [](ggml_backend_buffer_type_t) {
            return "other-host";
        };
        state.extras[0] = &state.packed;
        state.extras[1] = &state.other;
        devices.push_back(&state.device);
    }
    size_t cases = 0;
    for (int count : {2, 3, 4, 8, 16}) {
        auto * dev = ggml_backend_meta_device(
            devices.data(), count,
            [](const ggml_tensor *, void * data) { return *static_cast<ggml_backend_meta_split_state *>(data); },
            &split);
        check(ggml_backend_dev_backend_reg(dev) == nullptr, "test must exercise registry-less Meta");
        for (int q : {8, 10, 11, 12, 13, 14}) {
            for (int axis : {0, 1, 10}) {
                for (int pattern = 0; pattern < (axis == 10 ? 1 : 3); ++pattern) {
                    split            = {};
                    split.axis       = static_cast<ggml_backend_meta_split_axis>(axis);
                    split.n_segments = pattern == 2 ? 2 : 1;
                    int64_t ne[4]    = {512, 256, 1, 1};
                    int64_t total    = 0;
                    for (size_t s = 0; s < split.n_segments; ++s) {
                        split.nr[s] = pattern == 2 && s == 0 ? 2 : 1;
                        for (int rank = 0; rank < count; ++rank) {
                            const int factor           = pattern == 1 ? (rank == count - 1 ? 0 : rank + 1) : 1;
                            split.ne[s * count + rank] = (axis == 0 ? 512 : 256) * factor;
                            total += split.ne[s * count + rank] * split.nr[s];
                        }
                    }
                    if (axis != 10) {
                        ne[axis] = total;
                    }
                    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                    auto *           weight = ggml_new_tensor(ctx.get(), static_cast<ggml_type>(q), 4, ne);
                    ggml_set_name(weight, "tp-auto-weight");
                    ggml_set_param(weight);
                    const auto original = *weight;
                    for (auto & state : states) {
                        state.calls = 0;
                    }
                    auto * selected = preferred(dev, weight, GGML_OP_MUL_MAT);
                    check(selected && selected != ggml_backend_dev_buffer_type(dev),
                          "automatic TP storage not selected");
                    check(memcmp(weight, &original, sizeof(original)) == 0, "admission changed source metadata");
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), selected));
                    check(bool(buffer), "selected TP allocation");
                    for (int rank = 0; rank < count; ++rank) {
                        const auto * local = ggml_backend_meta_tensor_shard(weight, rank, nullptr);
                        check(local && ggml_backend_meta_buffer_type_at(selected, rank) == &states[rank].packed,
                              "selected family differs by rank");
                        if (ggml_nelements(local) == 0) {
                            check(states[rank].calls == 0, "empty shard queried for compute");
                        } else {
                            const auto & observed = states[rank].observed;
                            check(states[rank].calls == 1 && memcmp(local->ne, observed.ne, sizeof(local->ne)) == 0 &&
                                      memcmp(local->nb, observed.nb, sizeof(local->nb)) == 0,
                                  "queried geometry differs from allocated shard");
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    auto * dev = ggml_backend_meta_device(
        devices.data(), 2,
        [](const ggml_tensor *, void * data) { return *static_cast<ggml_backend_meta_split_state *>(data); }, &split);
    split = {GGML_BACKEND_SPLIT_AXIS_0, {512, 512}, {1}, 1};
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 3, nullptr, true}));
    auto *           weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 1024, 256);
    auto             query  = [&] {
        return preferred(dev, weight, GGML_OP_MUL_MAT);
    };
    check(query() != nullptr, "valid TP control");
    for (int rank : {0, 1}) {
        states[rank].enabled = false;
        check(!query(), "disabled rank selected packed storage");
        states[rank].enabled    = true;
        states[rank].device.reg = nullptr;
        check(!query(), "missing rank registry selected packed storage");
        states[rank].device.reg = &reg;
        states[rank].supported  = false;
        check(!query(), "unsupported rank buffer selected packed storage");
        states[rank].supported = true;
    }
    states[1].alternate = true;
    check(!query(), "different per-rank buffer families accepted");
    states[1].alternate = false;
    split.ne[0]         = 256;
    split.ne[1]         = 768;
    check(!query(), "globally supported but locally unsupported K accepted");
    split.ne[0] = 128;
    split.ne[1] = 896;
    check(!query(), "GGUF block split accepted");
    split.ne[0] = 512;
    split.ne[1] = 256;
    check(!query(), "incomplete split map accepted");
    split.ne[1] = 512;
    split.nr[0] = 0;
    check(!query(), "zero segment repetition accepted");
    split.nr[0]      = 1;
    split.n_segments = 17;
    check(!query(), "oversized split map accepted");
    split.n_segments = 1;
    split.axis       = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    check(!query(), "partial static weight accepted");
    split.axis = GGML_BACKEND_SPLIT_AXIS_0;
    check(!preferred(dev, weight, GGML_OP_MUL_MAT_ID), "grouped automatic placement enabled");
    auto * bf16 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 1024, 256);
    check(!preferred(dev, bf16, GGML_OP_MUL_MAT), "BF16 native control selected packed storage");
    auto * view = ggml_view_tensor(ctx.get(), weight);
    check(!preferred(dev, view, GGML_OP_MUL_MAT), "weight view accepted");
    check(query() != nullptr, "negative tests changed valid admission");
    printf("KPACK_TP_AUTO_HOST PASS cases=%zu devices=2,3,4,8,16 actual_shards=MATCH fallbacks=PASS\n", cases);
}

static void test_tp_raw_splits() {
    static ggml_backend_device           devices[GGML_BACKEND_META_MAX_DEVICES]{};
    static ggml_backend_meta_split_state split{};
    std::vector<ggml_backend_dev_t>      simple;
    for (auto & device : devices) {
        device.iface.get_name = [](ggml_backend_dev_t) {
            return "test-host";
        };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_buffer_type = [](ggml_backend_dev_t) {
            return ggml_backend_cpu_buffer_type();
        };
        simple.push_back(&device);
    }
    size_t cases = 0;
    for (int q : {0, 1, 8, 10, 11, 12, 13, 14}) {
        for (int count : {2, 3, 4, 8, 16}) {
            auto * device = ggml_backend_meta_device(
                simple.data(), count,
                [](const ggml_tensor *, void * data) { return *static_cast<ggml_backend_meta_split_state *>(data); },
                &split);
            auto * buft = ggml_backend_dev_buffer_type(device);
            check(ggml_backend_meta_buffer_type_count(buft) == size_t(count), "TP buffer count");
            for (int axis : {0, 1, 2, 10}) {
                for (int pattern = 0; pattern < 3; ++pattern) {
                    if ((axis == 2 && pattern == 2) || (axis == 10 && pattern != 0)) {
                        continue;
                    }
                    split        = {};
                    split.axis   = static_cast<ggml_backend_meta_split_axis>(axis);
                    int64_t ne[] = {512, 4, 2, 2};
                    if (pattern == 2) {
                        ne[3] = 1;
                        if (axis == 0) {
                            ne[2] = 1;
                        }
                    }
                    std::vector<std::vector<int64_t>> coordinates(count);
                    if (axis < GGML_MAX_DIMS) {
                        const int unit   = axis == 0 ? 512 : 1;
                        split.n_segments = pattern == 2 ? 2 : 1;
                        int64_t position = 0;
                        for (size_t s = 0; s < split.n_segments; ++s) {
                            split.nr[s] = pattern == 2 && s == 0 ? 2 : 1;
                            for (int d = 0; d < count; ++d) {
                                split.ne[s * count + d] = unit * (pattern == 1 ? (d == count - 1 ? 0 : d + 1) : 1);
                            }
                            for (size_t r = 0; r < split.nr[s]; ++r) {
                                for (int d = 0; d < count; ++d) {
                                    for (int64_t j = 0; j < split.ne[s * count + d]; ++j) {
                                        coordinates[d].push_back(position++);
                                    }
                                }
                            }
                        }
                        ne[axis] = position;
                    } else {
                        split.n_segments = split.nr[0] = 1;
                    }
                    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                    auto *           tensor =
                        ggml_new_tensor_4d(ctx.get(), static_cast<ggml_type>(q), ne[0], ne[1], ne[2], ne[3]);
                    ggml_set_param(tensor);
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
                    check(bool(buffer), "TP raw allocation");
                    std::vector<uint8_t> raw(ggml_nbytes(tensor));
                    for (size_t i = 0; i < raw.size(); ++i) {
                        raw[i] = (i * 11 + i / 17 + i / 256) % 251;
                    }
                    // Keep the native uploader's row-aligned range contract.
                    const size_t alignment = axis == 10 ? 1 : tensor->nb[axis + 1];
                    const size_t cut       = raw.size() > alignment ? alignment : 0;
                    ggml_backend_tensor_set(tensor, raw.data() + cut, cut, raw.size() - cut);
                    if (cut) {
                        ggml_backend_tensor_set(tensor, raw.data(), 0, cut);
                    }
                    for (int d = 0; d < count; ++d) {
                        auto * local = ggml_backend_meta_tensor_shard(tensor, d, nullptr);
                        check(local != nullptr, "missing raw shard");
                        std::vector<uint8_t> got(ggml_nbytes(local)), expected;
                        if (axis == 10) {
                            expected = raw;
                        } else {
                            const size_t block  = axis == 0 ? ggml_blck_size(tensor->type) : 1;
                            const size_t stride = tensor->nb[axis];
                            const size_t chunk  = stride * (tensor->ne[axis] / block);
                            for (size_t outer = 0; outer < raw.size(); outer += chunk) {
                                for (size_t x = 0; x < coordinates[d].size(); x += block) {
                                    const size_t begin = outer + coordinates[d][x] / block * stride;
                                    expected.insert(expected.end(), raw.begin() + begin, raw.begin() + begin + stride);
                                }
                            }
                        }
                        if (!got.empty()) {
                            ggml_backend_tensor_get(local, got.data(), 0, got.size());
                        }
                        check(got == expected, "raw GGUF partition differs from coordinate oracle");
                    }
                    ++cases;
                }
            }
        }
    }
    printf(
        "KPACK_TP_RAW PASS cases=%zu devices=2,3,4,8,16 axes=K,N,E,mirrored "
        "patterns=even,uneven-empty,segments-repeated\n",
        cases);
}

static void test_tp_sidecars(fixture & f) {
    std::string error;
    size_t      cases = 0;
    for (int count : {2, 3, 4, 8}) {
        for (int axis : {0, 1, 2, 3, 10}) {
            auto src        = f.inventory[7];  // Q4 grouped
            src.gguf_index  = 0;
            src.data_offset = 0;
            src.k           = 512;
            src.n           = 256;
            src.experts     = 2;
            src.rank        = axis == 3 ? 4 : 3;
            src.batches     = axis == 3 ? 2 : 1;
            llama_kpack_partition p;
            p.count = count;
            p.axis  = axis;
            if (axis < 4) {
                const int64_t unit = axis == 0 ? 512 : axis == 1 ? 256 : 1;
                p.widths.assign(count, unit);
                p.repeats = {1};
                if (axis == 0) {
                    src.k = unit * count;
                }
                if (axis == 1) {
                    src.n = unit * count;
                }
                if (axis == 2) {
                    src.experts = unit * count;
                }
                if (axis == 3) {
                    src.batches = unit * count;
                }
            }
            src.size_bytes    = ggml_row_size(GGML_TYPE_Q4_K, src.k) * src.n * src.experts * src.batches;
            const auto source = f.root / ("tp-source-" + std::to_string(cases));
            {
                std::ofstream out(source, std::ios::binary);
                out.seekp(src.size_bytes - 1);
                out.put(0);
                check(bool(out), "TP source fixture");
            }
            std::vector<llama_kpack_write_job> jobs;
            for (int d = 0; d < count; ++d) {
                p.index = d;
                llama_kpack_source_tensor local;
                check(llama_kpack_shard_source(src, p, local, error), error);
                auto planes        = f.jobs[7].planes;
                planes.low_bytes   = uint64_t(local.k) * local.n * local.experts * local.batches / 2;
                planes.high_bytes  = 0;
                planes.units_bytes = local.size_bytes - planes.low_bytes;
                auto bytes         = std::make_shared<std::vector<uint8_t>>(local.size_bytes, uint8_t(d + 1));
                jobs.push_back(
                    {src, planes, [bytes](size_t off, size_t, std::string &) { return bytes->data() + off; }, p});
            }
            const auto                    dir = f.root / ("tp-cache-" + std::to_string(cases));
            llama_kpack_background_writer writer;
            check(writer.prepare(dir.u8string(), source.u8string(), error), error);
            check(writer.start(jobs, {src}, 4096, error) && writer.wait(error), error);
            llama_kpack_sidecar_reader reader;
            check(reader.open(dir.u8string(), error) && reader.load_unchecked(source.u8string(), {src}, error), error);
            check(reader.size() == size_t(count) && !reader.find(src.name), "TP cache accepted as unsplit");
            for (int d = 0; d < count; ++d) {
                p.index          = d;
                const auto * rec = reader.find(src.name, p);
                check(rec && rec->planes.low[0] == d + 1 && rec->planes.units[0] == d + 1, "wrong TP shard bytes");
                auto wrong = p;
                ++wrong.count;
                check(!reader.find(src.name, wrong), "different TP count accepted");
                if (!p.widths.empty()) {
                    wrong = p;
                    ++wrong.widths[0];
                    check(!reader.find(src.name, wrong), "different split map accepted");
                    wrong = p;
                    ++wrong.repeats[0];
                    check(!reader.find(src.name, wrong), "different repeat map accepted");
                }
            }
            auto wrong = src;
            ++wrong.batches;
            check(!reader.load_unchecked(source.u8string(), {wrong}, error), "different source geometry accepted");
            if (axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                auto changed                                = json::parse(std::ifstream(dir / "manifest.json"));
                changed["tensors"][1]["partition"]["count"] = count + 1;
                {
                    std::ofstream out(dir / "manifest.json");
                    out << changed.dump();
                }
                llama_kpack_sidecar_reader mixed;
                check(!mixed.open(dir.u8string(), error), "mixed split maps accepted");
                auto invalid = jobs;
                ++invalid[1].partition.count;
                llama_kpack_background_writer bad;
                const auto                    bad_dir = f.root / ("mixed-tp-" + std::to_string(count));
                check(bad.prepare(bad_dir.u8string(), source.u8string(), error), error);
                check(bad.start(invalid, {src}, 4096, error) && !bad.wait(error), "mixed split maps published");
                check(!fs::exists(bad_dir), "invalid TP cache published");
            }
            ++cases;
        }
    }
    printf("KPACK_TP_SIDECAR PASS cases=%zu cross_topology=MISS rank4=PASS\n", cases);
}

int main() try {
    fixture     f;
    std::string error;
    test_cache_directory(f);
    test_tp_weight_selection();
    test_tp_raw_splits();
    test_tp_sidecars(f);
    test_prefetch_coverage();
#ifdef LLAMA_KPACK_TEST_FSYNC
    test_sync(f);
#endif
    const auto original = f.root / "complete";
    f.write(original);
    f.verify(original);
    {
        llama_kpack_sidecar_writer other;
        check(!other.begin(original.u8string(), error), "overwrite accepted");
        f.verify(original);
        fs::create_directory(f.root / "existing-empty");
        check(!other.begin((f.root / "existing-empty").u8string(), error), "empty directory overwritten");
        check(!other.begin(f.source.u8string(), error), "regular file overwritten");
#ifndef _WIN32
        fs::create_directory_symlink(f.root / "absent", f.root / "dangling");
        check(!other.begin((f.root / "dangling").u8string(), error), "dangling symlink overwritten");
#endif
    }
    {
        const auto                    dir = f.root / "partial";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), f.source.u8string(), error), error);
        check(writer.start({f.jobs.front()}, f.inventory, 4096, error), error);
        check(writer.wait(error), error);
        llama_kpack_sidecar_reader reader;
        check(reader.open(dir.u8string(), error) && reader.load_unchecked(f.source.u8string(), f.inventory, error),
              error);
        check(reader.size() == 1 && reader.find(f.inventory.front().name) && !reader.find(f.inventory.back().name),
              "partial cache inventory differs");
    }
    {
        const auto                    dir = f.root / "cancel";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), f.source.u8string(), error), error);
        auto              jobs = f.jobs;
        std::atomic<bool> entered{false}, cancel_called{false};
        jobs[0].read = [&](size_t, size_t, std::string & why) -> const uint8_t * {
            entered = true;
            while (!cancel_called) {
                std::this_thread::yield();
            }
            why = "injected read failure";
            return nullptr;
        };
        check(writer.start(jobs, f.inventory, 4096, error), error);
        while (!entered) {
            std::this_thread::yield();
        }
        cancel_called = true;
        writer.cancel();
        check(!fs::exists(dir), "cancel published");
        check(f.staging("cancel").empty(), "cancel leaked staging");
    }
    {
        const auto                 dir = f.root / "pending";
        llama_kpack_sidecar_writer owner;
        check(owner.begin(dir.u8string(), error), error);
        const auto staging = f.staging("pending");
        check(staging.size() == 1, "missing staging directory");
        {
            llama_kpack_sidecar_writer other;
            check(other.begin(dir.u8string(), error), error);
            check(f.staging("pending").size() == 2, "writers share staging directory");
        }
        check(f.staging("pending") == staging && fs::exists(staging[0] / "weights.bin"), "removed other writer");
    }
    check(f.staging("pending").empty(), "destructor leaked staging");
    {
        const auto                    dir = f.root / "concurrent";
        llama_kpack_background_writer a, b;
        check(a.prepare(dir.u8string(), f.source.u8string(), error), error);
        check(b.prepare(dir.u8string(), f.source.u8string(), error), error);
        check(a.start({f.jobs.front()}, f.inventory, 4096, error), error);
        check(b.start({f.jobs.back()}, f.inventory, 4096, error), error);
        std::string error_b;
        const bool  first = a.wait(error), second = b.wait(error_b);
        check(first != second, "concurrent publication must have exactly one winner");
        llama_kpack_sidecar_reader reader;
        check(reader.open(dir.u8string(), error) && reader.load_unchecked(f.source.u8string(), f.inventory, error),
              error);
        const size_t winner = first ? 0 : f.jobs.size() - 1;
        const auto * rec    = reader.find(f.inventory[winner].name);
        check(reader.size() == 1 && rec &&
                  memcmp(rec->planes.low, f.payloads[winner]->data(), f.payloads[winner]->size()) == 0,
              "mixed concurrent payloads");
        check(f.staging("concurrent").empty(), "publication leaked staging");
    }
    {
        llama_file                 loaded(f.source.u8string().c_str(), "rb");
        llama_kpack_sidecar_reader reader;
        check(reader.open(original.u8string(), error), error);
        check(reader.load_unchecked(f.source.u8string(), f.inventory, error, loaded.file_id()), error);
        const auto replacement = f.root / "replacement.gguf";
        fs::copy_file(f.source, replacement);
        fs::last_write_time(replacement, fs::last_write_time(f.source));
        llama_file different(replacement.u8string().c_str(), "rb");
        check(!reader.load_unchecked(f.source.u8string(), f.inventory, error, different.file_id()),
              "different loader file accepted");
        llama_kpack_sidecar_writer writer;
        check(!writer.bind_source(replacement.u8string(), error, loaded.file_id()), "loader file mismatch accepted");
    }
    const json manifest = json::parse(std::ifstream(original / "manifest.json"));
    for (const std::string kind : {"truncated", "offset", "overflow", "duplicate-key", "missing-manifest", "symlink"}) {
#ifdef _WIN32
        if (kind == "symlink") {
            continue;  // Creating symlinks may require elevated privileges on Windows.
        }
#endif
        const auto dir = f.root / kind;
        fs::create_directory(dir);
        fs::copy_file(original / "weights.bin", dir / "weights.bin");
        auto changed = manifest;
        if (kind == "truncated") {
            fs::resize_file(dir / "weights.bin", 1);
        }
        if (kind == "offset") {
            changed["tensors"][0]["spans"]["low"]["offset_bytes"] = 128;
        }
        if (kind == "overflow") {
            changed["tensors"][0]["n"] = uint64_t(-1);
        }
        if (kind != "missing-manifest") {
            std::ofstream out(dir / "manifest.json");
            const auto    text = changed.dump();
            out << (kind == "duplicate-key" ? "{\"schema\":\"duplicate\"," + text.substr(1) : text);
        }
        if (kind == "symlink") {
            fs::remove(dir / "weights.bin");
            fs::create_symlink(original / "weights.bin", dir / "weights.bin");
        }
        llama_kpack_sidecar_reader reader;
        check(!reader.open(dir.u8string(), error), kind + " accepted");
    }
    {
        llama_kpack_sidecar_reader reader;
        check(reader.open(original.u8string(), error), error);
        auto wrong = f.inventory;
        ++wrong[0].data_offset;
        check(!reader.load_unchecked(f.source.u8string(), wrong, error), "different inventory accepted");
        std::ofstream out(f.source, std::ios::binary | std::ios::app);
        out.put(0);
        out.close();
        check(!reader.load_unchecked(f.source.u8string(), f.inventory, error), "changed source accepted");
    }
    {
        const auto                    dir = f.root / "changed-during-write";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), f.source.u8string(), error), error);
        std::ofstream out(f.source, std::ios::binary | std::ios::app);
        out.put(0);
        out.close();
        check(writer.start(f.jobs, f.inventory, 4096, error), error);
        check(!writer.wait(error), "published after source changed");
        check(!fs::exists(dir), "invalid publication");
    }
#ifndef _WIN32
    {
        const auto                    dir = f.root / "replaced-during-write";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir.u8string(), f.source.u8string(), error), error);
        const auto old_time = fs::last_write_time(f.source);
        fs::copy_file(f.source, f.root / "next.gguf");
        fs::last_write_time(f.root / "next.gguf", old_time);
        fs::rename(f.root / "next.gguf", f.source);
        check(writer.start(f.jobs, f.inventory, 4096, error), error);
        check(!writer.wait(error) && !fs::exists(dir), "same-size same-mtime replacement accepted");
    }
#endif
    puts(
        "KPACK_SIDECAR PASS formats=6 dense=6 grouped=6 atomic=manifest-last concurrent=PASS directory=default+env "
        "prefetch=BYTES_95_PERCENT rejections=PASS");
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "KPACK_SIDECAR FAIL: %s\n", error.what());
    return 1;
}
