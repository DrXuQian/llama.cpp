#include "../src/llama-kpack-cache.h"
#include "../src/llama-mmap.h"
#include "../ggml/src/ggml-cuda/ncp-quactlize-lib.h"

#include <nlohmann/json.hpp>
#include <atomic>
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
                jobs.push_back({src, planes, [data](size_t off, size_t bytes, std::string &) {
                                    check(off <= data->size() && bytes <= data->size() - off && bytes <= 4096,
                                          "chunk bounds");
                                    return data->data() + off;
                                }});
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

int main() try {
    fixture     f;
    std::string error;
    test_cache_directory(f);
    test_prefetch_coverage();
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
