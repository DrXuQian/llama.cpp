#include "../src/llama-kpack-sidecar.h"
#include "../ggml/src/ggml-cuda/ncp-quactlize-lib.h"

#include <nlohmann/json.hpp>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json   = nlohmann::json;

static bool     manifest_last = false;
extern "C" long __real_syscall(long number, ...);
extern "C" long __wrap_syscall(long number, ...);

extern "C" long __wrap_syscall(long number, ...) {
    if (number != SYS_renameat2) {
        errno = ENOSYS;
        return -1;
    }
    if (manifest_last) {
        errno = EINVAL;
        return -1;
    }
    va_list args;
    va_start(args, number);
    int          a = va_arg(args, int);
    const char * b = va_arg(args, const char *);
    int          c = va_arg(args, int);
    const char * d = va_arg(args, const char *);
    int          e = va_arg(args, int);
    va_end(args);
    return __real_syscall(number, a, b, c, d, e);
}

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
        char         path[] = "kpack-sidecar.XXXXXX";
        const char * made   = mkdtemp(path);
        check(made != nullptr, "mkdtemp");
        root            = fs::absolute(made);
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
        // Only this test's mkdtemp directory; remove_all does not follow symlinks.
        std::error_code error;
        fs::remove_all(root, error);
    }

    void write(const fs::path & dir) {
        std::string                   error;
        llama_kpack_background_writer writer;
        check(writer.prepare(dir, source, error), error);
        check(writer.start(jobs, inventory, 4096, error), error);
        check(writer.wait(error), error);
    }

    void verify(const fs::path & dir) {
        std::string                error;
        llama_kpack_sidecar_reader reader;
        check(reader.open(dir, error), error);
        check(reader.load_unchecked(source, inventory, error), error);
        check(reader.size() == jobs.size(), "incomplete inventory");
        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto * rec = reader.find(inventory[i].name);
            check(rec && rec->planes.low && (rec->planes.high != nullptr) == (jobs[i].planes.high_bytes != 0),
                  "planes");
            check(memcmp(rec->planes.low, payloads[i]->data(), payloads[i]->size()) == 0, "roundtrip");
        }
    }
};

struct scoped_env {
    const char *      name;
    const bool        present;
    const std::string value;

    explicit scoped_env(const char * name) :
        name(name),
        present(getenv(name) != nullptr),
        value(present ? getenv(name) : "") {}

    ~scoped_env() {
        if (present) {
            setenv(name, value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }
};

static void test_cache_directory(fixture & f) {
    scoped_env configured("LLAMA_KPACK_CACHE_DIR"), tmp("TMPDIR");
    check(unsetenv(configured.name) == 0 && setenv(tmp.name, f.root.c_str(), 1) == 0, "default environment");
    std::string    error;
    const fs::path automatic = llama_kpack_cache_directory(f.source, error);
    const auto     root      = f.root / ("llama-kpack-" + std::to_string(geteuid()));
    check(!automatic.empty() && automatic.parent_path() == root, "default cache directory: " + error);

    struct stat st {};

    check(lstat(root.c_str(), &st) == 0 && st.st_uid == geteuid() && (st.st_mode & 0077) == 0,
          "default cache is not private");
    check(llama_kpack_cache_directory(f.source, error) == automatic, "unstable model key");
    f.write(automatic);
    f.verify(llama_kpack_cache_directory(f.source, error));

    const auto custom = f.root / "custom";
    check(setenv(configured.name, (custom.string() + "/").c_str(), 1) == 0, "custom environment");
    const fs::path overridden = llama_kpack_cache_directory(f.source, error);
    check(!overridden.empty() && overridden.parent_path() == custom && overridden.filename() == automatic.filename(),
          "cache root override: " + error);
    fs::create_symlink(f.source, f.root / "alias.gguf");
    check(llama_kpack_cache_directory(f.root / "alias.gguf", error) == overridden, "source alias changed key");
    fs::create_directory(f.root / "another");
    const auto other = f.root / "another/source.gguf";
    fs::copy_file(f.source, other);
    const auto other_key = llama_kpack_cache_directory(other, error);
    check(!other_key.empty() && other_key != overridden, "different models share a key");
    fs::resize_file(other, fs::file_size(other) + 1);
    const auto changed_key = llama_kpack_cache_directory(other, error);
    check(!changed_key.empty() && changed_key != other_key, "changed source reused key");

    const auto link = f.root / "cache-link";
    fs::create_directory_symlink(custom, link);
    check(setenv(configured.name, (link.string() + "/").c_str(), 1) == 0, "symlink environment");
    check(llama_kpack_cache_directory(f.source, error).empty(), "symlink cache root accepted");
    check(setenv(configured.name, custom.c_str(), 1) == 0 && chmod(custom.c_str(), 0777) == 0, "shared cache root");
    check(llama_kpack_cache_directory(f.source, error).empty(), "shared writable cache root accepted");
    check(chmod(custom.c_str(), 0700) == 0, "restore private cache root");
    check(setenv(configured.name, f.source.c_str(), 1) == 0, "file environment");
    check(llama_kpack_cache_directory(f.source, error).empty(), "cache root file accepted");
    check(setenv(configured.name, (f.root / "unused").c_str(), 1) == 0, "missing source environment");
    check(llama_kpack_cache_directory(f.root / "missing.gguf", error).empty() && !fs::exists(f.root / "unused"),
          "missing source created cache directory");
}

int main() try {
    fixture     f;
    std::string error;
    test_cache_directory(f);
    for (bool fallback : {false, true}) {
        manifest_last  = fallback;
        const auto dir = f.root / (fallback ? "manifest-last" : "rename");
        f.write(dir);
        f.verify(dir);
        llama_kpack_sidecar_writer other;
        check(!other.begin(dir, error), "overwrite accepted");
        f.verify(dir);
    }
    manifest_last = false;
    {
        const auto                    dir = f.root / "cancel";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir, f.source, error), error);
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
        check(!fs::exists(dir.string() + ".partial." + std::to_string(getpid())), "cancel leaked staging");
    }
    {
        const auto                 dir = f.root / "pending";
        llama_kpack_sidecar_writer owner;
        check(owner.begin(dir, error), error);
        {
            llama_kpack_sidecar_writer other;
            check(!other.begin(dir, error), "shared staging accepted");
        }
        check(fs::exists(dir.string() + ".partial." + std::to_string(getpid()) + "/weights.bin"),
              "removed other writer");
    }
    const auto original = f.root / "rename";
    const json manifest = json::parse(std::ifstream(original / "manifest.json"));
    for (const std::string kind : {"truncated", "offset", "overflow", "duplicate-key", "missing-manifest", "symlink"}) {
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
        check(!reader.open(dir, error), kind + " accepted");
    }
    {
        llama_kpack_sidecar_reader reader;
        check(reader.open(original, error), error);
        auto wrong = f.inventory;
        ++wrong[0].data_offset;
        check(!reader.load_unchecked(f.source, wrong, error), "different inventory accepted");
        std::ofstream out(f.source, std::ios::binary | std::ios::app);
        out.put(0);
        out.close();
        check(!reader.load_unchecked(f.source, f.inventory, error), "changed source accepted");
    }
    {
        const auto                    dir = f.root / "changed-during-write";
        llama_kpack_background_writer writer;
        check(writer.prepare(dir, f.source, error), error);
        std::ofstream out(f.source, std::ios::binary | std::ios::app);
        out.put(0);
        out.close();
        check(writer.start(f.jobs, f.inventory, 4096, error), error);
        check(!writer.wait(error), "published after source changed");
        check(!fs::exists(dir), "invalid publication");
    }
    puts(
        "KPACK_SIDECAR PASS formats=6 dense=6 grouped=6 atomic=rename+manifest-last directory=default+env "
        "rejections=PASS");
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "KPACK_SIDECAR FAIL: %s\n", error.what());
    return 1;
}
