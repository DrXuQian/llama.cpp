#include "llama-kpack-cache.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-cuda/quactlize-sidecar.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <set>
#include <sys/stat.h>

namespace {

constexpr size_t chunk_bytes = 8 * 1024 * 1024;

struct cache_api {
    decltype(&ggml_quactlize_tensor_is_kpack) is_kpack = nullptr;
    decltype(&ggml_quactlize_plane_layout) layout = nullptr;
    decltype(&ggml_quactlize_set_planes) set = nullptr;
    decltype(&ggml_quactlize_copy_range_async) copy = nullptr;
    decltype(&ggml_quactlize_copy_range_wait) wait = nullptr;
};

cache_api api_for(const ggml_tensor * t) {
    cache_api api;
    if (!t || !t->buffer) { return api; }
    auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    if (!dev) { return api; }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    api.is_kpack = (decltype(api.is_kpack)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_tensor_is_kpack");
    api.layout = (decltype(api.layout)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_plane_layout");
    api.set = (decltype(api.set)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_set_planes");
    api.copy = (decltype(api.copy)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_copy_range_async");
    api.wait = (decltype(api.wait)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_copy_range_wait");
    return api;
}

struct copy_slot {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_event_t event = nullptr;
    ~copy_slot() {
        if (event) { ggml_backend_event_free(event); }
        if (buffer) { ggml_backend_buffer_free(buffer); }
    }
    bool allocate(ggml_backend_dev_t dev) {
        auto * buft = ggml_backend_dev_host_buffer_type(dev);
        if (!buft) { return false; }
        buffer = ggml_backend_buft_alloc_buffer(buft, chunk_bytes);
        if (!buffer) { return false; }
        event = ggml_backend_event_new(dev);
        return event != nullptr;
    }
};

llama_kpack_planes describe(const ggml_quactlize_planes & p) {
    llama_kpack_planes out;
    out.low_bytes = p.low_bytes; out.high_bytes = p.high_bytes; out.units_bytes = p.units_bytes;
    const auto & a = p.arrangement;
    out.layout = a.layout; out.bits = a.bits; out.high_bits = a.high_bits;
    out.artifact_tile_k = a.artifact_tile_k; out.transport_tile_k = a.transport_tile_k;
    out.group_size = a.group_size; out.reserved = a.reserved; out.mapping_id = a.mapping_id;
    return out;
}

bool same_layout(const llama_kpack_planes & a, const llama_kpack_planes & b) {
    return a.low_bytes == b.low_bytes && a.high_bytes == b.high_bytes && a.units_bytes == b.units_bytes &&
        a.layout == b.layout && a.bits == b.bits && a.high_bits == b.high_bits &&
        a.artifact_tile_k == b.artifact_tile_k && a.transport_tile_k == b.transport_tile_k &&
        a.group_size == b.group_size && a.reserved == b.reserved && a.mapping_id == b.mapping_id;
}

} // namespace

struct llama_kpack_cache::impl {
    std::unique_ptr<llama_kpack_sidecar_reader> reader;
    llama_kpack_background_writer writer;
    std::vector<llama_kpack_source_tensor> inventory;
    std::map<std::string, llama_kpack_source_tensor> sources;
    std::map<ggml_backend_dev_t, std::shared_ptr<copy_slot>> slots;
    std::vector<llama_kpack_write_job> jobs;
    std::set<std::string> captured;
    size_t resident_requests = 0, cache_uploads = 0;
    bool writable = false, started = false;
};

llama_kpack_cache::llama_kpack_cache(const std::string & dir, const std::string & source,
        const std::vector<llama_kpack_source_tensor> & inventory, int loader_fd) : pimpl(new impl) {
    auto & I = *pimpl;
    I.inventory = inventory;
    for (const auto & src : inventory) { I.sources.emplace(src.name, src); }
    std::string error;
    const auto same_source = [&]() {
        if (loader_fd < 0) { return true; }
        struct stat loaded{}, named{};
        return fstat(loader_fd, &loaded) == 0 && stat(source.c_str(), &named) == 0 &&
            loaded.st_dev == named.st_dev && loaded.st_ino == named.st_ino && loaded.st_size == named.st_size &&
            loaded.st_mtim.tv_sec == named.st_mtim.tv_sec && loaded.st_mtim.tv_nsec == named.st_mtim.tv_nsec &&
            loaded.st_ctim.tv_sec == named.st_ctim.tv_sec && loaded.st_ctim.tv_nsec == named.st_ctim.tv_nsec;
    };
    if (!same_source()) { LLAMA_LOG_WARN("[kpack-cache] source path differs from the loader's file; cache disabled\n"); return; }
    struct stat st{};
    if (lstat(dir.c_str(), &st) == 0) {
        auto reader = std::make_unique<llama_kpack_sidecar_reader>();
        if (reader->open(dir, error) && reader->verify_source(source, inventory, 4, error) &&
            reader->verify_storage(4, error) && same_source()) {
            LLAMA_LOG_INFO("[kpack-cache] verified %zu tensors in %s\n", reader->size(), dir.c_str());
            I.reader = std::move(reader);
        } else {
            LLAMA_LOG_WARN("[kpack-cache] ignored invalid cache: %s; using GPU pack\n", error.c_str());
        }
        return;
    }
    if (errno != ENOENT) { LLAMA_LOG_WARN("[kpack-cache] cannot inspect %s: %s\n", dir.c_str(), strerror(errno)); return; }
    I.writable = I.writer.prepare(dir, source, error, loader_fd);
    if (!I.writable) { LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", error.c_str()); }
}

llama_kpack_cache::~llama_kpack_cache() {
    // Member buffers and tensor metadata are still owned by the model here.
    if (pimpl->started) {
        std::string error;
        if (!pimpl->writer.wait(error)) { LLAMA_LOG_WARN("[kpack-cache] not published: %s\n", error.c_str()); }
    } else {
        pimpl->writer.cancel();
    }
}

bool llama_kpack_cache::load(ggml_tensor * tensor) {
    auto & I = *pimpl;
    if (!I.reader) { return false; }
    const auto api = api_for(tensor);
    if (!api.is_kpack || !api.is_kpack(tensor)) { return false; }
    ++I.resident_requests;
    if (!api.layout || !api.set) { return false; }
    const auto * rec = I.reader->find(tensor->name);
    if (!rec) { return false; }
    const auto src = I.sources.find(tensor->name);
    if (src == I.sources.end() || tensor->type != src->second.ggml_type ||
        (src->second.rank != 2 && src->second.rank != 3) ||
        tensor->ne[0] != src->second.k || tensor->ne[1] != src->second.n ||
        tensor->ne[2] * tensor->ne[3] != std::max<int64_t>(1, src->second.experts)) { return false; }
    ggml_quactlize_planes p{};
    if (!api.layout(tensor, &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement) ||
        p.arrangement.version != 2 || !same_layout(describe(p), rec->planes)) { return false; }
    p.low = rec->planes.low; p.high = rec->planes.high; p.units = rec->planes.units;
    api.set(tensor, &p);
    ++I.cache_uploads;
    return true;
}

void llama_kpack_cache::capture(ggml_tensor * tensor) {
    auto & I = *pimpl;
    if (!I.writable || I.started || I.captured.count(tensor->name)) { return; }
    const auto api = api_for(tensor);
    if (!api.is_kpack || !api.is_kpack(tensor)) { return; }
    const auto disable = [&](const char * why) {
        LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", why);
        I.writable = false; I.jobs.clear(); I.writer.cancel();
    };
    if (!api.layout || !api.copy || !api.wait) { disable("resident backend lacks snapshot support"); return; }
    const auto src = I.sources.find(tensor->name);
    if (src == I.sources.end() || tensor->type != src->second.ggml_type ||
        (src->second.rank != 2 && src->second.rank != 3) ||
        tensor->ne[0] != src->second.k || tensor->ne[1] != src->second.n ||
        tensor->ne[2] * tensor->ne[3] != std::max<int64_t>(1, src->second.experts) ||
        ggml_nbytes(tensor) != src->second.size_bytes) { disable("resident tensor does not match its source record"); return; }
    ggml_quactlize_planes p{};
    if (!api.layout(tensor, &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement) || p.arrangement.version != 2) {
        disable("resident descriptor is unavailable"); return;
    }
    auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto & slot = I.slots[dev];
    if (!slot) {
        // Allocate before inference starts; never allocate/free pinned storage in
        // the running writer, where runtime allocation could synchronize a device.
        slot = std::make_shared<copy_slot>();
        if (!slot->allocate(dev)) {
            disable("pinned staging unavailable"); return;
        }
    }
    llama_kpack_write_job job{src->second, describe(p), {}};
    job.read = [tensor, slot, api](size_t offset, size_t bytes, std::string & error) -> const uint8_t * {
        void * data = ggml_backend_buffer_get_base(slot->buffer);
        if (bytes > chunk_bytes || !api.copy(tensor, data, offset, bytes, slot->event)) {
            error = "packed range copy rejected"; return nullptr;
        }
        // This is the writer thread, never a loader/compute callback.
        if (!api.wait(slot->event)) { error = "packed range copy failed"; return nullptr; }
        return (const uint8_t *) data;
    };
    I.jobs.push_back(std::move(job));
    I.captured.insert(tensor->name);
}

void llama_kpack_cache::start() {
    auto & I = *pimpl;
    if (I.reader) {
        LLAMA_LOG_INFO("[kpack-cache] cache_uploads=%zu resident_misses=%zu\n",
                       I.cache_uploads, I.resident_requests - I.cache_uploads);
        return;
    }
    if (!I.writable || I.started) { return; }
    if (I.jobs.empty()) {
        LLAMA_LOG_INFO("[kpack-cache] no resident snapshots; no cache will be published\n");
        I.writer.cancel(); return;
    }
    const size_t tensors = I.jobs.size();
    std::string error;
    I.started = I.writer.start(std::move(I.jobs), I.inventory, chunk_bytes, error);
    if (!I.started) { LLAMA_LOG_WARN("[kpack-cache] cannot start writer: %s\n", error.c_str()); return; }
    LLAMA_LOG_INFO("[kpack-cache] background write started: tensors=%zu pinned_MiB=%zu\n",
                   tensors, I.slots.size() * chunk_bytes / (1024 * 1024));
}
