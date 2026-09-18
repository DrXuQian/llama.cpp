#include "llama-kpack-cache.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-cuda/quactlize-sidecar.h"

#include <algorithm>
#include <array>
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
    if (!reg) { return api; }
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

using copy_slots = std::array<copy_slot, 2>;

// The writer consumes a returned slot before calling read() again. Prefetch the
// next range into the other slot so D2H can overlap that consumption (write()).
struct copy_reader {
    const ggml_tensor * tensor;
    std::shared_ptr<copy_slots> slots;
    cache_api api;
    std::array<size_t, 3> ends;
    size_t next_offset = 0;
    unsigned current = 0;
    bool pending[2] = {false, false};

    copy_reader(const ggml_tensor * t, std::shared_ptr<copy_slots> s, cache_api a, const llama_kpack_planes & p)
        : tensor(t), slots(std::move(s)), api(a),
          ends{{p.low_bytes, p.low_bytes + p.high_bytes, p.low_bytes + p.high_bytes + p.units_bytes}} {}

    ~copy_reader() {
        // Failed writes and cancellation can leave one prefetched range in
        // flight. Drain it on the writer before releasing its destination.
        for (unsigned i = 0; i < 2; ++i) {
            if (pending[i]) { api.wait((*slots)[i].event); }
        }
    }

    size_t count_at(size_t offset) const {
        for (size_t end : ends) {
            if (offset < end) { return std::min(chunk_bytes, end - offset); }
        }
        return 0;
    }

    bool submit(unsigned i, size_t offset, size_t bytes, std::string & error) {
        auto & slot = (*slots)[i];
        if (!api.copy(tensor, ggml_backend_buffer_get_base(slot.buffer), offset, bytes, slot.event)) {
            error = "packed range copy rejected"; return false;
        }
        pending[i] = true;
        return true;
    }

    const uint8_t * read(size_t offset, size_t bytes, std::string & error) {
        if (offset != next_offset || !bytes || bytes != count_at(offset)) {
            error = "nonsequential packed range request"; return nullptr;
        }
        if (!pending[current] && !submit(current, offset, bytes, error)) { return nullptr; }
        const size_t next = offset + bytes;
        if (next < ends.back() && !submit(current ^ 1, next, count_at(next), error)) { return nullptr; }
        auto & slot = (*slots)[current];
        const bool ok = api.wait(slot.event);
        pending[current] = false;
        if (!ok) { error = "packed range copy failed"; return nullptr; }
        next_offset = next;
        current ^= 1;
        return (const uint8_t *) ggml_backend_buffer_get_base(slot.buffer);
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

struct resident_shard {
    ggml_tensor * tensor;
    llama_kpack_partition partition;
    cache_api api;
};

std::vector<resident_shard> resident_shards(ggml_tensor * tensor) {
    std::vector<resident_shard> result;
    if (!tensor || !tensor->buffer) { return result; }
    const size_t count = ggml_backend_meta_buffer_type_count(ggml_backend_buffer_get_type(tensor->buffer));
    for (size_t index = 0; index < std::max<size_t>(1, count); ++index) {
        ggml_backend_meta_split_state split{};
        auto * local = count ? ggml_backend_meta_tensor_shard(tensor, index, &split) : tensor;
        if (!local) { return {}; }
        if (!ggml_nelements(local)) { continue; }
        const auto api = api_for(local);
        if (!api.is_kpack || !api.is_kpack(local)) { return {}; }
        llama_kpack_partition p;
        if (count) {
            p.axis = split.axis; p.count = count; p.index = index;
            if (split.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                if (split.axis < 0 || split.axis >= GGML_MAX_DIMS || split.n_segments > 16) { return {}; }
                p.widths.assign(split.ne, split.ne + split.n_segments * count);
                p.repeats.assign(split.nr, split.nr + split.n_segments);
            }
        }
        result.push_back({local, std::move(p), api});
    }
    return result;
}

bool matches_source(const ggml_tensor * tensor, const llama_kpack_source_tensor & src) {
    return tensor->type == src.ggml_type && (src.rank == 2 || src.rank == 3) &&
        tensor->ne[0] == src.k && tensor->ne[1] == src.n &&
        tensor->ne[2] * tensor->ne[3] == std::max<int64_t>(1, src.experts) && ggml_nbytes(tensor) == src.size_bytes;
}

} // namespace

struct llama_kpack_cache::impl {
    std::unique_ptr<llama_kpack_sidecar_reader> reader;
    llama_kpack_background_writer writer;
    std::vector<llama_kpack_source_tensor> inventory;
    std::map<std::string, llama_kpack_source_tensor> sources;
    std::map<ggml_backend_dev_t, std::shared_ptr<copy_slots>> slots;
    std::vector<llama_kpack_write_job> jobs;
    std::set<std::string> captured;
    size_t resident_requests = 0, cache_uploads = 0;
    bool writable = false, started = false;
};

llama_kpack_cache::llama_kpack_cache(const std::string & dir, const std::string & source,
        const std::vector<llama_kpack_source_tensor> & inventory, int loader_fd)
    : llama_kpack_cache(dir, std::vector<llama_kpack_source_file>{{source, loader_fd}}, inventory) {}

llama_kpack_cache::llama_kpack_cache(const std::string & dir, const std::vector<llama_kpack_source_file> & files,
        const std::vector<llama_kpack_source_tensor> & inventory) : pimpl(new impl) {
    auto & I = *pimpl;
    I.inventory = inventory;
    for (const auto & src : inventory) { I.sources.emplace(src.name, src); }
    std::string error;
    struct stat st{};
    if (lstat(dir.c_str(), &st) == 0) {
        auto reader = std::make_unique<llama_kpack_sidecar_reader>();
        if (reader->open(dir, error) && reader->load_unchecked(files, inventory, error)) {
            LLAMA_LOG_INFO("[kpack-cache] ready: tensors=%zu content_checks=disabled path=%s\n", reader->size(), dir.c_str());
            I.reader = std::move(reader);
        } else {
            LLAMA_LOG_WARN("[kpack-cache] ignored invalid cache: %s; using GPU pack\n", error.c_str());
        }
        return;
    }
    if (errno != ENOENT) { LLAMA_LOG_WARN("[kpack-cache] cannot inspect %s: %s\n", dir.c_str(), strerror(errno)); return; }
    I.writable = I.writer.prepare(dir, files, error);
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

bool llama_kpack_cache::has_cached_tensors() const {
    return pimpl->reader && pimpl->reader->size() > 0;
}

bool llama_kpack_cache::load(ggml_tensor * tensor) {
    auto & I = *pimpl;
    if (!I.reader) { return false; }
    const auto shards = resident_shards(tensor);
    if (shards.empty()) { return false; }
    I.resident_requests += shards.size();
    const auto src = I.sources.find(tensor->name);
    if (src == I.sources.end() || !matches_source(tensor, src->second)) { return false; }
    std::vector<ggml_quactlize_planes> planes;
    // Admit every local shard before installing any. A miss must leave the raw loader untouched.
    for (const auto & shard : shards) {
        const auto * rec = I.reader->find(tensor->name, shard.partition);
        const auto & api = shard.api;
        auto * local = shard.tensor;
        ggml_quactlize_planes p{};
        if (!rec || !api.layout || !api.set || rec->n != local->ne[1] || rec->k != local->ne[0] ||
            std::max<int64_t>(1, rec->experts) != local->ne[2] * local->ne[3] ||
            !api.layout(local, &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement) ||
            p.arrangement.version != 2 || !same_layout(describe(p), rec->planes)) { return false; }
        p.low = rec->planes.low; p.high = rec->planes.high; p.units = rec->planes.units;
        planes.push_back(p);
    }
    for (size_t index = 0; index < shards.size(); ++index) { shards[index].api.set(shards[index].tensor, &planes[index]); }
    I.cache_uploads += shards.size();
    return true;
}

void llama_kpack_cache::capture(ggml_tensor * tensor) {
    auto & I = *pimpl;
    if (!I.writable || I.started || I.captured.count(tensor->name)) { return; }
    const auto shards = resident_shards(tensor);
    if (shards.empty()) { return; }
    const auto disable = [&](const char * why) {
        LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", why);
        I.writable = false; I.jobs.clear(); I.writer.cancel();
    };
    const auto src = I.sources.find(tensor->name);
    if (src == I.sources.end() || !matches_source(tensor, src->second)) {
        disable("resident tensor does not match its source record"); return;
    }
    for (const auto & shard : shards) {
        const auto & api = shard.api;
        auto * local = shard.tensor;
        if (!api.layout || !api.copy || !api.wait) { disable("resident backend lacks snapshot support"); return; }
        ggml_quactlize_planes p{};
        if (!api.layout(local, &p.low_bytes, &p.high_bytes, &p.units_bytes, &p.arrangement) || p.arrangement.version != 2) {
            disable("resident descriptor is unavailable"); return;
        }
        auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(local->buffer));
        auto & slot = I.slots[dev];
        if (!slot) {
            // Allocate before inference starts, never from the snapshot worker.
            slot = std::make_shared<copy_slots>();
            if (!(*slot)[0].allocate(dev) || !(*slot)[1].allocate(dev)) {
                disable("pinned staging unavailable"); return;
            }
        }
        llama_kpack_write_job job{src->second, describe(p), {}, shard.partition};
        auto reader = std::make_shared<copy_reader>(local, slot, api, job.planes);
        job.read = [reader](size_t offset, size_t bytes, std::string & error) {
            return reader->read(offset, bytes, error);
        };
        I.jobs.push_back(std::move(job));
    }
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
    LLAMA_LOG_INFO("[kpack-cache] background write started: tensors=%zu pinned_MiB=%zu slots_per_device=2 content_checks=disabled\n",
                   tensors, I.slots.size() * 2 * chunk_bytes / (1024 * 1024));
}
