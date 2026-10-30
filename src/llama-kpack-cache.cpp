#include "llama-kpack-cache.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "../ggml/src/ggml-cuda/quactlize-buft.cuh"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <filesystem>
#include <set>
#include <sstream>

namespace {

constexpr size_t chunk_bytes = 8 * 1024 * 1024;

struct cache_api {
    decltype(&ggml_quactlize_tensor_is_kpack)  is_kpack = nullptr;
    decltype(&ggml_quactlize_can_load)         layout   = nullptr;
    decltype(&ggml_quactlize_set_packed)       set      = nullptr;
    decltype(&ggml_quactlize_copy_range_async) copy     = nullptr;
    decltype(&ggml_quactlize_copy_range_wait)  wait     = nullptr;
};

cache_api api_for(ggml_backend_buffer_type_t buft) {
    cache_api api;
    if (!buft) {
        return api;
    }
    auto * dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return api;
    }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return api;
    }
    auto is_kpack =
        (decltype(&ggml_cuda_buft_is_quactlize)) ggml_backend_reg_get_proc_address(reg, "ggml_cuda_buft_is_quactlize");
    if (!is_kpack || !is_kpack(buft)) {
        return api;
    }
    api.is_kpack = (decltype(api.is_kpack)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_tensor_is_kpack");
    api.layout   = (decltype(api.layout)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_can_load");
    api.set      = (decltype(api.set)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_set_packed");
    api.copy     = (decltype(api.copy)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_copy_range_async");
    api.wait     = (decltype(api.wait)) ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_copy_range_wait");
    return api;
}

cache_api api_for(const ggml_tensor * t) {
    return api_for(t && t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr);
}

struct copy_slot {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_event_t  event  = nullptr;

    ~copy_slot() {
        if (event) {
            ggml_backend_event_free(event);
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool allocate(ggml_backend_dev_t dev) {
        auto * buft = ggml_backend_dev_host_buffer_type(dev);
        if (!buft) {
            return false;
        }
        buffer = ggml_backend_buft_alloc_buffer(buft, chunk_bytes);
        if (!buffer) {
            return false;
        }
        event = ggml_backend_event_new(dev);
        return event != nullptr;
    }
};

using copy_slots = std::array<copy_slot, 2>;

// The writer consumes a returned slot before calling read() again. Prefetch the
// next range into the other slot so D2H can overlap that consumption (write()).
struct copy_reader {
    const ggml_tensor *         tensor;
    std::shared_ptr<copy_slots> slots;
    cache_api                   api;
    std::array<size_t, 3>       ends;
    size_t                      next_offset = 0;
    unsigned                    current     = 0;
    bool                        pending[2]  = {false, false};

    copy_reader(const ggml_tensor * t, std::shared_ptr<copy_slots> s, cache_api a, const llama_kpack_planes & p) :
        tensor(t),
        slots(std::move(s)),
        api(a),
        ends{{p.low_bytes, p.low_bytes + p.high_bytes, p.low_bytes + p.high_bytes + p.units_bytes}} {}

    ~copy_reader() {
        // Failed writes and cancellation can leave one prefetched range in
        // flight. Drain it on the writer before releasing its destination.
        for (unsigned i = 0; i < 2; ++i) {
            if (pending[i]) {
                api.wait((*slots)[i].event);
            }
        }
    }

    size_t count_at(size_t offset) const {
        for (size_t end : ends) {
            if (offset < end) {
                return std::min(chunk_bytes, end - offset);
            }
        }
        return 0;
    }

    bool submit(unsigned i, size_t offset, size_t bytes, std::string & error) {
        auto & slot = (*slots)[i];
        if (!api.copy(tensor, ggml_backend_buffer_get_base(slot.buffer), offset, bytes, slot.event)) {
            error = "packed range copy rejected";
            return false;
        }
        pending[i] = true;
        return true;
    }

    const uint8_t * read(size_t offset, size_t bytes, std::string & error) {
        if (offset != next_offset || !bytes || bytes != count_at(offset)) {
            error = "nonsequential packed range request";
            return nullptr;
        }
        if (!pending[current] && !submit(current, offset, bytes, error)) {
            return nullptr;
        }
        const size_t next = offset + bytes;
        if (next < ends.back() && !submit(current ^ 1, next, count_at(next), error)) {
            return nullptr;
        }
        auto &     slot  = (*slots)[current];
        const bool ok    = api.wait(slot.event);
        pending[current] = false;
        if (!ok) {
            error = "packed range copy failed";
            return nullptr;
        }
        next_offset = next;
        current ^= 1;
        return (const uint8_t *) ggml_backend_buffer_get_base(slot.buffer);
    }
};

llama_kpack_planes describe(const ggml_quactlize_packed_layout & p) {
    llama_kpack_planes out;
    out.low_bytes        = p.sizes.low_bytes;
    out.high_bytes       = p.sizes.high_bytes;
    out.units_bytes      = p.sizes.units_bytes;
    const auto & a       = p.arrangement;
    out.layout           = a.layout;
    out.bits             = a.bits;
    out.high_bits        = a.high_bits;
    out.artifact_tile_k  = a.artifact_tile_k;
    out.transport_tile_k = a.transport_tile_k;
    out.group_size       = a.group_size;
    out.reserved         = a.reserved;
    out.mapping_id       = a.mapping_id;
    return out;
}

bool same_layout(const llama_kpack_planes & a, const llama_kpack_planes & b) {
    return a.low_bytes == b.low_bytes && a.high_bytes == b.high_bytes && a.units_bytes == b.units_bytes &&
           a.layout == b.layout && a.bits == b.bits && a.high_bits == b.high_bits &&
           a.artifact_tile_k == b.artifact_tile_k && a.transport_tile_k == b.transport_tile_k &&
           a.group_size == b.group_size && a.reserved == b.reserved && a.mapping_id == b.mapping_id;
}

bool partition_for(const ggml_backend_meta_split_state & split, size_t count, size_t index, llama_kpack_partition & p) {
    if (!count || count > GGML_BACKEND_META_MAX_DEVICES || index >= count || split.n_segments > 16) {
        return false;
    }
    p.axis  = split.axis;
    p.count = count;
    p.index = index;
    if (split.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        p.widths.assign(split.ne, split.ne + split.n_segments * count);
        p.repeats.assign(split.nr, split.nr + split.n_segments);
    }
    return true;
}

bool partition_for(ggml_backend_buffer_type_t buft,
                   const ggml_tensor *        tensor,
                   size_t                     index,
                   llama_kpack_partition &    p) {
    const size_t                  count = ggml_backend_meta_buffer_type_count(buft);
    ggml_backend_meta_split_state split{};
    return count ? ggml_backend_meta_tensor_split(buft, tensor, &split) && partition_for(split, count, index, p) :
                   index == 0;
}

bool matches_source(const ggml_tensor * tensor, const llama_kpack_source_tensor & src) {
    return tensor->type == src.ggml_type && tensor->ne[0] == src.k && tensor->ne[1] == src.n &&
           tensor->ne[2] == std::max<int64_t>(1, src.experts) && tensor->ne[3] == src.batches &&
           ggml_nbytes(tensor) == src.size_bytes;
}

struct resident_shard {
    ggml_tensor *         tensor;
    llama_kpack_partition partition;
    cache_api             api;
};

std::vector<resident_shard> resident_shards(ggml_tensor * tensor) {
    std::vector<resident_shard> result;
    if (!tensor || !tensor->buffer) {
        return result;
    }
    auto *       buft  = ggml_backend_buffer_get_type(tensor->buffer);
    const size_t count = ggml_backend_meta_buffer_type_count(buft);
    for (size_t index = 0; index < std::max<size_t>(1, count); ++index) {
        ggml_backend_meta_split_state split{};
        auto *                        local = count ? ggml_backend_meta_tensor_shard(tensor, index, &split) : tensor;
        if (!local) {
            return {};
        }
        if (!ggml_nelements(local)) {
            continue;
        }
        const auto            api = api_for(local);
        llama_kpack_partition p;
        if (!api.is_kpack || !api.is_kpack(local) || (count && !partition_for(split, count, index, p))) {
            return {};
        }
        result.push_back({local, std::move(p), api});
    }
    return result;
}

}  // namespace

struct llama_kpack_cache::impl {
    std::unique_ptr<llama_kpack_sidecar_reader>               reader;
    llama_kpack_background_writer                             writer;
    std::vector<llama_kpack_source_tensor>                    inventory;
    std::map<std::string, llama_kpack_source_tensor>          sources;
    std::map<ggml_backend_dev_t, std::shared_ptr<copy_slots>> slots;
    std::vector<llama_kpack_write_job>                        jobs;
    std::set<std::string>                                     captured;
    size_t                                                    resident_requests = 0, cache_uploads = 0;
    bool                                                      writable = false, started = false;
    std::vector<std::unique_ptr<llama_kpack_cache>>           files;

    const llama_kpack_sidecar_record * matching_record(const ggml_tensor *            tensor,
                                                       const cache_api &              api,
                                                       ggml_quactlize_packed_layout & layout,
                                                       const llama_kpack_partition &  partition = {}) const {
        if (!reader || !api.is_kpack || !api.layout || !api.set) {
            return nullptr;
        }
        const auto *              rec = reader->find(tensor->name, partition);
        const auto   src = sources.find(tensor->name);
        llama_kpack_source_tensor local;
        std::string               error;
        if (!rec || src == sources.end() || !llama_kpack_shard_source(src->second, partition, local, error) ||
            !matches_source(tensor, local)) {
            return nullptr;
        }
        if (!api.layout(tensor, &layout) || layout.arrangement.version != 2 ||
            !same_layout(describe(layout), rec->planes)) {
            return nullptr;
        }
        return rec;
    }
};

llama_kpack_cache::llama_kpack_cache() : pimpl(new impl) {}

llama_kpack_cache::llama_kpack_cache(const std::string &                            dir,
                                     const std::string &                            source,
                                     const std::vector<llama_kpack_source_tensor> & inventory,
                                     int                                            loader_fd) :
    pimpl(new impl) {
    auto & I    = *pimpl;
    I.inventory = inventory;
    for (const auto & src : inventory) {
        I.sources.emplace(src.name, src);
    }
    std::string     error;
    std::error_code ec;
    const auto      status = std::filesystem::symlink_status(std::filesystem::u8path(dir), ec);
    if (ec && ec != std::errc::no_such_file_or_directory) {
        LLAMA_LOG_WARN("[kpack-cache] cannot inspect %s: %s\n", dir.c_str(), ec.message().c_str());
        return;
    }
    if (std::filesystem::exists(status)) {
        auto reader = std::make_unique<llama_kpack_sidecar_reader>();
        if (reader->open(dir, error) && reader->load_unchecked(source, inventory, error, loader_fd)) {
            LLAMA_LOG_INFO("[kpack-cache] ready: tensors=%zu content_checks=disabled path=%s\n", reader->size(),
                           dir.c_str());
            I.reader = std::move(reader);
        } else {
            LLAMA_LOG_WARN("[kpack-cache] ignored invalid cache: %s; using GPU pack\n", error.c_str());
        }
        return;
    }
    I.writable = I.writer.prepare(dir, source, error, loader_fd);
    if (!I.writable) {
        LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", error.c_str());
    }
}

llama_kpack_cache::~llama_kpack_cache() {
    // Join before the model releases tensor metadata and device buffers.
    if (pimpl->started) {
        std::string error;
        if (!pimpl->writer.wait(error)) {
            LLAMA_LOG_WARN("[kpack-cache] cache writer failed: %s\n", error.c_str());
        }
    } else {
        pimpl->writer.cancel();
    }
}

bool llama_kpack_cache::has_cached_tensors() const {
    for (const auto & file : pimpl->files) {
        if (file->has_cached_tensors()) {
            return true;
        }
    }
    return pimpl->reader && pimpl->reader->size() > 0;
}

llama_kpack_cache_coverage llama_kpack_cache::coverage(ggml_context * ctx, ggml_backend_buffer_type_t buft) const {
    llama_kpack_cache_coverage result;
    for (const auto & file : pimpl->files) {
        const auto part = file->coverage(ctx, buft);
        if (part.source_bytes > UINT64_MAX - result.source_bytes) {
            return {UINT64_MAX, 0};
        }
        result.source_bytes += part.source_bytes;
        result.cached_bytes += part.cached_bytes;
    }
    if (!pimpl->files.empty()) {
        return result;
    }
    const size_t count = ggml_backend_meta_buffer_type_count(buft);
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (!pimpl->sources.count(tensor->name)) {
            continue;
        }
        const uint64_t bytes = ggml_nbytes(tensor);
        if (bytes > UINT64_MAX - result.source_bytes) {
            return {UINT64_MAX, 0};
        }
        result.source_bytes += bytes;
        if (!matches_source(tensor, pimpl->sources.at(tensor->name))) {
            continue;
        }
        bool cached = true;
        for (size_t index = 0; index < std::max<size_t>(1, count); ++index) {
            llama_kpack_partition partition;
            if (!partition_for(buft, tensor, index, partition)) {
                cached = false;
                break;
            }
            llama_kpack_source_tensor local;
            std::string               error;
            if (!llama_kpack_shard_source(pimpl->sources.at(tensor->name), partition, local, error)) {
                cached = false;
                break;
            }
            if (!local.size_bytes) {
                continue;
            }
            ggml_tensor probe = *tensor;
            probe.ne[0]       = local.k;
            probe.ne[1]       = local.n;
            probe.ne[2]       = std::max<int64_t>(1, local.experts);
            probe.ne[3]       = local.batches;
            probe.nb[1]       = ggml_row_size(probe.type, probe.ne[0]);
            probe.nb[2]       = probe.nb[1] * probe.ne[1];
            probe.nb[3]       = probe.nb[2] * probe.ne[2];
            ggml_quactlize_packed_layout layout{};
            const auto                   api = api_for(count ? ggml_backend_meta_buffer_type_at(buft, index) : buft);
            if (!pimpl->matching_record(&probe, api, layout, partition)) {
                cached = false;
                break;
            }
        }
        if (cached) {
            result.cached_bytes += bytes;
        }
    }
    return result;
}

bool llama_kpack_cache::should_prefetch(const llama_model_loader & loader) const {
    llama_kpack_cache_coverage total;
    for (const auto & entry : loader.ctx_map) {
        const auto part = coverage(entry.second.get(), entry.first);
        if (part.source_bytes > UINT64_MAX - total.source_bytes) {
            return true;
        }
        total.source_bytes += part.source_bytes;
        total.cached_bytes += part.cached_bytes;
    }
    LLAMA_LOG_DEBUG("[kpack-cache] source_bytes=%" PRIu64 " cached_bytes=%" PRIu64 " prefetch=%d\n", total.source_bytes,
                    total.cached_bytes, int(total.should_prefetch()));
    return total.should_prefetch();
}

bool llama_kpack_cache::load(ggml_tensor * tensor) {
    auto & I = *pimpl;
    for (const auto & file : I.files) {
        if (file->load(tensor)) {
            return true;
        }
    }
    if (!I.reader) {
        return false;
    }
    const auto shards = resident_shards(tensor);
    const auto src    = I.sources.find(tensor->name);
    if (shards.empty() || src == I.sources.end() || !matches_source(tensor, src->second)) {
        return false;
    }
    ++I.resident_requests;
    std::vector<ggml_quactlize_packed_layout>       layouts(shards.size());
    std::vector<const llama_kpack_sidecar_record *> records;
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto & shard = shards[i];
        const auto * rec   = I.matching_record(shard.tensor, shard.api, layouts[i], shard.partition);
        if (!rec) {
            return false;
        }
        records.push_back(rec);
    }
    // All shards are admitted before installing any. A miss leaves raw loading untouched.
    for (size_t i = 0; i < shards.size(); ++i) {
        if (!shards[i].api.set(shards[i].tensor, &layouts[i], records[i]->planes.low, layouts[i].sizes.raw_bytes)) {
            throw std::runtime_error("K-pack shard cache upload failed");
        }
    }
    ++I.cache_uploads;
    return true;
}

void llama_kpack_cache::capture(ggml_tensor * tensor) {
    auto & I = *pimpl;
    for (const auto & file : I.files) {
        file->capture(tensor);
    }
    if (!I.writable || I.started || I.captured.count(tensor->name)) {
        return;
    }
    const auto shards = resident_shards(tensor);
    const auto src    = I.sources.find(tensor->name);
    if (shards.empty() || src == I.sources.end()) {
        return;
    }
    const auto disable = [&](const char * why) {
        LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", why);
        I.writable = false;
        I.jobs.clear();
        I.writer.cancel();
    };
    if (!matches_source(tensor, src->second)) {
        disable("resident tensor does not match its source record");
        return;
    }
    for (const auto & shard : shards) {
        const auto api   = shard.api;
        auto *     local = shard.tensor;
        if (!api.layout || !api.copy || !api.wait) {
            disable("resident backend lacks snapshot support");
            return;
        }
        ggml_quactlize_packed_layout p{};
        if (!api.layout(local, &p) || p.arrangement.version != 2) {
            disable("resident descriptor is unavailable");
            return;
        }
        auto * dev  = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(local->buffer));
        auto & slot = I.slots[dev];
        if (!slot) {
            // Allocate before inference starts; never allocate/free pinned storage in
            // the running writer, where runtime allocation could synchronize a device.
            slot = std::make_shared<copy_slots>();
            if (!(*slot)[0].allocate(dev) || !(*slot)[1].allocate(dev)) {
                disable("pinned staging unavailable");
                return;
            }
        }
        llama_kpack_write_job job{src->second, describe(p), {}, shard.partition};
        auto                  reader = std::make_shared<copy_reader>(local, slot, api, job.planes);
        job.read                     = [reader](size_t offset, size_t bytes, std::string & error) {
            return reader->read(offset, bytes, error);
        };
        I.jobs.push_back(std::move(job));
    }
    I.captured.insert(tensor->name);
}

void llama_kpack_cache::start() {
    auto & I = *pimpl;
    for (const auto & file : I.files) {
        file->start();
    }
    if (I.reader) {
        LLAMA_LOG_INFO("[kpack-cache] cache_uploads=%zu resident_misses=%zu\n", I.cache_uploads,
                       I.resident_requests - I.cache_uploads);
        return;
    }
    if (!I.writable || I.started) {
        return;
    }
    if (I.jobs.empty()) {
        LLAMA_LOG_INFO("[kpack-cache] no resident snapshots; no cache will be published\n");
        I.writer.cancel();
        return;
    }
    const size_t tensors = I.jobs.size();
    std::string  error;
    I.started = I.writer.start(std::move(I.jobs), I.inventory, chunk_bytes, error);
    if (!I.started) {
        LLAMA_LOG_WARN("[kpack-cache] cannot start writer: %s\n", error.c_str());
        return;
    }
    LLAMA_LOG_INFO(
        "[kpack-cache] background write started: tensors=%zu pinned_MiB=%zu slots_per_device=2 "
        "content_checks=disabled\n",
        tensors, I.slots.size() * 2 * chunk_bytes / (1024 * 1024));
}

std::unique_ptr<llama_kpack_cache> llama_kpack_cache::create(const llama_model_loader & loader) {
    if (loader.no_alloc) {
        return nullptr;
    }
    const bool has_kpack = std::any_of(loader.ctx_map.begin(), loader.ctx_map.end(), [](const auto & entry) {
        if (!ggml_get_first_tensor(entry.second.get())) {
            return false;
        }
        if (api_for(entry.first).is_kpack) {
            return true;
        }
        for (size_t i = 0; i < ggml_backend_meta_buffer_type_count(entry.first); ++i) {
            if (api_for(ggml_backend_meta_buffer_type_at(entry.first, i)).is_kpack) {
                return true;
            }
        }
        return false;
    });
    if (!has_kpack) {
        return nullptr;
    }
    if (loader.files.empty() || loader.source_paths.size() != loader.files.size()) {
        LLAMA_LOG_WARN("[kpack-cache] persistence requires named GGUF sources; cache disabled\n");
        return nullptr;
    }
    if (loader.check_tensors) {
        LLAMA_LOG_WARN("[kpack-cache] source validation requested; cache disabled\n");
        return nullptr;
    }
    std::vector<std::vector<llama_kpack_source_tensor>> inventories(loader.files.size());
    for (const auto & entry : loader.weights_map) {
        const auto &              w = entry.second;
        llama_kpack_source_tensor src;
        src.name        = entry.first;
        src.gguf_index  = w.tensor_index;
        src.data_offset = w.offs;
        src.size_bytes  = ggml_nbytes(w.tensor);
        src.ggml_type   = w.tensor->type;
        src.rank        = ggml_n_dims(w.tensor);
        src.n           = w.tensor->ne[1];
        src.k           = w.tensor->ne[0];
        src.experts     = src.rank >= 3 ? w.tensor->ne[2] : 0;
        src.batches     = w.tensor->ne[3];
        inventories.at(w.idx).push_back(std::move(src));
    }
    auto                                         result = std::unique_ptr<llama_kpack_cache>(new llama_kpack_cache);
    std::map<std::string, llama_kpack_partition> topology;
    for (const auto & entry : loader.ctx_map) {
        if (!ggml_backend_meta_buffer_type_count(entry.first)) {
            continue;
        }
        for (auto * t = ggml_get_first_tensor(entry.second.get()); t; t = ggml_get_next_tensor(entry.second.get(), t)) {
            llama_kpack_partition p;
            if (!partition_for(entry.first, t, 0, p)) {
                return nullptr;
            }
            topology.emplace(t->name, std::move(p));
        }
    }
    std::string variant;
    if (!topology.empty()) {
        // Namespace only. Cache admission still compares the full split map, not this hash.
        uint64_t   hash   = UINT64_C(14695981039346656037);
        const auto append = [&](uint64_t value) {
            for (int b = 0; b < 8; ++b) {
                hash = (hash ^ ((value >> (b * 8)) & 255)) * UINT64_C(1099511628211);
            }
        };
        for (const auto & entry : topology) {
            append(entry.first.size());
            for (unsigned char c : entry.first) {
                append(c);
            }
            const auto & p = entry.second;
            append(p.axis);
            append(p.count);
            append(p.widths.size());
            append(p.repeats.size());
            for (auto width : p.widths) {
                append(width);
            }
            for (auto repeat : p.repeats) {
                append(repeat);
            }
        }
        std::ostringstream suffix;
        suffix << "-tp-" << std::hex << hash;
        variant = suffix.str();
    }
    for (size_t i = 0; i < loader.files.size(); ++i) {
        std::string error;
        const auto  dir = llama_kpack_cache_directory(loader.source_paths[i], error);
        if (dir.empty()) {
            LLAMA_LOG_WARN("[kpack-cache] persistence disabled: %s\n", error.c_str());
            continue;
        }
        result->pimpl->files.push_back(std::make_unique<llama_kpack_cache>(dir + variant, loader.source_paths[i],
                                                                           inventories[i], loader.files[i]->file_id()));
    }
    return result;
}
