#include "../src/llama-kpack-cache.h"
#include "../ggml/src/ggml-cuda/quactlize-buft.cuh"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>

static ggml_backend_meta_split_state tp_split(const ggml_tensor * tensor, void * userdata) {
    const auto &                  split = static_cast<const nlohmann::json *>(userdata)->at(tensor->name);
    ggml_backend_meta_split_state result{};
    result.axis        = static_cast<ggml_backend_meta_split_axis>(split.at("axis").get<int>());
    const auto widths  = split.at("widths").get<std::vector<int64_t>>();
    const auto repeats = split.at("repeats").get<std::vector<uint32_t>>();
    GGML_ASSERT(widths.size() <= 16 * GGML_BACKEND_META_MAX_DEVICES && repeats.size() <= 16);
    std::copy(widths.begin(), widths.end(), result.ne);
    std::copy(repeats.begin(), repeats.end(), result.nr);
    result.n_segments = repeats.size();
    if (result.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        result.n_segments = result.nr[0] = 1;
    }
    return result;
}

static void check(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) try {
    check(argc == 5 || argc == 6,
          "usage: test-kpack-cache BACKEND REFERENCE.GGUF CACHE cold|hot|cold-partial|hot-partial|miss|incompatible "
          "[reject|tp|tp-reject]");
    const std::string option = argc == 6 ? argv[5] : "";
    const bool        tp     = option == "tp" || option == "tp-reject";
    const bool        reject = option == "reject" || option == "tp-reject";
    check(option.empty() || tp || reject, "invalid option");
    const std::string mode    = argv[4];
    const bool        partial = mode == "cold-partial" || mode == "hot-partial";
    const bool        cold    = mode == "cold" || mode == "cold-partial";
    const bool        hot     = mode == "hot" || mode == "hot-partial";
    check(cold || hot || mode == "miss" || mode == "incompatible", "invalid mode");
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    std::istringstream              device_names(argv[1]);
    for (std::string name; std::getline(device_names, name, ',');) {
        auto * item = ggml_backend_dev_by_name(name.c_str());
        check(item != nullptr, "requested TP device unavailable");
        devices.push_back(item);
    }
    check(!devices.empty() && devices.size() <= GGML_BACKEND_META_MAX_DEVICES, "invalid device count");
    auto * dev = devices[0];
    check(dev != nullptr, "device unavailable");
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto *           reg   = ggml_backend_dev_backend_reg(dev);
    auto             extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_override_bufts"));
    auto read = reinterpret_cast<decltype(&ggml_quactlize_read_packed)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_read_packed"));
    auto is_kpack = reinterpret_cast<decltype(&ggml_cuda_buft_is_quactlize)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_cuda_buft_is_quactlize"));
    check(backend && extra && read && is_kpack, "cache API unavailable");
    auto ** bufts = extra(dev);
    check(bufts && bufts[0], "packer unavailable");
    check(is_kpack(bufts[0]) && !is_kpack(ggml_backend_dev_buffer_type(dev)), "cache buffer isolation differs");

    ggml_context *   metadata = nullptr;
    gguf_context_ptr file(gguf_init_from_file(argv[2], {true, &metadata}));
    ggml_context_ptr metadata_owner(metadata);
    check(file && metadata, "reference GGUF unavailable");
    const auto key = gguf_find_key(file.get(), "quactlize.kpack.reference.manifest");
    check(key >= 0, "reference manifest missing");
    const auto                             manifest = nlohmann::json::parse(gguf_get_val_str(file.get(), key));
    nlohmann::json                         topology;
    ggml_backend_buffer_type_t             buft = bufts[0];
    if (tp) {
        for (const auto & record : manifest.at("tensors")) {
            check(record.at("shards").size() == devices.size(), "fixture device count differs");
            topology[record.at("source_name").get<std::string>()] = record.at("partition");
        }
        auto * meta = ggml_backend_meta_device(devices.data(), devices.size(), tp_split, &topology);
        buft        = ggml_backend_meta_buffer_type(meta, bufts[0]);
        check(buft && buft != ggml_backend_dev_buffer_type(meta), "TP override was not isolated");
        check(strcmp(ggml_backend_buft_name(buft), ggml_backend_buft_name(ggml_backend_dev_buffer_type(meta))) != 0,
              "TP override and default buffer names collide");
        check(ggml_backend_meta_buffer_type_count(buft) == devices.size(), "TP override device count");
    }
    std::vector<llama_kpack_source_tensor> inventory;
    std::vector<ggml_tensor *>             tensors;
    const size_t                           count = manifest.at("tensors").size();
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * (count + 8) + ggml_graph_overhead(), nullptr, true}));
    check(ctx != nullptr, "context allocation");
    for (const auto & record : manifest.at("tensors")) {
        const std::string name  = record.at("source_name");
        const int64_t     index = gguf_find_tensor(file.get(), name.c_str());
        check(index >= 0, "source tensor missing");
        const auto * source = ggml_get_tensor(metadata, name.c_str());
        auto *       t      = ggml_dup_tensor(ctx.get(), source);
        ggml_set_name(t, name.c_str());
        ggml_set_param(t);
        tensors.push_back(t);
        llama_kpack_source_tensor src;
        src.name        = name;
        src.gguf_index  = index;
        src.ggml_type   = t->type;
        src.data_offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), index);
        src.size_bytes  = ggml_nbytes(t);
        src.n           = t->ne[1];
        src.k           = t->ne[0];
        src.rank        = ggml_n_dims(t);
        src.experts     = src.rank >= 3 ? t->ne[2] : 0;
        src.batches     = t->ne[3];
        inventory.push_back(src);
    }
    ggml_backend_buffer_ptr buffer;
    // Cache dies before buffers and metadata, like the model-owned integration.
    auto                    cache = std::make_unique<llama_kpack_cache>(argv[3], argv[2], inventory);
    check(cache->has_cached_tensors() == (hot || mode == "incompatible"), "cache metadata admission differs");
    uint64_t source_bytes = 0;
    for (const auto & src : inventory) {
        source_bytes += src.size_bytes;
    }
    const auto planned = cache->coverage(ctx.get(), buft);
    check(planned.source_bytes == source_bytes &&
              planned.cached_bytes == (hot ? (partial ? inventory[0].size_bytes : source_bytes) : 0),
          "pre-allocation cache coverage differs");
    check(planned.should_prefetch() == (mode != "hot"), "source prefetch policy differs");
    const auto native = cache->coverage(ctx.get(), ggml_backend_dev_buffer_type(dev));
    check(native.source_bytes == source_bytes && native.cached_bytes == 0 && native.should_prefetch(),
          "ordinary GPU weights were treated as cached");
    const auto host = cache->coverage(ctx.get(), nullptr);
    check(host.source_bytes == source_bytes && host.cached_bytes == 0 && host.should_prefetch(),
          "host weights were treated as cached");
    {
        ggml_context_ptr probe(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        check(probe != nullptr, "coverage probe context");
        auto * changed = ggml_dup_tensor(probe.get(), tensors[0]);
        ggml_set_name(changed, tensors[0]->name);
        changed->ne[1] /= 2;
        const auto mismatched = cache->coverage(probe.get(), buft);
        check(mismatched.source_bytes == ggml_nbytes(changed) && mismatched.cached_bytes == 0 &&
                  mismatched.should_prefetch(),
              "different shape was treated as cached");
    }
    buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    check(buffer != nullptr, "buffer allocation");
    std::ifstream input(argv[2], std::ios::binary);
    auto          bytes = [&](const std::string & name) {
        const auto i = gguf_find_tensor(file.get(), name.c_str());
        check(i >= 0, "reference carrier missing");
        std::vector<uint8_t> result(gguf_get_tensor_size(file.get(), i));
        input.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), i));
        check(bool(input.read(reinterpret_cast<char *>(result.data()), result.size())), "reference truncated");
        return result;
    };
    size_t hits = 0, shuffles = 0;
    for (size_t i = 0; i < count; ++i) {
        auto *     tensor = tensors[i];
        const bool hit    = cache->load(tensor);
        check(hit == (hot && (!partial || i == 0)), "unexpected per-tensor cache result");
        if (hit) {
            ++hits;
        } else {
            auto raw = bytes(tensor->name);
            ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size());
            ++shuffles;
            if (!partial || i == 0) {
                cache->capture(tensor);
            }
        }
    }
    cache->start();
    // No explicit wait here: readback races the background snapshot on a separate stream.
    size_t snapshots = 0, stored = 0;
    for (size_t i = 0; i < count; ++i) {
        for (size_t rank = 0; rank < (tp ? devices.size() : 1); ++rank) {
            auto * local = tp ? ggml_backend_meta_tensor_shard(tensors[i], rank, nullptr) : tensors[i];
            check(local != nullptr, "TP local tensor missing");
            if (!ggml_nelements(local)) {
                continue;
            }
            ++snapshots;
            stored += !partial || i == 0;
            std::vector<uint8_t> expected;
            for (const char * plane : {"low", "high", "units"}) {
                const auto & record  = manifest.at("tensors")[i];
                const auto & carrier = (tp ? record.at("shards")[rank] : record).at("carriers").at(plane);
                if (!carrier.is_null()) {
                    auto part = bytes(carrier.at("name"));
                    expected.insert(expected.end(), part.begin(), part.end());
                }
            }
            std::vector<uint8_t> observed(expected.size());
            check(read(local, observed.data(), observed.size()), "readback failed");
            check(observed == expected, "cache bytes differ from independent offline reference");
        }
    }
    cache.reset();
    if (cold) {
        std::string                error;
        llama_kpack_sidecar_reader reader;
        check(reader.open(argv[3], error) && reader.load_unchecked(argv[2], inventory, error), error);
        check(reader.size() == stored, "publication incomplete");
    }
    if (reject) {
        auto * t = tp ? ggml_backend_meta_tensor_shard(tensors[0], 0, nullptr) : tensors[0];
        auto * a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, t->ne[0], 1, t->ne[2]);
        auto * g = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(g, ggml_mul_mat(ctx.get(), t, a));
        ggml_backend_graph_compute(backend.get(), g);
        throw std::runtime_error("cached forward was not rejected");
    }
    printf("KPACK_CACHE PASS mode=%s tensors=%zu hits=%zu gpu_shuffle=%zu inference=DISABLED\n", mode.c_str(), count,
           hits, shuffles);
    if (tp) {
        printf("KPACK_TP_CACHE PASS ranks=%zu snapshots=%zu\n", devices.size(), snapshots);
    }
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "KPACK_CACHE FAIL: %s\n", error.what());
    return 1;
}
