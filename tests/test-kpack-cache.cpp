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

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) try {
    check(argc == 5 || argc == 6,
          "usage: test-kpack-cache BACKEND REFERENCE.GGUF CACHE cold|hot|miss|incompatible [reject]");
    const std::string mode = argv[4];
    check(mode == "cold" || mode == "hot" || mode == "miss" || mode == "incompatible", "invalid mode");
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name(argv[1]);
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
        src.experts     = src.rank == 3 ? t->ne[2] : 0;
        inventory.push_back(src);
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), bufts[0]));
    check(buffer != nullptr, "buffer allocation");
    // Cache dies before buffers and metadata, like the model-owned integration.
    auto cache = std::make_unique<llama_kpack_cache>(argv[3], argv[2], inventory);
    check(cache->has_cached_tensors() == (mode == "hot" || mode == "incompatible"), "cache metadata admission differs");
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
        check(hit == (mode == "hot"), "unexpected per-tensor cache result");
        if (hit) {
            ++hits;
        } else {
            auto raw = bytes(tensor->name);
            ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size());
            ++shuffles;
            cache->capture(tensor);
        }
    }
    cache->start();
    // No explicit wait here: readback races the background snapshot on a separate stream.
    for (size_t i = 0; i < count; ++i) {
        std::vector<uint8_t> expected;
        for (const char * plane : {"low", "high", "units"}) {
            const auto & carrier = manifest.at("tensors")[i].at("carriers").at(plane);
            if (!carrier.is_null()) {
                auto part = bytes(carrier.at("name"));
                expected.insert(expected.end(), part.begin(), part.end());
            }
        }
        std::vector<uint8_t> observed(expected.size());
        check(read(tensors[i], observed.data(), observed.size()), "readback failed");
        check(observed == expected, "cache bytes differ from independent offline reference");
    }
    cache.reset();
    if (mode == "cold") {
        std::string                error;
        llama_kpack_sidecar_reader reader;
        check(reader.open(argv[3], error) && reader.load_unchecked(argv[2], inventory, error), error.c_str());
        check(reader.size() == count, "publication incomplete");
    }
    if (argc == 6) {
        auto * t = tensors[0];
        auto * a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, t->ne[0], 1, t->ne[2]);
        auto * g = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(g, ggml_mul_mat(ctx.get(), t, a));
        ggml_backend_graph_compute(backend.get(), g);
        throw std::runtime_error("cached forward was not rejected");
    }
    printf("KPACK_CACHE PASS mode=%s tensors=%zu hits=%zu gpu_shuffle=%zu inference=DISABLED\n", mode.c_str(), count,
           hits, shuffles);
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "KPACK_CACHE FAIL: %s\n", error.what());
    return 1;
}
