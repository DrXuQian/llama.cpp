// K-pack load-only buffer type test. Writes raw K-quant weights through the ordinary set_tensor path, reads back the
// GPU-shuffled planes and compares them byte-for-byte with an offline reference GGUF produced by gguf_kpack.py.
// Driven by test-kpack-load.py, which generates the reference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "../ggml/src/ggml-cuda/quactlize-buft.cuh"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static bool test_kpack_load(ggml_backend_t backend, const char * fixture, bool reject_forward) {
    auto require = [](bool ok, const char * message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    };
    try {
        auto * dev   = ggml_backend_get_device(backend);
        auto * reg   = ggml_backend_dev_backend_reg(dev);
        auto   extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_override_bufts"));
        auto query = reinterpret_cast<decltype(&ggml_quactlize_can_load)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_can_load"));
        auto read = reinterpret_cast<decltype(&ggml_quactlize_read_packed)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_read_packed"));
        auto install = reinterpret_cast<decltype(&ggml_quactlize_set_packed)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_set_packed"));
        require(extra && query && read && install, "load-only API is unavailable");
        auto ** bufts = extra(dev);
        require(bufts && bufts[0], "packer is unavailable");
        ggml_backend_buffer_type_t buft = bufts[0];
        require(ggml_backend_dev_buffer_type(dev) != buft, "K-pack is the default model buffer");
        auto automatic = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
        if (automatic) {
            auto ** candidates = automatic(dev);
            for (size_t i = 0; candidates && candidates[i]; ++i) {
                require(candidates[i] != buft, "K-pack is an automatic model buffer candidate");
            }
        }
        printf("KPACK_LOAD_DEFAULT native=PASS automatic_kpack=DISABLED\n");
        {
            ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 8, nullptr, true}));
            require(ctx != nullptr, "context allocation failed");
            require(!query(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 512, 256), nullptr), "float weights accepted");
            require(!query(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 512, 128), nullptr), "unsupported N accepted");
            require(!query(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q3_K, 256, 256), nullptr),
                    "unpaired metadata accepted");
            auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 512, 256);
            require(!query(ggml_view_tensor(ctx.get(), weight), nullptr), "weight view accepted");
        }
        ggml_context *                                      metadata = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(fixture, {true, &metadata}),
                                                                 gguf_free);
        ggml_context_ptr                                    metadata_owner(metadata);
        require(file != nullptr && metadata != nullptr, "cannot read reference GGUF");
        const int64_t key = gguf_find_key(file.get(), "quactlize.kpack.reference.manifest");
        require(key >= 0 && gguf_get_kv_type(file.get(), key) == GGUF_TYPE_STRING, "missing reference manifest");
        const auto manifest = nlohmann::json::parse(gguf_get_val_str(file.get(), key));
        require(manifest.at("schema") == "quactlize.kquant-kpack.reference-gguf" &&
                    manifest.at("schema_version") == 1 && !manifest.at("tensors").empty(),
                "invalid reference schema");
        std::ifstream input(fixture, std::ios::binary);
        auto          bytes = [&](const std::string & name) {
            const int64_t index = gguf_find_tensor(file.get(), name.c_str());
            require(index >= 0, "missing reference tensor");
            std::vector<uint8_t> out(gguf_get_tensor_size(file.get(), index));
            input.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), index));
            require(bool(input.read(reinterpret_cast<char *>(out.data()), out.size())), "truncated reference tensor");
            return out;
        };
        size_t tested = 0;
        for (const auto & record : manifest.at("tensors")) {
            const int qtype = record.at("qtype"), n = record.at("n"), k = record.at("k"),
                      experts        = record.at("experts");
            const std::string name   = record.at("source_name");
            const auto *      source = ggml_get_tensor(metadata, name.c_str());
            require(source && source->type == qtype && source->ne[0] == k && source->ne[1] == n &&
                        source->ne[2] == experts && source->ne[3] == 1,
                    "source geometry differs");
            const auto           raw = bytes(name);
            std::vector<uint8_t> expected;
            size_t               plane_bytes[3]{};
            int                  plane_index = 0;
            for (const char * plane : {"low", "high", "units"}) {
                const auto & carrier = record.at("carriers").at(plane);
                if (!carrier.is_null()) {
                    auto part                = bytes(carrier.at("name"));
                    plane_bytes[plane_index] = part.size();
                    expected.insert(expected.end(), part.begin(), part.end());
                }
                ++plane_index;
            }
            require(raw.size() == expected.size(), "reference is not byte-neutral");
            ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true}));
            require(ctx != nullptr, "context allocation failed");
            ggml_tensor * tensors[3];
            for (auto & t : tensors) {
                t = ggml_new_tensor_3d(ctx.get(), source->type, k, n, experts);
            }
            ggml_quactlize_packed_layout layout{};
            require(query(tensors[0], &layout), "producer rejects fixture");
            const auto &                        a = record.at("arrangement");
            quactlize_ppu_placed_arrangement_v2 declared{
                a.at("version"),    a.at("layout"),          a.at("bits"),
                a.at("high_bits"),  a.at("artifact_tile_k"), a.at("transport_tile_k"),
                a.at("group_size"), a.at("reserved"),        a.at("mapping_id")};
            require(memcmp(&declared, &layout.arrangement, sizeof(declared)) == 0, "canonical arrangement differs");
            require(layout.sizes.raw_bytes == raw.size() && layout.sizes.low_bytes == plane_bytes[0] &&
                        layout.sizes.high_bytes == plane_bytes[1] && layout.sizes.units_bytes == plane_bytes[2],
                    "plane sizes differ");
            ggml_backend_buffer_ptr packed(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            require(packed != nullptr, "packed allocation failed");
            std::vector<uint8_t> observed(expected.size());
            require(!read(tensors[0], observed.data(), observed.size()), "read accepted an uninitialized artifact");
            ggml_backend_tensor_set(tensors[0], raw.data(), 0, raw.size());
            const size_t cut = raw.size() / 3;
            ggml_backend_tensor_set(tensors[1], raw.data(), 0, cut);
            require(!read(tensors[1], observed.data(), observed.size()), "read accepted a partial artifact");
            ggml_backend_tensor_set(tensors[1], raw.data() + cut, cut, raw.size() - cut);
            auto invalid = layout;
            invalid.arrangement.mapping_id ^= 1;
            require(!install(tensors[2], &invalid, expected.data(), expected.size()), "wrong mapping accepted");
            require(!install(tensors[2], &layout, expected.data(), expected.size() - 1), "wrong byte count accepted");
            require(install(tensors[2], &layout, expected.data(), expected.size()), "offline upload rejected");
            for (auto * t : tensors) {
                require(read(t, observed.data(), observed.size()), "packed readback failed");
                require(observed == expected, "online/offline packed bytes differ");
                require(!install(t, &layout, expected.data(), expected.size()), "duplicate upload accepted");
            }
            printf("KPACK_LOAD q=%d n=%d k=%d experts=%d whole=PASS chunked=PASS offline=PASS\n", qtype, n, k, experts);
            ++tested;
            if (reject_forward) {
                auto * act   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, 1, experts);
                auto * out   = ggml_mul_mat(ctx.get(), tensors[0], act);
                auto * graph = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, out);
                // The load-only guard must run before allocation-dependent launch or fusion logic.
                ggml_backend_graph_compute(backend, graph);
                throw std::runtime_error("packed forward did not abort");
            }
        }
        printf("KPACK_LOAD PASS tensors=%zu inference=DISABLED\n", tested);
        return true;
    } catch (const std::exception & error) {
        fprintf(stderr, "KPACK_LOAD FAIL: %s\n", error.what());
        return false;
    }
}

static void usage(char ** argv) {
    printf("Usage: %s -b <backend> [--reject-forward] <reference.gguf>\n", argv[0]);
    printf("    --reject-forward additionally expects a load-only inference abort\n");
}

int main(int argc, char ** argv) {
    const char * backend_name   = nullptr;
    const char * fixture        = nullptr;
    bool         reject_forward = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--reject-forward") == 0) {
            reject_forward = true;
        } else if (!fixture && argv[i][0] != '-') {
            fixture = argv[i];
        } else {
            usage(argv);
            return 1;
        }
    }
    if (!backend_name || !fixture) {
        usage(argv);
        return 1;
    }

    ggml_backend_load_all();

    auto * dev = ggml_backend_dev_by_name(backend_name);
    if (!dev) {
        fprintf(stderr, "KPACK_LOAD FAIL: requested backend is unavailable\n");
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    return backend && test_kpack_load(backend.get(), fixture, reject_forward) ? 0 : 1;
}
