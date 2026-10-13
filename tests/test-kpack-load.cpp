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
#include <algorithm>
#include <cmath>
#include <cstdlib>
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
        auto * dev       = ggml_backend_get_device(backend);
        auto * reg       = ggml_backend_dev_backend_reg(dev);
        auto   preferred = reinterpret_cast<ggml_backend_dev_get_weight_buft_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_weight_buft"));
        auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
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
            require(!preferred || !preferred(dev, weight, GGML_OP_MUL_MAT), "disabled dense selected K-pack");
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

static bool test_kpack_dense(ggml_backend_t backend, bool reject_grouped) {
    auto require = [](bool ok, const char * message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    };
    try {
        auto * dev       = ggml_backend_get_device(backend);
        auto * reg       = ggml_backend_dev_backend_reg(dev);
        auto   preferred = reinterpret_cast<ggml_backend_dev_get_weight_buft_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_weight_buft"));
        auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_override_bufts"));
        auto read = reinterpret_cast<decltype(&ggml_quactlize_read_packed)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_read_packed"));
        auto install = reinterpret_cast<decltype(&ggml_quactlize_set_packed)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_set_packed"));
        auto query = reinterpret_cast<decltype(&ggml_quactlize_can_load)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_quactlize_can_load"));
        require(extra && read && install && query && preferred, "packed buffer API is unavailable");
        auto ** bufts = extra(dev);
        require(bufts && bufts[0], "packer is unavailable");
        int cases = 0;
        for (auto type :
             {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
            const int        n = 256, k = 512;
            ggml_context_ptr weights(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
            auto *           weight = ggml_new_tensor_2d(weights.get(), type, k, n);
            auto *           cached = ggml_new_tensor_2d(weights.get(), type, k, n);
            ggml_set_name(weight, "dense-online");
            ggml_set_name(cached, "dense-cached");
            auto * buft = preferred(dev, weight, GGML_OP_MUL_MAT);
            require(buft == bufts[0], "default dense weight placement is not K-pack");
            require(!preferred(dev, weight, GGML_OP_MUL_MAT_ID), "grouped weight was selected automatically");
            require(!preferred(dev, weight, GGML_OP_GET_ROWS), "embedding weight was selected automatically");
            require(!preferred(dev, weight, GGML_OP_NONE), "weight with no known consumer was selected automatically");
            ggml_backend_buffer_ptr packed(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
            require(packed != nullptr, "packed allocation failed");
            std::vector<float> source(n * k), decoded(n * k);
            for (size_t i = 0; i < source.size(); ++i) {
                source[i] = float(int((i * 17 + i / k * 7) % 127) - 63) / 128.0f;
            }
            std::vector<uint8_t> raw(ggml_nbytes(weight)), bytes(raw.size());
            require(ggml_quantize_chunk(type, source.data(), raw.data(), 0, n, k, nullptr) == raw.size(),
                    "fixture quantization failed");
            ggml_get_type_traits(type)->to_float(raw.data(), decoded.data(), n * k);
            ggml_backend_tensor_set(weight, raw.data(), 0, raw.size());
            require(read(weight, bytes.data(), bytes.size()), "packed read failed");
            ggml_quactlize_packed_layout layout{};
            require(query(cached, &layout) && install(cached, &layout, bytes.data(), bytes.size()),
                    "cached upload failed");
            for (int m : {1, 2, 4, 8, 16, 128}) {
                ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true}));
                auto *           input  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
                auto *           bias   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, m);
                auto *           mm     = ggml_mul_mat(ctx.get(), m % 2 ? weight : cached, input);
                auto *           output = ggml_add(ctx.get(), mm, bias);
                auto *           graph  = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, output);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                require(buffer && ggml_backend_supports_op(backend, mm), "dense operation not admitted");
                auto * ids           = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, m);
                auto * indexed_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, 1, m);
                auto * grouped       = ggml_mul_mat_id(ctx.get(), weight, indexed_input, ids);
                require(!ggml_backend_supports_op(backend, grouped), "grouped operation admitted by dense MR");
                if (reject_grouped) {
                    ggml_graph_clear(graph);
                    ggml_build_forward_expand(graph, grouped);
                    ggml_backend_graph_compute(backend, graph);
                    throw std::runtime_error("grouped packed forward did not abort");
                }
                std::vector<float> a(m * k), b(m * n), actual(m * n);
                double             worst = 0;
                // Repeated calls exercise graph capture and changing activation values.
                for (int replay = 0; replay < 6; ++replay) {
                    for (size_t i = 0; i < a.size(); ++i) {
                        a[i] = float(int((i * 13 + replay * 11) % 67) - 33) / 64.0f;
                    }
                    std::fill(b.begin(), b.end(), float(replay + 1) / 8.0f);
                    ggml_backend_tensor_set(input, a.data(), 0, a.size() * sizeof(float));
                    ggml_backend_tensor_set(bias, b.data(), 0, b.size() * sizeof(float));
                    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "dense forward failed");
                    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                    double error = 0, norm = 0;
                    for (int row = 0; row < m; ++row) {
                        for (int col = 0; col < n; ++col) {
                            double expected = b[row * n + col];
                            for (int j = 0; j < k; ++j) {
                                expected += double(a[row * k + j]) * decoded[col * k + j];
                            }
                            const float value = actual[row * n + col];
                            require(std::isfinite(value), "nonfinite dense output");
                            error += (value - expected) * (value - expected);
                            norm += expected * expected;
                        }
                    }
                    const double relative = sqrt(error / std::max(norm, 1e-30));
                    worst                 = std::max(worst, relative);
                    require(relative < .02, "dense output differs from independent GGUF reference");
                }
                printf("KPACK_DENSE q=%d m=%d n=%d k=%d replays=6 relative=%.8g status=PASS\n", type, m, n, k, worst);
                ++cases;
            }
        }
        printf("KPACK_DENSE PASS cases=%d formats=6 storage=F32 grouped=DISABLED\n", cases);
        return true;
    } catch (const std::exception & error) {
        fprintf(stderr, "KPACK_DENSE FAIL: %s\n", error.what());
        return false;
    }
}

static void usage(char ** argv) {
    printf("Usage: %s -b <backend> [--reject-forward] <reference.gguf>\n", argv[0]);
    printf("    --reject-forward additionally expects a load-only inference abort\n");
    printf("       %s -b <backend> --dense [--reject-grouped]\n", argv[0]);
}

int main(int argc, char ** argv) {
    const char * backend_name   = nullptr;
    const char * fixture        = nullptr;
    bool         reject_forward = false;
    bool         dense = false, reject_grouped = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--reject-forward") == 0) {
            reject_forward = true;
        } else if (strcmp(argv[i], "--dense") == 0) {
            dense = true;
        } else if (strcmp(argv[i], "--reject-grouped") == 0) {
            reject_grouped = true;
        } else if (!fixture && argv[i][0] != '-') {
            fixture = argv[i];
        } else {
            usage(argv);
            return 1;
        }
    }
    if (!backend_name || (dense ? fixture || reject_forward : !fixture || reject_grouped)) {
        usage(argv);
        return 1;
    }

    if (!dense) {
        setenv("GGML_QUACTLIZE_DENSE", "0", 1);
    }

    ggml_backend_load_all();

    auto * dev = ggml_backend_dev_by_name(backend_name);
    if (!dev) {
        fprintf(stderr, "KPACK_LOAD FAIL: requested backend is unavailable\n");
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    return backend && (dense ? test_kpack_dense(backend.get(), reject_grouped) :
                               test_kpack_load(backend.get(), fixture, reject_forward)) ?
               0 :
               1;
}
