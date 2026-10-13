// K-pack load and dense compute tests. The Python runner supplies offline packing references;
// dense tests compare GPU results with independent GGUF dequantization and a CPU dot product.

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
#include <limits>
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

static void dense_require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static double check_dense_output(const std::vector<float> &  actual,
                                 const std::vector<double> & expected,
                                 int                         n,
                                 int                         m,
                                 float                       sign,
                                 const std::vector<float> *  bias = nullptr) {
    double worst = 0;
    for (int row = 0; row < m; ++row) {
        double error = 0, norm = 0, max_error = 0, max_value = 0;
        for (int col = 0; col < n; ++col) {
            const size_t i    = size_t(row) * n + col;
            const double want = sign * expected[i] + (bias ? (*bias)[i] : 0.0);
            const double got  = actual[i];
            dense_require(std::isfinite(got) && std::isfinite(want), "nonfinite dense output/reference");
            const double delta = std::abs(got - want);
            error += delta * delta;
            norm += want * want;
            max_error = std::max(max_error, delta);
            max_value = std::max(max_value, std::abs(want));
        }
        const double relative = sqrt(error / std::max(norm, 1e-30));
        worst                 = std::max(worst, relative);
        // Check each row and its largest error; a correct large row must not hide a broken tail row.
        if (relative >= .02 || max_error > .02 * max_value + 1e-3) {
            fprintf(stderr, "KPACK_DENSE_ERROR row=%d relative=%.8g max_error=%.8g max_reference=%.8g\n", row, relative,
                    max_error, max_value);
            throw std::runtime_error("dense output differs from independent GGUF reference");
        }
    }
    return worst;
}

static void dense_fixture(const std::vector<float> & decoded,
                          int                        n,
                          int                        k,
                          int                        m,
                          std::vector<float> (&inputs)[6],
                          std::vector<double> (&expected)[6]) {
    for (int replay = 0; replay < 6; ++replay) {
        inputs[replay].resize(size_t(m) * k);
        expected[replay].resize(size_t(m) * n);
    }
    for (int row = 0; row < m; ++row) {
        const int first = (row * 257 + 31) % k, last = k - 1 - (row * 32 % k);
        inputs[1][size_t(row) * k + first] = 1.0f;
        inputs[5][size_t(row) * k + last]  = 32752.0f;
        for (int j = 0; j < k; ++j) {
            const size_t i = size_t(row) * k + j;
            inputs[2][i]   = float(int((i * 13 + i / 31 * 7) % 67) - 33) / 64.0f;
            inputs[3][i]   = -inputs[2][i];
            inputs[4][i]   = 16.0f * inputs[2][i];
        }
        for (int col = 0; col < n; ++col) {
            const size_t i   = size_t(row) * n + col;
            double       dot = 0;
            for (int j = 0; j < k; ++j) {
                dot += double(inputs[2][size_t(row) * k + j]) * decoded[size_t(col) * k + j];
            }
            expected[1][i] = decoded[size_t(col) * k + first];
            expected[2][i] = dot;
            expected[3][i] = -dot;
            expected[4][i] = 16.0 * dot;
            expected[5][i] = 32752.0 * decoded[size_t(col) * k + last];
        }
    }
}

static bool test_kpack_dense(ggml_backend_t backend, const std::string & suite, int qtype, bool reject_grouped) {
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
        ggml_backend_ptr second(ggml_backend_dev_init(dev, nullptr));
        require(second != nullptr, "second backend allocation failed");
        ggml_backend_t backends[] = {backend, second.get()};
        int cases = 0;
        for (auto type :
             {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
            if (qtype != -1 && type != qtype) {
                continue;
            }
            if (suite == "contract") {
                ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 48 + ggml_graph_overhead(), nullptr, true}));
                require(ctx != nullptr, "context allocation failed");
                for (int n : {128, 255, 257, 384}) {
                    auto * weight = ggml_new_tensor_2d(ctx.get(), type, 512, n);
                    require(!preferred(dev, weight, GGML_OP_MUL_MAT), "unaligned N selected K-pack");
                }
                if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K) {
                    const int k      = type == GGML_TYPE_Q8_0 ? 288 : 768;
                    auto *    weight = ggml_new_tensor_2d(ctx.get(), type, k, 256);
                    require(!preferred(dev, weight, GGML_OP_MUL_MAT), "unsupported K selected K-pack");
                }
                auto * weight = ggml_new_tensor_2d(ctx.get(), type, 512, 256);
                require(preferred(dev, weight, GGML_OP_MUL_MAT) == bufts[0], "aligned shape was rejected");
                const int min_k   = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 512 : 256;
                auto *    minimum = ggml_new_tensor_2d(ctx.get(), type, min_k, 256);
                require(preferred(dev, minimum, GGML_OP_MUL_MAT) == bufts[0], "minimum aligned K was rejected");
                require(!preferred(dev, ggml_view_tensor(ctx.get(), weight), GGML_OP_MUL_MAT),
                        "weight view selected K-pack");
                auto * experts = ggml_new_tensor_3d(ctx.get(), type, 512, 256, 2);
                require(!preferred(dev, experts, GGML_OP_MUL_MAT), "batched weight selected as dense");
                for (auto op : {GGML_OP_MUL_MAT_ID, GGML_OP_GET_ROWS, GGML_OP_NONE}) {
                    require(!preferred(dev, weight, op), "non-dense consumer selected K-pack");
                }
                // Bind just the weight; unsupported graph nodes must be rejected before launch or allocation.
                ggml_backend_buffer_ptr packed(ggml_backend_buft_alloc_buffer(bufts[0], ggml_nbytes(weight)));
                require(packed && ggml_backend_tensor_alloc(packed.get(), weight,
                                                            ggml_backend_buffer_get_base(packed.get())) ==
                                      GGML_STATUS_SUCCESS,
                        "packed allocation failed");
                auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 512, 3);
                auto * mm    = ggml_mul_mat(ctx.get(), weight, input);
                require(ggml_backend_supports_op(backend, mm), "M=3 tail was rejected");
                auto * ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 3);
                auto * indexed = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 512, 1, 3);
                auto * grouped = ggml_mul_mat_id(ctx.get(), weight, indexed, ids);
                require(!ggml_backend_supports_op(backend, grouped), "grouped operation admitted by dense MR");
                if (reject_grouped) {
                    auto * graph = ggml_new_graph(ctx.get());
                    ggml_build_forward_expand(graph, grouped);
                    ggml_backend_graph_compute(backend, graph);
                    throw std::runtime_error("grouped packed forward did not abort");
                }
                for (auto storage : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
                    auto * narrow = ggml_new_tensor_2d(ctx.get(), storage, 512, 3);
                    require(!ggml_backend_supports_op(backend, ggml_mul_mat(ctx.get(), weight, narrow)),
                            "unsupported activation storage admitted");
                }
                auto * strided = ggml_view_2d(ctx.get(), input, 512, 2, 1024 * sizeof(float), 0);
                require(!ggml_backend_supports_op(backend, ggml_mul_mat(ctx.get(), weight, strided)),
                        "noncontiguous input admitted");
                for (auto unsupported : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0}) {
                    auto * raw = ggml_new_tensor_2d(ctx.get(), unsupported, 512, 256);
                    require(!preferred(dev, raw, GGML_OP_MUL_MAT), "unsupported weight type selected K-pack");
                }
                printf("KPACK_DENSE_CONTRACT q=%d unaligned=REJECTED tail=ADMITTED grouped=REJECTED status=PASS\n",
                       type);
                ++cases;
                continue;
            }
            const int short_k     = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1024 : 768;
            const int shapes[][2] = {{256, 512}, {768, 1536}, {1024, 5120}, {512, short_k}};
            for (int shape = 0; shape < 4; ++shape) {
                const int        n = shapes[shape][0], k = shapes[shape][1];
                ggml_context_ptr weights(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
                require(weights != nullptr, "weight context allocation failed");
                auto * weight = ggml_new_tensor_2d(weights.get(), type, k, n);
                auto * cached = ggml_new_tensor_2d(weights.get(), type, k, n);
                ggml_set_name(weight, "dense-online");
                ggml_set_name(cached, "dense-cached");
                auto * buft = preferred(dev, weight, GGML_OP_MUL_MAT);
                require(buft == bufts[0], "default dense weight placement is not K-pack");
                require(!preferred(dev, weight, GGML_OP_MUL_MAT_ID), "grouped weight was selected automatically");
                require(!preferred(dev, weight, GGML_OP_GET_ROWS), "embedding weight was selected automatically");
                require(!preferred(dev, weight, GGML_OP_NONE),
                        "weight with no known consumer was selected automatically");
                ggml_backend_buffer_ptr packed(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
                require(packed != nullptr, "packed allocation failed");
                std::vector<float> source(n * k), decoded(n * k);
                uint32_t           state = 42 + type;
                for (size_t i = 0; i < source.size(); ++i) {
                    state     = state * 1664525u + 1013904223u;
                    source[i] = float(int(state >> 22) - 512) / float(1024 << ((i / 256 + i / k) % 4));
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
                std::vector<int> rows;
                if (suite != "gemm") {
                    const std::vector<int> decode[] = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 3, 7}, {1, 8}, {3}};
                    rows.insert(rows.end(), decode[shape].begin(), decode[shape].end());
                }
                if (suite != "gemv") {
                    const std::vector<int> prefill[] = {
                        {9, 15, 16, 17, 31, 32, 33, 127, 128, 129}, {17, 33, 129}, {16, 128}, {17}};
                    rows.insert(rows.end(), prefill[shape].begin(), prefill[shape].end());
                }
                for (int m : rows) {
                    printf("KPACK_DENSE_START suite=%s q=%d m=%d n=%d k=%d\n", suite.c_str(), type, m, n, k);
                    fflush(stdout);
                    std::vector<float>  a[6];
                    std::vector<double> expected[6];
                    dense_fixture(decoded, n, k, m, a, expected);
                    double worst = 0;
                    for (auto * source_weight : {weight, cached}) {
                        ggml_context_ptr        contexts[2];
                        ggml_backend_buffer_ptr buffers[2];
                        ggml_tensor *           input[2], *bias[2], *mm[2], *output[2];
                        ggml_cgraph *           graphs[2];
                        for (int lane = 0; lane < 2; ++lane) {
                            contexts[lane].reset(
                                ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true}));
                            auto * ctx = contexts[lane].get();
                            require(ctx != nullptr, "context allocation failed");
                            input[lane]  = m == 6 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 2, 3) :
                                                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
                            mm[lane]     = ggml_mul_mat(ctx, source_weight, input[lane]);
                            bias[lane]   = ggml_dup_tensor(ctx, mm[lane]);
                            output[lane] = ggml_add(ctx, mm[lane], bias[lane]);
                            // Keep both results observable, even if the backend has a MUL_MAT+ADD fusion.
                            ggml_set_output(mm[lane]);
                            ggml_set_output(output[lane]);
                            graphs[lane] = ggml_new_graph(ctx);
                            ggml_build_forward_expand(graphs[lane], output[lane]);
                            buffers[lane].reset(ggml_backend_alloc_ctx_tensors(ctx, backends[lane]));
                            require(buffers[lane] && ggml_backend_supports_op(backends[lane], mm[lane]),
                                    "dense operation not admitted");
                        }
                        require(input[0]->data != input[1]->data && mm[0]->data != mm[1]->data,
                                "dynamic-address fixture aliases");
                        std::vector<float>       activations[2], biases[2], actual(size_t(m) * n);
                        const std::vector<float> poison(actual.size(), std::numeric_limits<float>::quiet_NaN());
                        for (int replay = 0; replay < 6; ++replay) {
                            for (int lane = 0; lane < 2; ++lane) {
                                const float sign  = lane ? -1.0f : 1.0f;
                                activations[lane] = a[replay];
                                for (auto & value : activations[lane]) {
                                    value *= sign;
                                }
                                biases[lane].resize(actual.size());
                                for (size_t i = 0; i < actual.size(); ++i) {
                                    biases[lane][i] = sign * float(int(i % 17) - 8 + replay) / 32.0f;
                                }
                                ggml_backend_tensor_set(input[lane], activations[lane].data(), 0,
                                                        ggml_nbytes(input[lane]));
                                ggml_backend_tensor_set(bias[lane], biases[lane].data(), 0, ggml_nbytes(bias[lane]));
                                ggml_backend_tensor_set(mm[lane], poison.data(), 0, ggml_nbytes(mm[lane]));
                                ggml_backend_tensor_set(output[lane], poison.data(), 0, ggml_nbytes(output[lane]));
                            }
                            // One immutable weight plan serves two backend streams and different A/C addresses.
                            ggml_status status[2];
                            for (int lane = 0; lane < 2; ++lane) {
                                status[lane] = ggml_backend_graph_compute_async(backends[lane], graphs[lane]);
                            }
                            for (int lane = 0; lane < 2; ++lane) {
                                ggml_backend_synchronize(backends[lane]);
                            }
                            require(status[0] == GGML_STATUS_SUCCESS && status[1] == GGML_STATUS_SUCCESS,
                                    "dense forward failed");
                            for (int lane = 0; lane < 2; ++lane) {
                                const float sign = lane ? -1.0f : 1.0f;
                                for (bool with_bias : {false, true}) {
                                    ggml_backend_tensor_get(with_bias ? output[lane] : mm[lane], actual.data(), 0,
                                                            actual.size() * sizeof(float));
                                    worst = std::max(worst, check_dense_output(actual, expected[replay], n, m, sign,
                                                                               with_bias ? &biases[lane] : nullptr));
                                }
                                std::vector<float> unchanged(activations[lane].size());
                                ggml_backend_tensor_get(input[lane], unchanged.data(), 0, ggml_nbytes(input[lane]));
                                require(unchanged == activations[lane], "dense compute modified its input");
                            }
                        }
                    }
                    printf(
                        "KPACK_DENSE suite=%s q=%d m=%d n=%d k=%d replays=6 streams=2 weights=2 relative=%.8g "
                        "status=PASS\n",
                        suite.c_str(), type, m, n, k, worst);
                    ++cases;
                }
                for (auto * source_weight : {weight, cached}) {
                    std::vector<uint8_t> unchanged(bytes.size());
                    require(read(source_weight, unchanged.data(), unchanged.size()) && unchanged == bytes,
                            "dense compute modified packed weights");
                }
            }
        }
        const int formats    = qtype == -1 ? 6 : 1;
        const int per_format = suite == "contract" ? 1 : suite == "gemv" ? 14 : suite == "gemm" ? 16 : 30;
        require(cases == formats * per_format, "incomplete dense coverage");
        printf("KPACK_DENSE PASS suite=%s cases=%d formats=%d storage=F32 grouped=DISABLED\n", suite.c_str(), cases,
               formats);
        return true;
    } catch (const std::exception & error) {
        fprintf(stderr, "KPACK_DENSE FAIL: %s\n", error.what());
        return false;
    }
}

static void usage(char ** argv) {
    printf("Usage: %s -b <backend> [--reject-forward] <reference.gguf>\n", argv[0]);
    printf("    --reject-forward additionally expects a load-only inference abort\n");
    printf(
        "       %s -b <backend> --dense [--suite gemv|gemm|contract] [--qtype 8|10|11|12|13|14] [--reject-grouped]\n",
        argv[0]);
}

int main(int argc, char ** argv) {
    const char * backend_name   = nullptr;
    const char * fixture        = nullptr;
    bool         reject_forward = false;
    bool         dense = false, reject_grouped = false;
    std::string  suite = "all";
    int          qtype = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--reject-forward") == 0) {
            reject_forward = true;
        } else if (strcmp(argv[i], "--dense") == 0) {
            dense = true;
        } else if (strcmp(argv[i], "--reject-grouped") == 0) {
            reject_grouped = true;
        } else if (strcmp(argv[i], "--suite") == 0 && i + 1 < argc) {
            suite = argv[++i];
        } else if (strcmp(argv[i], "--qtype") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value != "8" && value != "10" && value != "11" && value != "12" && value != "13" && value != "14") {
                usage(argv);
                return 1;
            }
            qtype = std::stoi(value);
        } else if (!fixture && argv[i][0] != '-') {
            fixture = argv[i];
        } else {
            usage(argv);
            return 1;
        }
    }
    if (!backend_name ||
        (dense ? fixture || reject_forward : !fixture || reject_grouped || suite != "all" || qtype != -1) ||
        (suite != "all" && suite != "gemv" && suite != "gemm" && suite != "contract")) {
        usage(argv);
        return 1;
    }

    if (!dense) {
        setenv("GGML_QUACTLIZE_DENSE", "0", 1);
    }
    if (reject_grouped) {
        suite = "contract";
    }

    ggml_backend_load_all();

    auto * dev = ggml_backend_dev_by_name(backend_name);
    if (!dev) {
        fprintf(stderr, "KPACK_LOAD FAIL: requested backend is unavailable\n");
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    return backend && (dense ? test_kpack_dense(backend.get(), suite, qtype, reject_grouped) :
                               test_kpack_load(backend.get(), fixture, reject_forward)) ?
               0 :
               1;
}
