// Real GGUF/model-loader admission with host-only backend capabilities.
#include "llama-model-loader.h"
#include "ggml-backend-impl.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_backend_reg registry{};
static ggml_backend_device device{};
static ggml_backend_buffer_type ordinary{}, kpack{}, other_kpack{};
static bool producer_available = true, merged_supported = true;
static int pair_queries = 0;

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static bool pair_supported(ggml_backend_buffer_type_t buft, const ggml_tensor * merged) {
    ++pair_queries;
    return buft == &kpack && merged_supported && merged->type == GGML_TYPE_Q4_K &&
           merged->ne[0] == 256 && merged->ne[1] == 512 && merged->ne[2] == 2;
}

static void set_pair(ggml_tensor *, const void *, const void *, size_t) {
    throw std::runtime_error("metadata admission must not upload weights");
}

static void init_backend() {
    registry.api_version = GGML_BACKEND_API_VERSION;
    registry.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (!strcmp(name, "ggml_quactlize_pair_supported")) { return (void *) pair_supported; }
        if (!strcmp(name, "ggml_quactlize_set_gate_up") && producer_available) { return (void *) set_pair; }
        return nullptr;
    };
    device.reg = &registry;
    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    device.iface.get_name = [](ggml_backend_dev_t) { return "pair-test"; };
    for (auto * buft : {&ordinary, &kpack, &other_kpack}) {
        buft->device = &device;
        buft->iface.get_name = [](ggml_backend_buffer_type_t b) {
            return b == &ordinary ? "CUDA0" : b == &kpack ? "CUDA0_KPACK" : "CUDA1_KPACK";
        };
    }
}

struct options {
    ggml_type up_type = GGML_TYPE_Q4_K;
    int up_n = 256;
    bool bias = false, scale = false, enabled = true, premerged = false, missing_up = false;
};

static FILE * fixture(const options & opt) {
    ggml_context_ptr ctx(ggml_init({4 * 1024 * 1024, nullptr, false}));
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "qwen35moe");
    auto add = [&](const char * name, ggml_type type, int n) {
        auto * tensor = ggml_new_tensor_3d(ctx.get(), type, 256, n, 2);
        ggml_set_name(tensor, name);
        memset(tensor->data, 0, ggml_nbytes(tensor));
        gguf_add_tensor(meta.get(), tensor);
    };
    if (opt.premerged) {
        add("blk.0.ffn_gate_up_exps.weight", GGML_TYPE_Q4_K, 512);
    } else {
        add("blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 256);
        if (!opt.missing_up) { add("blk.0.ffn_up_exps.weight", opt.up_type, opt.up_n); }
        if (opt.bias) { add("blk.0.ffn_gate_exps.bias", GGML_TYPE_F32, 1); }
        if (opt.scale) { add("blk.0.ffn_up_exps.weight_scale", GGML_TYPE_F32, 1); }
    }
    FILE * file = tmpfile();
    require(file && gguf_write_to_file_ptr(meta.get(), file, false), "cannot write GGUF fixture");
    rewind(file);
    return file;
}

static void run(const char * label, std::vector<llama_model_tensor_buft_override> rules,
                bool want_merged, const options & opt = {}) {
    rules.push_back({nullptr, nullptr});
    require(setenv("QUACTLIZE_KPACK_PAIR_WEIGHTS", opt.enabled ? "1" : "0", 1) == 0, "setenv failed");
    pair_queries = 0;
    std::unique_ptr<FILE, decltype(&fclose)> file(fixture(opt), fclose);
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file.get(),
                              false, false, false, true, nullptr, rules.data());
    llama_hparams hp{};
    hp.n_layer_all = 1;
    hp.n_expert = 2;
    hp.n_expert_used = 1;
    buft_list_t buffers{{&device, &ordinary}, {&device, &kpack}};
    const LLM_TN tn(LLM_ARCH_QWEN35MOE);
    auto * merged = loader.create_tensor(hp, &buffers, &buffers, &buffers, &buffers,
        tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", 0), {256, 512, 2},
        llama_model_loader::TENSOR_NOT_REQUIRED);
    require(bool(merged) == want_merged, "merged tensor admission differs");
    if (merged) {
        require(loader.ctx_map.size() == 1 && loader.ctx_map.begin()->first == &kpack,
                "merged tensor did not inherit K-pack placement");
        const auto & weight = loader.require_weight(merged->name);
        if (opt.premerged) {
            require(weight.paired_sources.empty() && pair_queries == 0 && loader.n_created == 1,
                    "preconverted GGUF must use ordinary tensor intake");
        } else {
            require(weight.paired_sources == std::vector<std::string>{
                "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}, "paired source order differs");
            require(pair_queries == 1 && loader.n_created == 2, "missing merged capability check/count");
            const auto & gate = loader.require_weight(weight.paired_sources[0].c_str());
            const auto & up = loader.require_weight(weight.paired_sources[1].c_str());
            require(gate.tensor->ne[1] == 256 && up.tensor->ne[1] == 256,
                    "source metadata was changed");
            require(ggml_nbytes(merged) == ggml_nbytes(gate.tensor) + ggml_nbytes(up.tensor),
                    "pair changes byte count");
        }
    } else {
        require(loader.get_weight("blk.0.ffn_gate_up_exps.weight") == nullptr && loader.n_created == 0,
                "declined pair created a synthetic weight");
    }
    printf("KPACK_MODEL_PAIR case=%s merged=%d capability_queries=%d PASS\n", label, int(merged != nullptr), pair_queries);
}

int main() {
    try {
        init_backend();
        const char * gate = "^blk\\.0\\.ffn_gate_exps\\.weight$";
        const char * up = "^blk\\.0\\.ffn_up_exps\\.weight$";
        const char * pair = "^blk\\.0\\.ffn_gate_up_exps\\.weight$";
        const char * both = "^blk\\.0\\.ffn_(gate|up)_exps\\.weight$";
        const std::vector<llama_model_tensor_buft_override> exact{{gate, &kpack}, {up, &kpack}};
        run("source-exact-names", exact, true);
        run("source-combined-regex", {{both, &kpack}}, true);
        run("merged-name-only", {{pair, &kpack}}, true);
        run("merged-and-sources", {{pair, &kpack}, {both, &kpack}}, true);
        run("first-matching-source-rule", {{both, &kpack}, {gate, &ordinary}}, true);
        run("default-buffer", {}, false);
        run("mixed-source-devices", {{gate, &kpack}, {up, &other_kpack}}, false);
        run("mixed-source-buffers", {{gate, &kpack}, {up, &ordinary}}, false);
        run("only-one-source-override", {{gate, &kpack}}, false);
        run("merged-rule-conflict", {{pair, &ordinary}, {both, &kpack}}, false);
        run("earlier-source-rule-conflict", {{gate, &ordinary}, {both, &kpack}}, false);
        producer_available = false;
        run("no-pair-producer", exact, false);
        producer_available = true;
        merged_supported = false;
        run("no-merged-kernel", exact, false);
        merged_supported = true;
        options opt;
        opt.up_type = GGML_TYPE_Q5_K; run("different-qtypes", exact, false, opt);
        opt = {}; opt.up_n = 512; run("different-shapes", exact, false, opt);
        opt = {}; opt.bias = true; run("projection-bias", exact, false, opt);
        opt = {}; opt.scale = true; run("projection-scale", exact, false, opt);
        opt = {}; opt.missing_up = true; run("missing-up", exact, false, opt);
        opt = {}; opt.enabled = false; run("pair-disabled", exact, false, opt);
        opt.premerged = true; run("convert-script-merged-gguf", {{pair, &kpack}}, true, opt);
        require(unsetenv("QUACTLIZE_KPACK_PAIR_WEIGHTS") == 0, "unsetenv failed");
        puts("KPACK_MODEL_PAIR_ALL PASS cases=20 source=PRODUCTION_MODEL_LOADER gpu=0");
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "KPACK_MODEL_PAIR FAIL: %s\n", ex.what());
        return 1;
    }
}
