#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct weight_device {
    ggml_backend_device        device{};
    ggml_backend_buffer_type   normal{}, preferred{}, other{};
    ggml_backend_buffer_type_t extras[3]{};
    ggml_tensor                observed{};
    bool                       enabled = true, alternate = false, supported = true;
    size_t                     calls = 0;
};

static ggml_backend_buffer_type_t preferred_weight(ggml_backend_dev_t dev, const ggml_tensor * weight, ggml_op op) {
    auto & state = *static_cast<weight_device *>(dev->context);
    ++state.calls;
    state.observed = *weight;
    if (!state.enabled || op != GGML_OP_MUL_MAT || weight->type == GGML_TYPE_I32 || weight->ne[0] % 512 ||
        weight->ne[1] % 256) {
        return nullptr;
    }
    check(weight->buffer == nullptr && weight->data == nullptr && ggml_is_contiguous(weight),
          "unexpected placement metadata");
    return state.alternate ? &state.other : &state.preferred;
}

static void test_weight_selection() {
    static weight_device                 states[GGML_BACKEND_META_MAX_DEVICES];
    static ggml_backend_meta_split_state split{};
    static ggml_backend_reg              reg{};
    reg.api_version            = GGML_BACKEND_API_VERSION;
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (strcmp(name, "ggml_backend_dev_get_weight_buft") == 0) {
            return reinterpret_cast<void *>(preferred_weight);
        }
        if (strcmp(name, "ggml_backend_dev_get_override_bufts") == 0) {
            return reinterpret_cast<void *>(+[](ggml_backend_dev_t dev) -> ggml_backend_buffer_type_t * {
                return static_cast<weight_device *>(dev->context)->extras;
            });
        }
        return nullptr;
    };
    std::vector<ggml_backend_dev_t> devices;
    for (auto & state : states) {
        state.device.context        = &state;
        state.device.reg            = &reg;
        state.device.iface.get_name = [](ggml_backend_dev_t) {
            return "meta-weight-host";
        };
        state.device.iface.get_description = state.device.iface.get_name;
        state.device.iface.get_type        = [](ggml_backend_dev_t) {
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        };
        state.device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<weight_device *>(dev->context)->normal;
        };
        state.device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
            return buft->device == dev && static_cast<weight_device *>(dev->context)->supported;
        };
        state.device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) {
            return true;
        };
        for (auto * buft : {&state.normal, &state.preferred, &state.other}) {
            *buft        = *ggml_backend_cpu_buffer_type();
            buft->device = &state.device;
        }
        state.preferred.iface.get_name = [](ggml_backend_buffer_type_t) {
            return "preferred-host";
        };
        state.other.iface.get_name = [](ggml_backend_buffer_type_t) {
            return "other-host";
        };
        state.extras[0] = &state.preferred;
        state.extras[1] = &state.other;
        devices.push_back(&state.device);
    }
    auto get_split = [](const ggml_tensor *, void * data) {
        return *static_cast<ggml_backend_meta_split_state *>(data);
    };
    size_t cases = 0;
    for (int count : {2, 3, 4, 8, 16}) {
        auto * dev = ggml_backend_meta_device(devices.data(), count, get_split, &split);
        check(ggml_backend_dev_backend_reg(dev) == nullptr, "expected registry-less Meta device");
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
            for (int axis : {0, 1, 10}) {
                for (int pattern = 0; pattern < (axis == 10 ? 1 : 3); ++pattern) {
                    split            = {};
                    split.axis       = static_cast<ggml_backend_meta_split_axis>(axis);
                    split.n_segments = pattern == 2 ? 2 : 1;
                    int64_t ne[4]    = {512, 256, 1, 1};
                    int64_t total    = 0;
                    for (size_t s = 0; s < split.n_segments; ++s) {
                        split.nr[s] = pattern == 2 && s == 0 ? 2 : 1;
                        for (int rank = 0; rank < count; ++rank) {
                            const int factor           = pattern == 1 ? (rank == count - 1 ? 0 : rank + 1) : 1;
                            split.ne[s * count + rank] = (axis == 0 ? 512 : 256) * factor;
                            total += split.ne[s * count + rank] * split.nr[s];
                        }
                    }
                    if (axis != 10) {
                        ne[axis] = total;
                    }
                    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                    auto *           weight = ggml_new_tensor(ctx.get(), type, 4, ne);
                    ggml_set_name(weight, "meta-weight");
                    ggml_set_param(weight);
                    const auto original = *weight;
                    for (auto & state : states) {
                        state.calls = 0;
                    }
                    auto * selected = ggml_backend_meta_device_get_weight_buft(dev, weight, GGML_OP_MUL_MAT);
                    check(selected && selected != ggml_backend_dev_buffer_type(dev), "preferred storage not selected");
                    check(memcmp(weight, &original, sizeof(original)) == 0, "query changed source metadata");
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), selected));
                    check(bool(buffer), "preferred storage allocation failed");
                    for (int rank = 0; rank < count; ++rank) {
                        const auto * local = ggml_backend_meta_tensor_shard(weight, rank, nullptr);
                        check(local && ggml_backend_meta_buffer_type_at(selected, rank) == &states[rank].preferred,
                              "selected buffer family differs by rank");
                        if (ggml_nelements(local) == 0) {
                            check(states[rank].calls == 0, "empty shard queried for compute");
                        } else {
                            const auto & observed = states[rank].observed;
                            check(states[rank].calls == 1 && memcmp(local->ne, observed.ne, sizeof(local->ne)) == 0 &&
                                      memcmp(local->nb, observed.nb, sizeof(local->nb)) == 0,
                                  "queried geometry differs from allocated shard");
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    auto * dev = ggml_backend_meta_device(devices.data(), 2, get_split, &split);
    split      = {GGML_BACKEND_SPLIT_AXIS_0, {512, 512}, {1}, 1};
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 4, nullptr, true}));
    auto *           weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 1024, 256);
    auto             query  = [&] {
        return ggml_backend_meta_device_get_weight_buft(dev, weight, GGML_OP_MUL_MAT);
    };
    check(query() != nullptr, "valid TP control rejected");
    for (int rank : {0, 1}) {
        states[rank].enabled = false;
        check(!query(), "declined rank accepted");
        states[rank].enabled    = true;
        states[rank].device.reg = nullptr;
        check(!query(), "missing registry accepted");
        states[rank].device.reg = &reg;
        states[rank].supported  = false;
        check(!query(), "unsupported rank buffer accepted");
        states[rank].supported = true;
    }
    states[1].alternate = true;
    check(!query(), "different per-rank families accepted");
    states[1].alternate = false;
    split.ne[0]         = 256;
    split.ne[1]         = 768;
    check(!query(), "locally unsupported K accepted");
    split.ne[0] = 128;
    split.ne[1] = 896;
    check(!query(), "quantization block split accepted");
    split.ne[0] = 512;
    split.ne[1] = 256;
    check(!query(), "incomplete split map accepted");
    split.ne[1] = 512;
    split.nr[0] = 0;
    check(!query(), "zero segment repetition accepted");
    split.nr[0]      = 1;
    split.n_segments = 17;
    check(!query(), "oversized split map accepted");
    split.n_segments = 1;
    split.axis       = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    check(!query(), "partial static weight accepted");
    split.axis = GGML_BACKEND_SPLIT_AXIS_0;
    check(!ggml_backend_meta_device_get_weight_buft(dev, weight, GGML_OP_MUL_MAT_ID), "declined op accepted");
    auto * unsupported = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1024, 256);
    check(!ggml_backend_meta_device_get_weight_buft(dev, unsupported, GGML_OP_MUL_MAT), "declined type accepted");
    auto * view = ggml_view_tensor(ctx.get(), weight);
    check(!ggml_backend_meta_device_get_weight_buft(dev, view, GGML_OP_MUL_MAT), "weight view accepted");
    auto * empty = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1024, 0);
    check(!ggml_backend_meta_device_get_weight_buft(dev, empty, GGML_OP_MUL_MAT), "empty weight accepted");
    check(!ggml_backend_meta_device_get_weight_buft(&states[0].device, weight, GGML_OP_MUL_MAT),
          "non-Meta device accepted");
    check(!ggml_backend_meta_device_get_weight_buft(dev, nullptr, GGML_OP_MUL_MAT), "null weight accepted");
    check(query() != nullptr, "negative tests changed valid admission");
    printf("META_WEIGHT_BUFT PASS cases=%zu geometry=MATCH fallback=PASS\n", cases);
}

static void test_split_metadata() {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_val_str(gguf.get(), "general.architecture", "qwen35moe");
    auto * qkv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 512, 2048);
    auto * out = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 1024, 512);
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    ggml_set_name(out, "blk.0.ssm_out.weight");
    gguf_add_tensor(gguf.get(), qkv);
    gguf_add_tensor(gguf.get(), out);
    const auto close_file = [](FILE * fp) {
        if (fp) {
            fclose(fp);
        }
    };
    std::unique_ptr<FILE, decltype(close_file)> file(tmpfile(), close_file);
    const size_t bytes = gguf_get_meta_size(gguf.get()) + gguf_get_tensor_offset(gguf.get(), 1) + ggml_nbytes(out);
    check(file && gguf_write_to_file_ptr(gguf.get(), file.get(), true) &&
              fseek(file.get(), long(bytes - 1), SEEK_SET) == 0 && fputc(0, file.get()) != EOF &&
              fflush(file.get()) == 0,
          "metadata fixture write failed");
    rewind(file.get());
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file.get(), LLAMA_LOAD_MODE_NONE, false, true,
                              false, nullptr, nullptr);
    auto               params = llama_model_default_params();
    params.n_gpu_layers       = 0;
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35MOE, params));
    auto &                       hp = model->hparams;
    hp.n_layer_all                  = 1;
    hp.is_recr_impl[0]              = true;
    hp.n_embd                       = 512;
    hp.ssm_d_state                  = 128;
    hp.ssm_n_group                  = 4;
    hp.ssm_dt_rank                  = 8;
    for (size_t count : {2, 4}) {
        for (const auto * name : {qkv->name, out->name}) {
            const auto *                               weight = loader.require_tensor_meta(name);
            llama_meta_device_get_split_state_userdata data{count, model.get(), &loader};
            model->tensors_by_name.clear();
            const auto early       = llama_meta_device_get_split_state(weight, &data);
            model->tensors_by_name = {{qkv->name, loader.require_tensor_meta(qkv->name)},
                                      {out->name, loader.require_tensor_meta(out->name)}};
            data.loader            = nullptr;
            const auto late        = llama_meta_device_get_split_state(weight, &data);
            check(early.axis == late.axis && early.n_segments == late.n_segments &&
                      memcmp(early.ne, late.ne, sizeof(early.ne)) == 0 &&
                      memcmp(early.nr, late.nr, sizeof(early.nr)) == 0,
                  "split changed between placement and allocation");
        }
    }
    bool failed = false;
    try {
        model->load_tensors(loader);
    } catch (const std::exception &) {
        failed = true;
    }
    check(failed && model->get_split_state_ud.loader == nullptr, "metadata loader retained after failed load");
    puts("META_SPLIT_METADATA PASS early_late=IDENTICAL loader_lifetime=SCOPED");
}

int main() {
    try {
        test_weight_selection();
        test_split_metadata();
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "META_WEIGHT_BUFT FAIL: %s\n", error.what());
        return 1;
    }
}
