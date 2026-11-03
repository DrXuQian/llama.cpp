#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model.h"
#ifdef GGML_NCP_QUACTLIZE
#    include "../src/llama-kpack-merge.h"
#endif

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

#ifdef GGML_NCP_QUACTLIZE
static void test_shared_ffn_split() {
    ggml_backend_device host{};
    host.iface.get_name = [](ggml_backend_dev_t) {
        return "shared-ffn-host";
    };
    host.iface.get_description = host.iface.get_name;
    host.iface.get_buffer_type = [](ggml_backend_dev_t) {
        return ggml_backend_cpu_buffer_type();
    };
    size_t cases = 0;
    for (auto arch : {LLM_ARCH_QWEN35MOE, LLM_ARCH_QWEN3NEXT, LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK4}) {
        auto params         = llama_model_default_params();
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model> model(llama_model_create(arch, params));
        auto &                       hp = model->hparams;
        hp.n_layer_all                  = 2;
        hp.n_embd                       = 512;
        hp.n_head_arr.fill(8);
        hp.n_head_kv_arr.fill(8);
        hp.n_embd_head_k_full = 64;
        hp.ssm_d_state        = 128;
        hp.ssm_n_group        = 4;
        hp.ssm_dt_rank        = 8;
        for (size_t count : {2, 4}) {
            for (int layer : {0, 1}) {
                for (auto type :
                     {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
                    ggml_context_ptr weights(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
                    auto             weight = [&](const char * name, int64_t n, int64_t k) {
                        auto * t = ggml_new_tensor_2d(weights.get(), type, k, n);
                        ggml_format_name(t, "blk.%d.%s.weight", layer, name);
                        ggml_set_param(t);
                        return t;
                    };
                    auto * gate            = weight("ffn_gate_shexp", 512, 512);
                    auto * up              = weight("ffn_up_shexp", 512, 512);
                    auto * merged          = weight("ffn_gate_up_shexp", 1024, 512);
                    auto * down            = weight("ffn_down_shexp", 512, 512);
                    model->tensors_by_name = {
                        {gate->name, gate}, {up->name, up}, {merged->name, merged}, {down->name, down}};
                    llama_meta_device_get_split_state_userdata data{count, model.get(), nullptr};
                    const auto gate_split   = llama_meta_device_get_split_state(gate, &data);
                    const auto merged_split = llama_meta_device_get_split_state(merged, &data);
                    const auto down_split   = llama_meta_device_get_split_state(down, &data);
                    const bool sharded      = arch == LLM_ARCH_DEEPSEEK4;
                    check(
                        gate_split.axis == (sharded ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_MIRRORED) &&
                            down_split.axis == (sharded ? GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_MIRRORED),
                        "original shared FFN placement changed");
                    std::vector<ggml_backend_dev_t> devices(count, &host);
                    auto *                          meta =
                        ggml_backend_meta_device(devices.data(), count, llama_meta_device_get_split_state, &data);
                    auto *                  buft = ggml_backend_dev_buffer_type(meta);
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
                    check(bool(buffer), "shared FFN weight allocation failed");
                    ggml_context_ptr ctx(ggml_init(
                        {16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
                    auto *           input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 512, layer == 0 ? 1 : 16);
                    ggml_set_name(input, "shared-input");
                    ggml_set_input(input);
                    auto * separate   = ggml_swiglu_split(ctx.get(), ggml_mul_mat(ctx.get(), gate, input),
                                                          ggml_mul_mat(ctx.get(), up, input));
                    auto * activation = ggml_swiglu(ctx.get(), ggml_mul_mat(ctx.get(), merged, input));
                    ggml_set_name(activation, "ffn_swiglu-0");
                    auto * reference = ggml_mul_mat(ctx.get(), down, separate);
                    auto * output    = ggml_mul_mat(ctx.get(), down, activation);
                    ggml_set_name(output, "ffn_shexp-0");
                    ggml_set_output(reference);
                    ggml_set_output(output);
                    auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
                    ggml_build_forward_expand(graph, reference);
                    ggml_build_forward_expand(graph, output);
                    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(ggml_gallocr_new(buft),
                                                                                          ggml_gallocr_free);
                    check(ggml_gallocr_alloc_graph(allocator.get(), graph), "shared FFN graph allocation failed");
                    check(merged_split.axis == gate_split.axis, "merged shared FFN changed original placement");
                    for (size_t rank = 0; rank < count; ++rank) {
                        auto * local_gate       = ggml_backend_meta_tensor_shard(gate, rank, nullptr);
                        auto * local_merged     = ggml_backend_meta_tensor_shard(merged, rank, nullptr);
                        auto * local_down       = ggml_backend_meta_tensor_shard(down, rank, nullptr);
                        auto * local_activation = ggml_backend_meta_tensor_shard(activation, rank, nullptr);
                        ggml_backend_meta_split_state expected{}, actual{};
                        auto * local_reference = ggml_backend_meta_tensor_shard(reference, rank, &expected);
                        auto * local_output    = ggml_backend_meta_tensor_shard(output, rank, &actual);
                        check(local_merged->ne[1] == 2 * local_gate->ne[1] &&
                                  local_activation->ne[0] == local_down->ne[0] &&
                                  ggml_are_same_shape(local_reference, local_output) &&
                                  memcmp(&expected, &actual, sizeof(expected)) == 0,
                              "merged shared FFN local matmul or reduction differs");
                    }
                    ++cases;
                }
            }
        }
    }
    printf("META_SHARED_FFN PASS cases=%zu split_geometry=IDENTICAL tp2_tp4=PASS\n", cases);
}

static void test_paired_loading() {
    ggml_backend_device host{};
    ggml_backend_reg    reg{};
    auto                packed = *ggml_backend_cpu_buffer_type();
    packed.device = &host;
    host.reg                   = &reg;
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (strcmp(name, "ggml_cuda_buft_is_quactlize") == 0) {
            return reinterpret_cast<void *>(+[](ggml_backend_buffer_type_t) { return true; });
        }
        if (strcmp(name, "ggml_quactlize_tensor_is_kpack") == 0) {
            return reinterpret_cast<void *>(+[](const ggml_tensor *) { return true; });
        }
        return nullptr;
    };
    host.iface.get_name = [](ggml_backend_dev_t) {
        return "paired-host";
    };
    host.iface.get_description = host.iface.get_name;
    host.iface.get_buffer_type = [](ggml_backend_dev_t) {
        return ggml_backend_cpu_buffer_type();
    };
    size_t cases = 0;
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        for (int experts : {1, 4}) {
            ggml_context_ptr source_ctx(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
            gguf_context_ptr gguf(gguf_init_empty());
            gguf_set_val_str(gguf.get(), "general.architecture", "qwen35moe");
            std::vector<uint8_t> source_data[2];
            for (int side = 0; side < 2; ++side) {
                auto * tensor = ggml_new_tensor_3d(source_ctx.get(), type, 1024, 512, experts);
                ggml_set_name(tensor, side == 0 ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_up_exps.weight");
                source_data[side].resize(ggml_nbytes(tensor));
                for (size_t i = 0; i < source_data[side].size(); ++i) {
                    // Distinct row/expert bytes, with finite scales in every quantized block.
                    source_data[side][i] = (i % 251 + i / tensor->nb[1] + 17 * (i / tensor->nb[2]) + side * 13) % 64;
                }
                tensor->data = source_data[side].data();
                gguf_add_tensor(gguf.get(), tensor);
            }
            const auto close_file = [](FILE * fp) {
                if (fp) {
                    fclose(fp);
                }
            };
            std::unique_ptr<FILE, decltype(close_file)> file(tmpfile(), close_file);
            check(file && gguf_write_to_file_ptr(gguf.get(), file.get(), false) && fflush(file.get()) == 0,
                  "paired fixture write failed");
            for (int count : {2, 4, 1}) {
                for (auto axis : {GGML_BACKEND_SPLIT_AXIS_1, GGML_BACKEND_SPLIT_AXIS_0, GGML_BACKEND_SPLIT_AXIS_2,
                                  GGML_BACKEND_SPLIT_AXIS_MIRRORED}) {
                    if (count == 1 && axis != GGML_BACKEND_SPLIT_AXIS_1) {
                        continue;
                    }
                    for (bool mmap : {false, true}) {
                        if (mmap && !llama_mmap::SUPPORTED) {
                            continue;
                        }
                        rewind(file.get());
                        std::vector<std::string> splits;
                        llama_model_loader       loader(nullptr, nullptr, nullptr, "", splits, file.get(),
                                                  mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE, true, true, false,
                                                        nullptr, nullptr);
                        ggml_context_ptr         ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                        llama_hparams            hp{};
                        const auto name    = LLM_TN(LLM_ARCH_QWEN35MOE)(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", 0);
                        size_t     queries = 0;
                        auto       choose  = [&](ggml_tensor * weight) {
                            const char * expected[] = {"blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_gate_exps.weight",
                                                       "blk.0.ffn_up_exps.weight"};
                            check(queries < 3 && strcmp(weight->name, expected[queries]) == 0,
                                         "merged placement did not query both original sources");
                            ++queries;
                            return &packed;
                        };
                        auto context = [&](ggml_backend_buffer_type_t buft) {
                            check(buft == &packed, "merged placement changed buffer type");
                            return ctx.get();
                        };
                        check(!llama_kpack_create_merged_tensor(loader, hp, name, {1024, 512, experts}, 0, choose,
                                                                context),
                              "wrong merged dimensions accepted");
                        auto * tensor = llama_kpack_create_merged_tensor(loader, hp, name, {1024, 1024, experts}, 0,
                                                                         choose, context);
                        check(tensor && queries == 3 && loader.n_created == 2, "merged tensor creation changed");
                        ggml_set_param(tensor);
                        ggml_backend_meta_split_state split{};
                        split.axis       = axis;
                        split.n_segments = 1;
                        split.nr[0]      = axis == GGML_BACKEND_SPLIT_AXIS_1 ? 2 : 1;
                        if (axis < GGML_MAX_DIMS) {
                            const int64_t extent = tensor->ne[axis] / split.nr[0];
                            for (int rank = 0; rank < count; ++rank) {
                                split.ne[rank] = extent / count + (rank < extent % count);
                            }
                        }
                        auto *                          buft = ggml_backend_cpu_buffer_type();
                        std::vector<ggml_backend_dev_t> devices(count, &host);
                        if (count > 1) {
                            auto * meta = ggml_backend_meta_device(
                                devices.data(), count,
                                [](const ggml_tensor *, void * data) {
                                    return *static_cast<ggml_backend_meta_split_state *>(data);
                                },
                                &split);
                            buft = ggml_backend_dev_buffer_type(meta);
                        }
                        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
                        check(bool(buffer), "paired weight allocation failed");
                        llama_buf_map buffers;
                        loader.init_mappings(false, nullptr);
                        check(loader.size_data == source_data[0].size() + source_data[1].size(),
                              "paired source bytes counted twice");
                        if (mmap) {
                            size_t first, last;
                            void * address;
                            loader.get_mapping_range(&first, &last, &address, 0, ctx.get());
                            const auto & gate = loader.require_weight("blk.0.ffn_gate_exps.weight");
                            const auto & up   = loader.require_weight("blk.0.ffn_up_exps.weight");
                            check(address == loader.mappings[0]->addr() && first == std::min(gate.offs, up.offs) &&
                                      last ==
                                          std::max(gate.offs + source_data[0].size(), up.offs + source_data[1].size()),
                                  "paired mmap range differs from its sources");
                        }
                        std::vector<uint8_t> host_data;
                        if (count == 1) {
                            // load_all_data releases offloaded mmap ranges when it completes.
                            host_data.resize(ggml_nbytes(tensor));
                            ggml_tensor host_tensor = *tensor;
                            host_tensor.data        = host_data.data();
                            loader.load_data_for(&host_tensor);
                            std::vector<uint8_t> ordinary(source_data[0].size());
                            for (int side = 0; side < 2; ++side) {
                                const auto & source = loader.require_weight(side == 0 ? "blk.0.ffn_gate_exps.weight" :
                                                                                        "blk.0.ffn_up_exps.weight");
                                host_tensor         = *source.tensor;
                                host_tensor.data    = ordinary.data();
                                loader.load_data_for(&host_tensor);
                                check(ordinary == source_data[side], "ordinary source load changed");
                            }
                        }
                        check(loader.load_all_data(ctx.get(), buffers, nullptr, nullptr, nullptr),
                              "paired load failed");
                        check(loader.size_done == loader.size_data, "paired load progress differs");
                        std::vector<uint8_t> actual(ggml_nbytes(tensor));
                        if (count > 1 && axis == GGML_BACKEND_SPLIT_AXIS_2) {
                            size_t offset = 0;
                            for (int rank = 0; rank < count; ++rank) {
                                auto *       shard = ggml_backend_meta_tensor_shard(tensor, rank, nullptr);
                                const size_t bytes = ggml_nbytes(shard);
                                if (bytes) {
                                    ggml_backend_tensor_get(shard, actual.data() + offset, 0, bytes);
                                }
                                offset += bytes;
                            }
                            check(offset == actual.size(), "paired expert shard coverage differs");
                        } else {
                            ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
                        }
                        const size_t width = tensor->nb[2] / 2;
                        for (int e = 0; e < experts; ++e) {
                            for (int side = 0; side < 2; ++side) {
                                check(memcmp(actual.data() + (2 * e + side) * width,
                                             source_data[side].data() + e * width, width) == 0,
                                      "paired load bytes differ");
                            }
                        }
                        if (!host_data.empty()) {
                            check(host_data == actual, "paired legacy load differs from backend load");
                        }
                        if (count > 1 && axis == GGML_BACKEND_SPLIT_AXIS_1) {
                            for (int rank = 0; rank < count; ++rank) {
                                auto *       shard = ggml_backend_meta_tensor_shard(tensor, rank, nullptr);
                                const size_t part  = width / count;
                                actual.resize(ggml_nbytes(shard));
                                ggml_backend_tensor_get(shard, actual.data(), 0, actual.size());
                                for (int e = 0; e < experts; ++e) {
                                    for (int side = 0; side < 2; ++side) {
                                        check(memcmp(actual.data() + (2 * e + side) * part,
                                                     source_data[side].data() + e * width + rank * part, part) == 0,
                                              "paired shard mixed Gate/Up or expert slices");
                                    }
                                }
                            }
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    printf("META_PAIRED_LOAD PASS cases=%zu cold=BYTE_EXACT single_tp2_tp4=PASS\n", cases);
}

#endif

int main() {
    try {
        test_weight_selection();
        test_split_metadata();
#ifdef GGML_NCP_QUACTLIZE
        test_shared_ffn_split();
        test_paired_loading();
#endif
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "META_WEIGHT_BUFT FAIL: %s\n", error.what());
        return 1;
    }
}
