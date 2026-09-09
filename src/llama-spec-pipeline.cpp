#include "llama-spec-pipeline.h"

#include "llama-context.h"
#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "llama-sampler.h"

#include <algorithm>
#include <cstring>

struct llama_nextn_graph {
    llama_token seed_token = LLAMA_TOKEN_NULL;
    llama_pos seed_pos = -1;
    llama_seq_id seed_seq = -1;
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr res;
    ggml_backend_event_ptr done;
    std::vector<llm_graph_params> params;
    std::vector<size_t> input_ends;
    std::vector<ggml_tensor *> tokens;
    std::vector<ggml_tensor *> hidden;
    std::vector<ggml_tensor *> candidates;
    std::vector<ggml_tensor *> logits;
    bool prefetch = false;
    bool prefetched = false;
    float p_min = 0.0f;
    int n_verify = 0;
    ggml_context_ptr early_ctx;
    ggml_backend_buffer_ptr early_device;
    ggml_backend_buffer_ptr early_host;
    ggml_tensor * early_counts = nullptr;
    uint64_t generation = 0;
    ggml_tensor * accepted = nullptr;
    ggml_tensor * selected_token = nullptr;
    ggml_tensor * selected_hidden = nullptr;
    ggml_tensor * inp_draft = nullptr;
    ggml_tensor * inp_weights = nullptr;
    ggml_tensor * inp_positions = nullptr;
    ggml_tensor * inp_position_mask = nullptr;
    ggml_tensor * inp_kv_positions = nullptr;
    ggml_tensor * reject_mask = nullptr;
    std::vector<ggml_tensor *> positions;
};

struct llama_nextn_handoff {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device;
    ggml_backend_buffer_ptr host;
    ggml_backend_event_ptr consumed;
    std::vector<ggml_tensor *> rows;
};

struct llama_nextn_lookahead {
    llama_nextn_handoff sampled;
    ggml_backend_event_ptr ready;
    std::vector<llama_token> draft;
    llama_pos pos = -1;
    llama_seq_id seq = -1;
};

struct llama_nextn_target {
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr res;
    ggml_context_ptr meta_ctx;
    ggml_backend_sched_ptr meta_sched;
    ggml_cgraph * meta_gf = nullptr;
    llama_nextn_handoff input;
    ggml_backend_event_ptr ready;
    ggml_backend_event_ptr done;
    llm_graph_nextn_target meta;
    std::map<llama_seq_id, llama_sampler *> samplers;
    std::unique_ptr<llm_graph_params> controls;
    std::unique_ptr<llm_graph_params> build_controls;
    llama_memory_context_ptr graph_mctx;
    ggml_tensor * previous_count = nullptr;
    const llama_context * source = nullptr;
    uint64_t generation = 0;
    llama_pos previous_pos = -1;
    int previous_n = 0;
    int n_kv = 0;
    int n = 0;

    ~llama_nextn_target() {
        for (auto & entry : samplers) {
            llama_sampler_free(entry.second);
        }
    }
};

llama_spec_pipeline::llama_spec_pipeline(llama_context & context) : lctx(context) {}

llama_spec_pipeline::~llama_spec_pipeline() = default;

bool llama_spec_pipeline::prefetch_nextn_target(llama_context & source) {
    const bool eagle = source.model.arch == LLM_ARCH_EAGLE3;
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(lctx.memory.get());
    auto * mctx = this->mctx.get();
    auto * hctx = dynamic_cast<llama_memory_hybrid_context *>(mctx);
    auto * kv = hybrid ? hybrid->get_mem_attn() : dynamic_cast<llama_kv_cache *>(lctx.memory.get());
    auto * kctx = hctx ? hctx->get_attn() : dynamic_cast<llama_kv_cache_context *>(mctx);
    if (!this->prefetch_enabled || !kv || !kctx || !source.spec->graph || !source.spec->graph->prefetched ||
            !this->readback || this->target_pending >= 0 || lctx.sched_need_reserve || lctx.graph_reuse_disable ||
            lctx.cparams.n_seq_max != 1 || lctx.cparams.pipeline_parallel || lctx.cparams.cb_eval || lctx.opt_ctx ||
            !lctx.cparams.flash_attn || !lctx.cparams.causal_attn || lctx.cparams.embeddings ||
            lctx.cparams.embeddings_nextn_masked || lctx.cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
            lctx.model.hparams.n_swa || lctx.model.hparams.use_alibi || !lctx.loras->empty() ||
            (!eagle && std::any_of(lctx.cparams.embeddings_layer_inp.begin(), lctx.cparams.embeddings_layer_inp.end(), [](bool v) { return v; })) ||
            ggml_is_quantized(kv->type_k()) || ggml_is_quantized(kv->type_v()) ||
            (eagle ? (lctx.model.arch != LLM_ARCH_QWEN3 && lctx.model.arch != LLM_ARCH_QWEN3MOE) :
                     (lctx.model.arch != LLM_ARCH_QWEN35 && lctx.model.arch != LLM_ARCH_QWEN35MOE))) {
        return false;
    }
    auto & chain = *source.spec->graph;
    const auto & ubatch = mctx->get_ubatch();
    const int previous_n = ubatch.n_tokens;
    const bool early = chain.p_min > 0.0f;
    const int n_max = early ? chain.tokens.size() + 1 : previous_n;
    const int n_kv = kctx->get_n_kv();
    auto * rs = hybrid ? hybrid->get_mem_recr() : nullptr;
    if (previous_n < 1 || previous_n > 9 || (!early && previous_n != (int) chain.tokens.size() + 1) ||
            ubatch.n_seqs_unq != 1 || ubatch.seq_id[0][0] != 0 || !ubatch.token || ubatch.embd ||
            ubatch.pos[0] < 0 || ubatch.pos[0] + previous_n + n_max > (int) kv->get_size() ||
            (!early && (uint32_t) n_kv != kv->get_size()) || kv->get_n_stream() != 1 || lctx.sampling.samplers.size() != 1 ||
            !lctx.sampling.samplers.count(0) || !llama_sampler_backend_can_prefetch(lctx.sampling.samplers.at(0)) ||
            lctx.gf_res_prev->t_sampled_logits.size() != (size_t) previous_n) {
        return false;
    }
    if (rs && (!hctx || rs->size != 1 || hctx->get_recr()->get_n_rs() != 1 ||
            hctx->get_recr()->get_head() != 0 || hctx->get_recr()->get_rs_z() >= 0 || rs->n_rs_seq < (uint32_t) n_max - 1)) {
        return false;
    }
    for (int i = 0; i < previous_n; ++i) {
        auto * logits = lctx.gf_res_prev->t_sampled_logits[i];
        if (!ubatch.output[i] || !logits || ggml_nelements(logits) != 1) {
            return false;
        }
    }
    const auto & cells = kv->get_cells(0);
    for (int i = 0; i < ubatch.pos[0] + previous_n; ++i) {
        if (!cells.seq_has(i, 0) || cells.pos_get(i) != i) {
            return false;
        }
    }
    if (early && !kv->is_fixed_size() && (ubatch.pos[0] + previous_n + n_max > n_kv ||
            (n_kv > 256 && ubatch.pos[0] + 2 <= n_kv - 256))) {
        return false;
    }
    auto * backend = lctx.backend_ptrs[0];
    auto * producer = source.backend_ptrs[0];
    auto * dev = ggml_backend_get_device(backend);
    if (!dev || dev != ggml_backend_get_device(producer) ||
            strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
        return false;
    }
    auto get_batch = (ggml_backend_get_tensors_async_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_get_tensors_async");
    if (!get_batch) {
        return false;
    }
    const int bank = 1 - this->target_bank;
    const int n_begin = early ? 1 : n_max;
    auto * retired = this->targets[bank*10 + n_begin].get();
    if (retired && retired->done) {
        ggml_backend_event_synchronize(retired->done.get());
    }
    std::vector<ggml_backend_sched_t> prefixes, scheds;
    for (int n = n_begin; n <= n_max; ++n) {
        const int index = bank*10 + n;
        if (!this->targets[index]) {
            this->targets[index] = std::make_unique<llama_nextn_target>();
        }
        auto & state = *this->targets[index];
        auto & input = this->targets[bank*10 + n_begin]->input;
        if (!input.device) {
            input.ctx.reset(ggml_init({ 13*ggml_tensor_overhead(), nullptr, true }));
            auto * storage = ggml_new_tensor_1d(input.ctx.get(), GGML_TYPE_I32, 12);
            input.rows.push_back(ggml_view_1d(input.ctx.get(), storage, 1, 0));
            input.rows.push_back(ggml_view_1d(input.ctx.get(), storage, 1, sizeof(int32_t)));
            input.rows.push_back(ggml_view_1d(input.ctx.get(), storage, 1, 2*sizeof(int32_t)));
            for (int count = 1; count <= 9; ++count) {
                input.rows.push_back(ggml_view_1d(input.ctx.get(), storage, count, 3*sizeof(int32_t)));
            }
            input.device.reset(ggml_backend_alloc_ctx_tensors(input.ctx.get(), backend));
            input.consumed.reset(ggml_backend_event_new(dev));
            state.ready.reset(ggml_backend_event_new(dev));
            if (!input.device || !input.consumed || !state.ready) {
                return false;
            }
        }
        if (!state.res) {
            const size_t max_nodes = lctx.graph_max_nodes(n);
            state.res.reset(new llm_graph_result(max_nodes));
            state.sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                    max_nodes, false, lctx.cparams.op_offload));
        }
        if (!this->output_ready) {
            this->output_ready.reset(ggml_backend_event_new(dev));
            if (!this->output_ready) {
                return false;
            }
        }
        auto * res = state.res.get();
        llama_batch_allocr graph_balloc(lctx.model.hparams.n_pos_per_embd());
        auto graph_batch = graph_balloc.ubatch_reserve(n, 1);
        std::fill_n(graph_batch.output, n, true);
        for (int i = 0; i < n; ++i) {
            graph_batch.n_seq_id[i] = 1;
            graph_batch.seq_id[i] = graph_batch.seq_id_unq;
        }
        llama_memory_context_ptr graph_mctx;
        if (hctx) {
            graph_mctx = hctx->for_graph(graph_batch, n_kv);
        } else {
            graph_mctx = kctx->for_graph(graph_batch, n_kv);
        }
        auto controls = lctx.graph_params(res, graph_batch, graph_mctx.get(), LLM_GRAPH_TYPE_DEFAULT);
        controls.n_outputs = n;
        std::map<ggml_tensor *, ggml_tensor *> outputs;
        if (rs && !rs->nextn_state_outputs(outputs)) {
            return false;
        }
        auto params = controls;
        params.sched = state.sched.get();
        params.nextn_target = &state.meta;
        params.samplers = state.samplers;
        bool same_sampler = state.build_controls && llm_graph_params::samplers_equal(state.build_controls->samplers, controls.samplers);
        if (!same_sampler && state.samplers.size() == 1) {
            same_sampler = llama_sampler_backend_same_config(state.samplers.at(0), lctx.sampling.samplers.at(0));
            if (same_sampler && state.build_controls) {
                state.build_controls->samplers = controls.samplers;
            }
        }
        const bool reuse = same_sampler && state.n_kv == n_kv && state.build_controls && state.build_controls->allow_reuse(controls) &&
            state.meta.state_outputs == outputs && res->can_reuse(params);
        if (!reuse) {
            if (state.meta_sched) {
                ggml_backend_sched_reset(state.meta_sched.get());
            }
            ggml_backend_sched_reset(state.sched.get());
            res->reset();
            for (auto & entry : state.samplers) {
                llama_sampler_free(entry.second);
            }
            state.samplers.clear();
            auto * sampler = llama_sampler_clone(lctx.sampling.samplers.at(0));
            if (!sampler) {
                return false;
            }
            state.samplers.emplace(0, sampler);
            if (!sampler->iface->backend_init(sampler, ggml_backend_dev_buffer_type(dev), lctx.cparams.n_outputs_max_per_seq)) {
                return false;
            }
            params.samplers = state.samplers;
            state.meta = {};
            state.meta.state_outputs = std::move(outputs);
            auto * ctx = res->get_ctx();
            if (eagle) {
                state.meta_ctx.reset(ggml_init({ 128*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true }));
                ctx = state.meta_ctx.get();
                state.meta_gf = ggml_new_graph_custom(ctx, 128, false);
                if (!state.meta_sched) {
                    state.meta_sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                            128, false, lctx.cparams.op_offload));
                }
            }
            auto * base = ggml_cast(ctx, input.rows[0], GGML_TYPE_F32);
            if (eagle) {
                base = ggml_scale_bias(ctx, base, 1.0f, 1.0f);
            }
            auto * accepted = ggml_cast(ctx, input.rows[1], GGML_TYPE_F32);
            auto * positions = ggml_add(ctx, ggml_arange(ctx, 0, n, 1), base);
            state.meta.tokens = input.rows[n + 2];
            state.meta.kv_idxs = ggml_cast(ctx, positions, GGML_TYPE_I32);
            state.meta.positions = state.meta.kv_idxs;
            if (lctx.model.hparams.n_pos_per_embd() == 4) {
                auto * three = ggml_concat(ctx, ggml_concat(ctx, state.meta.kv_idxs, state.meta.kv_idxs, 0), state.meta.kv_idxs, 0);
                auto * zero = ggml_cast(ctx, ggml_scale(ctx, positions, 0), GGML_TYPE_I32);
                state.meta.positions = ggml_concat(ctx, three, zero, 0);
            } else if (lctx.model.hparams.n_pos_per_embd() != 1) {
                return false;
            }
            auto * columns = ggml_repeat_4d(ctx, ggml_arange(ctx, 0, n_kv, 1), n_kv, n, 1, 1);
            auto * query = ggml_reshape_2d(ctx, positions, 1, n);
            state.meta.mask = ggml_cast(ctx, ggml_scale(ctx, ggml_step(ctx, ggml_sub(ctx, columns, query)), -1e30f), GGML_TYPE_F16);
            if (rs) {
                state.previous_count = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_set_input(state.previous_count);
                state.meta.rs_copy = ggml_cast(ctx, ggml_sub(ctx, state.previous_count, accepted), GGML_TYPE_I32);
            }
            auto build_params = params;
            if (eagle) {
                // Preserve the ordinary target graph's allocation and CUDA fusion choices.
                build_params.nextn_target = nullptr;
            }
            if (!lctx.model.build_graph(build_params)) {
                state.controls.reset();
                return false;
            }
            if (eagle) {
                // The cache tag prevents CPU decoding from reusing these GPU-fed inputs.
                res->set_params(params);
                auto copy = [&](ggml_tensor * src, ggml_tensor * dst) {
                    ggml_backend_sched_set_tensor_backend(state.sched.get(), dst, backend);
                    auto * cur = ggml_cpy(ctx, src, dst);
                    ggml_set_name(cur, "nextn_target_metadata");
                    ggml_build_forward_expand(state.meta_gf, cur);
                };
                res->inputs.erase(std::remove_if(res->inputs.begin(), res->inputs.end(), [&](const llm_graph_input_ptr & input) {
                    if (auto * inp = dynamic_cast<llm_graph_input_embd *>(input.get())) {
                        copy(state.meta.tokens, inp->tokens);
                    } else if (auto * inp = dynamic_cast<llm_graph_input_pos *>(input.get())) {
                        copy(state.meta.positions, inp->pos);
                    } else if (auto * inp = dynamic_cast<llm_graph_input_attn_kv *>(input.get())) {
                        copy(positions, inp->self_k_idxs);
                        copy(positions, inp->self_v_idxs);
                        copy(state.meta.mask, inp->self_kq_mask);
                        if (kv->is_fixed_size()) {
                            for (int i = 0; i < ggml_graph_n_nodes(res->get_gf()); ++i) {
                                auto * node = ggml_graph_node(res->get_gf(), i);
                                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                                    ggml_flash_attn_ext_set_kv_indices(node, inp->self_k_idxs);
                                }
                            }
                        }
                    } else {
                        return false;
                    }
                    return true;
                }), res->inputs.end());
            }
            if (!ggml_backend_sched_alloc_graph_after(state.sched.get(), res->get_gf(), retired ? retired->done.get() : nullptr)) {
                state.controls.reset();
                return false;
            }
            if (eagle) {
                if (!ggml_backend_sched_alloc_graph_after(state.meta_sched.get(), state.meta_gf, retired ? retired->done.get() : nullptr)) {
                    state.controls.reset();
                    return false;
                }
                for (int i = 0; i < ggml_graph_n_nodes(state.meta_gf); ++i) {
                    auto * node = ggml_graph_node(state.meta_gf, i);
                    if (ggml_backend_sched_get_tensor_backend(state.meta_sched.get(), node) != backend) {
                        LLAMA_LOG_DEBUG("%s: unsupported metadata node %s\n", __func__, ggml_get_name(node));
                        state.controls.reset();
                        return false;
                    }
                }
            }
            for (int i = 0; i < ggml_graph_n_nodes(res->get_gf()); ++i) {
                auto * node = ggml_graph_node(res->get_gf(), i);
                if (ggml_backend_sched_get_tensor_backend(state.sched.get(), node) != backend) {
                    LLAMA_LOG_DEBUG("%s: unsupported node %s\n", __func__, ggml_get_name(node));
                    state.controls.reset();
                    return false;
                }
            }
            state.build_controls = std::make_unique<llm_graph_params>(controls);
            state.n_kv = n_kv;
        }
        for (auto & entry : state.samplers) {
            llama_sampler_copy(lctx.sampling.samplers.at(entry.first), entry.second);
            llama_sampler_backend_begin(entry.second);
        }
        res->set_inputs(&graph_batch);
        if (state.previous_count) {
            const float count = previous_n - 1;
            ggml_backend_tensor_set(state.previous_count, &count, 0, sizeof(count));
        }
        state.graph_mctx = std::move(graph_mctx);
        state.source = &source;
        state.generation = chain.generation;
        state.previous_pos = ubatch.pos[0];
        state.previous_n = previous_n;
        state.n = n;
        state.controls = std::make_unique<llm_graph_params>(lctx.graph_params(lctx.gf_res_prev.get(), ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT));
        prefixes.push_back(eagle ? state.meta_sched.get() : nullptr);
        scheds.push_back(state.sched.get());
    }
    auto & first = *this->targets[bank*10 + n_begin];
    if (early && !ggml_backend_sched_graph_select(prefixes.data(), scheds.data(), scheds.size(), first.input.rows[2])) {
        return false;
    }
    ggml_tensor position = *chain.positions[0];
    position.ne[0] = 1;
    ggml_tensor count = early ? *chain.early_counts : *chain.accepted;
    count.ne[0] = 1;
    std::vector<const ggml_tensor *> tensors = { &position, chain.accepted, &count, chain.selected_token };
    tensors.insert(tensors.end(), chain.tokens.begin(), chain.tokens.end());
    std::vector<void *> destinations;
    for (size_t i = 0; i < tensors.size(); ++i) {
        destinations.push_back((int32_t *) first.input.rows[0]->data + i);
    }
    ggml_backend_event_record(first.input.consumed.get(), backend);
    ggml_backend_event_wait(producer, first.input.consumed.get());
    if (!get_batch(producer, tensors.data(), destinations.data(), tensors.size())) {
        return false;
    }
    ggml_backend_event_record(first.ready.get(), producer);
    this->readback->submit();
    this->readback.reset();
    ggml_backend_event_record(this->output_ready.get(), backend);
    this->output_pending = true;
    if (early && source.spec->readback) {
        source.spec->readback->submit();
        source.spec->readback.reset();
        if (!source.spec->output_ready) {
            source.spec->output_ready.reset(ggml_backend_event_new(dev));
        }
        ggml_backend_event_record(source.spec->output_ready.get(), producer);
        source.spec->output_pending = true;
    }
    ggml_backend_event_wait(backend, first.ready.get());
    if (!early && eagle && ggml_backend_sched_graph_compute_async(first.meta_sched.get(), first.meta_gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (ggml_backend_sched_graph_compute_async(first.sched.get(), first.res->get_gf()) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!first.done) {
        first.done.reset(ggml_backend_event_new(dev));
    }
    ggml_backend_event_record(first.done.get(), backend);
    this->target_pending = bank*10 + n_begin;
    if (!this->prefetch_reported) {
        LLAMA_LOG_INFO("%s: GPU NextN pipeline active; target submitted before CPU acceptance\n", __func__);
        this->prefetch_reported = true;
    }
    LLAMA_LOG_DEBUG("%s: queued GPU target before CPU acceptance (%d..%d rows)\n", __func__, n_begin, n_max);
    return true;
}

bool llama_spec_pipeline::decode_nextn_verify(llama_context & source, const llama_batch & batch, int32_t * result) {
    const int n = batch.n_tokens;
    const int offset = source.model.arch == LLM_ARCH_EAGLE3 ? 1 : 0;
    bool consume_target = false;
    if (&source == &lctx || !source.spec->outputs || !source.spec->graph || n < 1 || n > 9 ||
            source.n_outputs < (uint32_t) (n - 1) || (int) source.spec->graph->tokens.size() < n - 1 ||
            lctx.cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || lctx.cparams.n_seq_max != 1 ||
            lctx.cparams.pipeline_parallel || lctx.cparams.cb_eval || lctx.opt_ctx || n > (int) lctx.cparams.n_ubatch || n > (int) lctx.cparams.n_batch ||
            (offset ? (lctx.model.arch != LLM_ARCH_QWEN3 && lctx.model.arch != LLM_ARCH_QWEN3MOE) :
                      (lctx.model.arch != LLM_ARCH_QWEN35 && lctx.model.arch != LLM_ARCH_QWEN35MOE)) ||
            !batch.token || batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id ||
            !lctx.model.tok_embd || ggml_backend_buffer_is_host(lctx.model.tok_embd->buffer) ||
            lctx.model.devices.size() != 1 || source.model.devices.size() != 1 || lctx.model.devices[0].dev != source.model.devices[0].dev ||
            lctx.model.n_gpu_layers() <= lctx.model.hparams.n_layer_all || lctx.model.vocab.n_tokens() != source.model.vocab.n_tokens()) {
        return false;
    }
    const auto & chain = *source.spec->graph;
    if (batch.pos[0] != chain.seed_pos + offset || batch.token[0] != chain.seed_token) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != chain.seed_seq || batch.pos[i] != batch.pos[0] + i) {
            return false;
        }
    }
    if (this->target_pending >= 0) {
        if (chain.p_min > 0.0f) {
            source.synchronize(true);
            const auto * counts = (const int32_t *) ggml_backend_buffer_get_base(chain.early_host.get());
            if (counts[0] + 1 != n) {
                return false;
            }
        }
        const int selected = (this->target_pending/10)*10 + n;
        if (!this->targets[selected]) {
            return false;
        }
        this->target_pending = selected;
        const auto & prepared = *this->targets[this->target_pending];
        const auto * lookahead = source.spec->lookahead.get();
        auto controls = lctx.graph_params(lctx.gf_res_prev.get(), this->last_ubatch, this->mctx.get(), LLM_GRAPH_TYPE_DEFAULT);
        bool outputs_match = batch.logits != nullptr;
        for (int i = 0; outputs_match && i < n; ++i) {
            outputs_match = batch.logits[i] != 0;
        }
        if (outputs_match && !lctx.sched_need_reserve && !lctx.graph_reuse_disable && prepared.n == n && prepared.source == &source &&
                prepared.generation == chain.generation && lookahead && prepared.previous_pos == lookahead->pos + offset &&
                lctx.memory->seq_pos_max(chain.seed_seq) == batch.pos[0] - 1 &&
                prepared.controls && prepared.controls->allow_reuse(controls)) {
            ggml_backend_event_synchronize(lookahead->ready.get());
            const auto * sampled = (const llama_token *) ggml_backend_buffer_get_base(lookahead->sampled.host.get());
            int accepted = 0;
            while (accepted + 1 < (int) lookahead->draft.size() && sampled[accepted] == lookahead->draft[accepted]) {
                ++accepted;
            }
            consume_target = batch.pos[0] == prepared.previous_pos + accepted + 1 && batch.token[0] == sampled[accepted];
            auto * hybrid = dynamic_cast<llama_memory_hybrid *>(lctx.memory.get());
            auto * kv = hybrid ? hybrid->get_mem_attn() : static_cast<llama_kv_cache *>(lctx.memory.get());
            const auto & cells = kv->get_cells(0);
            for (int i = 0; consume_target && i < batch.pos[0]; ++i) {
                consume_target = cells.seq_has(i, 0) && cells.pos_get(i) == i;
            }
        }
    }
    if (n == 1) {
        if (!consume_target) {
            return false;
        }
        this->target_consume = true;
        *result = lctx.decode(batch);
        this->target_consume = false;
        return true;
    }
    auto * dst_backend = lctx.backend_ptrs[0];
    auto * src_backend = source.backend_ptrs[0];
    auto * dev = ggml_backend_get_device(dst_backend);
    if (!dev || ggml_backend_get_device(src_backend) != dev) {
        return false;
    }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto get_batch = (ggml_backend_get_tensors_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_tensors_async");
    if (!get_batch) {
        return false;
    }
    if (!this->verify) {
        auto state = std::make_unique<llama_nextn_handoff>();
        state->ctx.reset(ggml_init({ 10*ggml_tensor_overhead(), nullptr, true }));
        auto * storage = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_I32, 8);
        for (int i = 1; i <= 8; i++) {
            state->rows.push_back(ggml_view_1d(state->ctx.get(), storage, i, 0));
        }
        state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), dst_backend));
        state->consumed.reset(ggml_backend_event_new(dev));
        if (!state->device || !state->consumed) {
            return false;
        }
        this->verify = std::move(state);
    }
    auto & state = *this->verify;
    auto * input = state.rows[n - 2];
    std::vector<const ggml_tensor *> tensors;
    std::vector<void *> destinations;
    for (int i = 0; i < n - 1; i++) {
        tensors.push_back(source.spec->graph->tokens[i]);
        destinations.push_back((llama_token *) input->data + i);
    }
    if (!consume_target) {
        ggml_backend_event_record(state.consumed.get(), dst_backend);
        ggml_backend_event_wait(src_backend, state.consumed.get());
        if (!get_batch(src_backend, tensors.data(), destinations.data(), tensors.size())) {
            return false;
        }
        ggml_backend_event_record(state.consumed.get(), src_backend);
        ggml_backend_event_wait(dst_backend, state.consumed.get());
    }
    this->verify_input = input;
    this->target_consume = consume_target;
    *result = lctx.decode(batch);
    this->verify_input = nullptr;
    this->target_consume = false;
    source.synchronize();
    for (int i = 0; i < n - 1; i++) {
        batch.token[i + 1] = source.get_sampled_candidates_ith(i)[0];
        if (*result == 0 && this->last_ubatch.token) {
            this->last_ubatch.token[i + 1] = batch.token[i + 1];
        }
    }
    return true;
}

bool llama_spec_pipeline::decode_nextn_catchup(llama_context & source, const llama_batch & batch, const float ** snapshot) {
    const int n = batch.n_tokens;
    const bool eagle = lctx.model.arch == LLM_ARCH_EAGLE3;
    if (&source == &lctx || (!eagle && (lctx.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || lctx.model.hparams.n_layer_nextn != 1 ||
            (lctx.model.arch != LLM_ARCH_QWEN35 && lctx.model.arch != LLM_ARCH_QWEN35MOE))) ||
            n < 1 || n > 9 || n > (int) lctx.cparams.n_ubatch || n > (int) lctx.cparams.n_batch ||
            !batch.token || !batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits ||
            lctx.cparams.pipeline_parallel || lctx.cparams.cb_eval || lctx.opt_ctx || !dynamic_cast<llama_kv_cache *>(lctx.memory.get()) ||
            source.cparams.embeddings_nextn_masked || source.cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
            lctx.model.devices.size() != 1 || source.model.devices.size() != 1 || lctx.model.devices[0].dev != source.model.devices[0].dev ||
            lctx.model.n_gpu_layers() <= lctx.model.hparams.n_layer_all || source.balloc->get_n_tokens() != (uint32_t) n) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != batch.seq_id[0][0] ||
                batch.pos[i] != batch.pos[0] + i || batch.logits[i]) {
            return false;
        }
    }
    if (eagle && (lctx.memory->seq_pos_max(batch.seq_id[0][0]) != batch.pos[0] || !this->graph ||
            this->graph->seed_pos != batch.pos[0] || this->graph->seed_token != batch.token[0])) {
        return false;
    }
    const auto & source_batch = source.spec->last_ubatch;
    if (source_batch.n_tokens != (uint32_t) n || !source_batch.token || !source_batch.pos || !source_batch.n_seq_id || !source_batch.seq_id) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (source_batch.token[i] != batch.token[i] || source_batch.pos[i] != batch.pos[i] + (int) eagle ||
                source_batch.n_seq_id[i] != 1 || source_batch.seq_id[i][0] != batch.seq_id[i][0]) {
            return false;
        }
    }
    const int64_t dim = lctx.model.hparams.n_embd_out();
    const int64_t feature_dim = eagle ? lctx.model.hparams.n_embd_inp_enc()/3 : dim;
    std::vector<ggml_tensor *> hidden;
    if (eagle) {
        if (lctx.model.target_layer_ids.size() != 3 || feature_dim != source.model.hparams.n_embd) {
            return false;
        }
        for (int id : lctx.model.target_layer_ids) {
            if (id >= 0 && id < (int) source.model.hparams.n_layer() && source.cparams.embeddings_layer_inp[id]) {
                hidden.push_back(source.gf_res_prev->t_layer_inp[id]);
            } else if (id == (int) source.model.hparams.n_layer() && source.cparams.embeddings_nextn) {
                hidden.push_back(source.gf_res_prev->get_h_nextn());
            } else {
                return false;
            }
        }
    } else {
        hidden.push_back(source.gf_res_prev->get_h_nextn());
    }
    auto * src_backend = source.backend_ptrs[0];
    for (auto * tensor : hidden) {
        if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != feature_dim || tensor->ne[1] != n ||
                !ggml_is_contiguous(tensor) || ggml_backend_buffer_is_host(tensor->buffer) ||
                ggml_backend_sched_get_tensor_backend(source.sched.get(), tensor) != src_backend) {
            return false;
        }
    }
    auto * dst_backend = lctx.backend_ptrs[0];
    auto * dev = ggml_backend_get_device(dst_backend);
    if (!dev || ggml_backend_get_device(src_backend) != dev ||
            strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
        return false;
    }
    if (!this->handoff) {
        auto state = std::make_unique<llama_nextn_handoff>();
        state->ctx.reset(ggml_init({ 12*ggml_tensor_overhead(), nullptr, true }));
        auto * storage = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, dim, 9);
        for (int i = 1; i <= 9; i++) {
            state->rows.push_back(ggml_view_2d(state->ctx.get(), storage, dim, i, storage->nb[1], 0));
        }
        state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), dst_backend));
        auto * host_type = ggml_backend_dev_host_buffer_type(dev);
        if (host_type) {
            state->host.reset(ggml_backend_buft_alloc_buffer(host_type, dim*9*sizeof(float)));
        }
        state->consumed.reset(ggml_backend_event_new(dev));
        if (!state->device || !state->host || !state->consumed) {
            return false;
        }
        this->handoff = std::move(state);
    }
    if (eagle && !this->features) {
        auto state = std::make_unique<llama_nextn_handoff>();
        state->ctx.reset(ggml_init({ 31*ggml_tensor_overhead(), nullptr, true }));
        for (int layer = 0; layer < 3; ++layer) {
            auto * storage = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, feature_dim, 9);
            for (int i = 1; i <= 9; ++i) {
                state->rows.push_back(ggml_view_2d(state->ctx.get(), storage, feature_dim, i, storage->nb[1], 0));
            }
        }
        state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), dst_backend));
        state->consumed.reset(ggml_backend_event_new(dev));
        if (!state->device || !state->consumed) {
            return false;
        }
        this->features = std::move(state);
    }
    auto & state = *this->handoff;
    auto * input = state.rows[n - 1];
    // Protect the destination from the preceding catch-up, then copy on the producer stream.
    ggml_backend_event_record(state.consumed.get(), dst_backend);
    ggml_backend_event_wait(src_backend, state.consumed.get());
    if (eagle) {
        auto get_batch = (ggml_backend_get_tensors_async_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_get_tensors_async");
        std::vector<const ggml_tensor *> tensors(hidden.begin(), hidden.end());
        std::vector<void *> destinations;
        for (int layer = 0; layer < 3; ++layer) {
            auto * features = this->features->rows[layer*9 + n - 1];
            destinations.push_back(features->data);
            this->catchup_features[layer] = features;
        }
        if (get_batch && get_batch(src_backend, tensors.data(), destinations.data(), tensors.size())) {
            // Wait once after all three feature copies on the producer stream.
            ggml_backend_event_record(this->features->consumed.get(), src_backend);
            ggml_backend_event_wait(dst_backend, this->features->consumed.get());
        } else {
            for (int layer = 0; layer < 3; ++layer) {
                ggml_backend_tensor_copy_async(src_backend, dst_backend, hidden[layer], this->catchup_features[layer]);
            }
        }
    } else {
        ggml_backend_tensor_copy_async(src_backend, dst_backend, hidden[0], input);
        ggml_backend_tensor_get_async(src_backend, hidden[0], ggml_backend_buffer_get_base(state.host.get()), 0, ggml_nbytes(hidden[0]));
    }
    this->catchup_input = input;
    auto decode_batch = batch;
    if (eagle) {
        --decode_batch.n_tokens;
        ++decode_batch.token;
        ++decode_batch.pos;
        ++decode_batch.n_seq_id;
        ++decode_batch.seq_id;
        ++decode_batch.logits;
        decode_batch.embd = nullptr;
    }
    std::vector<float> encoder_stub;
    if (eagle && n == 1) {
        decode_batch = batch;
        decode_batch.token = nullptr;
        encoder_stub.resize(lctx.model.hparams.n_embd_inp_enc());
        decode_batch.embd = encoder_stub.data();
    }
    const int ret = eagle && n == 1 ? lctx.encode(decode_batch) : lctx.decode(decode_batch);
    this->catchup_input = nullptr;
    this->catchup_features = {};
    if (ret != 0) {
        lctx.synchronize();
        lctx.memory->seq_rm(batch.seq_id[0][0], batch.pos[0], -1);
        return false;
    }
    if (eagle) {
        ggml_backend_tensor_get_async(dst_backend, input, ggml_backend_buffer_get_base(state.host.get()), 0, ggml_nbytes(input));
        ggml_backend_event_record(this->features->consumed.get(), dst_backend);
    }
    *snapshot = (const float *) ggml_backend_buffer_get_base(state.host.get());
    return true;
}

void llama_spec_pipeline::synchronize_nextn_catchup() {
    GGML_ASSERT(this->features && this->features->consumed);
    ggml_backend_event_synchronize(this->features->consumed.get());
}

bool llama_spec_pipeline::decode_nextn_prefetch(llama_context & source, const llama_batch & batch, int n_draft, float p_min) {
    const int n = batch.n_tokens;
    if (n_draft == 0) {
        n_draft = n - 1;
    }
    if (!this->prefetch_enabled || !source.spec->prefetch_enabled ||
            &source == &lctx || !source.spec->readback || !this->handoff || n < 1 || n > 9 || n_draft < 2 || n_draft > 8 ||
            lctx.cparams.n_seq_max != 1 || !batch.token || !batch.embd || !batch.pos || !batch.seq_id ||
            !batch.n_seq_id || !source.spec->last_ubatch.token || lctx.model.hparams.n_swa || lctx.model.hparams.use_alibi ||
            lctx.model.vocab.n_tokens() > (1u << 24) || batch.pos[0] < 0 || batch.pos[0] > (1 << 24) - 2*n ||
            (this->graph && this->graph->prefetched) || !source.gf_res_prev ||
            source.spec->last_ubatch.n_tokens != (uint32_t) n ||
            source.gf_res_prev->t_sampled.size() != (size_t) n ||
            lctx.memory->seq_pos_max(batch.seq_id[0][0]) != batch.pos[0] + n - 1) {
        LLAMA_LOG_DEBUG("%s: unavailable target outputs or catch-up state (n=%d)\n", __func__, n);
        return false;
    }
    if (lctx.model.arch == LLM_ARCH_EAGLE3) {
        auto * kv = static_cast<llama_kv_cache *>(lctx.memory.get());
        if (!lctx.cparams.flash_attn || kv->get_n_stream() != 1 || ggml_is_quantized(kv->type_k()) || ggml_is_quantized(kv->type_v())) {
            return false;
        }
        const auto & cells = kv->get_cells(0);
        for (int i = 0; i < batch.pos[0] + n; ++i) {
            if (!cells.seq_has(i, 0) || cells.pos_get(i) != i) {
                return false;
            }
        }
    }
    auto * src_backend = source.backend_ptrs[0];
    auto * dst_backend = lctx.backend_ptrs[0];
    auto * dev = ggml_backend_get_device(dst_backend);
    if (!dev || ggml_backend_get_device(src_backend) != dev) {
        return false;
    }
    auto get_batch = (ggml_backend_get_tensors_async_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_get_tensors_async");
    if (!get_batch) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        auto * tensor = source.gf_res_prev->t_sampled[i];
        if (!tensor || tensor->type != GGML_TYPE_I32 || ggml_nelements(tensor) != 1 ||
                ggml_backend_sched_get_tensor_backend(source.sched.get(), tensor) != src_backend ||
                source.spec->last_ubatch.token[i] != batch.token[i] ||
                source.spec->last_ubatch.pos[i] != batch.pos[i] + (lctx.model.arch == LLM_ARCH_EAGLE3 ? 1 : 0) ||
                batch.n_seq_id[i] != 1 || batch.pos[i] != batch.pos[0] + i || batch.seq_id[i][0] != batch.seq_id[0][0]) {
            LLAMA_LOG_DEBUG("%s: unsupported sampled row %d (%s)\n", __func__, i, tensor ? ggml_get_name(tensor) : "missing");
            return false;
        }
    }
    if (!this->lookahead) {
        auto state = std::make_unique<llama_nextn_lookahead>();
        auto & sampled = state->sampled;
        sampled.ctx.reset(ggml_init({ 11*ggml_tensor_overhead(), nullptr, true }));
        auto * storage = ggml_new_tensor_1d(sampled.ctx.get(), GGML_TYPE_I32, 9);
        for (int i = 1; i <= 9; i++) {
            sampled.rows.push_back(ggml_view_1d(sampled.ctx.get(), storage, i, 0));
        }
        sampled.device.reset(ggml_backend_alloc_ctx_tensors(sampled.ctx.get(), dst_backend));
        auto * host_type = ggml_backend_dev_host_buffer_type(dev);
        if (host_type) {
            sampled.host.reset(ggml_backend_buft_alloc_buffer(host_type, 9*sizeof(llama_token)));
        }
        sampled.consumed.reset(ggml_backend_event_new(dev));
        state->ready.reset(ggml_backend_event_new(dev));
        if (!sampled.device || !sampled.host || !sampled.consumed || !state->ready) {
            return false;
        }
        this->lookahead = std::move(state);
    }
    auto & state = *this->lookahead;
    if (state.pos >= 0) {
        ggml_backend_event_wait(src_backend, state.sampled.consumed.get());
    }
    auto * sampled = state.sampled.rows[n - 1];
    std::vector<const ggml_tensor *> tensors;
    std::vector<void *> destinations;
    for (int i = 0; i < n; i++) {
        tensors.push_back(source.gf_res_prev->t_sampled[i]);
        destinations.push_back((llama_token *) sampled->data + i);
    }
    if (!get_batch(src_backend, tensors.data(), destinations.data(), tensors.size())) {
        return false;
    }
    ggml_backend_event_record(state.ready.get(), src_backend);
    ggml_backend_event_wait(dst_backend, state.ready.get());
    ggml_backend_tensor_get_async(src_backend, sampled, ggml_backend_buffer_get_base(state.sampled.host.get()), 0, ggml_nbytes(sampled));
    ggml_backend_event_record(state.ready.get(), src_backend);
    state.pos = batch.pos[0];
    state.seq = batch.seq_id[0][0];
    state.draft.assign(batch.token + 1, batch.token + n);
    state.draft.push_back(LLAMA_TOKEN_NULL);
    llama_token token = 0;
    llama_pos pos = batch.pos[0] + n;
    int32_t n_seq = 1;
    int8_t output = 1;
    llama_batch seed = { 1, &token, batch.embd, &pos, &n_seq, batch.seq_id, &output };
    const bool result = decode_nextn(seed, n_draft, true, p_min);
    ggml_backend_event_record(state.sampled.consumed.get(), dst_backend);
    // The server rebuilds draft KV from verified hidden states after each verification.
    lctx.memory->seq_rm(state.seq, pos, -1);
    if (result) {
        source.prefetch_nextn_target(lctx);
    }
    return result;
}

bool llama_spec_pipeline::decode_nextn(const llama_batch & seed, int32_t n_draft, bool prefetch, float p_min) {
    if (p_min > 0.0f && !this->graph_cache_max) {
        return false;
    }
    if (!prefetch && this->graph && this->graph->prefetched) {
        auto & chain = *this->graph;
        chain.prefetched = false;
        auto & state = *this->lookahead;
        if (!lctx.sched_need_reserve && !lctx.graph_reuse_disable && n_draft == (int) chain.tokens.size() && chain.p_min == p_min &&
                seed.n_tokens == 1 && seed.token && seed.embd && seed.pos && seed.seq_id &&
                seed.n_seq_id && seed.n_seq_id[0] == 1 && seed.seq_id[0][0] == state.seq &&
                lctx.cparams.embeddings_nextn_masked && lctx.cparams.embeddings_nextn &&
                lctx.cparams.embeddings == chain.params[0].cparams.embeddings &&
                lctx.cparams.nextn_layer_offset == chain.params[0].cparams.nextn_layer_offset) {
            ggml_backend_event_synchronize(state.ready.get());
            if (this->features) {
                synchronize_nextn_catchup();
            }
            const auto * sampled = (const llama_token *) ggml_backend_buffer_get_base(state.sampled.host.get());
            int accepted = 0;
            while (accepted + 1 < (int) state.draft.size() && sampled[accepted] == state.draft[accepted]) {
                ++accepted;
            }
            const auto dim = lctx.model.hparams.n_embd_out();
            const auto * hidden = (const float *) ggml_backend_buffer_get_base(this->handoff->host.get());
            if (seed.pos[0] == state.pos + accepted + 1 && seed.token[0] == sampled[accepted] &&
                    !memcmp(seed.embd, hidden + (size_t) accepted*dim, dim*sizeof(float))) {
                if (lctx.model.arch == LLM_ARCH_EAGLE3) {
                    auto * kv = static_cast<llama_kv_cache *>(lctx.memory.get());
                    const auto & cells = kv->get_cells(0);
                    for (int i = 0; i < seed.pos[0]; ++i) {
                        if (!cells.seq_has(i, state.seq) || cells.pos_get(i) != i) {
                            return false;
                        }
                    }
                    if (lctx.memory->seq_pos_max(state.seq) != seed.pos[0] - 1 ||
                            !lctx.balloc->init(seed, lctx.model.vocab, lctx.memory.get(), dim, lctx.cparams.n_seq_max, false)) {
                        return false;
                    }
                    auto mctx = lctx.memory->init_batch(*lctx.balloc, 1, false);
                    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS || !mctx->apply()) {
                        return false;
                    }
                }
                chain.seed_token = seed.token[0];
                chain.seed_pos = seed.pos[0];
                chain.seed_seq = state.seq;
                LLAMA_LOG_DEBUG("%s: consuming GPU-prefetched draft after %d accepted tokens\n", __func__, accepted);
                return true;
            }
        }
    }
    this->readback.reset();
    this->output_pending = false;
    this->outputs = false;
    const bool supported = lctx.model.arch == LLM_ARCH_EAGLE3 ||
        (lctx.cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && lctx.model.hparams.n_layer_nextn == 1 &&
         (lctx.model.arch == LLM_ARCH_QWEN35 || lctx.model.arch == LLM_ARCH_QWEN35MOE));
    if (!supported || seed.n_tokens != 1 || !seed.token || !seed.embd || !seed.pos || !seed.seq_id ||
            !seed.n_seq_id || seed.n_seq_id[0] != 1 || n_draft < 2 || n_draft > 8 ||
            n_draft > (int32_t) lctx.cparams.n_outputs_max || n_draft > (int32_t) lctx.cparams.n_batch ||
            lctx.cparams.pipeline_parallel || lctx.cparams.cb_eval || lctx.opt_ctx || !lctx.cparams.embeddings_nextn_masked ||
            lctx.model.devices.size() != 1 || lctx.model.n_gpu_layers() <= lctx.model.hparams.n_layer_all ||
            !dynamic_cast<llama_kv_cache *>(lctx.memory.get()) || lctx.sampling.samplers.empty()) {
        return false;
    }
    const auto * tok_embd = lctx.model.tok_embd;
    if (lctx.cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        const auto * mtp_embd = lctx.model.layers[lctx.model.hparams.n_layer()].nextn.embed_tokens;
        if (mtp_embd) {
            tok_embd = mtp_embd;
        }
    } else if (!tok_embd && lctx.cparams.ctx_other) {
        tok_embd = llama_get_model(lctx.cparams.ctx_other)->tok_embd;
    }
    if (!tok_embd || ggml_backend_buffer_is_host(tok_embd->buffer)) {
        return false;
    }

    lctx.sched_reserve();
    lctx.memory_update(false);
    if (lctx.output_reserve(n_draft) < (uint32_t) n_draft) {
        return false;
    }

    const auto seq_id = seed.seq_id[0][0];
    const int64_t dim = lctx.model.hparams.n_embd_out();
    std::vector<llama_token> tokens(n_draft, seed.token[0]);
    std::vector<llama_pos> positions(n_draft);
    std::vector<float> hidden(n_draft * dim, 0.0f);
    std::copy_n(seed.embd, dim, hidden.data());
    std::vector<int32_t> n_seq_id(n_draft, 1);
    std::vector<llama_seq_id *> seq_ids(n_draft, seed.seq_id[0]);
    std::vector<int8_t> outputs(n_draft, 1);
    for (int i = 0; i < n_draft; i++) {
        positions[i] = seed.pos[0] + i;
    }
    llama_batch batch = { n_draft, tokens.data(), hidden.data(), positions.data(), n_seq_id.data(), seq_ids.data(), outputs.data() };
    if (!lctx.balloc->init(batch, lctx.model.vocab, lctx.memory.get(), dim, lctx.cparams.n_seq_max, false)) {
        return false;
    }
    auto mctx = lctx.memory->init_batch(*lctx.balloc, n_draft, false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return false;
    }

    auto fail = [&]() {
        lctx.synchronize();
        lctx.memory->seq_rm(seq_id, seed.pos[0], -1);
        this->graph.reset();
        return false;
    };
    std::vector<std::unique_ptr<llama_kv_cache_context>> contexts;
    if (mctx->get_ubatch().n_tokens != (uint32_t) n_draft || !mctx->apply()) {
        return fail();
    }
    // Reserve all positions once. Causal masks hide later steps until their KV is written.
    for (int i = 0; i < n_draft; i++) {
        contexts.push_back(static_cast<llama_kv_cache_context *>(mctx.get())->for_token(i, dim));
    }

    const int n_verify = prefetch ? this->lookahead->draft.size() : 0;
    const uint32_t cache_index = p_min > 0.0f ? n_verify : 0;
    if (cache_index != this->graph_active) {
        this->graph_cache[this->graph_active] = std::move(this->graph);
        this->graph = std::move(this->graph_cache[cache_index]);
        this->graph_active = cache_index;
    }
    if (!this->graph || (int) this->graph->params.size() != n_draft) {
        const size_t max_nodes = lctx.graph_max_nodes(1) * n_draft;
        this->graph.reset(new llama_nextn_graph);
        this->graph->res.reset(new llm_graph_result(max_nodes));
        this->graph->sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(), max_nodes, false, lctx.cparams.op_offload));
    }
    auto & chain = *this->graph;
    auto * res = chain.res.get();
    auto params_for = [&](int i) {
        auto params = lctx.graph_params(res, contexts[i]->get_ubatch(), contexts[i].get(), lctx.ctx_type_to_graph_type(lctx.cparams.ctx_type));
        params.sched = chain.sched.get();
        params.n_outputs = 1;
        params.samplers.clear();
        params.nextn_tokens = i ? chain.tokens[i - 1] : prefetch ? chain.selected_token : nullptr;
        params.nextn_hidden = i ? chain.hidden[i - 1] : prefetch ? chain.selected_hidden : nullptr;
        params.nextn_positions = prefetch && !chain.positions.empty() ? chain.positions[i] : nullptr;
        params.nextn_reject_mask = prefetch && lctx.model.arch != LLM_ARCH_EAGLE3 ? chain.reject_mask : nullptr;
        params.nextn_gpu_kv = prefetch && lctx.model.arch == LLM_ARCH_EAGLE3;
        params.cb = [](const llama_ubatch &, ggml_tensor * tensor, const char * name, int il) {
            ggml_format_name(tensor, "%s-%d", name, il);
        };
        return params;
    };
    bool reuse = !lctx.graph_reuse_disable && (int) chain.params.size() == n_draft && chain.prefetch == prefetch &&
        chain.p_min == p_min && chain.n_verify == n_verify;
    for (int i = 0; reuse && i < n_draft; i++) {
        const auto params = params_for(i);
        reuse = chain.params[i].allow_reuse(params);
        for (size_t j = i ? chain.input_ends[i - 1] : 0; reuse && j < chain.input_ends[i]; j++) {
            reuse = res->inputs[j]->can_reuse(params);
        }
    }
    if (!reuse) {
        if (chain.done) {
            ggml_backend_event_synchronize(chain.done.get());
        }
        ggml_backend_sched_reset(chain.sched.get());
        res->reset();
        chain.params.clear();
        chain.input_ends.clear();
        chain.tokens.clear();
        chain.hidden.clear();
        chain.candidates.clear();
        chain.logits.clear();
        chain.positions.clear();
        chain.prefetch = prefetch;
        chain.p_min = p_min;
        chain.n_verify = n_verify;
        if (prefetch) {
            auto * ctx = res->get_ctx();
            const int n_pos = lctx.model.hparams.n_pos_per_embd();
            auto input = [&](ggml_type type, int64_t n) {
                auto * tensor = ggml_new_tensor_1d(ctx, type, n);
                ggml_set_input(tensor);
                return tensor;
            };
            chain.inp_draft = input(GGML_TYPE_I32, n_verify);
            chain.inp_weights = input(GGML_TYPE_F32, n_verify);
            chain.inp_positions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pos, n_draft);
            ggml_set_input(chain.inp_positions);
            chain.inp_position_mask = input(GGML_TYPE_F32, n_pos);
            chain.inp_kv_positions = lctx.model.arch == LLM_ARCH_EAGLE3 ? nullptr : input(GGML_TYPE_F32, contexts[0]->get_n_kv());
            auto * sampled = this->lookahead->sampled.rows[n_verify - 1];
            auto * mismatch = ggml_step(ctx, ggml_abs(ctx, ggml_sub(ctx,
                    ggml_cast(ctx, sampled, GGML_TYPE_F32), ggml_cast(ctx, chain.inp_draft, GGML_TYPE_F32))));
            // Descending weights make the first mismatch unique, including the bonus row.
            chain.accepted = ggml_argmax(ctx, ggml_mul(ctx, mismatch, chain.inp_weights));
            chain.selected_token = ggml_get_rows(ctx, ggml_reshape_2d(ctx, sampled, 1, n_verify), chain.accepted);
            ggml_set_output(chain.accepted);
            ggml_set_output(chain.selected_token);
            chain.selected_hidden = ggml_get_rows(ctx, this->handoff->rows[n_verify - 1], chain.accepted);
            auto * accepted_f32 = ggml_cast(ctx, chain.accepted, GGML_TYPE_F32);
            auto * positions = ggml_cast(ctx, ggml_mul(ctx,
                    ggml_add(ctx, chain.inp_positions, accepted_f32), chain.inp_position_mask), GGML_TYPE_I32);
            ggml_set_output(positions);
            for (int i = 0; i < n_draft; i++) {
                chain.positions.push_back(ggml_view_1d(ctx, positions, n_pos, i*n_pos*sizeof(int32_t)));
            }
            chain.reject_mask = chain.inp_kv_positions ? ggml_scale(ctx, ggml_step(ctx,
                    ggml_sub(ctx, chain.inp_kv_positions, accepted_f32)), -1e30f) : nullptr;
        }
        for (int i = 0; i < n_draft; i++) {
            auto params = params_for(i);
            if (!lctx.model.build_graph(params) || !res->get_h_nextn() || !res->get_logits()) {
                return fail();
            }
            auto * ctx = res->get_ctx();
            auto * logits = ggml_reshape_1d(ctx, res->get_logits(), lctx.model.vocab.n_tokens());
            auto * ids = ggml_top_k(ctx, logits, std::min<uint32_t>(10, lctx.model.vocab.n_tokens()));
            auto * values = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, logits->ne[0]), ids);
            ggml_set_output(ids);
            ggml_set_output(values);
            ggml_build_forward_expand(res->get_gf(), values);
            chain.candidates.push_back(ids);
            chain.logits.push_back(values);
            chain.tokens.push_back(ggml_view_1d(ctx, ids, 1, 0));
            ggml_build_forward_expand(res->get_gf(), chain.tokens.back());
            chain.hidden.push_back(res->get_h_nextn());
            chain.params.push_back(std::move(params));
            chain.input_ends.push_back(res->inputs.size());
        }
        if (!ggml_backend_sched_alloc_graph_after(chain.sched.get(), res->get_gf(), chain.done.get())) {
            return fail();
        }
        for (int n = 0; n < ggml_graph_n_nodes(res->get_gf()); n++) {
            auto * node = ggml_graph_node(res->get_gf(), n);
            if (ggml_backend_sched_get_tensor_backend(chain.sched.get(), node) == lctx.backend_cpu) {
                LLAMA_LOG_DEBUG("%s: CPU node %s (%s)\n", __func__, ggml_get_name(node), ggml_op_desc(node));
                return fail();
            }
        }
    } else {
        lctx.n_reused++;
    }
    if (p_min > 0.0f) {
        if (!chain.early_device) {
            auto * backend = lctx.backend_ptrs[0];
            auto * dev = ggml_backend_get_device(backend);
            auto * host_type = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (!host_type) {
                return fail();
            }
            chain.early_ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
            chain.early_counts = ggml_new_tensor_1d(chain.early_ctx.get(), GGML_TYPE_I32, 2);
            chain.early_device.reset(ggml_backend_alloc_ctx_tensors(chain.early_ctx.get(), backend));
            chain.early_host.reset(ggml_backend_buft_alloc_buffer(host_type, 2*sizeof(int32_t)));
            if (!chain.early_device || !chain.early_host) {
                return fail();
            }
        }
        if (!ggml_backend_sched_graph_early_exit(chain.sched.get(), chain.logits.data(), n_draft, p_min, chain.early_counts)) {
            return fail();
        }
    }
    if (prefetch) {
        const auto & state = *this->lookahead;
        std::vector<float> weights(n_verify);
        std::vector<float> positions(lctx.model.hparams.n_pos_per_embd()*n_draft);
        std::vector<float> position_mask(lctx.model.hparams.n_pos_per_embd(), 1.0f);
        if (position_mask.size() == 4) {
            position_mask[3] = 0.0f;
        }
        std::vector<float> kv_positions(chain.inp_kv_positions ? chain.inp_kv_positions->ne[0] : 0, -1.0f);
        for (int i = 0; i < n_verify; i++) {
            weights[i] = n_verify - i;
        }
        for (int i = 0; i < n_draft; i++) {
            for (uint32_t j = 0; j < lctx.model.hparams.n_pos_per_embd(); j++) {
                positions[i*lctx.model.hparams.n_pos_per_embd() + j] = state.pos + 1 + i;
            }
        }
        const auto & cells = static_cast<llama_kv_cache *>(lctx.memory.get())->get_cells(seq_id);
        for (size_t i = 0; i < kv_positions.size(); i++) {
            if (cells.seq_has(i, seq_id) && cells.pos_get(i) >= state.pos && cells.pos_get(i) < state.pos + n_verify) {
                kv_positions[i] = cells.pos_get(i) - state.pos;
            }
        }
        ggml_backend_tensor_set(chain.inp_draft, state.draft.data(), 0, ggml_nbytes(chain.inp_draft));
        ggml_backend_tensor_set(chain.inp_weights, weights.data(), 0, ggml_nbytes(chain.inp_weights));
        ggml_backend_tensor_set(chain.inp_positions, positions.data(), 0, ggml_nbytes(chain.inp_positions));
        ggml_backend_tensor_set(chain.inp_position_mask, position_mask.data(), 0, ggml_nbytes(chain.inp_position_mask));
        if (chain.inp_kv_positions) {
            ggml_backend_tensor_set(chain.inp_kv_positions, kv_positions.data(), 0, ggml_nbytes(chain.inp_kv_positions));
        }
    }
    for (int i = 0; i < n_draft; i++) {
        for (size_t j = i ? chain.input_ends[i - 1] : 0; j < chain.input_ends[i]; j++) {
            res->inputs[j]->set_input(&contexts[i]->get_ubatch());
        }
    }
    if (ggml_backend_sched_graph_compute_async(chain.sched.get(), res->get_gf()) != GGML_STATUS_SUCCESS) {
        return fail();
    }
    if (!chain.done) {
        chain.done.reset(ggml_backend_event_new(ggml_backend_get_device(lctx.backend_ptrs[0])));
    }
    ggml_backend_event_record(chain.done.get(), lctx.backend_ptrs[0]);

    const auto stride = lctx.model.vocab.n_tokens();
    llama_output_copies copies;
    if (p_min > 0.0f) {
        copies.add(lctx.backend_ptrs[0], chain.early_counts, ggml_backend_buffer_get_base(chain.early_host.get()));
    }
    copy_tensor_async_rows(chain.logits, lctx.sampling.logits, stride, 0, chain.sched.get(), copies, &lctx.sampling.logits_count);
    copy_tensor_async_rows(chain.candidates, lctx.sampling.candidates, stride, 0, chain.sched.get(), copies, &lctx.sampling.candidates_count);
    this->readback = std::make_unique<llama_output_copies>(std::move(copies));
    lctx.logits.data = nullptr;
    lctx.embd_nextn.data = nullptr;
    lctx.output_swaps.clear();
    lctx.n_outputs = n_draft;
    for (int i = 0; i < n_draft; i++) {
        lctx.output_ids[i] = i;
    }
    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }
    lctx.n_queued_tokens += n_draft;
    chain.seed_token = seed.token[0];
    chain.seed_pos = seed.pos[0];
    chain.seed_seq = seed.seq_id[0][0];
    this->outputs = true;
    chain.prefetched = prefetch;
    ++chain.generation;
    return true;
}

int32_t llama_spec_pipeline::nextn_draft_length() {
    if (!this->graph || !this->graph->early_host || this->graph->p_min <= 0.0f) {
        return -1;
    }
    lctx.synchronize(true);
    const auto * counts = (const int32_t *) ggml_backend_buffer_get_base(this->graph->early_host.get());
    GGML_ASSERT(counts[0] >= 0 && counts[0] <= (int32_t) this->graph->tokens.size());
    LLAMA_LOG_DEBUG("%s: GPU early exit executed %d/%zu heads, kept %d\n", __func__, counts[1], this->graph->tokens.size(), counts[0]);
    return counts[0];
}

void llama_spec_pipeline::wait_for_snapshots() {
    if (this->lookahead && this->lookahead->pos >= 0) {
        // The acceptance snapshot is copied on the target context's stream.
        ggml_backend_event_synchronize(this->lookahead->ready.get());
    }
}

ggml_backend_sched_t llama_spec_pipeline::main_scheduler() const {
    return target_active >= 0 ? targets[0]->sched.get() :
        (lctx.graph_cache_active ? lctx.sched_cache[0].get() : lctx.sched.get());
}

void llama_spec_pipeline::reset_targets(bool sampler_only) {
    if (sampler_only && this->prefetch_enabled) {
        lctx.select_graph(0);
        for (auto & entry : this->targets) {
            if (entry && entry->build_controls) {
                // Request sampler addresses can be reused after the old sampler is freed.
                entry->build_controls->samplers.clear();
            }
        }
    } else {
        for (auto & entry : this->targets) {
            entry.reset();
        }
    }
    this->target_active = this->target_pending = -1;
    this->target_consume = false;
}

void llama_spec_pipeline::reset_draft() {
    this->graph.reset();
    for (auto & entry : this->graph_cache) {
        entry.reset();
    }
    this->graph_active = 0;
}

void llama_spec_pipeline::synchronize(bool outputs_only) {
    if (this->readback) {
        this->readback->submit();
        this->readback.reset();
    }
    if (outputs_only && this->output_pending) {
        ggml_backend_event_synchronize(this->output_ready.get());
    } else {
        ggml_backend_sched_synchronize(lctx.sched.get());
    }
}

void llama_spec_pipeline::set_nextn_prefetch(bool enabled, bool fixed_kv) {
    lctx.synchronize();
    this->prefetch_enabled = enabled;
    this->prefetch_reported = false;
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(lctx.memory.get());
    auto * kv = hybrid ? hybrid->get_mem_attn() : dynamic_cast<llama_kv_cache *>(lctx.memory.get());
    if (!kv) {
        return;
    }
    const bool supported = lctx.model.arch == LLM_ARCH_QWEN3 || lctx.model.arch == LLM_ARCH_QWEN3MOE ||
        lctx.model.arch == LLM_ARCH_QWEN35 || lctx.model.arch == LLM_ARCH_QWEN35MOE || lctx.model.arch == LLM_ARCH_EAGLE3;
    const bool fixed = enabled && fixed_kv && supported && lctx.cparams.n_seq_max == 1 && lctx.cparams.flash_attn &&
        !lctx.cparams.pipeline_parallel && !lctx.model.hparams.n_swa && kv->get_size() <= 8192 &&
        !ggml_is_quantized(kv->type_k()) && !ggml_is_quantized(kv->type_v()) &&
        lctx.model.devices.size() == 1 && lctx.model.n_gpu_layers() > lctx.model.hparams.n_layer_all &&
        lctx.model.tok_embd && !ggml_backend_buffer_is_host(lctx.model.tok_embd->buffer) &&
        strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(lctx.model.devices[0].dev)), "CUDA") == 0;
    if (kv->set_fixed_size(fixed)) {
        lctx.set_sched_need_reserve();
    }
}

void llama_spec_pipeline::set_nextn_graph_cache(int32_t n_max) {
    this->graph_cache_max = n_max >= 1 && n_max < (int32_t) lctx.graph_cache_stride ? n_max : 0;
    lctx.set_sched_need_reserve();
}

void llama_spec_pipeline::release_target() {
    if (this->target_active >= 0) {
        auto & active = *this->targets[this->target_active];
        active.sched = std::move(lctx.sched);
        active.res = std::move(lctx.gf_res_prev);
        lctx.sched = std::move(this->targets[0]->sched);
        lctx.gf_res_prev = std::move(this->targets[0]->res);
        this->target_active = -1;
    }
}

llm_graph_result * llama_spec_pipeline::consume_target(const llama_ubatch & ubatch, llama_memory_context_i * mctx) {
    if (!this->target_consume) {
        return nullptr;
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(lctx.memory.get());
    auto * hctx = dynamic_cast<llama_memory_hybrid_context *>(mctx);
    GGML_ASSERT(ubatch.n_tokens == (uint32_t) this->targets[this->target_pending]->n);
    if (hybrid) {
        GGML_ASSERT(hctx);
        const int rollback = hctx->get_recr()->s_copy(0);
        GGML_ASSERT(rollback == this->targets[this->target_pending]->previous_pos + this->targets[this->target_pending]->previous_n - ubatch.pos[0]);
        hybrid->get_mem_recr()->nextn_commit_states();
    }
    if (this->target_active < 0) {
        lctx.select_graph(0);
    }
    const int previous = this->target_active < 0 ? 0 : this->target_active;
    if (!this->targets[previous]) {
        this->targets[previous] = std::make_unique<llama_nextn_target>();
    }
    this->targets[previous]->sched = std::move(lctx.sched);
    this->targets[previous]->res = std::move(lctx.gf_res_prev);
    auto & prepared = *this->targets[this->target_pending];
    lctx.sched = std::move(prepared.sched);
    lctx.gf_res_prev = std::move(prepared.res);
    this->target_active = this->target_pending;
    this->target_bank = this->target_active/10;
    this->target_pending = -1;
    this->target_consume = false;
    this->last_ubatch = ubatch;
    ++lctx.n_reused;
    LLAMA_LOG_DEBUG("%s: consuming GPU-prefetched target (%u rows)\n", __func__, ubatch.n_tokens);
    return lctx.gf_res_prev.get();
}

void llama_spec_pipeline::set_graph_inputs(llm_graph_params & gparams) const {
    gparams.nextn_hidden = this->catchup_input;
    gparams.nextn_features = this->catchup_features;
    gparams.nextn_verify_tokens = this->verify_input;
}

void llama_spec_pipeline::record_batch(const llama_ubatch & ubatch) {
    if (this->verify_input || (this->prefetch_enabled && this->graph_cache_max && !lctx.cparams.embeddings_nextn_masked) ||
            (lctx.cparams.embeddings_nextn && !lctx.cparams.embeddings_nextn_masked)) {
        this->last_ubatch = ubatch;
    }
}

void llama_spec_pipeline::reset_outputs() {
    if (this->graph) {
        this->graph->prefetched = false;
    }
    this->readback.reset();
    this->output_pending = false;
    this->outputs = false;
    this->last_ubatch = {};
}

void llama_spec_pipeline::begin_decode() {
    if (this->target_pending >= 0 && !this->target_consume) {
        lctx.synchronize();
        this->target_pending = -1;
    }
    this->output_pending = false;
    this->mctx.reset();
    if (this->graph) {
        this->graph->prefetched = false;
    }
    this->readback.reset();
    this->outputs = false;
    this->last_ubatch = {};
}

bool llama_spec_pipeline::should_defer_outputs(const llama_ubatch & ubatch, uint32_t n_tokens) const {
    return (verify_input || (prefetch_enabled && graph_cache_max && ubatch.n_tokens <= 9)) && ubatch.n_tokens == n_tokens;
}

void llama_spec_pipeline::stage_outputs(llama_output_copies copies, llama_memory_context_ptr & mctx, bool defer) {
    if (defer) {
        readback = std::make_unique<llama_output_copies>(std::move(copies));
        this->mctx = std::move(mctx);
    } else {
        copies.submit();
    }
}

void llama_spec_pipeline::invalidate_graphs() {
    if (std::any_of(this->targets.begin(), this->targets.end(), [](const auto & entry) { return (bool) entry; })) {
        lctx.synchronize();
        this->target_pending = -1;
        for (auto & state : this->targets) {
            if (state) {
                state->controls.reset();
                state->build_controls.reset();
            }
        }
    }
    if (this->readback) {
        lctx.synchronize();
    }
    if (this->graph) {
        this->graph->prefetched = false;
    }
}

void llama_spec_pipeline::add_memory_usage(std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> & ret) const {
    if (!lctx.model.hparams.no_alloc) {
        for (const auto & backend_ptr : lctx.backends) {
            auto * backend = backend_ptr.get();
            auto * buft = ggml_backend_sched_get_buffer_type(lctx.sched.get(), backend);
            if (this->graph) {
                ret[buft].compute += ggml_backend_sched_get_buffer_size(this->graph->sched.get(), backend);
            }
            for (const auto & entry : this->graph_cache) {
                if (entry) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->sched.get(), backend);
                }
            }
            for (const auto & entry : this->targets) {
                if (entry && entry->sched) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->sched.get(), backend);
                }
                if (entry && entry->meta_sched) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->meta_sched.get(), backend);
                }
            }
        }
    }
    if (this->handoff) {
        ret[ggml_backend_buffer_get_type(this->handoff->device.get())].compute += ggml_backend_buffer_get_size(this->handoff->device.get());
        ret[ggml_backend_buffer_get_type(this->handoff->host.get())].compute += ggml_backend_buffer_get_size(this->handoff->host.get());
    }
    if (this->lookahead) {
        const auto & sampled = this->lookahead->sampled;
        ret[ggml_backend_buffer_get_type(sampled.device.get())].compute += ggml_backend_buffer_get_size(sampled.device.get());
        ret[ggml_backend_buffer_get_type(sampled.host.get())].compute += ggml_backend_buffer_get_size(sampled.host.get());
    }
    if (this->verify) {
        ret[ggml_backend_buffer_get_type(this->verify->device.get())].compute += ggml_backend_buffer_get_size(this->verify->device.get());
    }
    auto add_buffer = [&](const ggml_backend_buffer_ptr & buffer) {
        if (buffer) {
            ret[ggml_backend_buffer_get_type(buffer.get())].compute += ggml_backend_buffer_get_size(buffer.get());
        }
    };
    if (this->features) {
        add_buffer(this->features->device);
        add_buffer(this->features->host);
    }
    auto add_early = [&](const std::unique_ptr<llama_nextn_graph> & entry) {
        if (entry) {
            add_buffer(entry->early_device);
            add_buffer(entry->early_host);
        }
    };
    add_early(this->graph);
    for (const auto & entry : this->graph_cache) {
        add_early(entry);
    }
    for (const auto & entry : this->targets) {
        if (entry) {
            add_buffer(entry->input.device);
        }
    }
}
