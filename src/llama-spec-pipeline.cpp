#include "llama-spec-pipeline.h"

#include "llama-context.h"
#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "llama-sampler.h"
#include "llama-spec-tree.h"

#include <algorithm>
#include <cstring>

struct llama_nextn_control {
    enum field {
        POSITION, ACCEPTED, KEPT, EXECUTED, PREVIOUS_KEPT, EPOCH, STEP, N_FIELDS,
    };

    // Keep captured addresses outside the graph allocator's scratch buffers.
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device;
    ggml_backend_buffer_ptr host;
    ggml_backend_event_ptr consumed;
    ggml_backend_event_ptr ready;
    ggml_tensor * storage = nullptr;
    bool snapshot_pending = false;
    std::array<ggml_tensor *, N_FIELDS> fields = {};
    std::array<ggml_tensor *, 9> tokens = {};

    ~llama_nextn_control() {
        if (snapshot_pending) {
            ggml_backend_event_synchronize(ready.get());
        }
    }

    bool init(ggml_backend_t backend) {
        auto * dev = ggml_backend_get_device(backend);
        auto * host_type = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!host_type) {
            return false;
        }
        ctx.reset(ggml_init({ (1 + N_FIELDS + tokens.size())*ggml_tensor_overhead(), nullptr, true }));
        storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, N_FIELDS + tokens.size());
        for (int i = 0; i < N_FIELDS; ++i) {
            fields[i] = ggml_view_1d(ctx.get(), storage, 1, i*sizeof(int32_t));
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = ggml_view_1d(ctx.get(), storage, i + 1, N_FIELDS*sizeof(int32_t));
        }
        device.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        host.reset(ggml_backend_buft_alloc_buffer(host_type, ggml_nbytes(storage)));
        consumed.reset(ggml_backend_event_new(dev));
        ready.reset(ggml_backend_event_new(dev));
        return device && host && consumed && ready;
    }

    const int32_t * snapshot() const {
        ggml_backend_event_synchronize(ready.get());
        return (const int32_t *) ggml_backend_buffer_get_base(host.get());
    }
};

struct llama_nextn_graph {
    llama_token seed_token = LLAMA_TOKEN_NULL;
    llama_pos seed_pos = -1;
    llama_seq_id seed_seq = -1;
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr res;
    ggml_backend_event_ptr done;
    ggml_context_ptr constants_ctx;
    ggml_backend_buffer_ptr constants_device;
    std::vector<llm_graph_params> params;
    std::vector<size_t> input_ends;
    std::vector<ggml_tensor *> tokens;
    std::vector<ggml_tensor *> hidden;
    std::vector<ggml_tensor *> candidates;
    std::vector<ggml_tensor *> logits;
    std::vector<ggml_tensor *> tree_scores;
    bool composed = false;
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
    ggml_tensor * inp_bonus = nullptr;
    ggml_tensor * inp_weights = nullptr;
    ggml_tensor * inp_positions = nullptr;
    ggml_tensor * inp_position_mask = nullptr;
    ggml_tensor * inp_out_ids = nullptr;
    ggml_tensor * inp_kv_positions = nullptr;
    ggml_tensor * reject_mask = nullptr;
    ggml_tensor * step = nullptr;
    ggml_tensor * kept = nullptr;
    std::vector<ggml_tensor *> positions;
    std::vector<ggml_tensor *> kv_positions;
    std::vector<ggml_tensor *> kv_masks;
    std::shared_ptr<llama_nextn_control> input_owner;
    const llama_nextn_control * input = nullptr;
    ggml_tensor * input_sampled = nullptr;
    ggml_tensor * input_hidden = nullptr;
};

struct llama_nextn_handoff {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device;
    ggml_backend_buffer_ptr host;
    ggml_backend_event_ptr consumed;
    ggml_backend_event_ptr snapshot_ready;
    std::vector<ggml_tensor *> rows;
};

struct llama_nextn_lookahead {
    llama_nextn_handoff sampled;
    llama_nextn_control input;
    std::shared_ptr<llama_nextn_control> next;
    ggml_backend_event_ptr ready;
    std::vector<llama_token> draft;
    llama_pos pos = -1;
    llama_seq_id seq = -1;
    int32_t epoch = 0;
    int32_t step = 0;
};

struct llama_nextn_catchup {
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr res;
    ggml_context_ptr meta_ctx;
    ggml_backend_sched_ptr meta_sched;
    ggml_cgraph * meta_gf = nullptr;
    ggml_context_ptr output_ctx;
    ggml_backend_buffer_ptr output_device;
    ggml_tensor * hidden = nullptr;
    llama_memory_context_ptr mctx;
    std::unique_ptr<llm_graph_params> controls;
    bool consumed = false;
    bool handoff_deferred = false;
};

struct llama_nextn_draft_body {
    std::shared_ptr<llama_nextn_graph> graph;
    ggml_context_ptr input_ctx;
    ggml_backend_buffer_ptr input_device;
    ggml_tensor * sampled = nullptr;
    ggml_tensor * hidden = nullptr;
    std::vector<std::unique_ptr<llama_kv_cache_context>> contexts;
};

struct llama_nextn_draft {
    std::shared_ptr<llama_nextn_draft_body> body;
    ggml_context_ptr meta_ctx;
    ggml_backend_sched_ptr meta_sched;
    ggml_cgraph * meta_gf = nullptr;
    bool consumed = false;
};

struct llama_nextn_target {
    std::unique_ptr<llama_spec_tree> tree;
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr res;
    ggml_context_ptr constants_ctx;
    ggml_backend_buffer_ptr constants_device;
    ggml_tensor * out_ids = nullptr;
    ggml_context_ptr meta_ctx;
    ggml_backend_sched_ptr meta_sched;
    ggml_cgraph * meta_gf = nullptr;
    std::shared_ptr<llama_nextn_control> input;
    ggml_backend_event_ptr ready;
    ggml_backend_event_ptr done;
    llm_graph_nextn_target meta;
    std::map<llama_seq_id, llama_sampler *> samplers;
    std::unique_ptr<llm_graph_params> controls;
    std::unique_ptr<llm_graph_params> build_controls;
    llama_memory_context_ptr graph_mctx;
    std::shared_ptr<llama_nextn_handoff> catchup_seed;
    std::unique_ptr<llama_nextn_catchup> catchup;
    std::unique_ptr<llama_nextn_draft> draft;
    const llama_context * source = nullptr;
    uint64_t generation = 0;
    llama_pos previous_pos = -1;
    int previous_n = 0;
    int n_kv = 0;
    int n = 0;
    bool composed = false;

    ~llama_nextn_target() {
        for (auto & entry : samplers) {
            llama_sampler_free(entry.second);
        }
    }
};

llama_spec_pipeline::llama_spec_pipeline(llama_context & context) : lctx(context) {}

llama_spec_pipeline::~llama_spec_pipeline() {
    if (handoff && handoff->snapshot_ready) {
        ggml_backend_event_synchronize(handoff->snapshot_ready.get());
    }
    if (copy_backend) {
        ggml_backend_synchronize(copy_backend.get());
    }
}

bool llama_spec_pipeline::prepare_nextn_draft(llama_context & source, llama_nextn_target & target, int n, bool target_rebuilt,
        std::shared_ptr<llama_nextn_draft_body> & shared) {
    if (!target.catchup || !source.spec->graph || !source.spec->lookahead) {
        return false;
    }
    auto * kv = static_cast<llama_kv_cache *>(source.memory.get());
    const int n_draft = source.spec->graph->tokens.size();
    const float p_min = source.spec->graph->p_min;
    const auto & previous = this->mctx->get_ubatch();
    if (previous.pos[0] + previous.n_tokens + n + n_draft > kv->get_size() ||
            n_draft > (int) source.cparams.n_outputs_max || target.res->t_sampled.size() != (size_t) n) {
        return false;
    }
    if (target.draft && ((int) target.draft->body->graph->tokens.size() != n_draft ||
            target.draft->body->graph->p_min != p_min)) {
        target.draft.reset();
    }
    auto * backend = lctx.backend_ptrs[0];
    if (!target.draft) {
        target.draft = std::make_unique<llama_nextn_draft>();
        target.draft->meta_sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                128, false, source.cparams.op_offload));
    }
    auto & state = *target.draft;
    if (shared && state.body != shared) {
        state.body = shared;
        state.meta_gf = nullptr;
    }
    if (!state.body) {
        state.body = std::make_shared<llama_nextn_draft_body>();
        auto & body = *state.body;
        const size_t max_nodes = source.graph_max_nodes(1)*n_draft;
        body.graph = std::make_shared<llama_nextn_graph>();
        body.graph->composed = true;
        body.graph->input_owner = target.input;
        body.graph->res.reset(new llm_graph_result(max_nodes));
        body.graph->sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                max_nodes, false, source.cparams.op_offload));
        body.input_ctx.reset(ggml_init({ 2*ggml_tensor_overhead(), nullptr, true }));
        body.hidden = ggml_new_tensor_2d(body.input_ctx.get(), GGML_TYPE_F32, source.model.hparams.n_embd_out(), n_draft + 1);
        body.sampled = ggml_new_tensor_1d(body.input_ctx.get(), GGML_TYPE_I32, n_draft + 1);
        body.input_device.reset(ggml_backend_alloc_ctx_tensors(body.input_ctx.get(), backend));
        if (!body.input_device) {
            return false;
        }
    }
    auto & body = *state.body;
    if (target_rebuilt || !state.meta_gf || body.graph->input_owner != target.input) {
        body.graph->input_owner = target.input;
        ggml_backend_sched_reset(state.meta_sched.get());
        state.meta_ctx.reset(ggml_init({ 128*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true }));
        auto * ctx = state.meta_ctx.get();
        state.meta_gf = ggml_new_graph_custom(ctx, 128, false);
        auto alias = [&](ggml_tensor * tensor) {
            auto * leaf = ggml_dup_tensor(ctx, tensor);
            leaf->data = tensor->data;
            leaf->buffer = tensor->buffer;
            return leaf;
        };
        auto * rows = ggml_view_2d(ctx, body.hidden, body.hidden->ne[0], n, body.hidden->nb[1], 0);
        auto * hidden = ggml_cpy(ctx, alias(target.catchup->hidden), rows);
        ggml_set_name(hidden, "nextn_draft_hidden");
        ggml_build_forward_expand(state.meta_gf, hidden);
        for (int i = 0; i < n; ++i) {
            auto * sampled = target.res->t_sampled[i];
            if (!sampled || sampled->type != GGML_TYPE_I32 || ggml_nelements(sampled) != 1 ||
                    ggml_backend_sched_get_tensor_backend(target.sched.get(), sampled) != backend) {
                return false;
            }
            auto * copy = ggml_cpy(ctx, alias(sampled), ggml_view_1d(ctx, body.sampled, 1, i*sizeof(llama_token)));
            ggml_set_name(copy, "nextn_draft_sampled");
            ggml_build_forward_expand(state.meta_gf, copy);
        }
        if (!ggml_backend_sched_alloc_graph_after(state.meta_sched.get(), state.meta_gf, nullptr)) {
            return false;
        }
    }
    if (body.contexts.empty()) {
        const int dim = source.model.hparams.n_embd_out();
        llama_batch_allocr balloc(source.model.hparams.n_pos_per_embd());
        auto batch = balloc.ubatch_reserve(n_draft, 1);
        batch.data->embd.resize((size_t) dim*n_draft);
        batch.embd = batch.data->embd.data();
        for (int i = 0; i < n_draft; ++i) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = batch.seq_id_unq;
            batch.output[i] = 1;
        }
        llama_kv_cache_context full(kv);
        auto mctx = full.for_graph(batch, kv->get_size());
        for (int i = 0; i < n_draft; ++i) {
            body.contexts.push_back(mctx->for_token(i, dim));
        }
    }
    if (!shared && !source.spec->prepare_draft_graph(*body.graph, body.contexts, n_draft, true, p_min, n_draft + 1, true,
            target.input.get(), body.sampled, body.hidden, -1)) {
        return false;
    }
    shared = state.body;
    state.consumed = false;
    return true;
}

bool llama_spec_pipeline::prepare_nextn_catchup(llama_context & source, llama_nextn_target & target, int n, bool target_rebuilt) {
    const bool eagle = source.model.arch == LLM_ARCH_EAGLE3;
    const auto & target_meta = target.tree ? target.tree->linear : target.meta;
    auto * kv = dynamic_cast<llama_kv_cache *>(source.memory.get());
    if (!kv || !kv->is_fixed_size() || kv->get_size() != (uint32_t) target.n_kv || kv->get_n_stream() != 1 || kv->get_has_shift() ||
            !source.cparams.flash_attn || !source.cparams.causal_attn || source.cparams.embeddings ||
            !source.cparams.embeddings_nextn || !source.cparams.embeddings_nextn_masked ||
            source.cparams.n_seq_max != 1 || source.cparams.pipeline_parallel || source.cparams.cb_eval || source.opt_ctx ||
            !source.loras->empty() || source.model.hparams.n_swa || source.model.hparams.use_alibi ||
            ggml_is_quantized(kv->type_k()) || ggml_is_quantized(kv->type_v()) ||
            source.sched_need_reserve || source.graph_reuse_disable || !source.spec->handoff ||
            n > (int) source.cparams.n_ubatch || n > (int) source.cparams.n_batch ||
            (!eagle && (source.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || source.model.hparams.n_layer_nextn != 1))) {
        return false;
    }
    const auto & previous = this->mctx->get_ubatch();
    const int end = previous.pos[0] + previous.n_tokens - (int) eagle;
    if (end < 0 || end + n > (int) kv->get_size()) {
        return false;
    }
    const auto & cells = kv->get_cells(0);
    for (int i = 0; i < end; ++i) {
        if (!cells.seq_has(i, 0) || cells.pos_get(i) != i) {
            return false;
        }
    }
    auto * backend = lctx.backend_ptrs[0];
    const int64_t dim = source.model.hparams.n_embd_out();
    std::vector<ggml_tensor *> hidden;
    if (eagle) {
        if (source.model.target_layer_ids.size() != 3 || !source.spec->features) {
            return false;
        }
        for (int id : source.model.target_layer_ids) {
            if (id >= 0 && id < (int) lctx.model.hparams.n_layer() && lctx.cparams.embeddings_layer_inp[id]) {
                hidden.push_back(target.res->t_layer_inp[id]);
            } else if (id == (int) lctx.model.hparams.n_layer() && lctx.cparams.embeddings_nextn) {
                hidden.push_back(target.res->get_h_nextn());
            } else {
                return false;
            }
        }
    } else {
        hidden.push_back(target.res->get_h_nextn());
    }
    for (auto * tensor : hidden) {
        if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != (eagle ? (int64_t) lctx.model.hparams.n_embd : dim) ||
                tensor->ne[1] != n || !ggml_is_contiguous(tensor) ||
                ggml_backend_sched_get_tensor_backend(target.sched.get(), tensor) != backend) {
            return false;
        }
    }
    if (!target.catchup) {
        target.catchup = std::make_unique<llama_nextn_catchup>();
        auto & state = *target.catchup;
        const size_t max_nodes = source.graph_max_nodes(n);
        state.res.reset(new llm_graph_result(max_nodes));
        state.sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                max_nodes, false, source.cparams.op_offload));
        state.meta_sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                128, false, source.cparams.op_offload));
    }
    auto & state = *target.catchup;
    const int n_decode = eagle && n > 1 ? n - 1 : n;
    llama_batch_allocr balloc(source.model.hparams.n_pos_per_embd());
    auto batch = balloc.ubatch_reserve(n_decode, 1);
    for (int i = 0; i < n_decode; ++i) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = batch.seq_id_unq;
    }
    if (!eagle || n == 1) {
        batch.data->embd.resize((eagle ? source.model.hparams.n_embd_inp_enc() : dim)*n_decode);
        batch.embd = batch.data->embd.data();
        if (eagle) {
            batch.token = nullptr;
        }
    }
    llama_kv_cache_context full(kv);
    auto mctx = full.for_graph(batch, kv->get_size());
    const auto type = eagle && n == 1 ? LLM_GRAPH_TYPE_ENCODER : source.ctx_type_to_graph_type(source.cparams.ctx_type);
    auto controls = source.graph_params(state.res.get(), batch, mctx.get(), type);
    controls.sched = state.sched.get();
    controls.n_outputs = 0;
    controls.samplers.clear();
    controls.cb = [](const llama_ubatch &, ggml_tensor * tensor, const char * name, int il) {
        ggml_format_name(tensor, "%s-%d", name, il);
    };
    if (target_rebuilt || !state.controls || !state.controls->allow_reuse(controls)) {
        ggml_backend_sched_reset(state.meta_sched.get());
        ggml_backend_sched_reset(state.sched.get());
        state.res->reset();
        state.meta_ctx.reset(ggml_init({ 128*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true }));
        auto * ctx = state.meta_ctx.get();
        state.meta_gf = ggml_new_graph_custom(ctx, 128, false);
        auto params = controls;
        params.nextn_out_ids = ggml_view_1d(ctx, target.input->tokens[n - 1], 0, 0);
        auto alias = [&](ggml_tensor * tensor) {
            auto * leaf = ggml_dup_tensor(ctx, tensor);
            leaf->data = tensor->data;
            leaf->buffer = tensor->buffer;
            return leaf;
        };
        if (eagle) {
            if (!state.output_device) {
                state.output_ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
                state.hidden = ggml_new_tensor_2d(state.output_ctx.get(), GGML_TYPE_F32, dim, n);
                state.output_device.reset(ggml_backend_alloc_ctx_tensors(state.output_ctx.get(), backend));
                if (!state.output_device) {
                    return false;
                }
            }
            for (int i = 0; i < 3; ++i) {
                params.nextn_features[i] = alias(hidden[i]);
            }
            params.nextn_hidden = state.hidden;
        } else {
            params.nextn_hidden = alias(hidden[0]);
            state.hidden = params.nextn_hidden;
        }
        if (!source.model.build_graph(params)) {
            return false;
        }
        if (!eagle || n > 1) {
            auto * positions = ggml_cast(ctx, ggml_view_1d(ctx, alias(target_meta.kv_idxs), n_decode, 0), GGML_TYPE_F32);
            auto * rope = eagle ? ggml_view_1d(ctx, alias(target_meta.positions), n_decode, 0) : alias(target_meta.positions);
            auto * tokens = ggml_view_1d(ctx, target.input->tokens[n - 1], n_decode, eagle ? sizeof(llama_token) : 0);
            auto * target_mask = alias(target_meta.mask);
            auto * mask = ggml_view_2d(ctx, target_mask, kv->get_size(), n_decode, target_mask->nb[1], 0);
            auto copy = [&](ggml_tensor * src, ggml_tensor * dst) {
                ggml_backend_sched_set_tensor_backend(state.sched.get(), dst, backend);
                auto * cur = ggml_cpy(ctx, src, dst);
                ggml_set_name(cur, "nextn_catchup_metadata");
                ggml_build_forward_expand(state.meta_gf, cur);
            };
            auto & inputs = state.res->inputs;
            inputs.erase(std::remove_if(inputs.begin(), inputs.end(), [&](const llm_graph_input_ptr & input) {
                if (auto * inp = dynamic_cast<llm_graph_input_embd_h *>(input.get())) {
                    copy(tokens, inp->tokens);
                    ggml_backend_sched_set_tensor_backend(state.sched.get(), inp->h, backend);
                    copy(target.catchup_seed->rows[0], ggml_view_2d(ctx, inp->h, dim, 1, inp->h->nb[1], 0));
                } else if (auto * inp = dynamic_cast<llm_graph_input_embd *>(input.get())) {
                    copy(tokens, inp->tokens);
                } else if (auto * inp = dynamic_cast<llm_graph_input_pos *>(input.get())) {
                    copy(rope, inp->pos);
                } else if (auto * inp = dynamic_cast<llm_graph_input_attn_kv *>(input.get())) {
                    copy(positions, inp->self_k_idxs);
                    copy(positions, inp->self_v_idxs);
                    copy(mask, inp->self_kq_mask);
                } else {
                    return false;
                }
                return true;
            }), inputs.end());
        }
        if (!ggml_backend_sched_alloc_graph_after(state.sched.get(), state.res->get_gf(), nullptr)) {
            return false;
        }
        if (ggml_graph_n_nodes(state.meta_gf) && !ggml_backend_sched_alloc_graph_after(state.meta_sched.get(), state.meta_gf, nullptr)) {
            return false;
        }
        for (const auto & stage : { std::make_pair(state.sched.get(), state.res->get_gf()), std::make_pair(state.meta_sched.get(), state.meta_gf) }) {
            for (int i = 0; i < ggml_graph_n_nodes(stage.second); ++i) {
                auto * node = ggml_graph_node(stage.second, i);
                if (!ggml_is_empty(node) && ggml_backend_sched_get_tensor_backend(stage.first, node) != backend) {
                    LLAMA_LOG_DEBUG("%s: unsupported catch-up node %s\n", __func__, ggml_get_name(node));
                    return false;
                }
            }
        }
        state.res->set_inputs(&batch);
        state.controls = std::make_unique<llm_graph_params>(controls);
        state.mctx = std::move(mctx);
        LLAMA_LOG_DEBUG("%s: prepared GPU catch-up (%d rows)\n", __func__, n);
    }
    state.consumed = false;
    state.handoff_deferred = false;
    return true;
}

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
    auto * producer = ggml_backend_sched_get_tensor_backend(chain.sched.get(), chain.logits[0]);
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
    const bool can_compose = ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_graph_select") != nullptr;
    const bool tree = tree_enabled && eagle && !early && n_max == 4 && chain.tree_scores.size() == 3 &&
        kv->is_fixed_size() && (uint32_t) n_kv == kv->get_size() && can_compose;
    const int bank = 1 - this->target_bank;
    const int n_begin = early ? 1 : n_max;
    auto * retired = this->targets[bank*10 + n_begin].get();
    if (this->target_copied[bank]) {
        ggml_backend_event_synchronize(this->target_copied[bank].get());
    }
    if (retired && retired->done) {
        ggml_backend_event_synchronize(retired->done.get());
    }
    std::vector<ggml_backend_sched_t> stages;
    std::shared_ptr<llama_nextn_draft_body> draft_body;
    bool has_catchup = false;
    for (int n = n_begin; n <= n_max; ++n) {
        const int index = bank*10 + n;
        if (!this->targets[index]) {
            this->targets[index] = std::make_unique<llama_nextn_target>();
        }
        auto & state = *this->targets[index];
        if ((bool) state.tree != tree) {
            state.build_controls.reset();
            state.tree.reset();
            if (tree) {
                state.tree = std::make_unique<llama_spec_tree>();
                if (!state.tree->init(backend)) {
                    state.tree.reset();
                    return false;
                }
            }
        }
        auto & shared_input = this->targets[bank*10 + n_begin]->input;
        if (!shared_input) {
            auto input = std::make_shared<llama_nextn_control>();
            if (!input->init(backend)) {
                return false;
            }
            shared_input = std::move(input);
        }
        if (state.input != shared_input) {
            state.build_controls.reset();
            state.input = shared_input;
        }
        auto & shared_seed = this->targets[bank*10 + n_begin]->catchup_seed;
        if (!eagle && !shared_seed) {
            shared_seed = std::make_shared<llama_nextn_handoff>();
            shared_seed->ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
            shared_seed->rows.push_back(ggml_new_tensor_2d(shared_seed->ctx.get(), GGML_TYPE_F32, source.model.hparams.n_embd_out(), 1));
            shared_seed->device.reset(ggml_backend_alloc_ctx_tensors(shared_seed->ctx.get(), backend));
            if (!shared_seed->device) {
                return false;
            }
        }
        state.catchup_seed = shared_seed;
        auto & input = *state.input;
        if (!state.ready) {
            state.ready.reset(ggml_backend_event_new(dev));
            if (!state.ready) {
                return false;
            }
        }
        if (!state.res) {
            const size_t max_nodes = lctx.graph_max_nodes(n);
            state.res.reset(new llm_graph_result(max_nodes));
            state.sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                    max_nodes, false, lctx.cparams.op_offload));
        }
        if (!state.out_ids) {
            state.constants_ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
            state.out_ids = ggml_new_tensor_1d(state.constants_ctx.get(), GGML_TYPE_I32, n);
            state.constants_device.reset(ggml_backend_alloc_ctx_tensors(state.constants_ctx.get(), backend));
            if (!state.constants_device) {
                state.out_ids = nullptr;
                return false;
            }
            // Every row is an output in this width's verification graph.
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) {
                ids[i] = i;
            }
            ggml_backend_tensor_set(state.out_ids, ids.data(), 0, n*sizeof(int32_t));
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
        params.nextn_out_ids = state.out_ids;
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
                state.meta_gf = ggml_new_graph_custom(state.meta_ctx.get(), 128, false);
                if (!tree) {
                    ctx = state.meta_ctx.get();
                }
                if (!state.meta_sched) {
                    state.meta_sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(),
                            128, false, lctx.cparams.op_offload));
                }
            }
            auto * base = ggml_cast(ctx, input.fields[llama_nextn_control::POSITION], GGML_TYPE_F32);
            if (eagle) {
                base = ggml_scale_bias(ctx, base, 1.0f, 1.0f);
            }
            auto * accepted = ggml_cast(ctx, input.fields[llama_nextn_control::ACCEPTED], GGML_TYPE_F32);
            auto * positions = ggml_add(ctx, ggml_arange(ctx, 0, n, 1), base);
            state.meta.tokens = input.tokens[n - 1];
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
            ggml_set_output(state.meta.positions);
            ggml_set_output(state.meta.kv_idxs);
            ggml_set_output(state.meta.mask);
            if (rs) {
                auto * previous_count = ggml_cast(ctx, input.fields[llama_nextn_control::PREVIOUS_KEPT], GGML_TYPE_F32);
                state.meta.rs_copy = ggml_cast(ctx, ggml_sub(ctx, previous_count, accepted), GGML_TYPE_I32);
            }
            auto build_params = params;
            if (tree) {
                state.tree->build_inputs(*res, state.meta, n_kv);
            } else if (eagle) {
                // Preserve the ordinary target graph's allocation and CUDA fusion choices.
                build_params.nextn_target = nullptr;
            }
            if (!lctx.model.build_graph(build_params)) {
                state.controls.reset();
                return false;
            }
            if (tree) {
                for (int i = 0; i < ggml_graph_n_nodes(res->get_gf()); ++i) {
                    auto * node = ggml_graph_node(res->get_gf(), i);
                    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                        ggml_flash_attn_ext_set_kv_indices(node, state.meta.kv_idxs);
                    }
                }
                if (!state.tree->build_outputs(*res, *kv, input.tokens[3], lctx.model.vocab.n_tokens())) {
                    return false;
                }
            } else if (eagle) {
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
            if (eagle && !tree) {
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
                    LLAMA_LOG_DEBUG("%s: unsupported node %d %s (%s, %s)\n", __func__, i, ggml_get_name(node),
                            ggml_op_name(node->op), ggml_type_name(node->type));
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
        if (!can_compose || !prepare_nextn_catchup(source, state, n, !reuse)) {
            state.catchup.reset();
        }
        if (!prepare_nextn_draft(source, state, n, !reuse, draft_body)) {
            state.draft.reset();
        }
        if (tree && (!state.catchup || !state.draft)) {
            return false;
        }
        has_catchup |= (bool) state.catchup;
        state.graph_mctx = std::move(graph_mctx);
        state.source = &source;
        state.generation = chain.generation;
        state.previous_pos = ubatch.pos[0];
        state.previous_n = previous_n;
        state.n = n;
        state.controls = std::make_unique<llm_graph_params>(lctx.graph_params(lctx.gf_res_prev.get(), ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT));
        stages.push_back(eagle && !tree ? state.meta_sched.get() : nullptr);
        stages.push_back(state.sched.get());
        stages.push_back(state.catchup && ggml_graph_n_nodes(state.catchup->meta_gf) ? state.catchup->meta_sched.get() : nullptr);
        stages.push_back(state.catchup ? state.catchup->sched.get() : nullptr);
        stages.push_back(state.draft ? state.draft->meta_sched.get() : nullptr);
        stages.push_back(state.draft ? state.draft->body->graph->sched.get() : nullptr);
    }
    auto & first = *this->targets[bank*10 + n_begin];
    auto & input = *first.input;
    const bool composed = early || first.catchup || first.composed;
    if (composed && !ggml_backend_sched_graph_select(first.sched.get(), stages.data(), n_max - n_begin + 1, 6,
            early ? input.fields[llama_nextn_control::KEPT] : nullptr)) {
        return false;
    }
    first.composed = composed;
    ggml_tensor position = *chain.positions[0];
    position.ne[0] = 1;
    ggml_tensor count = early ? *chain.early_counts : *chain.kept;
    count.ne[0] = 1;
    ggml_tensor executed = count;
    if (early) {
        executed.data = (int32_t *) executed.data + 1;
    }
    const auto & previous = *chain.input;
    std::vector<const ggml_tensor *> tensors = { &position, chain.accepted, &count, &executed,
        previous.fields[llama_nextn_control::KEPT], previous.fields[llama_nextn_control::EPOCH], chain.step, chain.selected_token };
    tensors.insert(tensors.end(), chain.tokens.begin(), chain.tokens.end());
    std::vector<void *> destinations;
    for (size_t i = 0; i < tensors.size(); ++i) {
        destinations.push_back((int32_t *) input.storage->data + i);
    }
    const size_t n_control = tensors.size();
    std::array<ggml_tensor, 4> tree_candidates;
    std::array<ggml_tensor, 4> tree_probabilities;
    if (tree) {
        tensors.push_back(chain.selected_token);
        destinations.push_back(first.tree->tokens->data);
        for (int i = 0; i < 3; ++i) {
            tree_candidates[i] = *chain.candidates[i];
            tree_candidates[i].ne[0] = 2;
            tensors.push_back(&tree_candidates[i]);
            destinations.push_back((int32_t *) first.tree->tokens->data + 1 + 2*i);
            tree_probabilities[i] = *chain.tree_scores[i];
            tree_probabilities[i].ne[0] = 2;
            tensors.push_back(&tree_probabilities[i]);
            destinations.push_back((float *) first.tree->scores->data + 1 + 2*i);
        }
        tree_candidates[3] = tree_candidates[0];
        tree_candidates[3].ne[0] = 1;
        tree_candidates[3].data = (int32_t *) tree_candidates[3].data + 2;
        tree_probabilities[3] = tree_probabilities[0];
        tree_probabilities[3].ne[0] = 1;
        tree_probabilities[3].data = (float *) tree_probabilities[3].data + 2;
        tensors.push_back(&tree_candidates[3]);
        destinations.push_back((int32_t *) first.tree->tokens->data + 7);
        tensors.push_back(&tree_probabilities[3]);
        destinations.push_back((float *) first.tree->scores->data + 7);
    }
    if (first.catchup_seed) {
        tensors.push_back(chain.selected_hidden);
        destinations.push_back(first.catchup_seed->rows[0]->data);
    }
    if (producer != backend) {
        ggml_backend_event_record(input.consumed.get(), backend);
        ggml_backend_event_wait(producer, input.consumed.get());
    }
    // Keep scalar control packing and tree input copies on the small device-copy paths.
    const size_t n_first = tree ? n_control : tensors.size();
    if (!get_batch(producer, tensors.data(), destinations.data(), n_first) ||
            (tree && !get_batch(producer, tensors.data() + n_first, destinations.data() + n_first, tensors.size() - n_first))) {
        return false;
    }
    auto same_backend = [](const std::unique_ptr<llama_output_copies> & copies, ggml_backend_t current) {
        return !copies || (copies->data.size() == 1 && copies->data.begin()->first == current);
    };
    // Only alternating graph banks keep their outputs alive during the next target.
    const bool can_overlap = chain.composed && this->target_active >= 0 && producer == backend &&
        same_backend(this->readback, backend) && same_backend(source.spec->readback, producer);
    if (can_overlap && !this->copy_backend) {
        this->copy_backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (this->copy_backend) {
            LLAMA_LOG_DEBUG("%s: GPU result readback uses a separate stream\n", __func__);
        }
    }
    auto * transfer = can_overlap && this->copy_backend ? this->copy_backend.get() : nullptr;
    if (producer != backend || transfer) {
        ggml_backend_event_record(first.ready.get(), producer);
    }
    if (transfer) {
        ggml_backend_event_wait(transfer, first.ready.get());
    }
    ggml_backend_tensor_get_async(transfer ? transfer : producer, input.storage,
            ggml_backend_buffer_get_base(input.host.get()), 0, ggml_nbytes(input.storage));
    ggml_backend_event_record(input.ready.get(), transfer ? transfer : producer);
    input.snapshot_pending = true;
    this->readback->submit(transfer);
    this->readback.reset();
    ggml_backend_event_record(this->output_ready.get(), transfer ? transfer : backend);
    this->output_pending = true;
    if ((early || first.catchup) && source.spec->readback) {
        source.spec->readback->submit(transfer);
        source.spec->readback.reset();
        if (!source.spec->output_ready) {
            source.spec->output_ready.reset(ggml_backend_event_new(dev));
        }
        ggml_backend_event_record(source.spec->output_ready.get(), transfer ? transfer : producer);
        source.spec->output_pending = true;
    }
    if (transfer) {
        auto & copied = this->target_copied[this->target_bank];
        if (!copied) {
            copied.reset(ggml_backend_event_new(dev));
        }
        ggml_backend_event_record(copied.get(), transfer);
    }
    if (producer != backend) {
        ggml_backend_event_wait(backend, first.ready.get());
    }
    if (!composed && eagle && ggml_backend_sched_graph_compute_async(first.meta_sched.get(), first.meta_gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (ggml_backend_sched_graph_compute_async(first.sched.get(), first.res->get_gf()) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!first.done) {
        first.done.reset(ggml_backend_event_new(dev));
    }
    ggml_backend_event_record(first.done.get(), backend);
    if (has_catchup) {
        ggml_backend_event_wait(source.backend_ptrs[0], first.done.get());
    }
    this->target_pending = bank*10 + n_begin;
    source.spec->lookahead->next = first.input;
    if (!this->prefetch_reported) {
        LLAMA_LOG_INFO("%s: GPU NextN pipeline active; target submitted before CPU acceptance\n", __func__);
        if (tree) {
            LLAMA_LOG_INFO("%s: experimental EAGLE3 tree active (4 target rows, GPU path selection and KV commit)\n", __func__);
        }
        this->prefetch_reported = true;
    }
    LLAMA_LOG_DEBUG("%s: queued GPU target before CPU acceptance (%d..%d rows)\n", __func__, n_begin, n_max);
    if (tree) {
        LLAMA_LOG_DEBUG("%s: EAGLE3 tree verification and accepted-path KV commit queued\n", __func__);
    }
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
            const auto * control = prepared.input->snapshot();
            const int accepted = control[llama_nextn_control::ACCEPTED];
            consume_target = control[llama_nextn_control::EPOCH] == lookahead->epoch &&
                control[llama_nextn_control::STEP] == lookahead->step + 1 &&
                control[llama_nextn_control::KEPT] + 1 == n &&
                control[llama_nextn_control::PREVIOUS_KEPT] + 1 == prepared.previous_n &&
                accepted >= 0 && accepted < prepared.previous_n &&
                batch.pos[0] == prepared.previous_pos + accepted + 1 &&
                batch.pos[0] == control[llama_nextn_control::POSITION] + offset &&
                batch.token[0] == control[llama_nextn_control::N_FIELDS];
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
    source.synchronize(true);
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
    auto * active = source.spec->target_active >= 0 ? source.spec->targets[source.spec->target_active].get() : nullptr;
    if (active && active->source == &lctx && active->catchup && !active->catchup->consumed &&
            active->n == n && this->graph && active->generation == this->graph->generation &&
            this->graph->seed_pos == batch.pos[0] && this->graph->seed_token == batch.token[0] &&
            !lctx.sched_need_reserve && !lctx.graph_reuse_disable && this->lookahead && active->input == this->lookahead->next) {
        auto & prepared = *active->catchup;
        auto controls = lctx.graph_params(prepared.res.get(), prepared.controls->ubatch, prepared.mctx.get(), prepared.controls->gtype);
        controls.n_outputs = 0;
        controls.samplers.clear();
        auto * kv = static_cast<llama_kv_cache *>(lctx.memory.get());
        bool match = prepared.controls->allow_reuse(controls) && !kv->get_has_shift() && kv->is_fixed_size() &&
            batch.seq_id[0][0] == 0 && lctx.memory->seq_pos_max(0) == batch.pos[0] - 1 + (int) eagle;
        const auto & cells = kv->get_cells(0);
        for (int i = 0; match && i < batch.pos[0] + (int) eagle; ++i) {
            match = cells.seq_has(i, 0) && cells.pos_get(i) == i;
        }
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
        llama_ubatch ubatch;
        llama_kv_cache::slot_info slots = {};
        if (match && decode_batch.n_tokens > 0) {
            match = lctx.balloc->init(decode_batch, lctx.model.vocab, lctx.memory.get(), dim, 1, false);
            if (match) {
                ubatch = lctx.balloc->split_simple(decode_batch.n_tokens);
                slots = kv->find_slot(ubatch, true);
                match = ubatch.n_tokens == (uint32_t) decode_batch.n_tokens && !slots.empty() &&
                    slots.is_contiguous() && slots.head() == (uint32_t) decode_batch.pos[0];
            }
        }
        if (match) {
            this->begin_decode();
            lctx.n_outputs = 0;
            if (decode_batch.n_tokens > 0) {
                kv->apply_ubatch(slots, ubatch);
            }
            if (active->draft) {
                prepared.handoff_deferred = true;
                if (!source.spec->copy_backend) {
                    source.spec->copy_backend.reset(ggml_backend_dev_init(dev, nullptr));
                }
            }
            auto * transfer = prepared.handoff_deferred ? source.spec->copy_backend.get() : nullptr;
            if (transfer) {
                // The composed draft already consumed these hidden rows on the target stream.
                ggml_backend_event_record(state.consumed.get(), src_backend);
                ggml_backend_event_wait(transfer, state.consumed.get());
                auto * tensor = eagle ? prepared.hidden : hidden[0];
                ggml_backend_tensor_get_async(transfer, tensor, ggml_backend_buffer_get_base(state.host.get()), 0, ggml_nbytes(tensor));
                if (!state.snapshot_ready) {
                    state.snapshot_ready.reset(ggml_backend_event_new(dev));
                }
                ggml_backend_event_record(state.snapshot_ready.get(), transfer);
                if (eagle) {
                    ggml_backend_event_record(this->features->consumed.get(), transfer);
                }
                auto & copied = source.spec->target_copied[source.spec->target_bank];
                if (!copied) {
                    copied.reset(ggml_backend_event_new(dev));
                }
                ggml_backend_event_record(copied.get(), transfer);
            } else {
                prepared.handoff_deferred = false;
                ggml_backend_event_record(state.consumed.get(), dst_backend);
                ggml_backend_event_wait(src_backend, state.consumed.get());
                ggml_backend_tensor_copy_async(src_backend, dst_backend, prepared.hidden, input);
                if (eagle) {
                    ggml_backend_tensor_get_async(dst_backend, input, ggml_backend_buffer_get_base(state.host.get()), 0, ggml_nbytes(input));
                    ggml_backend_event_record(this->features->consumed.get(), dst_backend);
                } else {
                    ggml_backend_tensor_get_async(src_backend, hidden[0], ggml_backend_buffer_get_base(state.host.get()), 0, ggml_nbytes(hidden[0]));
                }
            }
            prepared.consumed = true;
            *snapshot = (const float *) ggml_backend_buffer_get_base(state.host.get());
            LLAMA_LOG_DEBUG("%s: consuming GPU-prefetched catch-up (%d rows)\n", __func__, n);
            return true;
        }
    }
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
        if (!sampled.device || !sampled.host || !sampled.consumed || !state->ready || !state->input.init(dst_backend)) {
            return false;
        }
        this->lookahead = std::move(state);
    }
    auto & state = *this->lookahead;
    auto * prepared = source.spec->target_active >= 0 ? source.spec->targets[source.spec->target_active].get() : nullptr;
    if (prepared && prepared->input == state.next && prepared->catchup && prepared->catchup->consumed &&
            prepared->draft && !prepared->draft->consumed &&
            prepared->draft->body->graph->p_min == p_min && (int) prepared->draft->body->graph->tokens.size() == n_draft &&
            lctx.output_reserve(n_draft) >= (uint32_t) n_draft) {
        auto & next = *prepared->draft;
        ++state.step;
        state.next.reset();
        state.pos = batch.pos[0];
        state.seq = batch.seq_id[0][0];
        state.draft.assign(batch.token + 1, batch.token + n);
        state.draft.push_back(LLAMA_TOKEN_NULL);
        auto * transfer = source.spec->copy_backend.get();
        ggml_backend_event_record(state.sampled.consumed.get(), src_backend);
        if (transfer) {
            ggml_backend_event_wait(transfer, state.sampled.consumed.get());
        }
        ggml_backend_tensor_get_async(transfer ? transfer : src_backend, next.body->sampled,
                ggml_backend_buffer_get_base(state.sampled.host.get()), 0, n*sizeof(llama_token));
        ggml_backend_event_record(state.ready.get(), transfer ? transfer : src_backend);
        if (transfer) {
            auto & copied = source.spec->target_copied[source.spec->target_bank];
            if (!copied) {
                copied.reset(ggml_backend_event_new(dev));
            }
            ggml_backend_event_record(copied.get(), transfer);
        }
        if (this->graph && !this->graph->composed) {
            this->graph_cache[this->graph_active] = std::move(this->graph);
        } else {
            this->graph.reset();
        }
        this->graph_active = p_min > 0.0f ? n : 0;
        this->graph = next.body->graph;
        llama_token token = 0;
        llama_pos pos = batch.pos[0] + n;
        int32_t n_seq = 1;
        int8_t output = 1;
        llama_batch seed = { 1, &token, batch.embd, &pos, &n_seq, batch.seq_id, &output };
        collect_draft_outputs(*this->graph, seed, true);
        next.consumed = true;
        LLAMA_LOG_DEBUG("%s: GPU control feedback epoch=%d step=%d\n", __func__, state.epoch, state.step);
        LLAMA_LOG_DEBUG("%s: consuming GPU-composed acceptance and draft (%d verified rows)\n", __func__, n);
        source.prefetch_nextn_target(lctx);
        return true;
    }
    if (prepared && prepared->input == state.next && prepared->catchup && prepared->catchup->handoff_deferred) {
        ggml_backend_event_record(this->handoff->consumed.get(), dst_backend);
        ggml_backend_event_wait(src_backend, this->handoff->consumed.get());
        ggml_backend_tensor_copy_async(src_backend, dst_backend, prepared->catchup->hidden, this->handoff->rows[n - 1]);
        prepared->catchup->handoff_deferred = false;
    }
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
    const auto * active = source.spec->target_active >= 0 ? source.spec->targets[source.spec->target_active].get() : nullptr;
    if (active && active->input == state.next) {
        tensors.push_back(active->input->storage);
        destinations.push_back(state.input.storage->data);
        ++state.step;
        LLAMA_LOG_DEBUG("%s: GPU control feedback epoch=%d step=%d\n", __func__, state.epoch, state.step);
    } else {
        if (state.pos >= 0) {
            ggml_backend_event_synchronize(state.sampled.consumed.get());
        }
        state.epoch = state.epoch == INT32_MAX ? 1 : state.epoch + 1;
        state.step = 0;
        std::array<int32_t, llama_nextn_control::N_FIELDS + 9> initial = {};
        initial[llama_nextn_control::POSITION] = batch.pos[0];
        initial[llama_nextn_control::KEPT] = n - 1;
        initial[llama_nextn_control::EXECUTED] = n - 1;
        initial[llama_nextn_control::EPOCH] = state.epoch;
        std::copy_n(batch.token, n, initial.data() + llama_nextn_control::N_FIELDS);
        ggml_backend_tensor_set(state.input.storage, initial.data(), 0, sizeof(initial));
        LLAMA_LOG_DEBUG("%s: GPU control bootstrap epoch=%d pos=%d rows=%d\n", __func__, state.epoch, batch.pos[0], n);
    }
    state.next.reset();
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

bool llama_spec_pipeline::prepare_draft_graph(llama_nextn_graph & chain,
        const std::vector<std::unique_ptr<llama_kv_cache_context>> & contexts, int n_draft, bool prefetch,
        float p_min, int n_verify, bool gpu_kv, const llama_nextn_control * input,
        ggml_tensor * sampled, ggml_tensor * input_hidden, llama_pos verify_pos) {
    const bool eagle = lctx.model.arch == LLM_ARCH_EAGLE3;
    const auto seq_id = contexts[0]->get_ubatch().seq_id[0][0];
    const bool tree = tree_enabled && eagle && prefetch && n_draft == 3 && p_min == 0.0f && lctx.model.vocab.n_tokens() >= 10;
    auto * res = chain.res.get();
    auto params_for = [&](int i) {
        auto params = lctx.graph_params(res, contexts[i]->get_ubatch(), contexts[i].get(), lctx.ctx_type_to_graph_type(lctx.cparams.ctx_type));
        params.sched = chain.sched.get();
        params.n_outputs = 1;
        params.samplers.clear();
        params.nextn_tokens = i ? chain.tokens[i - 1] : prefetch ? chain.selected_token : nullptr;
        params.nextn_hidden = i ? chain.hidden[i - 1] : prefetch ? chain.selected_hidden : nullptr;
        params.nextn_positions = prefetch && !chain.positions.empty() ? chain.positions[i] : nullptr;
        params.nextn_kv_positions = gpu_kv && !chain.kv_positions.empty() ? chain.kv_positions[i] : nullptr;
        params.nextn_kq_mask = gpu_kv && !chain.kv_masks.empty() ? chain.kv_masks[i] : nullptr;
        params.nextn_out_ids = prefetch ? chain.inp_out_ids : nullptr;
        params.nextn_reject_mask = prefetch && lctx.model.arch != LLM_ARCH_EAGLE3 ? chain.reject_mask : nullptr;
        params.nextn_gpu_kv = gpu_kv;
        params.cb = [](const llama_ubatch &, ggml_tensor * tensor, const char * name, int il) {
            ggml_format_name(tensor, "%s-%d", name, il);
        };
        return params;
    };
    bool reuse = !lctx.graph_reuse_disable && (int) chain.params.size() == n_draft && chain.prefetch == prefetch &&
        chain.p_min == p_min && chain.n_verify == n_verify && chain.input == input &&
        chain.input_sampled == sampled && chain.input_hidden == input_hidden && tree == !chain.tree_scores.empty();
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
        chain.tree_scores.clear();
        chain.positions.clear();
        chain.kv_positions.clear();
        chain.kv_masks.clear();
        chain.prefetch = prefetch;
        chain.p_min = p_min;
        chain.n_verify = n_verify;
        chain.input = input;
        chain.input_sampled = sampled;
        chain.input_hidden = input_hidden;
        if (prefetch) {
            auto * ctx = res->get_ctx();
            const int n_pos = lctx.model.hparams.n_pos_per_embd();
            chain.constants_device.reset();
            chain.constants_ctx.reset(ggml_init({ 5*ggml_tensor_overhead(), nullptr, true }));
            auto constant = [&](ggml_type type, int64_t n) {
                auto * tensor = ggml_new_tensor_1d(chain.constants_ctx.get(), type, n);
                ggml_set_input(tensor);
                return tensor;
            };
            chain.inp_bonus = constant(GGML_TYPE_I32, 1);
            chain.inp_weights = constant(GGML_TYPE_F32, chain.composed ? 2*n_verify*n_verify : n_verify);
            chain.inp_positions = ggml_new_tensor_2d(chain.constants_ctx.get(), GGML_TYPE_F32, n_pos, n_draft);
            ggml_set_input(chain.inp_positions);
            chain.inp_position_mask = constant(GGML_TYPE_F32, n_pos);
            chain.inp_out_ids = eagle ? nullptr : constant(GGML_TYPE_I32, 1);
            chain.inp_kv_positions = gpu_kv ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, contexts[0]->get_n_kv());
            if (chain.inp_kv_positions) {
                ggml_set_input(chain.inp_kv_positions);
            }
            // Graph scratch storage can be reused after an input's last reader.
            chain.constants_device.reset(ggml_backend_alloc_ctx_tensors(chain.constants_ctx.get(), lctx.backend_ptrs[0]));
            if (!chain.constants_device) {
                return false;
            }
            const auto & control = *input;
            auto * draft = chain.inp_bonus;
            if (n_verify > 1) {
                auto * proposals = ggml_view_1d(ctx, control.tokens[n_verify - 1], n_verify - 1, sizeof(llama_token));
                draft = ggml_concat(ctx, proposals, chain.inp_bonus, 0);
            }
            auto * mismatch = ggml_step(ctx, ggml_abs(ctx, ggml_sub(ctx,
                    ggml_cast(ctx, sampled, GGML_TYPE_F32), ggml_cast(ctx, draft, GGML_TYPE_F32))));
            // Descending weights make the first mismatch unique, including the bonus row.
            if (chain.composed) {
                // Ignore unused rows and force a mismatch at the GPU-selected bonus row.
                auto * weights = ggml_get_rows(ctx, ggml_reshape_2d(ctx, chain.inp_weights, 2*n_verify, n_verify),
                        control.fields[llama_nextn_control::KEPT]);
                auto * prefix = ggml_view_1d(ctx, weights, n_verify, 0);
                auto * bonus = ggml_view_1d(ctx, weights, n_verify, n_verify*sizeof(float));
                chain.accepted = ggml_argmax(ctx, ggml_add(ctx, ggml_mul(ctx, mismatch, prefix), bonus));
            } else {
                chain.accepted = ggml_argmax(ctx, ggml_mul(ctx, mismatch, chain.inp_weights));
            }
            chain.selected_token = ggml_get_rows(ctx, ggml_reshape_2d(ctx, sampled, 1, n_verify), chain.accepted);
            ggml_set_output(chain.accepted);
            ggml_set_output(chain.selected_token);
            chain.selected_hidden = ggml_get_rows(ctx, input_hidden, chain.accepted);
            // Catch-up reads the seed after the draft chain has finished.
            ggml_set_output(chain.selected_hidden);
            auto * accepted_f32 = ggml_cast(ctx, chain.accepted, GGML_TYPE_F32);
            auto * base_pos = ggml_cast(ctx, control.fields[llama_nextn_control::POSITION], GGML_TYPE_F32);
            auto * base = ggml_add(ctx, base_pos, accepted_f32);
            auto * positions = ggml_cast(ctx, ggml_mul(ctx,
                    ggml_add(ctx, chain.inp_positions, base), chain.inp_position_mask), GGML_TYPE_I32);
            ggml_set_output(positions);
            for (int i = 0; i < n_draft; i++) {
                chain.positions.push_back(ggml_view_1d(ctx, positions, n_pos, i*n_pos*sizeof(int32_t)));
            }
            chain.reject_mask = chain.inp_kv_positions ? ggml_scale(ctx, ggml_step(ctx,
                    ggml_sub(ctx, chain.inp_kv_positions, accepted_f32)), -1e30f) : nullptr;
            if (gpu_kv && !eagle) {
                // Keep the original physical slots; only the RoPE positions follow acceptance.
                auto * count = ggml_cast(ctx, control.fields[llama_nextn_control::KEPT], GGML_TYPE_F32);
                auto * physical = ggml_add(ctx, ggml_arange(ctx, 1, n_draft + 1, 1), ggml_add(ctx, base_pos, count));
                for (int i = 0; i < n_draft; ++i) {
                    chain.kv_positions.push_back(ggml_view_1d(ctx, physical, 1, i*sizeof(float)));
                }
                auto * relative = ggml_sub(ctx, ggml_arange(ctx, 0, contexts[0]->get_n_kv(), 1), base_pos);
                auto * rejected = ggml_step(ctx, ggml_mul(ctx, ggml_sub(ctx, relative, accepted_f32),
                        ggml_scale_bias(ctx, ggml_sub(ctx, relative, count), -1.0f, 1.0f)));
                const int n_kv = contexts[0]->get_n_kv();
                auto * columns = ggml_repeat_4d(ctx, ggml_arange(ctx, 0, n_kv, 1), n_kv, n_draft, 1, 1);
                auto * causal = ggml_step(ctx, ggml_sub(ctx, columns, ggml_reshape_2d(ctx, physical, 1, n_draft)));
                auto * masks = ggml_cast(ctx, ggml_scale(ctx, ggml_add(ctx, causal, rejected), -1e30f), GGML_TYPE_F16);
                for (int i = 0; i < n_draft; ++i) {
                    chain.kv_masks.push_back(ggml_view_1d(ctx, masks, n_kv, i*masks->nb[1]));
                }
                LLAMA_LOG_DEBUG("%s: GPU draft KV indices and rejection mask (%d rows)\n", __func__, n_verify);
            }
            chain.step = ggml_cast(ctx, ggml_scale_bias(ctx,
                    ggml_cast(ctx, control.fields[llama_nextn_control::STEP], GGML_TYPE_F32), 1.0f, 1.0f), GGML_TYPE_I32);
            ggml_set_output(chain.step);
            ggml_build_forward_expand(res->get_gf(), chain.step);
        }
        for (int i = 0; i < n_draft; i++) {
            auto params = params_for(i);
            if (!lctx.model.build_graph(params) || !res->get_h_nextn() || !res->get_logits()) {
                return false;
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
            if (tree) {
                auto * candidates = ggml_view_1d(ctx, ids, i ? 2 : 3, 0);
                auto * probabilities = llama_spec_tree::probabilities(ctx, logits, candidates, llama_spec_tree::row(ctx, values, 0));
                if (i) {
                    probabilities = ggml_mul(ctx, probabilities, llama_spec_tree::row(ctx, chain.tree_scores.back(), 0));
                }
                ggml_set_output(probabilities);
                ggml_build_forward_expand(res->get_gf(), probabilities);
                chain.tree_scores.push_back(probabilities);
            }
            chain.params.push_back(std::move(params));
            chain.input_ends.push_back(res->inputs.size());
        }
        if (!ggml_backend_sched_alloc_graph_after(chain.sched.get(), res->get_gf(), chain.done.get())) {
            return false;
        }
        for (int n = 0; n < ggml_graph_n_nodes(res->get_gf()); n++) {
            auto * node = ggml_graph_node(res->get_gf(), n);
            if (ggml_backend_sched_get_tensor_backend(chain.sched.get(), node) == lctx.backend_cpu) {
                LLAMA_LOG_DEBUG("%s: CPU node %s (%s)\n", __func__, ggml_get_name(node), ggml_op_desc(node));
                return false;
            }
        }
    } else {
        lctx.n_reused++;
    }
    if (prefetch || p_min > 0.0f) {
        if (!chain.early_device) {
            auto * backend = lctx.backend_ptrs[0];
            auto * dev = ggml_backend_get_device(backend);
            auto * host_type = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (!host_type) {
                return false;
            }
            chain.early_ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
            chain.early_counts = ggml_new_tensor_1d(chain.early_ctx.get(), GGML_TYPE_I32, 2);
            chain.early_device.reset(ggml_backend_alloc_ctx_tensors(chain.early_ctx.get(), backend));
            chain.early_host.reset(ggml_backend_buft_alloc_buffer(host_type, 2*sizeof(int32_t)));
            if (!chain.early_device || !chain.early_host) {
                return false;
            }
        }
        chain.kept = chain.early_counts;
        if (p_min <= 0.0f && !reuse) {
            const int32_t counts[] = { n_draft, n_draft };
            ggml_backend_tensor_set(chain.early_counts, counts, 0, sizeof(counts));
        }
        if (p_min > 0.0f && !ggml_backend_sched_graph_early_exit(chain.sched.get(), chain.logits.data(), n_draft, p_min, chain.early_counts)) {
            return false;
        }
    }
    if (prefetch && !reuse) {
        std::vector<float> weights(ggml_nelements(chain.inp_weights), 0.0f);
        std::vector<float> positions(lctx.model.hparams.n_pos_per_embd()*n_draft);
        std::vector<float> position_mask(lctx.model.hparams.n_pos_per_embd(), 1.0f);
        if (position_mask.size() == 4) {
            position_mask[3] = 0.0f;
        }
        for (int i = 0; i < n_verify; i++) {
            if (chain.composed) {
                for (int j = 0; j < i; ++j) {
                    weights[i*2*n_verify + j] = n_verify - j;
                }
                weights[i*2*n_verify + n_verify + i] = n_verify - i;
            } else {
                weights[i] = n_verify - i;
            }
        }
        for (int i = 0; i < n_draft; i++) {
            for (uint32_t j = 0; j < lctx.model.hparams.n_pos_per_embd(); j++) {
                positions[i*lctx.model.hparams.n_pos_per_embd() + j] = 1 + i;
            }
        }
        const llama_token bonus = LLAMA_TOKEN_NULL;
        ggml_backend_tensor_set(chain.inp_bonus, &bonus, 0, sizeof(bonus));
        ggml_backend_tensor_set(chain.inp_weights, weights.data(), 0, ggml_nbytes(chain.inp_weights));
        ggml_backend_tensor_set(chain.inp_positions, positions.data(), 0, ggml_nbytes(chain.inp_positions));
        ggml_backend_tensor_set(chain.inp_position_mask, position_mask.data(), 0, ggml_nbytes(chain.inp_position_mask));
        if (chain.inp_out_ids) {
            const int32_t zero = 0;
            ggml_backend_tensor_set(chain.inp_out_ids, &zero, 0, sizeof(zero));
        }
    }
    if (prefetch && chain.inp_kv_positions) {
        std::vector<float> kv_positions(chain.inp_kv_positions->ne[0], -1.0f);
        const auto & cells = static_cast<llama_kv_cache *>(lctx.memory.get())->get_cells(seq_id);
        for (size_t i = 0; i < kv_positions.size(); i++) {
            if (cells.seq_has(i, seq_id) && cells.pos_get(i) >= verify_pos && cells.pos_get(i) < verify_pos + n_verify) {
                kv_positions[i] = cells.pos_get(i) - verify_pos;
            }
        }
        ggml_backend_tensor_set(chain.inp_kv_positions, kv_positions.data(), 0, ggml_nbytes(chain.inp_kv_positions));
    }
    for (int i = 0; i < n_draft; i++) {
        for (size_t j = i ? chain.input_ends[i - 1] : 0; j < chain.input_ends[i]; j++) {
            res->inputs[j]->set_input(&contexts[i]->get_ubatch());
        }
    }
    return true;
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
            if (this->handoff->snapshot_ready) {
                ggml_backend_event_synchronize(this->handoff->snapshot_ready.get());
            }
            if (this->features) {
                synchronize_nextn_catchup();
            }
            const auto * sampled = (const llama_token *) ggml_backend_buffer_get_base(state.sampled.host.get());
            int accepted = 0;
            if (state.next) {
                const auto * control = state.next->snapshot();
                if (control[llama_nextn_control::EPOCH] != state.epoch || control[llama_nextn_control::STEP] != state.step + 1 ||
                        control[llama_nextn_control::PREVIOUS_KEPT] + 1 != (int) state.draft.size()) {
                    return false;
                }
                accepted = control[llama_nextn_control::ACCEPTED];
                if (accepted < 0 || accepted >= (int) state.draft.size() ||
                        control[llama_nextn_control::POSITION] != seed.pos[0] ||
                        control[llama_nextn_control::N_FIELDS] != seed.token[0]) {
                    return false;
                }
            } else {
                while (accepted + 1 < (int) state.draft.size() && sampled[accepted] == state.draft[accepted]) {
                    ++accepted;
                }
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

    const bool eagle = lctx.model.arch == LLM_ARCH_EAGLE3;
    bool gpu_kv = prefetch && eagle;
    if (prefetch && !eagle) {
        const auto * kv = static_cast<const llama_kv_cache *>(lctx.memory.get());
        gpu_kv = lctx.cparams.flash_attn && lctx.cparams.causal_attn && seq_id == 0 && kv->get_n_stream() == 1 &&
            !ggml_is_quantized(kv->type_k()) && !ggml_is_quantized(kv->type_v()) &&
            seed.pos[0] >= 0 && seed.pos[0] <= (int64_t) kv->get_size() - n_draft;
        const auto & cells = kv->get_cells(seq_id);
        for (int i = 0; gpu_kv && i < seed.pos[0] + n_draft; ++i) {
            gpu_kv = cells.seq_has(i, seq_id) && cells.pos_get(i) == i;
        }
    }

    const int n_verify = prefetch ? this->lookahead->draft.size() : 0;
    const uint32_t cache_index = p_min > 0.0f ? n_verify : 0;
    if (cache_index != this->graph_active || (this->graph && this->graph->composed)) {
        if (this->graph && !this->graph->composed) {
            this->graph_cache[this->graph_active] = std::move(this->graph);
        } else {
            this->graph.reset();
        }
        this->graph = std::move(this->graph_cache[cache_index]);
        this->graph_active = cache_index;
    }
    if (!this->graph || this->graph->composed || (int) this->graph->params.size() != n_draft) {
        const size_t max_nodes = lctx.graph_max_nodes(1) * n_draft;
        this->graph.reset(new llama_nextn_graph);
        this->graph->res.reset(new llm_graph_result(max_nodes));
        this->graph->sched.reset(ggml_backend_sched_new(lctx.backend_ptrs.data(), lctx.backend_buft.data(), lctx.backend_ptrs.size(), max_nodes, false, lctx.cparams.op_offload));
    }
    auto & chain = *this->graph;
    auto * res = chain.res.get();
    if (!prepare_draft_graph(chain, contexts, n_draft, prefetch, p_min, n_verify, gpu_kv,
            prefetch ? &this->lookahead->input : nullptr,
            prefetch ? this->lookahead->sampled.rows[n_verify - 1] : nullptr,
            prefetch ? this->handoff->rows[n_verify - 1] : nullptr,
            prefetch ? this->lookahead->pos : -1)) {
        return fail();
    }
    if (ggml_backend_sched_graph_compute_async(chain.sched.get(), res->get_gf()) != GGML_STATUS_SUCCESS) {
        return fail();
    }
    collect_draft_outputs(chain, seed, prefetch);
    return true;
}

void llama_spec_pipeline::collect_draft_outputs(llama_nextn_graph & chain, const llama_batch & seed, bool prefetch) {
    const int n_draft = chain.tokens.size();
    const float p_min = chain.p_min;
    auto * backend = ggml_backend_sched_get_tensor_backend(chain.sched.get(), chain.logits[0]);
    if (!chain.done) {
        chain.done.reset(ggml_backend_event_new(ggml_backend_get_device(backend)));
    }
    ggml_backend_event_record(chain.done.get(), backend);

    const auto stride = lctx.model.vocab.n_tokens();
    llama_output_copies copies;
    if (p_min > 0.0f) {
        copies.add(backend, chain.early_counts, ggml_backend_buffer_get_base(chain.early_host.get()));
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
        // Wait for acceptance snapshots before their host buffers are released.
        ggml_backend_event_synchronize(this->lookahead->ready.get());
    }
    if (this->handoff && this->handoff->snapshot_ready) {
        ggml_backend_event_synchronize(this->handoff->snapshot_ready.get());
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
        if (this->outputs && this->graph) {
            auto * backend = ggml_backend_sched_get_tensor_backend(this->graph->sched.get(), this->graph->logits[0]);
            if (!this->output_ready) {
                this->output_ready.reset(ggml_backend_event_new(ggml_backend_get_device(backend)));
            }
            ggml_backend_event_record(this->output_ready.get(), backend);
            this->output_pending = true;
        }
    }
    if (!outputs_only || !this->output_pending) {
        ggml_backend_sched_synchronize(lctx.sched.get());
        if (this->outputs && this->graph &&
                ggml_backend_sched_get_tensor_backend(this->graph->sched.get(), this->graph->logits[0]) != lctx.backend_ptrs[0]) {
            ggml_backend_sched_synchronize(this->graph->sched.get());
        }
    }
    if (this->output_pending) {
        ggml_backend_event_synchronize(this->output_ready.get());
    }
    if (!outputs_only && this->copy_backend) {
        ggml_backend_synchronize(this->copy_backend.get());
    }
    if (this->handoff && this->handoff->snapshot_ready) {
        ggml_backend_event_synchronize(this->handoff->snapshot_ready.get());
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
    if (target_active >= 0 && targets[target_active]->tree) {
        auto & state = *targets[target_active];
        copies.add(lctx.backend_ptrs[0], state.input->tokens[3], ggml_backend_buffer_get_base(state.tree->host.get()));
    }
    if (defer) {
        readback = std::make_unique<llama_output_copies>(std::move(copies));
        this->mctx = std::move(mctx);
    } else {
        copies.submit();
    }
}

bool llama_spec_pipeline::verified_draft(llama_token * tokens, int32_t n) {
    if (!tokens || n != 3 || target_active < 0 || !targets[target_active]->tree) {
        return false;
    }
    lctx.synchronize(true);
    const auto * resolved = (const llama_token *) ggml_backend_buffer_get_base(targets[target_active]->tree->host.get());
    std::copy_n(resolved + 1, n, tokens);
    return true;
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
    std::set<const llama_nextn_graph *> graphs;
    auto add_graph = [&](const std::shared_ptr<llama_nextn_graph> & entry) {
        if (entry && !entry->logits.empty() &&
                ggml_backend_sched_get_tensor_backend(entry->sched.get(), entry->logits[0]) == lctx.backend_ptrs[0]) {
            graphs.insert(entry.get());
        }
    };
    add_graph(this->graph);
    for (const auto & entry : this->graph_cache) {
        add_graph(entry);
    }
    for (const auto & entry : this->targets) {
        if (entry && entry->draft) {
            add_graph(entry->draft->body->graph);
        }
    }
    if (!lctx.model.hparams.no_alloc) {
        for (const auto & backend_ptr : lctx.backends) {
            auto * backend = backend_ptr.get();
            auto * buft = ggml_backend_sched_get_buffer_type(lctx.sched.get(), backend);
            for (const auto * entry : graphs) {
                ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->sched.get(), backend);
            }
            for (const auto & entry : this->targets) {
                if (entry && entry->sched) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->sched.get(), backend);
                }
                if (entry && entry->meta_sched) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->meta_sched.get(), backend);
                }
                if (entry && entry->draft) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->draft->meta_sched.get(), backend);
                }
                if (entry && entry->catchup) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->catchup->sched.get(), backend);
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(entry->catchup->meta_sched.get(), backend);
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
    if (this->lookahead) {
        add_buffer(this->lookahead->input.device);
        add_buffer(this->lookahead->input.host);
    }
    for (const auto * entry : graphs) {
        add_buffer(entry->early_device);
        add_buffer(entry->early_host);
        add_buffer(entry->constants_device);
    }
    std::set<const llama_nextn_control *> controls;
    std::set<const llama_nextn_handoff *> seeds;
    std::set<const llama_nextn_draft_body *> drafts;
    for (const auto & entry : this->targets) {
        if (entry) {
            add_buffer(entry->constants_device);
            if (entry->tree) {
                add_buffer(entry->tree->device);
                add_buffer(entry->tree->host);
            }
        }
        if (entry && entry->input && controls.insert(entry->input.get()).second) {
            add_buffer(entry->input->device);
            add_buffer(entry->input->host);
        }
        if (entry && entry->catchup_seed && seeds.insert(entry->catchup_seed.get()).second) {
            add_buffer(entry->catchup_seed->device);
        }
        if (entry && entry->draft && drafts.insert(entry->draft->body.get()).second) {
            add_buffer(entry->draft->body->input_device);
        }
        if (entry && entry->catchup) {
            add_buffer(entry->catchup->output_device);
        }
    }
}
