#include "llama-spec-opt-chain.h"
#include "llama-spec-opt-api.h"
#include "llama-spec-opt-eagle3-graph.h"

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>

void llama_spec_chain_pipeline::finish(llama_output_readback & readback) {
    if (!chain || !chain->prefetched) {
        return;
    }
    readback.submit_async();
    ggml_backend_sched_synchronize(chain->sched.get());
    chain->prefetched = false;
}

void llama_spec_chain_pipeline::reset(llama_output_readback & readback) {
    finish(readback);
    chain.reset();
}

bool llama_spec_chain_pipeline::can_defer_readback(const llama_batch & batch,
                                                   uint32_t            n_tokens,
                                                   uint32_t            n_outputs,
                                                   uint32_t            n_ubatch) const {
    // a single-ubatch verify batch holds its D2H copies back until the draft side has queued its D2D copies
    return verify_input && batch.token && n_outputs == n_tokens && n_tokens <= n_ubatch;
}

void llama_spec_chain_pipeline::set_graph_params(llm_graph_params & params) const {
    if (is_eagle3) {
        params.spec_inputs.features = draft_state_copy.dst;
    } else {
        // mtp also reads the seed row in front of the copied rows
        params.spec_inputs.features = draft_state_copy.active() ? std::vector<ggml_tensor *>{draft_state_io->mtp_h} :
                                                                  std::vector<ggml_tensor *>{};
    }
    params.spec_inputs.verify_tokens = verify_copy.active() ? verify_input->tokens : nullptr;
}

void llama_spec_chain_pipeline::bind_inputs(llm_graph_result * res) const {
    if (!verify_copy.active()) {
        return;
    }
    // set_input writes the host tokens here, then the chain tokens overwrite rows 1..n_draft
    ggml_tensor * tokens = res->get_inp_tokens();
    GGML_ASSERT(tokens && tokens->type == GGML_TYPE_I32 && tokens->ne[0] == verify_input->tokens->ne[0]);
    ggml_backend_tensor_alloc(verify_input->device.get(), tokens, verify_input->tokens->data);
}

void llama_spec_chain_pipeline::memory_breakdown(llama_memory_breakdown &              ret,
                                                 ggml_backend_sched_t                  sched,
                                                 const std::vector<ggml_backend_ptr> & backends) const {
    // draft chain runs on its own sched, so add its compute buffers too
    if (chain) {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched, backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(chain->sched.get(), backend);
        }
    }
    auto add_device = [&](const ggml_backend_buffer_ptr & buf) {
        ret[ggml_backend_buffer_get_type(buf.get())].compute += ggml_backend_buffer_get_size(buf.get());
    };
    if (input) {
        add_device(input->device);
    }
    if (verify_input) {
        add_device(verify_input->device);
    }
    if (draft_state_io) {
        add_device(draft_state_io->device);
    }
}

bool llama_spec_chain_pipeline::submit_draft_state_update(llama_context &     draft,
                                                          llama_context &     target,
                                                          const llama_batch & batch,
                                                          const float **      eagle3_g_out) {
    // copy the fresh target hidden states draft-side by D2D, so the fused encoder and the draft state update never touch the host
    // the encoder gets one row more than the decoder: that row is the g of the new deferred boundary, kept for the next round
    // MTP has no encoder: row 0 is the seed, its hidden comes from batch.embd
    GGML_ASSERT(draft.spec_opt.chain_pipeline);
    auto & pipeline = *draft.spec_opt.chain_pipeline;
    GGML_ASSERT(pipeline.draft_state_io);
    if (eagle3_g_out) {
        *eagle3_g_out = nullptr;
    }

    const int n_features = pipeline.is_eagle3 ? batch.n_tokens + 1 : batch.n_tokens;
    if (draft.sched_need_reserve || target.balloc->get_n_tokens() != (uint32_t) n_features ||
        (pipeline.chain && pipeline.chain->prefetched) || target.gf_res_prev->t_sampled.size() != (size_t) n_features ||
        !pipeline.input || pipeline.input->sampled_rows.size() != (size_t) n_features) {
        return false;
    }

    auto * target_backend = target.backend_ptrs[0];
    auto * draft_backend  = draft.backend_ptrs[0];
    for (int i = 0; i < n_features; ++i) {
        auto * sampled = target.gf_res_prev->t_sampled[i];
        if (!sampled || !sampled->buffer || sampled->type != GGML_TYPE_I32 || !ggml_is_contiguous(sampled) ||
            ggml_nbytes(sampled) != sizeof(llama_token) || ggml_backend_buffer_is_host(sampled->buffer) ||
            ggml_backend_sched_get_tensor_backend(target.sched.get(), sampled) != target_backend) {
            return false;
        }
    }

    const llama_seq_id seq_id   = batch.seq_id[0][0];
    const llama_pos    seed_pos = batch.pos[0] - 1;
    if (seed_pos < 0) {
        return false;
    }

    if (draft.memory->seq_pos_max(seq_id) != seed_pos) {
        return false;
    }

    auto &        state       = *pipeline.draft_state_io;
    const size_t  n_layers    = state.features.size();
    const int64_t feature_dim = state.features[0]->ne[0];

    std::vector<ggml_tensor *> hidden(n_layers);
    for (size_t layer = 0; layer < n_layers; ++layer) {
        const int id = pipeline.is_eagle3 ? draft.model.target_layer_ids[layer] : (int) target.model.hparams.n_layer();
        ggml_tensor * tensor = id < (int) target.model.hparams.n_layer() ? target.gf_res_prev->get_layer_inp(id) :
                                                                           target.gf_res_prev->get_h_nextn();
        if (!tensor || !tensor->buffer || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != feature_dim ||
            tensor->ne[1] != n_features || !ggml_is_contiguous(tensor) || ggml_backend_buffer_is_host(tensor->buffer) ||
            ggml_backend_sched_get_tensor_backend(target.sched.get(), tensor) != target_backend) {
            return false;
        }
        hidden[layer] = tensor;
    }

    if (!pipeline.is_eagle3) {
        ggml_backend_tensor_set_async(draft_backend, state.mtp_h, batch.embd, 0, state.mtp_h->nb[1]);
    }

    pipeline.draft_state_copy = {target_backend, draft_backend, std::move(hidden), state.features};
    auto decode_batch         = batch;
    decode_batch.embd         = nullptr;
    const int  ret            = llama_spec_opt::decode_with_copy(draft, decode_batch, pipeline.draft_state_copy);
    const bool copy_submitted = pipeline.draft_state_copy.submitted;
    pipeline.draft_state_copy = {};

    if (ret != 0 || !copy_submitted || (pipeline.is_eagle3 && !readback_eagle3_g(draft, n_features, eagle3_g_out))) {
        draft.synchronize();
        draft.memory->seq_rm(seq_id, pipeline.is_eagle3 ? seed_pos : batch.pos[0], -1);
        return false;
    }
    return true;
}

// eagle3 keeps g of the draft state update for the next round, false if g is missing or has a wrong shape
bool llama_spec_chain_pipeline::readback_eagle3_g(llama_context & draft, int n_features, const float ** eagle3_g_out) {
    auto * g = ggml_graph_get_tensor(draft.gf_res_prev->get_gf(), llama_eagle3_graph::draft_state_output_name);
    if (!g || g->ne[0] != draft.model.hparams.n_embd_out() || g->ne[1] != n_features) {
        return false;
    }
    auto & state         = *draft.spec_opt.chain_pipeline->draft_state_io;
    auto * draft_backend = draft.backend_ptrs[0];
    ggml_backend_tensor_copy_async(draft_backend, draft_backend, g, state.eagle3_g);
    auto * host = ggml_backend_buffer_get_base(state.eagle3_g_host.get());
    ggml_backend_tensor_get_async(draft_backend, state.eagle3_g, host, 0, ggml_nbytes(state.eagle3_g));
    ggml_backend_event_record(state.eagle3_g_ready.get(), draft_backend);
    *eagle3_g_out = static_cast<const float *>(host);
    return true;
}

// prepare the KV metadata of n rows from pos0, without token or embd data
llama_memory_context_ptr llama_spec_chain_pipeline::prepare_kv_metadata_rows(llama_context & ctx,
                                                                             llama_seq_id    seq,
                                                                             llama_pos       pos0,
                                                                             int32_t         n) {
    const int32_t               n_pos = ctx.model.hparams.n_pos_per_embd();
    std::vector<llama_pos>      positions((size_t) n * n_pos);
    std::vector<int32_t>        n_seq_id(n, 1);
    std::vector<llama_seq_id *> seq_ids(n, &seq);
    std::vector<int8_t>         outputs(n, 1);
    for (int j = 0; j < n_pos; ++j) {
        for (int i = 0; i < n; ++i) {
            positions[(size_t) j * n + i] = pos0 + i;
        }
    }
    llama_batch batch = {n, nullptr, nullptr, positions.data(), n_seq_id.data(), seq_ids.data(), outputs.data()};
    if (!ctx.balloc->init(batch, ctx.model.vocab, ctx.memory.get(), ctx.model.hparams.n_embd_out(),
                          ctx.cparams.n_seq_max, false)) {
        return nullptr;
    }
    auto mctx = ctx.memory->init_batch(*ctx.balloc, n, false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return nullptr;
    }
    GGML_ASSERT(mctx->get_ubatch().n_tokens == (uint32_t) n);
    if (!mctx->apply()) {
        ctx.memory->seq_rm(seq, pos0, -1);
        return nullptr;
    }
    return mctx;
}

// the old chain graph can be reused if every step has the same params and inputs
static bool chain_can_reuse(const llama_spec_draft_chain & chain, const std::vector<llm_graph_params> & steps) {
    if (chain.params.size() != steps.size()) {
        return false;
    }
    // every step adds the same number of inputs
    const auto & inputs       = chain.res->inputs;
    const size_t n_step_input = inputs.size() / steps.size();
    for (size_t i = 0; i < steps.size(); i++) {
        if (!chain.params[i].allow_reuse(steps[i])) {
            return false;
        }
        for (size_t j = 0; j < n_step_input; j++) {
            if (!inputs[i * n_step_input + j]->can_reuse(steps[i])) {
                return false;
            }
        }
    }
    return true;
}

bool llama_spec_chain_pipeline::build_chain(llama_context & draft, llama_kv_cache_context * kv_ctx, int32_t n_draft) {
    auto & pipeline = *draft.spec_opt.chain_pipeline;
    if (!pipeline.chain) {
        const size_t max_nodes = draft.graph_max_nodes(1) * n_draft;
        pipeline.chain         = std::make_unique<llama_spec_draft_chain>();
        pipeline.chain->res.reset(new llm_graph_result(max_nodes));
        pipeline.chain->sched.reset(ggml_backend_sched_new(draft.backend_ptrs.data(), draft.backend_buft.data(),
                                                           draft.backend_ptrs.size(), max_nodes, false,
                                                           draft.cparams.op_offload));
    }
    auto & chain = *pipeline.chain;
    auto * res   = chain.res.get();

    const auto & chain_ubatch = kv_ctx->get_ubatch();
    GGML_ASSERT((uint32_t) n_draft <= chain_ubatch.n_tokens &&
                chain_ubatch.n_pos == draft.model.hparams.n_pos_per_embd() && !chain_ubatch.token &&
                !chain_ubatch.embd);

    const auto gtype = pipeline.is_eagle3 ? LLM_GRAPH_TYPE_DEFAULT : LLM_GRAPH_TYPE_DECODER_MTP;

    std::vector<llm_graph_params> steps;
    for (int i = 0; i < n_draft; i++) {
        auto step_ubatch     = chain_ubatch;
        step_ubatch.n_tokens = step_ubatch.n_seq_tokens = step_ubatch.n_seqs = 1;
        step_ubatch.pos += i;
        step_ubatch.n_seq_id += i;
        step_ubatch.seq_id += i;
        step_ubatch.output += i;

        auto params      = draft.graph_params(res, step_ubatch, kv_ctx, gtype);
        params.sched     = chain.sched.get();
        params.n_outputs = 1;
        params.samplers.clear();
        params.cb = [](const llama_ubatch &, ggml_tensor * tensor, const char * name, int il) {
            ggml_format_name(tensor, "%s-%d", name, il);
        };
        steps.push_back(std::move(params));
    }
    if (!draft.graph_reuse_disable && chain_can_reuse(chain, steps)) {
        draft.n_reused++;
        return true;
    }

    // wait for any in-flight chain before resetting its graph buffers
    ggml_backend_sched_synchronize(chain.sched.get());
    ggml_backend_sched_reset(chain.sched.get());
    res->reset();
    chain.params.clear();
    chain.tokens.clear();

    // accepted = first row where the target sample differs from the draft token
    // the last draft token is LLAMA_TOKEN_NULL, so a mismatch always exists; decreasing weights make argmax pick the first one
    const auto & input    = *pipeline.input;
    auto *       gctx     = res->get_ctx();
    auto *       mismatch = ggml_step(gctx, ggml_abs(gctx, ggml_sub(gctx, ggml_cast(gctx, input.sampled, GGML_TYPE_F32),
                                                                    ggml_cast(gctx, input.draft_tokens, GGML_TYPE_F32))));
    auto *       accepted = ggml_argmax(gctx, ggml_mul(gctx, mismatch, input.accept_weights));

    // step 0 starts from the accepted row: its target sample and hidden state
    ggml_tensor * token = ggml_get_rows(gctx, ggml_reshape_2d(gctx, input.sampled, 1, n_draft + 1), accepted);
    ggml_tensor * state = pipeline.is_eagle3 ? pipeline.draft_state_io->eagle3_g : pipeline.draft_state_io->features[0];
    ggml_tensor * hidden = ggml_get_rows(gctx, state, accepted);
    ggml_tensor * pos =
        ggml_cast(gctx, ggml_add(gctx, input.positions, ggml_cast(gctx, accepted, GGML_TYPE_F32)), GGML_TYPE_I32);

    size_t n_step_input = 0;
    for (int i = 0; i < n_draft; i++) {
        auto & params                = steps[i];
        params.spec_inputs.tokens    = token;
        params.spec_inputs.hidden    = hidden;
        params.spec_inputs.positions = ggml_view_1d(gctx, pos, 1, i * sizeof(int32_t));
        if (!draft.model.build_graph(params) || !res->get_h_nextn() || !res->get_logits()) {
            return false;
        }
        // greedy pick, same as the top token of the CPU top-k draft sampler
        token  = ggml_argmax(gctx, res->get_logits());
        hidden = res->get_h_nextn();
        ggml_set_output(token);
        ggml_build_forward_expand(res->get_gf(), token);
        chain.tokens.push_back(token);

        // chain_can_reuse() expects the same number of inputs per step
        if (i == 0) {
            n_step_input = res->inputs.size();
        }
        GGML_ASSERT(res->inputs.size() == (size_t) (i + 1) * n_step_input);
    }
    chain.params = std::move(steps);

    if (!ggml_backend_sched_alloc_graph(chain.sched.get(), res->get_gf())) {
        return false;
    }
    // reject graphs with any node assigned to CPU despite full model offload
    for (int n = 0; n < ggml_graph_n_nodes(res->get_gf()); n++) {
        auto * node = ggml_graph_node(res->get_gf(), n);
        if (ggml_backend_sched_get_tensor_backend(chain.sched.get(), node) == draft.backend_cpu) {
            LLAMA_LOG_DEBUG("%s: CPU node %s (%s)\n", __func__, ggml_get_name(node), ggml_op_desc(node));
            return false;
        }
    }
    return true;
}

void llama_spec_chain_pipeline::upload_chain_input(llama_context &     draft,
                                                   llama_context &     target,
                                                   const llama_batch & batch) {
    auto *    target_backend = target.backend_ptrs[0];
    auto *    draft_backend  = draft.backend_ptrs[0];
    const int n_draft        = batch.n_tokens;
    const int n_verify       = n_draft + 1;
    GGML_ASSERT(draft.spec_opt.chain_pipeline && draft.spec_opt.chain_pipeline->input);
    auto & input = *draft.spec_opt.chain_pipeline->input;
    for (int i = 0; i < n_verify; ++i) {
        ggml_backend_tensor_copy_async(target_backend, draft_backend, target.gf_res_prev->t_sampled[i],
                                       input.sampled_rows[i]);
    }
    input.seq = batch.seq_id[0][0];
    input.draft_tokens_host.assign(batch.token, batch.token + n_draft);
    input.draft_tokens_host.push_back(LLAMA_TOKEN_NULL);
    input.positions_host.resize(n_draft);
    for (int i = 0; i < n_draft; ++i) {
        input.positions_host[i] = batch.pos[0] + i;
    }
    ggml_backend_tensor_set_async(draft_backend, input.draft_tokens, input.draft_tokens_host.data(), 0,
                                  ggml_nbytes(input.draft_tokens));
    ggml_backend_tensor_set_async(draft_backend, input.positions, input.positions_host.data(), 0,
                                  ggml_nbytes(input.positions));
}

// the mtp seed row has no draft token, the chain starts after it
static llama_batch mtp_chain_batch(const llama_batch & batch) {
    auto chain_batch = batch;
    chain_batch.n_tokens -= 1;
    chain_batch.token += 1;
    chain_batch.pos += 1;
    chain_batch.n_seq_id += 1;
    chain_batch.seq_id += 1;
    chain_batch.logits = chain_batch.logits ? chain_batch.logits + 1 : nullptr;
    chain_batch.embd   = nullptr;
    return chain_batch;
}

bool llama_spec_chain_pipeline::submit_draft_chain(llama_context &     draft,
                                                   llama_context &     target,
                                                   const llama_batch & state_batch) {
    const auto batch   = draft.spec_opt.chain_pipeline->is_eagle3 ? state_batch : mtp_chain_batch(state_batch);
    const int  n_draft = batch.n_tokens;

    auto * kv = static_cast<llama_kv_cache *>(draft.memory.get());
    if (draft.sched_need_reserve || kv->get_has_shift()) {
        return false;
    }

    // the chain starts right after the draft state rows; logical positions are used directly as physical KV slots
    const auto      seq_id   = batch.seq_id[0][0];
    const llama_pos seed_pos = batch.pos[0] + n_draft;
    const auto &    cells    = kv->get_cells(0);
    if (draft.memory->seq_pos_min(seq_id) != 0 || draft.memory->seq_pos_max(seq_id) != seed_pos - 1 ||
        cells.used_max_p1() != (uint32_t) seed_pos || !cells.is_prefix_ordered()) {
        return false;
    }

    if (draft.output_reserve(n_draft) < (uint32_t) n_draft) {
        return false;
    }
    auto & sampled = draft.sampling.sampled;
    if (!sampled.has_data() || sampled.size < (size_t) n_draft) {
        return false;
    }
    std::fill_n(sampled.data, n_draft, LLAMA_TOKEN_NULL);

    // KV metadata for the all-accepted case, only to build the chain graph
    auto mctx = prepare_kv_metadata_rows(draft, seq_id, seed_pos, n_draft);
    if (!mctx) {
        return false;
    }
    auto * kv_ctx = static_cast<llama_kv_cache_context *>(mctx.get());

    auto & pipeline = *draft.spec_opt.chain_pipeline;
    auto   fail     = [&]() {
        draft.synchronize();
        draft.memory->seq_rm(seq_id, seed_pos, -1);
        pipeline.chain.reset();
        return false;
    };

    upload_chain_input(draft, target, batch);

    if (draft.t_compute_start_us == 0) {
        draft.t_compute_start_us = ggml_time_us();
    }

    if (!build_chain(draft, kv_ctx, n_draft)) {
        return fail();
    }
    auto & chain = *pipeline.chain;
    if (ggml_backend_sched_graph_compute_async(chain.sched.get(), chain.res->get_gf()) != GGML_STATUS_SUCCESS) {
        return fail();
    }
    // after chain launch: last target->draft D2D is queued, so D2H no longer delays draft inputs
    target.spec_opt.submit_readback();

    // chain tokens are the output of this round, read back at the next finish()
    for (size_t i = 0; i < chain.tokens.size(); ++i) {
        auto * token = chain.tokens[i];
        GGML_ASSERT(ggml_is_contiguous(token) && ggml_nelements(token) == 1);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(chain.sched.get(), token);
        draft.spec_opt.readback.copy_async(backend, token, sampled.data + i, ggml_nbytes(token));
    }
    draft.logits.data     = nullptr;
    draft.embd_nextn.data = nullptr;
    draft.output_swaps.clear();
    draft.n_outputs = n_draft;
    for (int i = 0; i < n_draft; i++) {
        draft.output_ids[i] = i;
    }
    draft.n_queued_tokens += n_draft;
    chain.prefetched = true;

    // the host path sets the KV metadata again once the accepted boundary is known
    draft.memory->seq_rm(seq_id, seed_pos, -1);
    return true;
}

bool llama_spec_chain_pipeline::submit_draft_generation(llama_context &     draft,
                                                        llama_context &     target,
                                                        const llama_batch & batch,
                                                        const float **      eagle3_g_out) {
    if (!submit_draft_state_update(draft, target, batch, eagle3_g_out)) {
        return false;
    }
    if (!submit_draft_chain(draft, target, batch)) {
        draft.synchronize();
        draft.memory->seq_rm(batch.seq_id[0][0], batch.pos[0], -1);
        if (eagle3_g_out) {
            *eagle3_g_out = nullptr;
        }
        return false;
    }
    return true;
}

void llama_spec_chain_pipeline::synchronize_eagle3_g(llama_context & draft) {
    GGML_ASSERT(draft.spec_opt.chain_pipeline);
    const auto & io = draft.spec_opt.chain_pipeline->draft_state_io;
    GGML_ASSERT(io && io->eagle3_g_host && io->eagle3_g_ready && io->eagle3_g);
    ggml_backend_event_synchronize(io->eagle3_g_ready.get());
}

// the EAGLE3 draft model encodes target hidden states of target_layer_ids
bool llama_spec_chain_pipeline::eagle3_supported(const llama_context & target, const llama_context & draft) {
    const auto & tparams = target.cparams;
    const auto & tmodel  = target.model;
    const auto & dmodel  = draft.model;
    if (draft.cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
        (tmodel.arch != LLM_ARCH_QWEN3 && tmodel.arch != LLM_ARCH_QWEN3MOE) || dmodel.hparams.n_pos_per_embd() != 1 ||
        !dynamic_cast<const llama_kv_cache *>(target.memory.get())) {
        return false;
    }

    const size_t n_layers = dmodel.target_layer_ids.size();
    if (n_layers == 0 || dmodel.hparams.n_embd_inp_enc() % (int64_t) n_layers != 0 ||
        dmodel.hparams.n_embd_inp_enc() / (int64_t) n_layers != tmodel.hparams.n_embd) {
        return false;
    }
    const int n_layer_tgt = (int) tmodel.hparams.n_layer();
    for (const int id : dmodel.target_layer_ids) {
        if (id < 0 || id > n_layer_tgt || (id < n_layer_tgt && !tparams.embeddings_layer_inp[id]) ||
            (id == n_layer_tgt && !tparams.embeddings_nextn)) {
            return false;
        }
    }
    return true;
}

// the MTP draft context runs the nextn layer of the target model
bool llama_spec_chain_pipeline::mtp_supported(const llama_context & target,
                                              const llama_context & draft,
                                              int32_t               n_draft) {
    const auto & tparams = target.cparams;
    const auto & tmodel  = target.model;
    if (draft.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || &draft.model != &tmodel ||
        (tmodel.arch != LLM_ARCH_QWEN35 && tmodel.arch != LLM_ARCH_QWEN35MOE) || tmodel.hparams.n_layer_nextn != 1 ||
        tmodel.hparams.n_pos_per_embd() != 4 || !tparams.embeddings_nextn ||
        tmodel.hparams.n_embd_out() != tmodel.hparams.n_embd || tparams.n_rs_seq < (uint32_t) n_draft) {
        return false;
    }
    const ggml_tensor * embed = tmodel.layers[tmodel.hparams.n_layer()].nextn.embed_tokens;
    return !embed || (embed->buffer && !ggml_backend_buffer_is_host(embed->buffer));
}

llama_spec_chain_pipeline::pipeline_type llama_spec_chain_pipeline::supported_type(const llama_context & target,
                                                                                   const llama_context & draft,
                                                                                   int32_t               n_draft) {
    if (n_draft <= 0) {
        return pipeline_type::none;
    }

    const bool is_eagle3 = draft.model.arch == LLM_ARCH_EAGLE3;
    if (is_eagle3 ? !eagle3_supported(target, draft) : !mtp_supported(target, draft, n_draft)) {
        return pipeline_type::none;
    }

    const auto & tparams = target.cparams;
    const auto & dparams = draft.cparams;
    if (tparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || tparams.n_seq_max != 1 || tparams.pipeline_parallel ||
        tparams.cb_eval || target.opt_ctx || tparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
        tparams.embeddings_nextn_masked || n_draft + 1 > (int32_t) tparams.n_ubatch) {
        return pipeline_type::none;
    }
    if (dparams.n_seq_max != 1 || dparams.pipeline_parallel || dparams.cb_eval || draft.opt_ctx ||
        !dparams.flash_attn || n_draft > (int32_t) dparams.n_ubatch) {
        return pipeline_type::none;
    }

    const auto & tmodel = target.model;
    const auto & dmodel = draft.model;
    if (!tmodel.tok_embd || !tmodel.tok_embd->buffer || !dmodel.tok_embd || !dmodel.tok_embd->buffer ||
        ggml_backend_buffer_is_host(tmodel.tok_embd->buffer) || ggml_backend_buffer_is_host(dmodel.tok_embd->buffer) ||
        tmodel.devices.size() != 1 || dmodel.devices.size() != 1 ||
        tmodel.n_gpu_layers() <= tmodel.hparams.n_layer_all || dmodel.n_gpu_layers() <= dmodel.hparams.n_layer_all ||
        dmodel.hparams.n_swa || dmodel.hparams.use_alibi || dmodel.vocab.n_tokens() > (1u << 24) ||
        tmodel.vocab.n_tokens() != dmodel.vocab.n_tokens()) {
        return pipeline_type::none;
    }

    const auto * kv = dynamic_cast<const llama_kv_cache *>(draft.memory.get());
    if (!kv || kv->get_n_stream() != 1 || ggml_is_quantized(kv->type_k()) || ggml_is_quantized(kv->type_v())) {
        return pipeline_type::none;
    }

    auto * dev = ggml_backend_get_device(target.backend_ptrs[0]);
    if (!dev || ggml_backend_get_device(draft.backend_ptrs[0]) != dev ||
        strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
        return pipeline_type::none;
    }
    return is_eagle3 ? pipeline_type::eagle3 : pipeline_type::mtp;
}

// the ensure functions keep an existing buffer only if it has the requested size

static bool eagle3_draft_state_io_ensure(std::unique_ptr<llama_spec_draft_state_io> & io,
                                         ggml_backend_t                               backend,
                                         int64_t                                      feature_dim,
                                         int64_t                                      g_dim,
                                         int64_t                                      n_verify,
                                         size_t                                       n_layers) {
    if (io) {
        return io->eagle3_g->ne[1] == n_verify;
    }
    auto * dev   = ggml_backend_get_device(backend);
    auto   state = std::make_unique<llama_spec_draft_state_io>();
    state->ctx.reset(ggml_init({(n_layers + 1) * ggml_tensor_overhead(), nullptr, true}));
    state->features.resize(n_layers);
    for (auto & feature : state->features) {
        feature = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, feature_dim, n_verify);
    }
    state->eagle3_g = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, g_dim, n_verify);
    state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), backend));
    auto * host_type = ggml_backend_dev_host_buffer_type(dev);
    if (host_type) {
        state->eagle3_g_host.reset(ggml_backend_buft_alloc_buffer(host_type, ggml_nbytes(state->eagle3_g)));
    }
    state->eagle3_g_ready.reset(ggml_backend_event_new(dev));
    if (!state->device || !state->eagle3_g_host || !state->eagle3_g_ready) {
        return false;
    }
    io = std::move(state);
    return true;
}

// row 0 of mtp_h is the seed hidden, features[0] is a view of the next rows
static bool mtp_draft_state_io_ensure(std::unique_ptr<llama_spec_draft_state_io> & io,
                                      ggml_backend_t                               backend,
                                      int64_t                                      feature_dim,
                                      int64_t                                      n_verify) {
    if (io) {
        return io->features[0]->ne[1] == n_verify;
    }
    auto state = std::make_unique<llama_spec_draft_state_io>();
    state->ctx.reset(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
    auto * h     = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, feature_dim, n_verify + 1);
    state->mtp_h = h;
    state->features.push_back(ggml_view_2d(state->ctx.get(), h, feature_dim, n_verify, h->nb[1], h->nb[1]));
    state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), backend));
    if (!state->device) {
        return false;
    }
    io = std::move(state);
    return true;
}

static bool chain_input_ensure(std::unique_ptr<llama_spec_chain_input> & input,
                               ggml_backend_t                            backend,
                               int64_t                                   n_verify) {
    if (input) {
        return input->sampled->ne[0] == n_verify;
    }
    auto         state     = std::make_unique<llama_spec_chain_input>();
    const size_t n_tensors = 4 + (size_t) n_verify;
    state->ctx.reset(ggml_init({n_tensors * ggml_tensor_overhead(), nullptr, true}));
    state->sampled        = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_I32, n_verify);
    state->draft_tokens   = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_I32, n_verify);
    state->accept_weights = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_F32, n_verify);
    state->positions      = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_F32, n_verify - 1);
    for (int64_t i = 0; i < n_verify; ++i) {
        state->sampled_rows.push_back(ggml_view_1d(state->ctx.get(), state->sampled, 1, i * sizeof(llama_token)));
    }
    state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), backend));
    if (!state->device) {
        return false;
    }
    std::vector<float> weights(n_verify);
    for (int64_t i = 0; i < n_verify; ++i) {
        weights[i] = n_verify - i;
    }
    ggml_backend_tensor_set(state->accept_weights, weights.data(), 0, ggml_nbytes(state->accept_weights));
    input = std::move(state);
    return true;
}

static bool verify_input_ensure(std::unique_ptr<llama_spec_verify_input> & input,
                                ggml_backend_t                             backend,
                                int64_t                                    n_verify) {
    if (input) {
        return input->tokens->ne[0] == n_verify;
    }
    auto         state     = std::make_unique<llama_spec_verify_input>();
    const size_t n_tensors = (size_t) n_verify;
    state->ctx.reset(ggml_init({n_tensors * ggml_tensor_overhead(), nullptr, true}));
    state->tokens = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_I32, n_verify);
    for (int64_t i = 1; i < n_verify; i++) {
        state->draft_rows.push_back(ggml_view_1d(state->ctx.get(), state->tokens, 1, i * sizeof(llama_token)));
    }
    state->device.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), backend));
    if (!state->device) {
        return false;
    }
    input = std::move(state);
    return true;
}

bool llama_spec_chain_pipeline::prepare(llama_context & target, llama_context & draft, int32_t n_draft) {
    const auto type = supported_type(target, draft, n_draft);
    if (type == pipeline_type::none) {
        return false;
    }

    if (!draft.spec_opt.chain_pipeline) {
        draft.spec_opt.chain_pipeline = std::make_unique<llama_spec_chain_pipeline>();
    }
    if (!target.spec_opt.chain_pipeline) {
        target.spec_opt.chain_pipeline = std::make_unique<llama_spec_chain_pipeline>();
    }

    const bool    is_eagle3     = type == pipeline_type::eagle3;
    const int64_t n_verify      = (int64_t) n_draft + 1;
    const int64_t feature_dim   = target.model.hparams.n_embd;
    auto *        draft_backend = draft.backend_ptrs[0];
    auto &        dpipeline     = *draft.spec_opt.chain_pipeline;
    bool          io_ready      = false;
    if (is_eagle3) {
        io_ready = eagle3_draft_state_io_ensure(dpipeline.draft_state_io, draft_backend, feature_dim,
                                                draft.model.hparams.n_embd_out(), n_verify,
                                                draft.model.target_layer_ids.size());
    } else {
        io_ready = mtp_draft_state_io_ensure(dpipeline.draft_state_io, draft_backend, feature_dim, n_verify);
    }
    if (!io_ready || !chain_input_ensure(dpipeline.input, draft_backend, n_verify) ||
        !verify_input_ensure(target.spec_opt.chain_pipeline->verify_input, target.backend_ptrs[0], n_verify)) {
        return false;
    }

    dpipeline.is_eagle3 = is_eagle3;

    // the chain returns one output row per draft step
    if ((uint32_t) n_draft > draft.cparams.n_outputs_max) {
        LLAMA_LOG_DEBUG("%s: draft n_outputs_max %u -> %d\n", __func__, draft.cparams.n_outputs_max, n_draft);
        draft.cparams.n_outputs_max = n_draft;
    }

    return true;
}

int32_t llama_spec_chain_pipeline::submit_target_verification(llama_context &     target,
                                                              llama_context &     draft,
                                                              const llama_batch & batch,
                                                              llama_token *       draft_tokens) {
    GGML_ASSERT(target.spec_opt.chain_pipeline && draft.spec_opt.chain_pipeline);
    auto & pipeline = *target.spec_opt.chain_pipeline;
    GGML_ASSERT(pipeline.verify_input && draft.spec_opt.chain_pipeline->chain &&
                draft.spec_opt.chain_pipeline->chain->prefetched);

    auto &    verify_input = *pipeline.verify_input;
    auto &    chain        = *draft.spec_opt.chain_pipeline->chain;
    const int n_draft      = (int) chain.tokens.size();
    GGML_ASSERT(n_draft > 0 && batch.n_tokens == n_draft + 1);
    GGML_ASSERT((int) verify_input.draft_rows.size() == n_draft);

    pipeline.verify_copy = {draft.backend_ptrs[0], target.backend_ptrs[0], chain.tokens, verify_input.draft_rows, true};

    int32_t result = llama_spec_opt::decode_with_copy(target, batch, pipeline.verify_copy);

    const bool copy_submitted = pipeline.verify_copy.submitted;
    pipeline.verify_copy      = {};
    // after verify launch: draft->target token copy is queued, so D2H no longer delays verify inputs
    draft.spec_opt.finish();
    if (draft_tokens) {
        const llama_token * sampled = draft.get_sampled_tokens();
        GGML_ASSERT(sampled);
        std::memcpy(draft_tokens, sampled, (size_t) n_draft * sizeof(llama_token));
    }
    if (result == 0 && !copy_submitted) {
        LLAMA_LOG_ERROR("%s: target token copy was not submitted\n", __func__);
        result = -3;  // llama_decode maps GGML_STATUS_FAILED to -3
    }
    return result;
}

bool llama_spec_chain_pipeline::commit_chain_kv_metadata(llama_context & ctx, llama_pos seed_pos) {
    if (!ctx.spec_opt.chain_pipeline) {
        return false;
    }
    auto & pipeline = *ctx.spec_opt.chain_pipeline;
    if (!pipeline.chain || !pipeline.chain->prefetched || !pipeline.input) {
        return false;
    }

    return prepare_kv_metadata_rows(ctx, pipeline.input->seq, seed_pos, (int32_t) pipeline.chain->params.size()) !=
           nullptr;
}

bool llama_spec_submit_draft_generation(llama_context * draft,
                                        llama_context * target,
                                        llama_batch     batch,
                                        const float **  eagle3_g_out) {
    return llama_spec_chain_pipeline::submit_draft_generation(*draft, *target, batch, eagle3_g_out);
}

bool llama_spec_prepare_chain(llama_context * target, llama_context * draft, int32_t n_draft) {
    return llama_spec_chain_pipeline::prepare(*target, *draft, n_draft);
}

int32_t llama_spec_submit_target_verification(llama_context * target,
                                              llama_context * draft,
                                              llama_batch     batch,
                                              llama_token *   draft_tokens) {
    return llama_spec_chain_pipeline::submit_target_verification(*target, *draft, batch, draft_tokens);
}

bool llama_spec_commit_chain_kv_metadata(llama_context * ctx, llama_pos seed_pos) {
    return llama_spec_chain_pipeline::commit_chain_kv_metadata(*ctx, seed_pos);
}

void llama_spec_synchronize_eagle3_g(llama_context * draft) {
    llama_spec_chain_pipeline::synchronize_eagle3_g(*draft);
}
