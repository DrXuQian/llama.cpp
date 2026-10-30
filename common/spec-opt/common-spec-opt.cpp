#include "common-spec-opt.h"

#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include "../../src/spec-opt/llama-spec-opt-api.h"

#include <algorithm>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)

common_spec_opt::~common_spec_opt() {
    llama_batch_free(batch_update);
}

bool common_spec_opt::init_pipeline(const common_params_speculative & params,
                                    common_speculative_type           type,
                                    uint32_t                          n_seq,
                                    common_sampler *                  smpl,
                                    bool                              has_backend_chain) {
    const auto & dparams   = params.draft;
    const bool   only_type = std::all_of(params.types.begin(), params.types.end(),
                                         [type](auto t) { return t == type || t == COMMON_SPECULATIVE_TYPE_NONE; });
    if (!only_type || n_seq != 1 || !dparams.backend_sampling || dparams.p_min != 0.0f || !has_backend_chain) {
        return false;
    }

    ctx_tgt = dparams.ctx_tgt;
    ctx_dft = dparams.ctx_dft;
    n_max   = dparams.n_max;

    enabled = llama_spec_prepare_chain(ctx_tgt, ctx_dft, n_max);
    SPC_TRC("GPU pipeline = %s\n", enabled ? "enabled" : "disabled");
    if (!enabled) {
        return false;
    }

    chain_tokens.resize(n_max);

    this->smpl = smpl;
    return true;
}

bool common_spec_opt::consume_process_skip() {
    const bool skip   = skip_next_process;
    skip_next_process = false;
    return skip;
}

bool common_spec_opt::prepare_host_kv_metadata(llama_seq_id, llama_pos seed_pos, common_speculative_draft_params & dp) {
    if (!enabled || dp.n_max < n_max) {
        return false;
    }
    GGML_ASSERT(dp.result->empty());
    if (!llama_spec_commit_chain_kv_metadata(ctx_dft, seed_pos)) {
        return false;
    }
    SPC_DBG("GPU draft chain: %d steps\n", n_max);
    dp.result->assign(n_max, 0);
    result = dp.result;
    return true;
}

int32_t common_spec_opt::decode_target(llama_context * ctx, const llama_batch & batch_tgt) {
    GGML_ASSERT(ctx == ctx_tgt);
    if (!result) {
        return llama_decode(ctx_tgt, batch_tgt);
    }
    const int32_t ret = llama_spec_submit_target_verification(ctx_tgt, ctx_dft, batch_tgt, chain_tokens.data());
    if (ret != 0) {
        LOG_ERR("%s: failed to decode, ret = %d\n", __func__, ret);
    }
    return ret;
}

void common_spec_opt::consume_chain_tokens_on_host(llama_batch & batch_tgt) {
    if (!result) {
        return;
    }
    // llama_spec_submit_target_verification() finished the chain and filled chain_tokens after one synchronization
    result->clear();
    for (const llama_token token : chain_tokens) {
        GGML_ASSERT(token != LLAMA_TOKEN_NULL);
        common_sampler_accept(smpl, token, true);
        result->push_back(token);
        batch_tgt.token[result->size()] = token;
    }
    result = nullptr;
}
