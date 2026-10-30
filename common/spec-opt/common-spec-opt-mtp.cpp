#include "common-spec-opt-mtp.h"

#include "log.h"

#include "../../src/spec-opt/llama-spec-opt-api.h"

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)

common_spec_opt_mtp::~common_spec_opt_mtp() {
    batch_update.embd = nullptr;  // borrowed from the mtp impl
}

void common_spec_opt_mtp::init(const common_params_speculative & params,
                               uint32_t                          n_seq,
                               common_sampler *                  smpl,
                               bool                              has_backend_chain,
                               std::vector<float> &              pending_h) {
    if (!init_pipeline(params, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, smpl, has_backend_chain)) {
        return;
    }

    batch_update      = llama_batch_init(n_max + 1, 0, 1);
    batch_update.embd = pending_h.data();
}

bool common_spec_opt_mtp::advance_pipeline(llama_context * ctx, llama_batch batch_tgt, int32_t & ret) {
    if (!enabled) {
        return false;
    }

    ret = decode_target(ctx, batch_tgt);
    consume_chain_tokens_on_host(batch_tgt);
    const bool built  = ret == 0 && build_draft_batch_from_host(batch_tgt);
    skip_next_process = built && submit_draft_generation();

    return true;
}

bool common_spec_opt_mtp::build_draft_batch_from_host(const llama_batch & batch_tgt) {
    const int32_t n_tokens = batch_tgt.n_tokens;
    if (n_tokens != n_max + 1 || !batch_tgt.token || batch_tgt.embd) {
        return false;
    }

    // F32 acceptance arithmetic requires exact positions.
    if (batch_tgt.pos[0] > (1 << 24) - 2 * n_tokens) {
        return false;
    }

    // same rows as the host process()
    common_batch_clear(batch_update);
    for (int32_t k = 0; k < n_tokens; ++k) {
        common_batch_add(batch_update, batch_tgt.token[k], batch_tgt.pos[k], {batch_tgt.seq_id[0][0]},
                         /*logits=*/false);
    }
    return true;
}

// mtp decodes the seed row again with the seed hidden in batch_update.embd, same as the host process()
bool common_spec_opt_mtp::submit_draft_generation() {
    if (!llama_spec_submit_draft_generation(ctx_dft, ctx_tgt, batch_update, nullptr)) {
        return false;
    }
    SPC_DBG("GPU MTP draft state update and next-draft prefetch: %d rows\n", batch_update.n_tokens);
    return true;
}
