#include "common-spec-opt-eagle3.h"

#include "log.h"

#include "../../src/spec-opt/llama-spec-opt-api.h"

#include <cstring>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)

void common_spec_opt_eagle3::init(const common_params_speculative & params,
                                  uint32_t                          n_seq,
                                  common_sampler *                  smpl,
                                  bool                              has_backend_chain,
                                  std::vector<float> &              verify_g,
                                  int32_t &                         verify_g_rows,
                                  llama_pos &                       verify_pos_first,
                                  llama_pos &                       pos_last) {
    if (!init_pipeline(params, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, smpl, has_backend_chain)) {
        return;
    }

    batch_update = llama_batch_init(n_max, 0, 1);

    this->verify_g         = &verify_g;
    this->verify_g_rows    = &verify_g_rows;
    this->verify_pos_first = &verify_pos_first;
    this->pos_last         = &pos_last;
}

bool common_spec_opt_eagle3::consume_process_skip(const llama_batch & batch_tgt) {
    const bool skip = common_spec_opt::consume_process_skip();
    if (skip) {
        const int32_t n_tokens = batch_tgt.n_tokens;
        verify_g->resize((size_t) n_tokens * llama_model_n_embd(llama_get_model(ctx_dft)));
        *verify_g_rows    = n_tokens;
        *verify_pos_first = batch_tgt.pos[0];
        *pos_last         = batch_tgt.pos[n_tokens - 1];
    }
    return skip;
}

bool common_spec_opt_eagle3::advance_pipeline(llama_context * ctx, llama_batch batch_tgt, int32_t & ret) {
    if (!enabled) {
        return false;
    }

    pending_g = nullptr;

    ret = decode_target(ctx, batch_tgt);
    consume_chain_tokens_on_host(batch_tgt);
    const bool built  = ret == 0 && build_draft_batch_from_host(batch_tgt);
    skip_next_process = built && submit_draft_generation();

    return true;
}

void common_spec_opt_eagle3::sync_pending_g_to_verify() {
    if (!pending_g) {
        return;
    }
    llama_spec_synchronize_eagle3_g(ctx_dft);
    std::memcpy(verify_g->data(), pending_g, verify_g->size() * sizeof(float));
    pending_g = nullptr;
}

bool common_spec_opt_eagle3::build_draft_batch_from_host(const llama_batch & batch_tgt) {
    const int32_t n_tokens = batch_tgt.n_tokens;
    if (n_tokens != n_max + 1 || !batch_tgt.token || batch_tgt.embd) {
        return false;
    }

    // F32 acceptance arithmetic requires exact positions.
    if (*pos_last + 1 != batch_tgt.pos[0] || batch_tgt.pos[0] > (1 << 24) - 2 * n_tokens) {
        return false;
    }

    // token k + 1 pairs with g at pos k
    common_batch_clear(batch_update);
    for (int32_t k = 0; k < n_tokens - 1; ++k) {
        common_batch_add(batch_update, batch_tgt.token[k + 1], batch_tgt.pos[k], {batch_tgt.seq_id[0][0]},
                         /*logits=*/false);
    }
    return true;
}

// the draft state update reuses the seed row that draft() left at the deferred boundary, so it only decodes batch_tgt[1..]
// its g output is kept only when the next chain is also prefetched
bool common_spec_opt_eagle3::submit_draft_generation() {
    const float * g = nullptr;
    if (!llama_spec_submit_draft_generation(ctx_dft, ctx_tgt, batch_update, &g)) {
        return false;
    }
    pending_g = g;
    SPC_DBG("GPU EAGLE3 async encoder, draft state update and next-draft prefetch: %d rows\n",
            batch_update.n_tokens + 1);
    return true;
}
