#pragma once

#include "common-spec-opt.h"

struct common_spec_opt_eagle3 : common_spec_opt {
    // enables the pipeline when eagle3 is the only speculator, n_seq == 1 and the params are supported
    // the state refs are owned by the seq 0 eagle3 impl
    void init(const common_params_speculative & params,
              uint32_t                          n_seq,
              common_sampler *                  smpl,
              bool                              has_backend_chain,
              std::vector<float> &              verify_g,
              int32_t &                         verify_g_rows,
              llama_pos &                       verify_pos_first,
              llama_pos &                       pos_last);

    // also sets the verify state from the skipped batch
    bool consume_process_skip(const llama_batch & batch_tgt);

    // verify the current speculative tokens and queue the next generation, false if not enabled
    bool advance_pipeline(llama_context * ctx_tgt, llama_batch batch_tgt, int32_t & ret);

    // wait for the async g rows and copy them into verify_g, no-op if the impl process() used the host path
    void sync_pending_g_to_verify();

private:
    // state owned by the eagle3 impl
    std::vector<float> * verify_g         = nullptr;
    int32_t *            verify_g_rows    = nullptr;
    llama_pos *          verify_pos_first = nullptr;
    llama_pos *          pos_last         = nullptr;

    // g rows of the async draft state update in pinned memory, readable after sync_pending_g_to_verify()
    const float * pending_g = nullptr;

    // batch_update = batch_tgt[1..n_max]
    bool build_draft_batch_from_host(const llama_batch & batch_tgt);
    bool submit_draft_generation();
};
