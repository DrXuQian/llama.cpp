#pragma once

#include "common-spec-opt.h"

struct common_spec_opt_mtp : common_spec_opt {
    ~common_spec_opt_mtp();

    // enables the pipeline when mtp is the only speculator, n_seq == 1 and the params are supported
    // the state ref is owned by the seq 0 mtp impl
    void init(const common_params_speculative & params,
              uint32_t                          n_seq,
              common_sampler *                  smpl,
              bool                              has_backend_chain,
              std::vector<float> &              pending_h);

    // verify the current speculative tokens and queue the next generation, false if not enabled
    bool advance_pipeline(llama_context * ctx_tgt, llama_batch batch_tgt, int32_t & ret);

private:
    // batch_update = batch_tgt, its embd is the seed hidden row of the mtp impl, not owned
    bool build_draft_batch_from_host(const llama_batch & batch_tgt);
    bool submit_draft_generation();
};
