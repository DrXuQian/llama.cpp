#pragma once

// GPU speculative pipeline, common side: one seq, greedy draft chain on the backend

#include "llama.h"
#include "common.h"

struct common_sampler;
struct common_speculative_draft_params;

// shared part of common_spec_opt_eagle3 and common_spec_opt_mtp
struct common_spec_opt {
    ~common_spec_opt();

    bool consume_process_skip();

    // prepare host KV metadata for the prefetched draft; tokens remain placeholders until advance_pipeline() reads them
    bool prepare_host_kv_metadata(llama_seq_id seq_id, llama_pos seed_pos, common_speculative_draft_params & dp);

protected:
    llama_context *  ctx_tgt           = nullptr;
    llama_context *  ctx_dft           = nullptr;
    common_sampler * smpl              = nullptr;
    int32_t          n_max             = 0;
    bool             enabled           = false;
    bool             skip_next_process = false;

    // draft state update input, reused every round, allocated by the derived init
    llama_batch batch_update = {};

    // enables the pipeline when type is the only speculator, n_seq == 1 and the params are supported
    bool init_pipeline(const common_params_speculative & params,
                       common_speculative_type           type,
                       uint32_t                          n_seq,
                       common_sampler *                  smpl,
                       bool                              has_backend_chain);

    // verify the prepared chain tokens, or decode batch_tgt directly if no chain is prepared
    int32_t decode_target(llama_context * ctx, const llama_batch & batch_tgt);

    // accept the verified chain tokens and write them to batch_tgt[1..], no-op if no chain is prepared
    void consume_chain_tokens_on_host(llama_batch & batch_tgt);

private:
    // non-null while a chain is prepared as a draft, placeholders until advance_pipeline() reads the chain tokens
    llama_tokens * result = nullptr;

    // host destination of the deferred chain readback; one synchronization fills all rows
    llama_tokens chain_tokens;
};
