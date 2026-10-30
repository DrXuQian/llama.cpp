#pragma once

#include "llama-spec-opt.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

#include <memory>
#include <vector>

class llama_kv_cache_context;

// n_draft decoder steps in one graph, each step feeds its token and hidden state to the next
struct llama_spec_draft_chain {
    ggml_backend_sched_ptr        sched;
    llm_graph_result_ptr          res;
    std::vector<llm_graph_params> params;
    std::vector<ggml_tensor *>    tokens;
    // the chain is launched and its tokens are not consumed yet
    bool                          prefetched = false;
};

// device inputs of the draft chain, draft side
struct llama_spec_chain_input {
    ggml_context_ptr           ctx;
    ggml_backend_buffer_ptr    device;
    // target samples of the verify batch, copied D2D row by row
    ggml_tensor *              sampled = nullptr;
    std::vector<ggml_tensor *> sampled_rows;
    // draft tokens of the verify batch, with LLAMA_TOKEN_NULL at the end
    ggml_tensor *              draft_tokens   = nullptr;
    ggml_tensor *              accept_weights = nullptr;
    ggml_tensor *              positions      = nullptr;
    // host sources of the async uploads, kept alive until the chain is done
    std::vector<llama_token>   draft_tokens_host;
    std::vector<float>         positions_host;
    llama_seq_id               seq = -1;
};

// token input of the verify graph: row 0 is the seed from the host, rows 1..n_draft come from the chain
struct llama_spec_verify_input {
    ggml_context_ptr           ctx;
    ggml_backend_buffer_ptr    device;
    ggml_tensor *              tokens = nullptr;
    std::vector<ggml_tensor *> draft_rows;
};

// draft state update io: draft-side buffer for target features, plus the EAGLE3 g output and its host copy
struct llama_spec_draft_state_io {
    ggml_context_ptr           ctx;
    ggml_backend_buffer_ptr    device;
    std::vector<ggml_tensor *> features;
    // EAGLE3 only
    ggml_tensor *              eagle3_g = nullptr;
    ggml_backend_buffer_ptr    eagle3_g_host;
    ggml_backend_event_ptr     eagle3_g_ready;
    // MTP only: row 0 is the seed hidden from the host, rows 1.. are features[0]
    ggml_tensor *              mtp_h = nullptr;
};

struct llama_spec_chain_pipeline {
    void finish(llama_output_readback & readback);
    void reset(llama_output_readback & readback);
    bool can_defer_readback(const llama_batch & batch, uint32_t n_tokens, uint32_t n_outputs, uint32_t n_ubatch) const;
    void set_graph_params(llm_graph_params & params) const;
    void bind_inputs(llm_graph_result * res) const;
    void memory_breakdown(llama_memory_breakdown &              ret,
                          ggml_backend_sched_t                  sched,
                          const std::vector<ggml_backend_ptr> & backends) const;

    static bool    prepare(llama_context & target, llama_context & draft, int32_t n_draft);
    static bool    submit_draft_generation(llama_context &     draft,
                                           llama_context &     target,
                                           const llama_batch & batch,
                                           const float **      eagle3_g_out);
    static int32_t submit_target_verification(llama_context &     target,
                                              llama_context &     draft,
                                              const llama_batch & batch,
                                              llama_token *       draft_tokens);
    static bool    commit_chain_kv_metadata(llama_context & ctx, llama_pos seed_pos);
    static void    synchronize_eagle3_g(llama_context & draft);

private:
    enum class pipeline_type {
        none,
        eagle3,
        mtp,
    };

    bool                                       is_eagle3 = false;
    std::unique_ptr<llama_spec_draft_chain>    chain;
    std::unique_ptr<llama_spec_chain_input>    input;
    std::unique_ptr<llama_spec_draft_state_io> draft_state_io;
    llama_deferred_copy                        draft_state_copy;
    std::unique_ptr<llama_spec_verify_input>   verify_input;
    llama_deferred_copy                        verify_copy;

    static pipeline_type supported_type(const llama_context & target, const llama_context & draft, int32_t n_draft);
    static bool          eagle3_supported(const llama_context & target, const llama_context & draft);
    static bool          mtp_supported(const llama_context & target, const llama_context & draft, int32_t n_draft);
    static llama_memory_context_ptr prepare_kv_metadata_rows(llama_context & ctx,
                                                             llama_seq_id    seq,
                                                             llama_pos       pos0,
                                                             int32_t         n);
    static bool                     submit_draft_state_update(llama_context &     draft,
                                                              llama_context &     target,
                                                              const llama_batch & batch,
                                                              const float **      eagle3_g_out);
    static bool readback_eagle3_g(llama_context & draft, int n_features, const float ** eagle3_g_out);
    static bool submit_draft_chain(llama_context & draft, llama_context & target, const llama_batch & state_batch);
    static void upload_chain_input(llama_context & draft, llama_context & target, const llama_batch & batch);
    static bool build_chain(llama_context & draft, llama_kv_cache_context * kv_ctx, int32_t n_draft);
};
