#pragma once

#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-output.h"

#include <array>
#include <memory>
#include <vector>

struct llama_context;
struct llama_memory_breakdown_data;

struct llama_nextn_graph;
struct llama_nextn_handoff;
struct llama_nextn_lookahead;
struct llama_nextn_target;

struct llama_spec_pipeline {
    explicit llama_spec_pipeline(llama_context & context);
    ~llama_spec_pipeline();

    bool prefetch_nextn_target(llama_context & source);
    bool decode_nextn_verify(llama_context & source, const llama_batch & batch, int32_t * result);
    bool decode_nextn_catchup(llama_context & source, const llama_batch & batch, const float ** snapshot);
    void synchronize_nextn_catchup();
    bool decode_nextn_prefetch(llama_context & source, const llama_batch & batch, int n_draft, float p_min);
    bool decode_nextn(const llama_batch & seed, int32_t n_draft, bool prefetch, float p_min);
    int32_t nextn_draft_length();

private:
    friend struct llama_context;

    void wait_for_snapshots();
    ggml_backend_sched_t main_scheduler() const;
    void reset_targets(bool sampler_only);
    void reset_draft();
    void synchronize(bool outputs_only);
    void set_nextn_prefetch(bool enabled, bool fixed_kv);
    void set_nextn_graph_cache(int32_t n_max);
    void release_target();
    bool prepare_nextn_catchup(llama_context & source, llama_nextn_target & target, int n, bool target_rebuilt);
    llm_graph_result * consume_target(const llama_ubatch & ubatch, llama_memory_context_i * mctx);
    uint32_t graph_cache_limit() const { return graph_cache_max; }
    int recurrent_bank() const { return target_bank; }
    bool consuming_target() const { return target_consume; }
    void set_graph_inputs(llm_graph_params & gparams) const;
    void record_batch(const llama_ubatch & ubatch);
    void reset_outputs();
    void begin_decode();
    bool should_defer_outputs(const llama_ubatch & ubatch, uint32_t n_tokens) const;
    void stage_outputs(llama_output_copies copies, llama_memory_context_ptr & mctx, bool defer);
    void invalidate_graphs();
    void invalidate_target() { target_pending = -1; }
    void add_memory_usage(std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> & ret) const;

    llama_context & lctx;
    uint32_t graph_cache_max = 0;

    std::unique_ptr<llama_nextn_graph> graph;
    std::array<std::unique_ptr<llama_nextn_graph>, 10> graph_cache;
    uint32_t graph_active = 0;
    std::unique_ptr<llama_nextn_handoff> handoff;
    std::unique_ptr<llama_nextn_handoff> features;
    std::array<ggml_tensor *, 3> catchup_features = {};
    ggml_tensor * catchup_input = nullptr;
    std::unique_ptr<llama_nextn_handoff> verify;
    ggml_tensor * verify_input = nullptr;
    bool outputs = false;
    llama_ubatch last_ubatch = {};
    std::unique_ptr<llama_output_copies> readback;
    std::unique_ptr<llama_nextn_lookahead> lookahead;
    std::array<std::unique_ptr<llama_nextn_target>, 20> targets;
    llama_memory_context_ptr mctx;
    ggml_backend_event_ptr output_ready;
    bool output_pending = false;
    int target_active = -1;
    int target_bank = 0;
    int target_pending = -1;
    bool target_consume = false;
    bool prefetch_enabled = true;
    bool prefetch_reported = false;
};
