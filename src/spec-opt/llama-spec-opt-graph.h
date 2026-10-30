#pragma once

#include <vector>

struct ggml_tensor;
struct llm_graph_context;
class llm_graph_input_attn_kv;

// Graph tensors used in place of host inputs, null otherwise.
struct llama_spec_graph_params {
    // Draft chain step: token, hidden state and position come from the previous step.
    ggml_tensor * tokens    = nullptr;
    ggml_tensor * hidden    = nullptr;
    ggml_tensor * positions = nullptr;

    // Verify graph token input, only checked for graph reuse.
    ggml_tensor * verify_tokens = nullptr;

    // Draft state update: target hidden states copied draft-side, mtp puts the seed hidden in row 0.
    std::vector<ggml_tensor *> features;

    bool has_decoder_inputs() const { return tokens || !features.empty(); }

    bool is_chain() const { return tokens != nullptr; }

    bool allow_reuse(const llama_spec_graph_params & other) const {
        return verify_tokens == other.verify_tokens && features == other.features;
    }
};

struct llama_spec_chain_graph {
    // Chain step: position and KQ mask computed from the step position tensor.
    static void build_attn_input(llm_graph_context &             graph,
                                 const llama_spec_graph_params & inputs,
                                 llm_graph_input_attn_kv *       inp_attn);
};
