#pragma once

#include "llama-spec-opt-graph.h"

struct ggml_tensor;
struct llama_model;
struct llm_graph_context;
class llm_graph_input_embd;

struct llama_eagle3_graph {
    static void bind_inputs(llm_graph_context &             graph,
                            const llama_model &             model,
                            const llama_spec_graph_params & inputs,
                            llm_graph_input_embd &          inp);

    static ggml_tensor * build_position(llm_graph_context & graph, const llama_spec_graph_params & inputs);

    static void select_output_rows(llm_graph_context &             graph,
                                   const llama_spec_graph_params & inputs,
                                   ggml_tensor *&                  cur,
                                   ggml_tensor *&                  residual);

    static constexpr const char * draft_state_output_name = "g_draft_state";
};
