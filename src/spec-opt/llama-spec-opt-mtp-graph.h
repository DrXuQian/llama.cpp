#pragma once

#include "llama-spec-opt-graph.h"

struct ggml_tensor;
struct llm_graph_context;
class llm_graph_input_embd_h;

struct llama_mtp_graph {
    static void bind_token(const llama_spec_graph_params & inputs, llm_graph_input_embd_h & inp);
    static void bind_hidden(llm_graph_context &             graph,
                            const llama_spec_graph_params & inputs,
                            llm_graph_input_embd_h &        inp);

    static ggml_tensor * build_position(llm_graph_context & graph, const llama_spec_graph_params & inputs);
    static ggml_tensor * build_output_ids(llm_graph_context & graph, const llama_spec_graph_params & inputs);

    static void select_output_rows(llm_graph_context &             graph,
                                   const llama_spec_graph_params & inputs,
                                   ggml_tensor *&                  inp_out_ids,
                                   ggml_tensor *&                  cur,
                                   ggml_tensor *&                  residual);
};
