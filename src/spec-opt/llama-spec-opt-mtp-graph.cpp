#include "llama-spec-opt-mtp-graph.h"

#include "llama-graph.h"

void llama_mtp_graph::bind_token(const llama_spec_graph_params & inputs, llm_graph_input_embd_h & inp) {
    if (inputs.is_chain()) {
        inp.tokens = inputs.tokens;
    }
}

void llama_mtp_graph::bind_hidden(llm_graph_context &             graph,
                                  const llama_spec_graph_params & inputs,
                                  llm_graph_input_embd_h &        inp) {
    if (inputs.is_chain()) {
        inp.h = inputs.hidden;
    } else if (!inputs.features.empty()) {
        // draft state update: row 0 is the seed hidden, the next rows are the target hidden states
        auto * h = inputs.features[0];
        inp.h    = ggml_view_2d(graph.ctx0, h, graph.n_embd, graph.n_tokens, h->nb[1], 0);
    }
}

ggml_tensor * llama_mtp_graph::build_position(llm_graph_context & graph, const llama_spec_graph_params & inputs) {
    if (!inputs.positions) {
        return graph.build_inp_pos();
    }

    // text tokens use [p, p, p, 0], see llm_graph_input_pos::set_input
    auto *        ctx0 = graph.ctx0;
    ggml_tensor * pos  = inputs.positions;
    ggml_tensor * zero = ggml_cast(ctx0, ggml_scale(ctx0, ggml_cast(ctx0, pos, GGML_TYPE_F32), 0.0f), GGML_TYPE_I32);

    ggml_tensor * res = pos;
    for (int i = 1; i < 3; ++i) {
        res = ggml_concat(ctx0, res, pos, 0);
    }
    return ggml_concat(ctx0, res, zero, 0);
}

ggml_tensor * llama_mtp_graph::build_output_ids(llm_graph_context & graph, const llama_spec_graph_params & inputs) {
    return inputs.is_chain() ? nullptr : graph.build_inp_out_ids();
}

// draft state update only needs the kv, so keep only the output rows after attention
void llama_mtp_graph::select_output_rows(llm_graph_context &             graph,
                                         const llama_spec_graph_params & inputs,
                                         ggml_tensor *&                  inp_out_ids,
                                         ggml_tensor *&                  cur,
                                         ggml_tensor *&                  residual) {
    if (inputs.features.empty() || !inp_out_ids || !graph.cparams.embeddings_nextn_masked ||
        graph.n_outputs >= graph.n_tokens) {
        return;
    }
    cur         = ggml_get_rows(graph.ctx0, cur, inp_out_ids);
    residual    = ggml_get_rows(graph.ctx0, residual, inp_out_ids);
    inp_out_ids = nullptr;
}
