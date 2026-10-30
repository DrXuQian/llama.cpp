#include "llama-spec-opt-eagle3-graph.h"

#include "llama-graph.h"
#include "llama-model.h"

void llama_eagle3_graph::bind_inputs(llm_graph_context &             graph,
                                     const llama_model &             model,
                                     const llama_spec_graph_params & inputs,
                                     llm_graph_input_embd &          inp) {
    auto * ctx0 = graph.ctx0;

    if (inputs.tokens) {
        inp.tokens = inputs.tokens;
        inp.embd   = inputs.hidden;
        return;
    }

    ggml_tensor * features = inputs.features[0];
    for (size_t i = 1; i < inputs.features.size(); ++i) {
        features = ggml_concat(ctx0, features, inputs.features[i], 0);
    }
    if (graph.hparams.norm_before_fc) {
        features = graph.build_norm(features, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    }
    auto * encoded = graph.build_lora_mm(model.fc, features);
    ggml_set_output(encoded);
    ggml_set_name(encoded, draft_state_output_name);

    // the encoder has one row more than the decoder, see submit_draft_state_update
    inp.embd = ggml_view_2d(ctx0, encoded, graph.n_embd, graph.n_tokens, encoded->nb[1], 0);
}

ggml_tensor * llama_eagle3_graph::build_position(llm_graph_context & graph, const llama_spec_graph_params & inputs) {
    return inputs.positions ? inputs.positions : graph.build_inp_pos();
}

void llama_eagle3_graph::select_output_rows(llm_graph_context &             graph,
                                            const llama_spec_graph_params & inputs,
                                            ggml_tensor *&                  cur,
                                            ggml_tensor *&                  residual) {
    if (!inputs.has_decoder_inputs() || !graph.cparams.embeddings_nextn_masked || graph.n_outputs >= graph.n_tokens) {
        return;
    }
    ggml_tensor * inp_out_ids = graph.build_inp_out_ids();

    cur      = ggml_get_rows(graph.ctx0, cur, inp_out_ids);
    residual = ggml_get_rows(graph.ctx0, residual, inp_out_ids);
}
