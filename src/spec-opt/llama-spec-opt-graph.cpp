#include "llama-spec-opt-graph.h"

#include "llama-graph.h"

void llama_spec_chain_graph::build_attn_input(llm_graph_context &             graph,
                                              const llama_spec_graph_params & inputs,
                                              llm_graph_input_attn_kv *       inp_attn) {
    if (!inputs.positions) {
        return;
    }

    auto * ctx0 = graph.ctx0;
    auto * pos  = inputs.positions;

    // all chain rows are applied up front, so attend only to the rows up to this position
    const float n_kv     = inp_attn->self_kq_mask->ne[0];
    auto *      position = ggml_cast(ctx0, pos, GGML_TYPE_F32);
    auto *      distance = ggml_sub(ctx0, ggml_arange(ctx0, 0, n_kv, 1), position);

    inp_attn->self_k_idxs      = pos;
    inp_attn->self_v_idxs      = pos;
    inp_attn->self_kq_mask     = ggml_cast(ctx0, ggml_scale(ctx0, ggml_step(ctx0, distance), -1e30f), GGML_TYPE_F16);
    inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    if (inp_attn->self_kv_used) {
        inp_attn->self_kv_used = ggml_cast(ctx0, ggml_scale_bias(ctx0, position, 1.0f, 1.0f), GGML_TYPE_I32);
    }
}
