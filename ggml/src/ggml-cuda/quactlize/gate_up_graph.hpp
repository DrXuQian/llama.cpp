#pragma once
#include "ggml.h"
#include "ggml-impl.h"
#include <initializer_list>

namespace quactlize::llama {
struct SharedGateUpGraph {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * output = nullptr;
    int count = 0;
};

inline SharedGateUpGraph match_shared_gate_up(ggml_cgraph const * graph, int start) {
    if (!graph || start < 0 || start + 3 > graph->n_nodes) return {};
    auto * first = graph->nodes[start];
    auto * second = graph->nodes[start + 1];
    auto * glu = graph->nodes[start + 2];
    if (first->op != GGML_OP_MUL_MAT || second->op != GGML_OP_MUL_MAT ||
        glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) return {};
    auto * gate = glu->src[0];
    auto * up = glu->src[1];
    if (!((gate == first && up == second) || (gate == second && up == first)) ||
        gate->src[1] != up->src[1] || !ggml_are_same_shape(gate, up) ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) return {};
    for (auto * node : {gate, up}) {
        auto * weight = node->src[0];
        auto * input = node->src[1];
        if (!weight || !input || weight->type != GGML_TYPE_Q8_0 || weight->ne[0] != 2048 ||
            weight->ne[1] != 512 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            input->type != GGML_TYPE_F32 || input->ne[1] < 1 || input->ne[1] > 8 ||
            input->ne[2] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return {};
    }
    ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU};
    int output = start + 2;
    if (!ggml_can_fuse_subgraph(graph, start, 3, ops, &output, 1)) return {};
    return {gate, up, glu, 3};
}
} // namespace quactlize::llama
