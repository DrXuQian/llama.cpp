#pragma once
#include "ggml.h"
#include "ggml-impl.h"

namespace quactlize::llama {
struct MoeGraph {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    int count = 0;
};

inline MoeGraph match_moe(ggml_cgraph const * graph, int start) {
    if (!graph || start < 0 || start + 4 > graph->n_nodes) return {};
    ggml_tensor * first = graph->nodes[start];
    if (first->op != GGML_OP_MUL_MAT_ID) return {};
    MoeGraph match;
    ggml_tensor * glu = nullptr;
    if (graph->nodes[start + 1]->op == GGML_OP_MUL_MAT_ID &&
        graph->nodes[start + 2]->op == GGML_OP_GLU) {
        glu = graph->nodes[start + 2];
        match = {glu->src[0], glu->src[1], graph->nodes[start + 3], 4};
        if (!((match.gate == first && match.up == graph->nodes[start + 1]) ||
              (match.up == first && match.gate == graph->nodes[start + 1]))) return {};
        if (match.gate->src[1] != match.up->src[1] || match.gate->src[2] != match.up->src[2] ||
            !ggml_are_same_shape(match.gate,match.up)) return {};
    } else if (start + 5 <= graph->n_nodes && graph->nodes[start + 1]->op == GGML_OP_VIEW &&
               graph->nodes[start + 2]->op == GGML_OP_VIEW && graph->nodes[start + 3]->op == GGML_OP_GLU) {
        glu = graph->nodes[start + 3];
        auto * gate = glu->src[0]; auto * up = glu->src[1];
        if (!gate || !up || !((gate == graph->nodes[start + 1] && up == graph->nodes[start + 2]) ||
            (gate == graph->nodes[start + 2] && up == graph->nodes[start + 1])) ||
            gate->view_src != first || up->view_src != first || gate->view_offs != 0 ||
            up->view_offs != size_t(gate->ne[0])*sizeof(float) || first->ne[0] != 2*gate->ne[0] ||
            !ggml_are_same_shape(gate,up) || gate->nb[1] != first->nb[1] || up->nb[1] != first->nb[1] ||
            gate->nb[2] != first->nb[2] || up->nb[2] != first->nb[2]) return {};
        match = {first,nullptr,graph->nodes[start + 4],5};
    } else return {};
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->type != GGML_TYPE_F32 ||
        match.down->op != GGML_OP_MUL_MAT_ID || match.down->src[1] != glu ||
        match.down->src[2] != match.gate->src[2] || !ggml_is_contiguous(glu)) return {};
    auto * ids = match.gate->src[2]; auto * input = match.gate->src[1];
    if (!ids || !input || ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[0] > 32/ids->ne[1] ||
        input->ne[2] != ids->ne[1] || input->ne[3] != 1) return {};
    for (auto * node : {match.gate,match.up,match.down}) {
        if (!node) continue;
        if (!node->src[0] || node->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(node) || !ggml_is_contiguous(node->src[1]) ||
            node->src[0]->ne[2] != match.gate->src[0]->ne[2] || node->src[0]->ne[2] > 1024) return {};
    }
    ggml_op ops[5];
    for (int j=0;j<match.count;++j) ops[j]=graph->nodes[start+j]->op;
    int output = start+match.count-1;
    // Views and intermediate projections must have no outside consumers.
    if (!ggml_can_fuse_subgraph(graph,start,match.count,ops,&output,1)) return {};
    return match;
}
} // namespace quactlize::llama
