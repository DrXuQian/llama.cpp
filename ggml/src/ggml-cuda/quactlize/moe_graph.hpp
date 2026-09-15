#pragma once
#include "ggml.h"
#include "ggml-impl.h"
#include <vector>

namespace quactlize::llama {
struct MoeGraph {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    int count = 0;
    ggml_tensor * weights = nullptr;
    ggml_tensor * finish = nullptr;
};

inline MoeGraph extend_moe_finish(ggml_cgraph const * graph,int start,MoeGraph base) {
    auto * down=base.down;
    if (!down || down->ne[1]!=8 || down->ne[2]<1 || down->ne[2]>8) return base;
    int next=start+base.count;
    if (next+16>graph->n_nodes) return base;
    auto * mul=graph->nodes[next];
    if (mul->op!=GGML_OP_MUL || mul->src[0]!=down || !ggml_is_contiguous(mul)) return base;
    auto * weights=mul->src[1];
    if (!weights || weights->type!=GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
        weights->ne[0]!=1 || weights->ne[1]!=8 || weights->ne[2]!=down->ne[2] || weights->ne[3]!=1)
        return base;
    ggml_tensor * views[8];
    for (int slot=0;slot<8;++slot) {
        auto * view=graph->nodes[next+1+slot];views[slot]=view;
        if (view->op!=GGML_OP_VIEW || view->type!=GGML_TYPE_F32 || view->view_src!=mul ||
            view->view_offs!=size_t(slot)*mul->nb[1] || view->ne[0]!=down->ne[0] ||
            view->ne[1]!=down->ne[2] || view->ne[2]!=1 || view->ne[3]!=1 ||
            view->nb[0]!=sizeof(float) || view->nb[1]!=mul->nb[2]) return base;
    }
    auto * result=views[0];
    for (int slot=1;slot<8;++slot) {
        auto * add=graph->nodes[next+8+slot];
        if (add->op!=GGML_OP_ADD || add->src[0]!=result || add->src[1]!=views[slot] ||
            add->type!=GGML_TYPE_F32 || !ggml_is_contiguous(add)) return base;
        result=add;
    }
    int count=base.count+16,output=start+count-1;
    std::vector<ggml_op> ops;
    for (int i=0;i<count;++i) ops.push_back(graph->nodes[start+i]->op);
    if (!ggml_can_fuse_subgraph(graph,start,count,ops.data(),&output,1)) return base;
    base.count=count;base.weights=weights;base.finish=result;
    return base;
}

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
    if (!ids || !input || ids->ne[0] <= 0 || ids->ne[1] <= 0 ||
        ids->ne[0] > (ids->ne[1]<=8 ? 64 : 32)/ids->ne[1] ||
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
    return extend_moe_finish(graph,start,match);
}

struct MoeRouterSpan {
    int chain_start = 0;
    int count = 0;
};

inline MoeRouterSpan match_moe_router(ggml_cgraph const * graph, int start,
    std::vector<ggml_op> ops, int ids_index, int weights_index) {
    if (!graph || start < 0 || ops.empty() || start + int(ops.size()) > graph->n_nodes ||
        ids_index < start || ids_index >= start + int(ops.size()) ||
        weights_index < start || weights_index >= start + int(ops.size())) return {};
    std::vector<int> outputs{ids_index, weights_index};
    int next = start + int(ops.size());
    while (next < graph->n_nodes && (graph->nodes[next]->op == GGML_OP_VIEW ||
                                    graph->nodes[next]->op == GGML_OP_RESHAPE)) {
        // Input views remain aliases, not elided intermediate storage. Their
        // parent may be outside the fused span and have other consumers.
        outputs.push_back(next);
        ops.push_back(graph->nodes[next++]->op);
    }
    auto chain = match_moe(graph, next);
    if (!chain.count || chain.gate->src[2] != graph->nodes[ids_index]) return {};
    for (int j = 0; j < chain.count; ++j) ops.push_back(graph->nodes[next+j]->op);
    outputs.push_back(next + chain.count - 1);
    if (!ggml_can_fuse_subgraph(graph, start, int(ops.size()), ops.data(), outputs.data(), int(outputs.size()))) return {};
    return {next, int(ops.size())};
}
} // namespace quactlize::llama
