#pragma once
#include "../ggml-impl.h"
#include <array>

struct ggml_quactlize_ffn_graph {
    ggml_tensor * gate    = nullptr;
    ggml_tensor * up      = nullptr;
    ggml_tensor * down    = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * output  = nullptr;
    int           count   = 0;
};

// Graph allocation can reuse A/IDs/routing storage for the final output.
// Such bindings require the library's lifetime-aware execution entry.
inline bool ggml_quactlize_ffn_input_aliases(const ggml_quactlize_ffn_graph & match, uint32_t & mask) {
    mask = 0;
    if (!match.gate || !match.output || !match.output->data) {
        return false;
    }
    const uintptr_t out  = reinterpret_cast<uintptr_t>(match.output->data);
    const size_t    size = ggml_nbytes(match.output);
    if (size > UINTPTR_MAX - out) {
        return false;
    }
    const std::array<const ggml_tensor *, 3> inputs = {match.gate->src[1], match.gate->src[2], match.weights};
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto * input = inputs[i];
        if (!input) {
            continue;
        }
        const uintptr_t address = reinterpret_cast<uintptr_t>(input->data);
        const size_t    bytes   = ggml_nbytes(input);
        if (!address || bytes > UINTPTR_MAX - address) {
            return false;
        }
        if (address < out + size && out < address + bytes) {
            mask |= 1u << i;
        }
    }
    return true;
}

// Match only ordinary SwiGLU and the original ordered expert sum. Bias,
// clamps, adapters and collectives interrupt the match and remain unfused.
inline ggml_quactlize_ffn_graph ggml_quactlize_match_ffn(const ggml_cgraph * graph, int start) {
    ggml_quactlize_ffn_graph result;
    if (!graph || start < 0 || start >= graph->n_nodes) {
        return {};
    }
    auto *     first = graph->nodes[start];
    const auto op    = first->op;
    if ((op != GGML_OP_MUL_MAT && op != GGML_OP_MUL_MAT_ID) || ggml_is_empty(first)) {
        return {};
    }
    int  at   = start + 1;
    auto take = [&](ggml_op expected) -> ggml_tensor * {
        return at < graph->n_nodes && graph->nodes[at]->op == expected ? graph->nodes[at++] : nullptr;
    };
    ggml_tensor * glu = nullptr;
    if (auto * second = take(op)) {
        glu = take(GGML_OP_GLU);
        if (!glu ||
            !((glu->src[0] == first && glu->src[1] == second) || (glu->src[0] == second && glu->src[1] == first))) {
            return {};
        }
        result.gate = glu->src[0];
        result.up   = glu->src[1];
        if (result.gate->src[1] != result.up->src[1] || result.gate->src[2] != result.up->src[2]) {
            return {};
        }
    } else {
        result.gate = first;
        if (auto * gate = take(GGML_OP_VIEW)) {
            auto * up = take(GGML_OP_VIEW);
            glu       = take(GGML_OP_GLU);
            if (!up || !glu || glu->src[0] != gate || glu->src[1] != up || gate->view_src != first ||
                up->view_src != first || gate->src[0] != first || up->src[0] != first || gate->view_offs != 0 ||
                first->ne[0] != 2 * gate->ne[0] || up->view_offs != gate->ne[0] * first->nb[0] ||
                !ggml_are_same_layout(gate, up)) {
                return {};
            }
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                if (gate->nb[dim] != first->nb[dim] || (dim && gate->ne[dim] != first->ne[dim])) {
                    return {};
                }
            }
        } else {
            glu = take(GGML_OP_GLU);
            if (!glu || glu->src[0] != first || glu->src[1]) {
                return {};
            }
        }
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return {};
    }
    result.down = take(op);
    if (!result.down || result.down->src[1] != glu || result.down->src[2] != first->src[2]) {
        return {};
    }
    result.output = result.down;
    if (op == GGML_OP_MUL_MAT_ID) {
        auto * weighted = take(GGML_OP_MUL);
        if (!weighted || !ggml_are_same_shape(weighted, result.down)) {
            return {};
        }
        result.weights     = weighted->src[0] == result.down ? weighted->src[1] :
                             weighted->src[1] == result.down ? weighted->src[0] :
                                                               nullptr;
        const int64_t topk = result.down->ne[1], tokens = result.down->ne[2];
        auto *        w = result.weights;
        if (!w || w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != topk || w->ne[2] != tokens ||
            w->ne[3] != 1 || w->nb[1] != sizeof(float) || w->nb[2] % sizeof(float) || topk < 1 || topk > 32) {
            return {};
        }
        std::array<ggml_tensor *, 32> views{};
        for (int slot = 0; slot < topk; ++slot) {
            auto * view = take(GGML_OP_VIEW);
            if (!view || view->view_src != weighted || view->src[0] != weighted ||
                view->view_offs != slot * weighted->nb[1] || view->ne[0] != weighted->ne[0] || view->ne[1] != tokens ||
                view->ne[2] != 1 || view->ne[3] != 1 || view->nb[0] != sizeof(float) ||
                view->nb[1] != weighted->nb[2]) {
                return {};
            }
            views[slot] = view;
        }
        result.output = views[0];
        for (int slot = 1; slot < topk; ++slot) {
            auto * sum = take(GGML_OP_ADD);
            if (!sum || sum->src[0] != result.output || sum->src[1] != views[slot]) {
                return {};
            }
            result.output = sum;
        }
        if (topk == 1) {
            auto * cont = take(GGML_OP_CONT);
            if (!cont || cont->src[0] != result.output) {
                return {};
            }
            result.output = cont;
        }
    }
    if (result.output->type != GGML_TYPE_F32 || !ggml_is_contiguous(result.output)) {
        return {};
    }
    result.count = at - start;
    // The indexed form accepts the long view/add tail without the sequential
    // helper's 32-node limit; its external-use/output checks remain unchanged.
    std::array<int, 72>     indices{};
    std::array<ggml_op, 72> ops{};
    if (result.count > int(indices.size())) {
        return {};
    }
    for (int i = 0; i < result.count; ++i) {
        indices[i] = start + i;
        ops[i]     = graph->nodes[start + i]->op;
    }
    const int output = at - 1;
    if (!ggml_can_fuse_subgraph_ext(graph, indices.data(), result.count, ops.data(), &output, 1)) {
        return {};
    }
    return result;
}
