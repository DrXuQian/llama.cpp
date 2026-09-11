#pragma once

#include "llama-graph.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <array>
#include <vector>

// A primary chain with side leaves, pruned to a fixed number of ancestor-closed rows.
struct llama_spec_tree {
    int n_rows = 4;
    int n_candidates = 8;
    int n_branches = 2;
    ggml_context_ptr constants;
    ggml_backend_buffer_ptr device;
    ggml_backend_buffer_ptr host;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * scores = nullptr;
    ggml_tensor * topologies = nullptr;
    ggml_tensor * memberships = nullptr;
    ggml_tensor * ties = nullptr;
    ggml_tensor * depths = nullptr;
    ggml_tensor * ancestors = nullptr;
    ggml_tensor * paths = nullptr;
    ggml_tensor * parents = nullptr;
    ggml_tensor * original = nullptr;
    ggml_tensor * membership = nullptr;
    llm_graph_nextn_target linear = {};

    bool init(ggml_backend_t backend, int n = 4) {
        GGML_ASSERT(n == 4 || n == 8);
        n_rows = n;
        n_branches = n == 4 ? 2 : n - 1;
        n_candidates = n == 4 ? 8 : 1 + (n - 1)*n_branches;
        constants.reset(ggml_init({ 8*ggml_tensor_overhead(), nullptr, true }));
        auto * ctx = constants.get();
        tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_candidates);
        scores = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_candidates);
        if (n_rows == 4) {
            topologies = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 4);
            memberships = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        } else {
            ties = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_candidates, n_candidates);
        }
        depths = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_candidates);
        ancestors = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_candidates, n_candidates + 2);
        paths = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_rows, n_candidates);
        parents = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_candidates);
        device.reset(ggml_backend_alloc_ctx_tensors(ctx, backend));
        host.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend)), n_rows*sizeof(llama_token)));
        if (!device || !host) {
            return false;
        }
        std::vector<int32_t> parent(n_candidates, 0);
        std::vector<float> depth(n_candidates, 0);
        for (int q = 1; q < n_candidates; ++q) {
            const int d = n_rows == 4 && q == 7 ? 0 : (q - 1)/n_branches;
            parent[q] = d ? 1 + (d - 1)*n_branches : 0;
            depth[q] = d + 1;
        }
        if (n_rows == 4) {
            const int32_t topology[] = {0, 1, 2, 3, 0, 1, 3, 4, 0, 1, 3, 5, 0, 1, 2, 7};
            std::array<float, 32> members = {};
            for (int t = 0; t < 4; ++t) {
                for (int q = 0; q < 4; ++q) {
                    members[t*8 + topology[t*4 + q]] = 1.0f;
                }
            }
            ggml_backend_tensor_set(topologies, topology, 0, sizeof(topology));
            ggml_backend_tensor_set(memberships, members.data(), 0, sizeof(members));
        } else {
            std::vector<float> priority(n_candidates*n_candidates, 0);
            for (int q = 0; q < n_candidates; ++q) {
                std::fill_n(priority.data() + q*n_candidates, q, 1.0f);
            }
            ggml_backend_tensor_set(ties, priority.data(), 0, priority.size()*sizeof(float));
        }
        std::vector<float> ancestor(n_candidates*(n_candidates + 2), -1e30f);
        std::fill_n(ancestor.data(), n_candidates, 0.0f);
        std::vector<int32_t> path(n_rows*n_candidates);
        for (int q = 0; q < n_candidates; ++q) {
            std::vector<int> branch;
            for (int p = q;; p = parent[p]) {
                branch.push_back(p);
                ancestor[(p + 1)*n_candidates + q] = 0.0f;
                if (!p) {
                    break;
                }
            }
            std::reverse(branch.begin(), branch.end());
            for (int i = 0; i < n_rows; ++i) {
                path[q*n_rows + i] = branch[std::min(i, (int) depth[q])];
            }
        }
        const float root_score = 1.0f;
        ggml_backend_tensor_set(scores, &root_score, 0, sizeof(root_score));
        ggml_backend_tensor_set(depths, depth.data(), 0, depth.size()*sizeof(float));
        ggml_backend_tensor_set(ancestors, ancestor.data(), 0, ancestor.size()*sizeof(float));
        ggml_backend_tensor_set(paths, path.data(), 0, path.size()*sizeof(int32_t));
        ggml_backend_tensor_set(parents, parent.data(), 0, parent.size()*sizeof(int32_t));
        return true;
    }

    static ggml_tensor * flat(ggml_context * ctx, ggml_tensor * t) {
        return ggml_reshape_1d(ctx, t, ggml_nelements(t));
    }

    static ggml_tensor * at(ggml_context * ctx, ggml_tensor * t, ggml_tensor * ids) {
        return flat(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, t, 1, ggml_nelements(t)), ids));
    }

    static ggml_tensor * row(ggml_context * ctx, ggml_tensor * t, int i) {
        return ggml_view_1d(ctx, t, 1, i*ggml_element_size(t));
    }

    static ggml_tensor * join(ggml_context * ctx, const std::vector<ggml_tensor *> & rows) {
        auto * result = flat(ctx, rows[0]);
        for (size_t i = 1; i < rows.size(); ++i) {
            result = ggml_concat(ctx, result, flat(ctx, rows[i]), 0);
        }
        return result;
    }

    static ggml_tensor * probabilities(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * ids, ggml_tensor * maximum) {
        // Normalize over the vocabulary without a cooperative softmax launch in the composed graph.
        int64_t width = 256;
        while (logits->ne[0] % width) {
            width /= 2;
        }
        auto * weights = ggml_exp(ctx, ggml_sub(ctx, logits, maximum));
        auto * partial = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, weights, width, logits->ne[0]/width));
        auto * total = ggml_sum_rows(ctx, flat(ctx, partial));
        return ggml_div(ctx, at(ctx, weights, ids), total);
    }

    void build_inputs(llm_graph_result & res, llm_graph_nextn_target & meta, int n_kv) {
        auto * ctx = res.get_ctx();
        auto * gf = res.get_gf();
        linear = meta;
        for (auto * t : {linear.positions, linear.kv_idxs, linear.mask}) {
            ggml_build_forward_expand(gf, t);
        }
        if (n_rows == 4) {
            auto * a = row(ctx, scores, 2);
            auto * b = row(ctx, scores, 4);
            auto * c = row(ctx, scores, 5);
            // Strict comparisons preserve the lower original ID on probability ties.
            auto * ba = ggml_step(ctx, ggml_sub(ctx, b, a));
            auto * ca = ggml_step(ctx, ggml_sub(ctx, c, a));
            auto * cb = ggml_step(ctx, ggml_sub(ctx, c, b));
            auto * choice = ggml_add(ctx, ba,
                    ggml_mul(ctx, ggml_mul(ctx, ca, cb), ggml_scale_bias(ctx, ba, -1, 2)));
            // Three root children win only when the third child beats the second primary node.
            auto * wide = ggml_step(ctx, ggml_sub(ctx, row(ctx, scores, 7), row(ctx, scores, 3)));
            choice = ggml_cast(ctx, ggml_add(ctx, ggml_mul(ctx, choice, ggml_scale_bias(ctx, wide, -1, 1)),
                    ggml_scale(ctx, wide, 3)), GGML_TYPE_I32);
            original = flat(ctx, ggml_get_rows(ctx, topologies, choice));
            membership = flat(ctx, ggml_get_rows(ctx, memberships, choice));
        } else {
            // Cumulative scores are monotone. Exact ranks keep ancestors first on ties and underflow.
            auto * delta = ggml_sub(ctx, ggml_repeat_4d(ctx, scores, n_candidates, n_candidates, 1, 1),
                    ggml_reshape_2d(ctx, scores, 1, n_candidates));
            auto * equal = ggml_scale_bias(ctx, ggml_step(ctx, ggml_abs(ctx, delta)), -1, 1);
            auto * rank = flat(ctx, ggml_sum_rows(ctx, ggml_add(ctx, ggml_step(ctx, delta), ggml_mul(ctx, equal, ties))));
            membership = ggml_step(ctx, ggml_scale_bias(ctx, rank, -1, n_rows - 0.5f));
            auto * order = ggml_add(ctx, ggml_arange(ctx, 0, n_candidates, 1),
                    ggml_scale_bias(ctx, membership, -n_candidates, n_candidates));
            original = ggml_view_1d(ctx, ggml_argsort(ctx, order, GGML_SORT_ORDER_ASC), n_rows, 0);
        }
        meta.tokens = at(ctx, tokens, original);
        auto * base = ggml_cast(ctx, row(ctx, linear.positions, 0), GGML_TYPE_F32);
        meta.positions = ggml_cast(ctx, ggml_add(ctx, at(ctx, depths, original), base), GGML_TYPE_I32);
        meta.kv_idxs = ggml_cast(ctx, ggml_add(ctx, ggml_arange(ctx, 0, n_rows, 1), base), GGML_TYPE_I64);
        auto * query = ggml_cont(ctx, ggml_transpose(ctx, ggml_get_rows(ctx,
                ggml_cont(ctx, ggml_transpose(ctx, ancestors)), original)));
        auto * columns = join(ctx, {ggml_fill(ctx, base, 0),
                ggml_scale_bias(ctx, ggml_cast(ctx, original, GGML_TYPE_F32), 1, 1), ggml_fill(ctx, base, n_candidates + 1)});
        auto * relative = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sub(ctx, ggml_arange(ctx, 0, n_kv, 1), base), 1, 1), 0, n_rows + 1);
        auto * indices = ggml_cast(ctx, at(ctx, columns, ggml_cast(ctx, relative, GGML_TYPE_I32)), GGML_TYPE_I32);
        meta.mask = ggml_cast(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_get_rows(ctx, query, indices))), GGML_TYPE_F16);
        for (auto * t : {meta.positions, meta.kv_idxs, meta.mask}) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
    }

    bool build_outputs(llm_graph_result & res, llama_kv_cache & kv, ggml_tensor * proposals, int n_vocab) {
        auto * ctx = res.get_ctx();
        auto * gf = res.get_gf();
        auto expand = [&](ggml_tensor * t) { ggml_set_output(t); ggml_build_forward_expand(gf, t); };
        auto f32 = [&](ggml_tensor * t) { return ggml_cast(ctx, t, GGML_TYPE_F32); };
        auto i32 = [&](ggml_tensor * t) { return ggml_cast(ctx, t, GGML_TYPE_I32); };
        auto scatter = [&](ggml_tensor * values) {
            auto * empty = ggml_reshape_2d(ctx, ggml_fill(ctx, ggml_arange(ctx, 0, n_candidates, 1), -1), 1, n_candidates);
            return flat(ctx, ggml_set_rows(ctx, empty, ggml_reshape_2d(ctx, values, 1, n_rows), original));
        };
        auto * sampled = scatter(f32(join(ctx, res.t_sampled)));
        auto * compact = scatter(ggml_arange(ctx, 0, n_rows, 1));
        auto * mismatch = ggml_step(ctx, ggml_abs(ctx, ggml_sub(ctx, f32(tokens), at(ctx, sampled, parents))));
        auto * matches = ggml_mul(ctx, ggml_scale_bias(ctx, mismatch, -1, 1), membership);
        matches = join(ctx, {ggml_fill(ctx, row(ctx, sampled, 0), 1), ggml_view_1d(ctx, matches, n_candidates - 1, sizeof(float))});
        // A padded path is valid only if every ancestor matches, including repeated leaf rows.
        auto * path_matches = ggml_reshape_2d(ctx, at(ctx, matches, flat(ctx, paths)), n_rows, n_candidates);
        auto * valid = flat(ctx, ggml_step(ctx, ggml_scale_bias(ctx, ggml_sum_rows(ctx, path_matches), 1, 0.5f - n_rows)));
        auto * selected = ggml_argmax(ctx, ggml_mul(ctx, valid, ggml_scale_bias(ctx, depths, 1, 1)));
        auto * accepted = at(ctx, depths, selected);
        auto * path = flat(ctx, ggml_get_rows(ctx, paths, selected));
        auto * path_rows = i32(at(ctx, compact, path));
        auto * path_samples = at(ctx, sampled, path);
        auto * path_tokens = at(ctx, tokens, path);

        std::vector<ggml_tensor *> matrices;
        llama_kv_cache::slot_info slots = {};
        for (uint32_t il : kv.get_layer_ids()) {
            for (bool key : {true, false}) {
                auto * t = key ? kv.get_k(ctx, il, kv.get_size(), slots) : kv.get_v(ctx, il, kv.get_size(), slots);
                if (t->nb[2] != t->ne[0]*t->ne[1]*ggml_element_size(t)) {
                    return false;
                }
                matrices.push_back(ggml_reshape_2d(ctx, t, t->ne[0]*t->ne[1], kv.get_size()));
            }
        }
        auto * base = f32(row(ctx, linear.kv_idxs, 0));
        auto * source = ggml_add(ctx, f32(path_rows), base);
        auto * destination = linear.kv_idxs;
        auto buffer_of = [](ggml_tensor * t) { return t->view_src ? t->view_src->buffer : t->buffer; };
        for (size_t begin = 0; begin < matrices.size();) {
            auto * first = matrices[begin];
            auto * buffer = buffer_of(first);
            size_t end = begin + 1;
            for (; end < matrices.size(); ++end) {
                auto * next = matrices[end];
                if (buffer_of(next) != buffer || next->type != first->type || !ggml_are_same_shape(first, next) ||
                        !ggml_is_contiguous(next) || (uintptr_t) next->data != (uintptr_t) first->data + (end - begin)*ggml_nbytes(first)) {
                    break;
                }
            }
            auto * group = ggml_new_tensor_3d(ctx, first->type, first->ne[0], kv.get_size(), end - begin);
            if (!buffer || !ggml_is_contiguous(first) || ggml_backend_tensor_alloc(buffer, group, first->data) != GGML_STATUS_SUCCESS) {
                return false;
            }
            auto * rows = i32(ggml_repeat_4d(ctx, source, n_rows, end - begin, 1, 1));
            auto * snapshot = ggml_get_rows(ctx, group, rows);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, group, snapshot, destination));
            begin = end;
        }
        auto remap = [&](ggml_tensor * & t) {
            if (t) {
                t = ggml_get_rows(ctx, t, path_rows);
                expand(t);
            }
        };
        remap(res.t_logits);
        remap(res.t_h_nextn);
        for (auto & t : res.t_layer_inp) {
            if (t && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                remap(t);
            }
        }
        for (auto * rows : {&res.t_sampled, &res.t_sampled_probs, &res.t_sampled_logits, &res.t_candidates}) {
            if (rows->empty() || !(*rows)[0]) {
                continue;
            }
            auto * mapped = at(ctx, join(ctx, *rows), path_rows);
            expand(mapped);
            for (int i = 0; i < n_rows; ++i) {
                (*rows)[i] = row(ctx, mapped, i);
                expand((*rows)[i]);
            }
        }
        // Padding stays in the vocabulary and forces rejection at the path's bonus row.
        auto * alternate = ggml_scale_bias(ctx, path_samples, 1, 1);
        alternate = ggml_sub(ctx, alternate, ggml_scale(ctx,
                ggml_step(ctx, ggml_scale_bias(ctx, alternate, 1, 0.5f - n_vocab)), n_vocab));
        auto * padding = ggml_step(ctx, ggml_sub(ctx, ggml_arange(ctx, 0, n_rows, 1), accepted));
        auto * resolved = i32(ggml_add(ctx, ggml_mul(ctx, f32(path_tokens), ggml_scale_bias(ctx, padding, -1, 1)),
                ggml_mul(ctx, alternate, padding)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, resolved, proposals));
        return true;
    }
};
