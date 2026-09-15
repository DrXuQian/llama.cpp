#pragma once

#include "llama-graph.h"

struct llama_spec_sampling {
    static ggml_tensor * at(ggml_context * ctx, ggml_tensor * values, ggml_tensor * index) {
        return ggml_get_rows(ctx, ggml_reshape_2d(ctx, values, 1, ggml_nelements(values)), index);
    }

    static ggml_tensor * row(ggml_context * ctx, ggml_tensor * values, int index) {
        return ggml_view_1d(ctx, values, 1, index*ggml_type_size(values->type));
    }

    static ggml_tensor * sample(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * ids, ggml_tensor * uniform) {
        auto * cdf = ggml_cumsum(ctx, weights);
        auto * threshold = ggml_mul(ctx, ggml_sum(ctx, weights), uniform);
        auto * above = ggml_sum(ctx, ggml_step(ctx, ggml_sub(ctx, cdf, threshold)));
        auto * index = ggml_cast(ctx, ggml_scale_bias(ctx,
                ggml_clamp(ctx, above, 1, weights->ne[0]), -1, weights->ne[0]), GGML_TYPE_I32);
        return at(ctx, ids, index);
    }

    // Greedy drafts have q(draft)=1; recovery removes that point mass from the target.
    static ggml_tensor * block_verify(llm_graph_result & res, ggml_tensor * tokens,
            ggml_tensor * uniforms, ggml_tensor * recovery) {
        auto * ctx = res.get_ctx();
        const int n = res.t_sampled.size();
        std::vector<ggml_tensor *> proposals, recovered;
        ggml_tensor * log_ratio = nullptr;
        ggml_tensor * legal = nullptr;
        for (int i = 0; i + 1 < n; ++i) {
            auto * probabilities = res.t_sampled_probs[i];
            auto * candidates = res.t_candidates[i];
            if (!probabilities || !candidates || ggml_nelements(probabilities) != ggml_nelements(candidates)) {
                return nullptr;
            }
            probabilities = ggml_reshape_1d(ctx, probabilities, ggml_nelements(probabilities));
            candidates = ggml_reshape_1d(ctx, candidates, ggml_nelements(candidates));
            auto * proposal = row(ctx, tokens, i + 1);
            auto * different = ggml_step(ctx, ggml_abs(ctx, ggml_sub(ctx,
                    ggml_cast(ctx, candidates, GGML_TYPE_F32), ggml_cast(ctx, proposal, GGML_TYPE_F32))));
            auto * same = ggml_scale_bias(ctx, different, -1, 1);
            auto * probability = ggml_sum(ctx, ggml_mul(ctx, probabilities, same));
            auto * ratio = ggml_sub(ctx, ggml_log(ctx, row(ctx, uniforms, i)), ggml_log(ctx, probability));
            log_ratio = log_ratio ? ggml_add(ctx, log_ratio, ratio) : ratio;
            auto * prefix = ggml_scale_bias(ctx, ggml_step(ctx, log_ratio), -1, 1);
            legal = legal ? ggml_concat(ctx, legal, prefix, 0) : prefix;
            proposals.push_back(proposal);
            recovered.push_back(sample(ctx, ggml_mul(ctx, probabilities, different), candidates, row(ctx, recovery, i)));
        }
        // Scalar log sums handle zero probabilities without inf-inf in a parallel scan.
        // A later legal prefix can include an earlier failed token.
        auto * ranks = ggml_mul(ctx, legal, ggml_arange(ctx, 1, n, 1));
        auto * zero = ggml_fill(ctx, row(ctx, uniforms, 0), 0);
        auto * accepted = ggml_argmax(ctx, ggml_concat(ctx, zero, ranks, 0));
        ggml_set_name(accepted, "magic_mtp_accepted");
        ggml_set_output(accepted);
        ggml_build_forward_expand(res.get_gf(), accepted);
        for (int i = 0; i + 1 < n; ++i) {
            auto * keep = ggml_step(ctx, ggml_scale_bias(ctx, ggml_cast(ctx, accepted, GGML_TYPE_F32), 1, -i));
            auto * token = ggml_add(ctx,
                    ggml_mul(ctx, keep, ggml_cast(ctx, proposals[i], GGML_TYPE_F32)),
                    ggml_mul(ctx, ggml_scale_bias(ctx, keep, -1, 1), ggml_cast(ctx, recovered[i], GGML_TYPE_F32)));
            res.t_sampled[i] = ggml_cast(ctx, token, GGML_TYPE_I32);
            ggml_format_name(res.t_sampled[i], "magic_mtp_sampled_%d", i);
            ggml_set_output(res.t_sampled[i]);
            ggml_build_forward_expand(res.get_gf(), res.t_sampled[i]);
        }
        return accepted;
    }
};
