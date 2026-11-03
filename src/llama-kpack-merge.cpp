#include "llama-kpack-merge.h"

#ifdef GGML_NCP_QUACTLIZE
#    include "llama-kpack-cache.h"
#    include "llama-model.h"

#    include <algorithm>
#    include <cinttypes>

ggml_tensor * llama_kpack_create_merged_tensor(
    llama_model_loader &                                              loader,
    const llama_hparams &                                             hparams,
    const LLM_TN_IMPL &                                               tn,
    const std::initializer_list<int64_t> &                            ne,
    int                                                               flags,
    const std::function<ggml_backend_buffer_type_t(ggml_tensor *)> &  buft_for_tensor,
    const std::function<ggml_context *(ggml_backend_buffer_type_t)> & ctx_for_buft) {
    if (loader.get_tensor_meta(tn.str().c_str()) || !tn.suffix || strcmp(tn.suffix, "weight") ||
        (flags & (llama_model_loader::TENSOR_SKIP | llama_model_loader::TENSOR_DUPLICATED)) ||
        (tn.tensor != LLM_TENSOR_FFN_GATE_UP && tn.tensor != LLM_TENSOR_FFN_GATE_UP_SHEXP &&
         tn.tensor != LLM_TENSOR_FFN_GATE_UP_EXPS) ||
        tn.bid < 0 || tn.bid >= LLAMA_MAX_LAYERS ||
        (tn.tensor != LLM_TENSOR_FFN_GATE_UP_EXPS && !(hparams.swiglu_clamp_shexp[tn.bid] <= 1e-6f))) {
        return nullptr;
    }
    const bool experts   = tn.tensor == LLM_TENSOR_FFN_GATE_UP_EXPS;
    const bool shared    = tn.tensor == LLM_TENSOR_FFN_GATE_UP_SHEXP;
    const auto gate_name = LLM_TN_IMPL(tn.arch,
                                       experts ? LLM_TENSOR_FFN_GATE_EXPS :
                                       shared  ? LLM_TENSOR_FFN_GATE_SHEXP :
                                                 LLM_TENSOR_FFN_GATE,
                                       "weight", tn.bid, tn.xid)
                               .str();
    const auto up_name = LLM_TN_IMPL(tn.arch,
                                     experts ? LLM_TENSOR_FFN_UP_EXPS :
                                     shared  ? LLM_TENSOR_FFN_UP_SHEXP :
                                               LLM_TENSOR_FFN_UP,
                                     "weight", tn.bid, tn.xid)
                             .str();
    const auto * gate         = loader.get_weight(gate_name.c_str());
    const auto * up           = loader.get_weight(up_name.c_str());
    auto         plain_weight = [&](const std::string & name) {
        const auto prefix = name.substr(0, name.size() - strlen("weight"));
        for (auto it = loader.weights_map.lower_bound(prefix);
             it != loader.weights_map.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it) {
            if (it->first != name) {
                return false;  // Bias, scales and embedded adapters keep their separate projections.
            }
        }
        return true;
    };
    // Keep a merged cache record within one source file's stat identity.
    if (!gate || !up || gate->idx != up->idx || gate->tensor->type != up->tensor->type ||
        !ggml_are_same_shape(gate->tensor, up->tensor) || !ggml_are_same_stride(gate->tensor, up->tensor) ||
        !plain_weight(gate_name) || !plain_weight(up_name) ||
        (gate->tensor->type != GGML_TYPE_Q8_0 &&
         (gate->tensor->type < GGML_TYPE_Q2_K || gate->tensor->type > GGML_TYPE_Q6_K))) {
        return nullptr;
    }
    ggml_tensor merged = *gate->tensor;
    merged.ne[1] *= 2;
    merged.nb[2] *= 2;
    merged.nb[3] *= 2;
    ggml_set_name(&merged, tn.str().c_str());
    bool shape = ne.size() <= GGML_MAX_DIMS;
    for (size_t d = 0; d < GGML_MAX_DIMS; ++d) {
        shape &= merged.ne[d] == (d < ne.size() ? ne.begin()[d] : 1);
    }
    if (!shape) {
        return nullptr;
    }
    auto * buft = buft_for_tensor(&merged);
    if (!buft || !llama_kpack_buffer_type(buft) || buft_for_tensor(gate->tensor) != buft ||
        buft_for_tensor(up->tensor) != buft) {
        return nullptr;
    }
    auto * tensor = ggml_dup_tensor(ctx_for_buft(buft), &merged);
    ggml_set_name(tensor, merged.name);
    auto weight           = *gate;
    weight.tensor         = tensor;
    weight.paired_sources = {gate_name, up_name};
    loader.weights_map.emplace(tn.str(), std::move(weight));
    loader.n_created += 2;
    LLAMA_LOG_INFO("[quactlize-merge] tensor=%s sources=%s,%s n=%" PRId64 " k=%" PRId64 " experts=%" PRId64 "\n",
                   tensor->name, gate_name.c_str(), up_name.c_str(), tensor->ne[1], tensor->ne[0], tensor->ne[2]);
    return tensor;
}

bool llama_kpack_mapping_range(const llama_model_loader &                      loader,
                               const llama_model_loader::llama_tensor_weight & weight,
                               size_t *                                        first,
                               size_t *                                        last) {
    if (weight.paired_sources.empty()) {
        return false;
    }
    for (const auto & name : weight.paired_sources) {
        const auto & source = loader.require_weight(name.c_str());
        *first              = std::min(*first, source.offs);
        *last               = std::max(*last, source.offs + ggml_nbytes(source.tensor));
    }
    return true;
}

bool llama_kpack_load_data_for(const llama_model_loader &                      loader,
                               ggml_tensor *                                   tensor,
                               const llama_model_loader::llama_tensor_weight & weight) {
    if (weight.paired_sources.empty()) {
        return false;
    }
    GGML_ASSERT(tensor->data && weight.paired_sources.size() == 2);
    const size_t width = tensor->nb[2] / 2;
    for (size_t side = 0; side < 2; ++side) {
        const auto & source = loader.require_weight(weight.paired_sources[side].c_str());
        for (int64_t e = 0; e < tensor->ne[2] * tensor->ne[3]; ++e) {
            auto * dst = static_cast<uint8_t *>(tensor->data) + e * 2 * width + side * width;
            if (loader.use_mmap) {
                memcpy(dst,
                       static_cast<const uint8_t *>(loader.mappings.at(source.idx)->addr()) + source.offs + e * width,
                       width);
            } else {
                loader.files.at(source.idx)->seek(source.offs + e * width, SEEK_SET);
                loader.files.at(source.idx)->read_raw(dst, width);
            }
            if (loader.check_tensors && !ggml_validate_row_data(tensor->type, dst, width)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", source.tensor->name));
            }
        }
    }
    return true;
}

bool llama_kpack_load_tensor(llama_model_loader &                            loader,
                             ggml_tensor *                                   tensor,
                             const llama_model_loader::llama_tensor_weight & weight,
                             std::vector<no_init<uint8_t>> &                 read_buf) {
    if (weight.paired_sources.empty()) {
        return false;
    }
    GGML_ASSERT(weight.paired_sources.size() == 2);
    const size_t                  width = tensor->nb[2] / 2;
    const size_t                  rows  = tensor->ne[2] * tensor->ne[3];
    ggml_backend_meta_split_state split{};
    if (ggml_backend_meta_tensor_split(ggml_backend_buffer_get_type(tensor->buffer), tensor, &split)) {
        // Meta accepts complete split planes, not separate Gate/Up halves.
        const size_t planes = split.axis == GGML_BACKEND_SPLIT_AXIS_2 ? tensor->ne[2] : 1;
        read_buf.resize(2 * width * planes);
        for (size_t first = 0; first < rows; first += planes) {
            for (size_t side = 0; side < 2; ++side) {
                const auto & source = loader.require_weight(weight.paired_sources[side].c_str());
                for (size_t plane = 0; plane < planes; ++plane) {
                    auto *       dst    = read_buf.data() + (2 * plane + side) * width;
                    const size_t offset = source.offs + (first + plane) * width;
                    if (loader.use_mmap) {
                        memcpy(dst, static_cast<const uint8_t *>(loader.mappings.at(source.idx)->addr()) + offset,
                               width);
                    } else {
                        loader.files.at(source.idx)->seek(offset, SEEK_SET);
                        loader.files.at(source.idx)->read_raw(dst, width);
                    }
                    if (loader.check_tensors && !ggml_validate_row_data(tensor->type, dst, width)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", source.tensor->name));
                    }
                }
            }
            ggml_backend_tensor_set(tensor, read_buf.data(), first * 2 * width, read_buf.size());
        }
    } else {
        for (size_t side = 0; side < weight.paired_sources.size(); ++side) {
            const auto & source = loader.require_weight(weight.paired_sources[side].c_str());
            const void * data;
            if (loader.use_mmap) {
                data = static_cast<const uint8_t *>(loader.mappings.at(source.idx)->addr()) + source.offs;
            } else {
                read_buf.resize(ggml_nbytes(source.tensor));
                loader.files.at(source.idx)->seek(source.offs, SEEK_SET);
                loader.files.at(source.idx)->read_raw(read_buf.data(), read_buf.size());
                data = read_buf.data();
            }
            if (loader.check_tensors && !ggml_validate_row_data(tensor->type, data, width * rows)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", source.tensor->name));
            }
            ggml_backend_tensor_set_2d(tensor, data, side * width, width, rows, 2 * width, width);
        }
    }
    if (loader.kpack_cache) {
        loader.kpack_cache->capture(tensor);
    }
    return true;
}

void llama_model_base::create_tensor_gate_up(llama_layer & layer,
                                             int           bid,
                                             int64_t       n_embd_,
                                             int64_t       n_ff_,
                                             int           flags,
                                             bool          shared) {
    auto * merged = create_tensor(tn(shared ? LLM_TENSOR_FFN_GATE_UP_SHEXP : LLM_TENSOR_FFN_GATE_UP, "weight", bid),
                                  {n_embd_, 2 * n_ff_}, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    auto & gate   = shared ? layer.ffn_gate_shexp : layer.ffn_gate;
    auto & up     = shared ? layer.ffn_up_shexp : layer.ffn_up;
    if (merged) {
        up = merged;
    } else {
        gate = create_tensor(tn(shared ? LLM_TENSOR_FFN_GATE_SHEXP : LLM_TENSOR_FFN_GATE, "weight", bid),
                             {n_embd_, n_ff_}, flags);
        up   = create_tensor(tn(shared ? LLM_TENSOR_FFN_UP_SHEXP : LLM_TENSOR_FFN_UP, "weight", bid), {n_embd_, n_ff_},
                             flags);
    }
}
#endif
