#pragma once

#ifdef GGML_NCP_QUACTLIZE
#    include "llama-model-loader.h"

#    include <functional>

ggml_tensor * llama_kpack_create_merged_tensor(
    llama_model_loader &                                              loader,
    const llama_hparams &                                             hparams,
    const LLM_TN_IMPL &                                               tn,
    const std::initializer_list<int64_t> &                            ne,
    int                                                               flags,
    const std::function<ggml_backend_buffer_type_t(ggml_tensor *)> &  buft_for_tensor,
    const std::function<ggml_context *(ggml_backend_buffer_type_t)> & ctx_for_buft);

bool llama_kpack_mapping_range(const llama_model_loader &                      loader,
                               const llama_model_loader::llama_tensor_weight & weight,
                               size_t *                                        first,
                               size_t *                                        last);
bool llama_kpack_load_data_for(const llama_model_loader &                      loader,
                               ggml_tensor *                                   tensor,
                               const llama_model_loader::llama_tensor_weight & weight);
bool llama_kpack_load_tensor(llama_model_loader &                            loader,
                             ggml_tensor *                                   tensor,
                             const llama_model_loader::llama_tensor_weight & weight,
                             std::vector<no_init<uint8_t>> &                 read_buf);
#endif
