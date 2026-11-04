#pragma once

#include "ggml-backend.h"

#include <climits>

// argsort_top_k keeps the full expert-count stride between token rows.
inline bool ggml_ncp_mxfp4_ids_layout_supported(const ggml_tensor * ids) {
    return ids && ids->type == GGML_TYPE_I32 && ids->ne[0] > 0 && ids->nb[0] == sizeof(int32_t) &&
           ids->nb[1] % sizeof(int32_t) == 0 && ids->nb[1] / sizeof(int32_t) >= size_t(ids->ne[0]) &&
           ids->nb[1] / sizeof(int32_t) <= INT_MAX;
}

ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_buffer_type(ggml_backend_dev_t dev);

// Private Sail layout; ordinary get/set still exchange GGUF bytes.
ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_weight_buft(ggml_backend_dev_t  dev,
                                                           const ggml_tensor * weight,
                                                           ggml_op             op);
bool                       ggml_cuda_buft_is_ncp_mxfp4(ggml_backend_buffer_type_t buft);
bool                       ggml_cuda_ncp_mxfp4_tensor(const ggml_tensor * tensor);
bool                       ggml_cuda_ncp_mxfp4_alias(const ggml_tensor * tensor);
