#pragma once

#include "ggml-backend.h"

ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_buffer_type(ggml_backend_dev_t dev);

// Private Sail layout; ordinary get/set still exchange GGUF bytes.
ggml_backend_buffer_type_t ggml_cuda_ncp_mxfp4_weight_buft(ggml_backend_dev_t  dev,
                                                           const ggml_tensor * weight,
                                                           ggml_op             op);
bool                       ggml_cuda_buft_is_ncp_mxfp4(ggml_backend_buffer_type_t buft);
bool                       ggml_cuda_ncp_mxfp4_tensor(const ggml_tensor * tensor);
bool                       ggml_cuda_ncp_mxfp4_alias(const ggml_tensor * tensor);
