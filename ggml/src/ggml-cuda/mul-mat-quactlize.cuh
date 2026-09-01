#pragma once
// GGML_OP_MUL_MAT for weights that live in the K-pack buffer type. The mul_mat_id counterpart is
// mmid-quactlize.cuh, and the same rule holds here: a tensor in that buffer no longer carries GGUF blocks, so this
// is not one route among several -- it is the only one, and a decline aborts rather than falling through.
#include "common.cuh"

bool ggml_cuda_mul_mat_is_quactlize(const ggml_tensor * src0);

void ggml_cuda_mul_mat_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
