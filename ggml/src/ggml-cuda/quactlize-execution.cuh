#pragma once
#include "common.cuh"

// Prepare native plans and one-time SF metadata before CUDA graph capture.
void ggml_quactlize_execution_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph);
// True means this call was enqueued by the native path. A policy/package miss
// returns false before launch so the existing K-pack-capable fallback can run.
bool ggml_quactlize_execution_run(ggml_backend_cuda_context & ctx,
    const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output);
