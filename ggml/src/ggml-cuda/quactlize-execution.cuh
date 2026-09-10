#pragma once
#include "common.cuh"
#include "quactlize/kpack_indexed.h"

// Prepare native plans and one-time SF metadata before CUDA graph capture.
void ggml_quactlize_execution_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph);
// Match an exact closed gate/up/SwiGLU/down graph. The CUDA graph walker also
// checks allocator overlap before skipping its nodes. Unsupported graphs stay
// on the ordinary per-node path.
int ggml_quactlize_execution_moe_nodes(const ggml_cgraph * graph, int start);
bool ggml_quactlize_execution_moe_run(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int start);
bool ggml_quactlize_execution_moe_router_run(ggml_backend_cuda_context & ctx, ggml_cgraph * graph,
    int start, const qk_llama_router_v1 & router, const ggml_tensor * ids);
// True means this call was enqueued by the native path. A policy/package miss
// returns false before launch so the existing K-pack-capable fallback can run.
bool ggml_quactlize_execution_run(ggml_backend_cuda_context & ctx,
    const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output);
