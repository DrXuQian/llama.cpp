#pragma once

// mul_mat_id through the external NCP DeepGemm grouped-GEMM .so, kept out of ggml-cuda.cu the way fattn.cu keeps its
// variants out (fattn-tile, fattn-wmma-f16) and mmvf-ppu.cuh keeps the PPU GEMV out of mmvf.cu: the dispatcher should
// carry the decision, not the implementation.
//
// Both entries exist only under GGML_NCP_MOE and their call sites are guarded to match, so a plain CUDA build links
// nothing from here -- and an unguarded call is a link error rather than a silent no-op.
#include "common.cuh"

// Runs the node on the .so. false -> caller falls through to the inline path, having changed nothing.
bool ggml_cuda_mul_mat_id_ncp_lib(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// Whether the above would take this node. Asked by the GLU fusion gate, which has to decline fusion BEFORE the node
// ever reaches the mul_mat_id dispatcher -- fusion consumes the gate/up pair itself and launches the generic mmvf, so
// without this two thirds of a MoE layer's matmuls never see the .so. Lives next to the hook so the two conditions
// cannot drift apart unnoticed.
bool ggml_cuda_mul_mat_id_ncp_lib_supported(const ggml_tensor * tensor);
