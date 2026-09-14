#pragma once

// FlashAttention through the external NCP .so, kept out of fattn.cu the same way its other backends are: fattn.cu
// picks a kernel, fattn-tile.cu / fattn-wmma-f16.cu / this file implement one. Before the split the .so hook was 126
// of fattn.cu's 730 lines -- a fifth of the dispatcher was one backend's body.
//
// Defined only under GGML_NCP_FA, and the single call site in fattn.cu is guarded to match, so a plain CUDA build
// links nothing from here and an unguarded call is a link error rather than a silent no-op.
#include "common.cuh"

// Runs dst on the .so. false -> nothing was written and the caller falls through to the inline kernels.
bool ggml_cuda_flash_attn_ext_ncp_lib(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
