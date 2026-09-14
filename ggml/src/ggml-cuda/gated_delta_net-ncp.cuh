#pragma once

// GDN through the external NCP .so, kept out of gated_delta_net.cu the same way fattn-ncp is kept out of
// fattn.cu: gated_delta_net.cu dispatches to the inline kernel, this file implements one path through the
// dlopen'd .so. Before the split the .so hook would have been a large fraction of the dispatcher.
//
// Defined only under GGML_NCP_GDN, and the single call site in gated_delta_net.cu is guarded to match, so a
// plain CUDA build links nothing from here and an unguarded call is a link error rather than a silent no-op.
#include "common.cuh"

// Runs dst on the .so. false -> nothing was written and the caller falls through to the inline kernel.
bool ggml_cuda_op_gated_delta_net_ncp_so(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
