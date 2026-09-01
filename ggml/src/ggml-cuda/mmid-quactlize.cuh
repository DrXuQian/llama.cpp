#pragma once
// mul_mat_id for expert weights that live in the K-pack buffer type, through the quactlize grouped-GEMM .so.
//
// Kept out of ggml-cuda.cu the same way mmid-ncp.cuh is: the dispatcher carries the decision, not the implementation.
//
// UNLIKE THE NCP HOOK, THIS ONE MAY NOT DECLINE SILENTLY. A tensor in the K-pack buffer no longer holds GGUF blocks,
// so every other branch in ggml_cuda_mul_mat_id -- to_bf16, MMQ, MMF, the D2H fallback -- would read the artifact as
// if it were the quantised format it claims to be and return plausible garbage. The entry therefore aborts on a
// decline rather than returning false, and supports_op is what has to be conservative.
#include "common.cuh"

// True if src0 is a converted K-pack artifact, i.e. if this node MUST go through here.
bool ggml_cuda_mul_mat_id_is_quactlize(const ggml_tensor * src0);

void ggml_cuda_mul_mat_id_quactlize(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
