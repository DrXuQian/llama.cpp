#pragma once
// The K-pack extra buffer type: a CUDA buffer that stores k-quant weights in quactlize's resident artifact layout
// instead of the GGUF block layout, converting once, at load, inside set_tensor.
//
// WHY A BUFFER TYPE AND NOT A CACHE IN mul_mat_id. Converting lazily in the forward pass would have to keep the
// original GGUF bytes as well -- llama.cpp's model buffer owns them and other paths still read them -- and the
// K-pack form is byte-neutral, so the quantised weights would be resident TWICE. Owning the buffer is what makes
// the artifact the only copy: get_alloc_size is exactly ggml_nbytes (no MATRIX_ROW_PADDING tail, the K-pack reader
// does not use one), so the resident footprint is bit-for-bit what it is today.
//
// set_tensor uploads raw bytes to reusable scratch and packs directly into the final device allocation.
// Compute waits on packed-ready; optional backcopy uses an independent nonblocking stream.
//
// THE TENSOR BECOMES UNREADABLE, DELIBERATELY. get_tensor and cpy_tensor are null, exactly as ggml-cpu's repack
// buffer does it: once a tensor is here its GGUF bytes are gone, so anything but the K-pack kernels reading it
// would read a different format silently. That also means TAKING THIS BUFFER TYPE IS A PROMISE: supports_op must
// only say yes when the library can serve the tensor for every shape the graph can present, because there is no
// un-K-packed copy left to fall back to.

#include "common.cuh"
#include "quactlize-lib.h"
#include "quactlize-sidecar.h"

// The K-pack buffer type for one CUDA/PPU device. Returns nullptr when the build has no quactlize support.
ggml_backend_buffer_type_t ggml_backend_cuda_quactlize_buffer_type(int device);

bool ggml_backend_buft_is_cuda_quactlize(ggml_backend_buffer_type_t buft);

// Would a tensor of this type/shape be servable from the K-pack path? Answered from the library's own tactic
// inventory, never from a table kept here -- a second list can disagree with the kernels, and this one decides
// whether the GGUF bytes get thrown away. Used by ggml_backend_cuda_device_supports_op.
bool ggml_quactlize_can_serve(const ggml_tensor * weight, ggml_op op);

// The resident artifact of a tensor that took this buffer type. false when the tensor is in an ordinary buffer, or
// before its conversion has been enqueued. Consumers must wait on ready before reading the planes.
struct ggml_quactlize_artifact {
    const uint8_t * low;
    const uint8_t * high;    // null when the format has no high plane
    const uint8_t * units;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    int     qtype;
    int64_t n;
    int64_t k;
    int64_t experts;
    cudaEvent_t ready;
};

bool ggml_quactlize_artifact_for(const ggml_tensor * tensor, ggml_quactlize_artifact * out);

#ifdef GGML_NCP_QUACTLIZE
inline void ggml_quactlize_wait_ready(const ggml_quactlize_artifact & art, cudaStream_t stream) {
    // Recorded once by the pack/upload stream, outside any inference capture.
    // Retained with the weights for all graph replays; no dependency on backcopy.
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    // PPU rejects the external flag outside capture, including eager warmup.
    const unsigned int flags = status == cudaStreamCaptureStatusNone ? 0 : cudaEventWaitExternal;
    CUDA_CHECK(cudaStreamWaitEvent(stream, art.ready, flags));
}
#endif

// Does this node read a K-pack artifact through ANY of its sources?
//
// For the fusion gates. Fusion runs inside the backend's graph_compute, after supports_op has already admitted each
// node one at a time, and it launches mmvf/mmvq/mmf directly -- so a fused node never reaches ggml_cuda_mul_mat or
// ggml_cuda_mul_mat_id and never reaches the K-pack branch there. A K-pack tensor still reports its GGUF type, so
// ggml_cuda_should_fuse_mul_mat_vec_q would happily take it and hand the artifact to a Q4_K reader.
//
// Any source, not just src[0]: refusing a fusion that would have been fine costs one intermediate write and read,
// while missing one costs a wrong answer that looks plausible. That asymmetry is the whole design of this check.
bool ggml_quactlize_node_reads_artifact(const ggml_tensor * node);
