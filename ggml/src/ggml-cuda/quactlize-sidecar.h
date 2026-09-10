#pragma once

// Host-only proc-address contract; no CUDA or PPU SDK dependency.
#include "ggml-backend.h"
#include "quactlize/quactlize_ppu_config.h"

struct ggml_quactlize_planes {
    const uint8_t * low;   size_t low_bytes;
    const uint8_t * high;  size_t high_bytes;
    const uint8_t * units; size_t units_bytes;
    quactlize_ppu_placed_arrangement_v2 arrangement;
};

bool ggml_quactlize_tensor_is_kpack(const ggml_tensor * tensor);
bool ggml_quactlize_plane_layout(const ggml_tensor * tensor, size_t * low_bytes, size_t * high_bytes,
                                 size_t * units_bytes, quactlize_ppu_placed_arrangement_v2 * arrangement);
// Loading-thread only. CPU inputs are consumed before return; uploads may still
// be in flight in buffer-owned pinned slots. Consumers must use artifact.ready.
void ggml_quactlize_set_planes(ggml_tensor * tensor, const ggml_quactlize_planes * planes);
// Whole same-qtype gate/up sources, each [E,N/2,K]. No CPU concatenation.
bool ggml_quactlize_pair_supported(ggml_backend_buffer_type_t, const ggml_tensor * merged);
void ggml_quactlize_set_gate_up(ggml_tensor * merged, const void * gate, const void * up, size_t bytes_each);

// Offset addresses the resident [low][high][units] allocation. The destination
// must be pinned, and completion must belong to the tensor's device. Caller owns
// both until completion. No CPU wait on success; only the copy stream is drained
// on an enqueue error. Submit from the snapshot worker after loading is sealed.
// Published weights are immutable; join the worker before buffer teardown.
bool ggml_quactlize_copy_range_async(const ggml_tensor * tensor, void * pinned,
                                    size_t offset, size_t bytes, ggml_backend_event_t completion);
// Worker-only completion, on the device selected by copy_range_async().
bool ggml_quactlize_copy_range_wait(ggml_backend_event_t completion);
