#pragma once

#include "ggml-backend.h"
#include "quactlize/packing/api.h"

struct ggml_quactlize_packed_layout {
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1        sizes;
};

ggml_backend_buffer_type_t * ggml_cuda_quactlize_extra_bufts(ggml_backend_dev_t dev);
bool                         ggml_cuda_buft_is_quactlize(ggml_backend_buffer_type_t buft);
bool                         ggml_quactlize_can_load(const ggml_tensor * tensor, ggml_quactlize_packed_layout * layout);
void                         ggml_quactlize_assert_no_compute(const ggml_cgraph * graph);

// Explicit packed-byte I/O. The ordinary get_tensor/copy interfaces stay disabled.
bool ggml_quactlize_read_packed(const ggml_tensor * tensor, void * data, size_t size);
bool ggml_quactlize_set_packed(ggml_tensor *                        tensor,
                               const ggml_quactlize_packed_layout * layout,
                               const void *                         data,
                               size_t                               size);
