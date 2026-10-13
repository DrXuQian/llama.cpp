#pragma once

#include "ggml-backend.h"
#include "ncp-quactlize-lib.h"

struct ggml_quactlize_packed_layout {
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1        sizes;
};

struct ggml_backend_cuda_context;

ggml_backend_buffer_type_t * ggml_cuda_quactlize_extra_bufts(ggml_backend_dev_t dev);
ggml_backend_buffer_type_t   ggml_cuda_quactlize_weight_buft(ggml_backend_dev_t  dev,
                                                             const ggml_tensor * weight,
                                                             ggml_op             op);
bool                         ggml_cuda_buft_is_quactlize(ggml_backend_buffer_type_t buft);
bool                         ggml_quactlize_can_load(const ggml_tensor * tensor, ggml_quactlize_packed_layout * layout);
bool                         ggml_quactlize_dense_enabled();
bool                         ggml_quactlize_can_compute(const ggml_tensor * op);
void                         ggml_quactlize_prepare_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph);
void                         ggml_quactlize_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Explicit packed-byte I/O. The ordinary get_tensor/copy interfaces stay disabled.
bool ggml_quactlize_read_packed(const ggml_tensor * tensor, void * data, size_t size);
bool ggml_quactlize_set_packed(ggml_tensor *                        tensor,
                               const ggml_quactlize_packed_layout * layout,
                               const void *                         data,
                               size_t                               size);

// Snapshot after loading is sealed. Caller owns pinned memory and event until completion.
// Submit and wait on the writer, and join it before freeing the tensor buffer.
bool ggml_quactlize_tensor_is_kpack(const ggml_tensor * tensor);
bool ggml_quactlize_copy_range_async(const ggml_tensor *  tensor,
                                     void *               pinned,
                                     size_t               offset,
                                     size_t               bytes,
                                     ggml_backend_event_t completion);
bool ggml_quactlize_copy_range_wait(ggml_backend_event_t completion);
