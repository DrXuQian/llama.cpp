#pragma once

#include "llama-impl.h"

#include "ggml-backend.h"

#include <map>
#include <vector>

struct llama_output_copies {
    std::map<ggml_backend_t, std::pair<std::vector<const ggml_tensor *>, std::vector<void *>>> data;

    void add(ggml_backend_t backend, const ggml_tensor * tensor, void * dst) {
        auto & entry = data[backend];
        entry.first.push_back(tensor);
        entry.second.push_back(dst);
    }

    void submit() {
        for (auto & [backend, entry] : data) {
            auto * dev = ggml_backend_get_device(backend);
            auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto get_batch = reg ? (ggml_backend_get_tensors_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_tensors_async") : nullptr;
            if (get_batch && get_batch(backend, entry.first.data(), entry.second.data(), entry.first.size())) {
                continue;
            }
            for (size_t i = 0; i < entry.first.size(); i++) {
                ggml_backend_tensor_get_async(backend, entry.first[i], entry.second[i], 0, ggml_nbytes(entry.first[i]));
            }
        }
    }
};

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    llama_output_copies & copies,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        copies.add(backend, tensor, row_ptr);

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}
