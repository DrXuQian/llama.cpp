#pragma once

#include "llama.h"
#include "llama-ext.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <memory>
#include <vector>

struct llama_context;
struct llm_graph_params;
class llm_graph_result;
struct llama_spec_chain_pipeline;

// D2D copy submitted by the sched callback during a decode
struct llama_deferred_copy {
    ggml_backend_t             src_backend = nullptr;
    ggml_backend_t             dst_backend = nullptr;
    std::vector<ggml_tensor *> src;
    std::vector<ggml_tensor *> dst;
    // a split on another backend may copy dst as its input, so the copy must finish before that
    bool                       read_by_other = false;
    bool                       submitted     = false;
    bool                       synced        = false;

    bool active() const { return src_backend != nullptr; }
};

// install the before-compute callback only for the decode that owns the copy
struct llama_deferred_copy_scope {
    llama_deferred_copy_scope(ggml_backend_sched_t sched, llama_deferred_copy & copy);
    ~llama_deferred_copy_scope();

    llama_deferred_copy_scope(const llama_deferred_copy_scope &)             = delete;
    llama_deferred_copy_scope & operator=(const llama_deferred_copy_scope &) = delete;

private:
    static bool submit(ggml_backend_t backend, bool inputs_ready, void * user_data);

    ggml_backend_sched_t sched;
};

// D2H output copies held back until submit_async(), so they do not delay a later D2D copy on the same stream
struct llama_output_readback {
    void copy_async(ggml_backend_t backend, const ggml_tensor * tensor, void * dst, size_t size);

    void submit_async();
    bool pending() const;

    // queue the D2H copy in the readback of the running decode, else submit it now
    static void tensor_get_async(ggml_backend_t backend, const ggml_tensor * tensor, void * dst, size_t size);

private:
    struct copy_desc {
        ggml_backend_t      backend;
        const ggml_tensor * tensor;
        void *              dst;
        size_t              size;
    };

    std::vector<copy_desc> copies;
};

// sets the readback of the running decode on this thread, restores the previous one when the scope ends
struct llama_output_readback_scope {
    explicit llama_output_readback_scope(llama_output_readback * readback);
    ~llama_output_readback_scope();

    llama_output_readback_scope(const llama_output_readback_scope &)             = delete;
    llama_output_readback_scope & operator=(const llama_output_readback_scope &) = delete;

private:
    llama_output_readback * prev;
};

struct llama_spec_opt {
    llama_spec_opt();
    ~llama_spec_opt();

    // hooks for llama_context
    void flush_readback(llama_context & ctx);
    void finish();
    void reset();
    void submit_readback();

    llama_output_readback_scope begin_readback(const llama_batch & batch,
                                               uint32_t            n_tokens,
                                               uint32_t            n_outputs,
                                               uint32_t            n_ubatch);

    void set_graph_params(llm_graph_params & params) const;
    void bind_inputs(llm_graph_result * res) const;

    void memory_breakdown(llama_memory_breakdown &              ret,
                          ggml_backend_sched_t                  sched,
                          const std::vector<ggml_backend_ptr> & backends) const;

    static int32_t decode_with_copy(llama_context & ctx, const llama_batch & batch, llama_deferred_copy & copy);

private:
    friend struct llama_spec_chain_pipeline;

    std::unique_ptr<llama_spec_chain_pipeline> chain_pipeline;
    llama_output_readback                      readback;
};
