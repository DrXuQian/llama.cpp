#include "llama-spec-opt.h"

#include "llama-spec-opt-chain.h"

#include "llama-context.h"

static thread_local llama_output_readback * cur_readback = nullptr;

llama_deferred_copy_scope::llama_deferred_copy_scope(ggml_backend_sched_t sched, llama_deferred_copy & copy) :
    sched(sched) {
    ggml_backend_sched_set_before_compute_callback(sched, submit, &copy);
}

llama_deferred_copy_scope::~llama_deferred_copy_scope() {
    ggml_backend_sched_set_before_compute_callback(sched, nullptr, nullptr);
}

// called twice per split: before its input copies (inputs_ready = false) and after them
bool llama_deferred_copy_scope::submit(ggml_backend_t backend, bool inputs_ready, void * user_data) {
    auto * copy = static_cast<llama_deferred_copy *>(user_data);

    const bool dst_split   = backend == copy->dst_backend && inputs_ready;
    const bool other_split = backend != copy->dst_backend && !inputs_ready && copy->read_by_other;
    if (!dst_split && !other_split) {
        return true;
    }
    if (!copy->submitted) {
        for (size_t i = 0; i < copy->src.size(); ++i) {
            ggml_backend_tensor_copy_async(copy->src_backend, copy->dst_backend, copy->src[i], copy->dst[i]);
        }
        copy->submitted = true;
    }
    // the input copy of the other backend does not wait for the dst stream
    if (other_split && !copy->synced) {
        ggml_backend_synchronize(copy->dst_backend);
        copy->synced = true;
    }
    return true;
}

void llama_output_readback::copy_async(ggml_backend_t backend, const ggml_tensor * tensor, void * dst, size_t size) {
    copies.push_back({backend, tensor, dst, size});
}

void llama_output_readback::submit_async() {
    for (const auto & c : copies) {
        ggml_backend_tensor_get_async(c.backend, c.tensor, c.dst, 0, c.size);
    }
    copies.clear();
}

bool llama_output_readback::pending() const {
    return !copies.empty();
}

void llama_output_readback::tensor_get_async(ggml_backend_t      backend,
                                             const ggml_tensor * tensor,
                                             void *              dst,
                                             size_t              size) {
    if (cur_readback) {
        cur_readback->copy_async(backend, tensor, dst, size);
    } else {
        ggml_backend_tensor_get_async(backend, tensor, dst, 0, size);
    }
}

llama_output_readback_scope::llama_output_readback_scope(llama_output_readback * readback) : prev(cur_readback) {
    cur_readback = readback;
}

llama_output_readback_scope::~llama_output_readback_scope() {
    cur_readback = prev;
}

llama_spec_opt::llama_spec_opt()  = default;
llama_spec_opt::~llama_spec_opt() = default;

void llama_spec_opt::flush_readback(llama_context & ctx) {
    if (readback.pending()) {
        ctx.synchronize();
    }
}

void llama_spec_opt::finish() {
    if (chain_pipeline) {
        chain_pipeline->finish(readback);
    }
}

void llama_spec_opt::reset() {
    if (chain_pipeline) {
        chain_pipeline->reset(readback);
    }
}

void llama_spec_opt::submit_readback() {
    readback.submit_async();
}

llama_output_readback_scope llama_spec_opt::begin_readback(const llama_batch & batch,
                                                           uint32_t            n_tokens,
                                                           uint32_t            n_outputs,
                                                           uint32_t            n_ubatch) {
    const bool defer = chain_pipeline && chain_pipeline->can_defer_readback(batch, n_tokens, n_outputs, n_ubatch);
    return llama_output_readback_scope(defer ? &readback : nullptr);
}

void llama_spec_opt::set_graph_params(llm_graph_params & params) const {
    if (chain_pipeline) {
        chain_pipeline->set_graph_params(params);
    }
}

void llama_spec_opt::bind_inputs(llm_graph_result * res) const {
    if (chain_pipeline) {
        chain_pipeline->bind_inputs(res);
    }
}

void llama_spec_opt::memory_breakdown(llama_memory_breakdown &              ret,
                                      ggml_backend_sched_t                  sched,
                                      const std::vector<ggml_backend_ptr> & backends) const {
    if (chain_pipeline) {
        chain_pipeline->memory_breakdown(ret, sched, backends);
    }
}

int32_t llama_spec_opt::decode_with_copy(llama_context & ctx, const llama_batch & batch, llama_deferred_copy & copy) {
    // decode() may rebuild the sched, so do it before the callback is installed
    ctx.sched_reserve();

    llama_deferred_copy_scope scope(ctx.sched.get(), copy);
    return ctx.decode(batch);
}
