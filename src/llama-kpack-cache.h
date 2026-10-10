#pragma once

#include "llama-kpack-sidecar.h"

// Model-owned cache integration. Initialize and capture while loading; start
// once all tensors are installed. Destruction drains the writer before weights
// are released. No method is called from the inference submission path.
struct llama_model_loader;

class llama_kpack_cache {
public:
    static std::unique_ptr<llama_kpack_cache> create(const llama_model_loader & loader);
    llama_kpack_cache(const std::string &                            dir,
                      const std::string &                            source,
                      const std::vector<llama_kpack_source_tensor> & inventory,
                      int                                            loader_fd = -1);
    ~llama_kpack_cache();
    // Metadata only; individual tensor loads can still fall back to the source.
    bool has_cached_tensors() const;
    bool load(ggml_tensor * tensor);
    void capture(ggml_tensor * tensor);
    void start();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
