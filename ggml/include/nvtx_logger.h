
#if defined(GGML_USE_PPU)

#include <nvtx3/nvToolsExt.h>

namespace {

class llama_context_nvtx_range {
public:
    explicit llama_context_nvtx_range(const char * name) {
        nvtxRangePushA(name);
    }

    ~llama_context_nvtx_range() {
        nvtxRangePop();
    }

    llama_context_nvtx_range(const llama_context_nvtx_range &) = delete;
    llama_context_nvtx_range & operator=(const llama_context_nvtx_range &) = delete;
};

} // namespace

#define LLAMA_CONTEXT_NVTX_CONCAT_IMPL(a, b) a ## b
#define LLAMA_CONTEXT_NVTX_CONCAT(a, b) LLAMA_CONTEXT_NVTX_CONCAT_IMPL(a, b)
#define LLAMA_CONTEXT_NVTX_RANGE(name) \
    const llama_context_nvtx_range LLAMA_CONTEXT_NVTX_CONCAT(nvtx_range_, __LINE__)(name)
#else
#define LLAMA_CONTEXT_NVTX_RANGE(name) ((void) 0)
#endif
