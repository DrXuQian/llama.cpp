#pragma once

// API of the GPU autoregressive speculative chain pipeline

#include "llama.h"

// Update the draft state from the target context, then schedule an async draft chain with GPU-side acceptance.
// The eagle3_g_out pointer becomes readable after llama_spec_synchronize_eagle3_g completes, MTP sets it to null.
LLAMA_API bool llama_spec_submit_draft_generation(struct llama_context * draft,
                                                  struct llama_context * target,
                                                  struct llama_batch     batch,
                                                  const float **         eagle3_g_out);

// Validate and allocate fixed resources for an autoregressive speculative chain pipeline.
LLAMA_API bool llama_spec_prepare_chain(struct llama_context * target, struct llama_context * draft, int32_t n_draft);

// Submit target verification using device-resident draft chain tokens.
// Returns the same status as llama_decode.
LLAMA_API int32_t llama_spec_submit_target_verification(struct llama_context * target,
                                                        struct llama_context * draft,
                                                        struct llama_batch     batch,
                                                        llama_token *          draft_tokens);

// Commit host KV metadata for rows written by the prefetched draft chain.
LLAMA_API bool llama_spec_commit_chain_kv_metadata(struct llama_context * ctx, llama_pos seed_pos);

// Block until the pending EAGLE3 g output reaches host pinned memory.
LLAMA_API void llama_spec_synchronize_eagle3_g(struct llama_context * draft);
