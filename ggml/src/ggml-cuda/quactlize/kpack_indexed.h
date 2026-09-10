#pragma once
#include <stdint.h>

// Binding for llama.cpp's MUL_MAT_ID layout. Strides count elements, not bytes.
// IDs are [tokens,topk] with ids_stride >= topk and unique experts per token.
// A is F32 [tokens,channels,K]; source channel = slot % channels. Output is
// F32 [tokens,topk,N]. row_ids[m] is caller-owned scratch (m=tokens*topk).
// Binding occurs once, outside capture. All pointers/lifetimes remain fixed;
// IDs and A values may change on every replay. run performs no host data read.
// The first fused implementation accepts m<=32 and experts<=1024. Larger
// requests explicitly decline and use the existing unfused preparation.
typedef struct {
  uint32_t version, size;
  int32_t tokens, topk, channels, reserved;
  int64_t ids_stride, a_row_stride, a_token_stride, out_row_stride;
  int32_t const* ids;
  float const* a;
  float* output;
  int32_t* row_ids;
} qk_llama_indexed_v1;

// Optional 256-expert router for a complete, closed llama MoE graph. Input
// logits are F32 [tokens,256], selection bias is optional F32 [256]. Weights
// are written in token/slot order; IDs use the indexed binding's stride.
// softmax is before selection unless delayed_softmax is set. Normalization
// and delayed softmax are mutually exclusive. No weighted-input variant.
typedef struct {
  uint32_t version, size;
  int32_t use_sigmoid, with_norm, delayed_softmax, reserved;
  float clamp, scale;
  float const *logits, *bias;
  float *weights;
} qk_llama_router_v1;

#ifdef __cplusplus
extern "C" {
#endif
// Additive module entry. Existing run_v1 executes prepare+producer+completion
// after a successful bind. Old modules need not export this optional entry.
int quactlize_kpack_bind_llama_indexed_v1(void*, qk_llama_indexed_v1 const*);
#ifdef __cplusplus
}
#endif
