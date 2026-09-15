#pragma once
#include "kpack_module.h"

// Additive decode-only dense endpoints. The existing qk_call_v1 FP16 entry
// is unchanged. This call names actual storage, never reinterprets BF16 as
// FP16. The mixed-input TC core still uses FP16 A/B with FP32 accumulation.
// F32/BF16 A conversion occurs inside each mainloop load; final output is
// written by the epilogue or the ordered Split-K reducer. No A workspace,
// gather, scatter, or standalone cast is allocated/launched by this API.
enum { QKD_F32=1, QKD_BF16=2 };
typedef struct {
  uint32_t version,size;
  qk_call_v1 call;
  int32_t input_type,output_type;
} qkd_dense_call_v1;

enum { QKD_COMPUTE_F16=0, QKD_COMPUTE_BF16=1 };
// Compute precision is not inferred from input/output storage. A v2 module
// advertises one compute identity; a mismatched request must be rejected.
// FP16 scale/zero and packed-unit bytes retain their original format. BF16
// TC reconstructs integer codes exactly, then rounds scale multiplication
// and zero addition separately into BF16, and accumulates in FP32.
typedef struct {
  uint32_t version,size;
  qkd_dense_call_v1 dense;
  int32_t compute_type;
} qkd_dense_call_v2;
typedef struct {
  uint32_t version,size;
  qk_identity_v1 const* parent;
  int32_t compute_type;
} qkd_compute_identity_v2;
// Dense E=1, M1..8, contiguous [M,K] / [M,N]. Matching F32/F32 and
// BF16/BF16 endpoints; all other combinations explicitly decline. Input
// values must be representable by the FP16 compute boundary. No input-type
// inference from an untyped pointer and no use by prefill.
#ifdef __cplusplus
extern "C" {
#endif
qk_identity_v1 const* quactlize_kpack_decode_dense_identity_v1(void);
int quactlize_kpack_decode_dense_device_v1(char*,int,int32_t*,int32_t*);
int quactlize_kpack_decode_dense_query_v1(qkd_dense_call_v1 const*,qk_recipe_v1 const*,qk_resources_v1*);
int quactlize_kpack_decode_dense_prepare_v1(qkd_dense_call_v1 const*,qk_recipe_v1 const*,void**);
int quactlize_kpack_decode_dense_run_v1(void*,void* stream);
void quactlize_kpack_decode_dense_destroy_v1(void*);
qkd_compute_identity_v2 const* quactlize_kpack_decode_dense_identity_v2(void);
int quactlize_kpack_decode_dense_query_v2(qkd_dense_call_v2 const*,qk_recipe_v1 const*,qk_resources_v1*);
int quactlize_kpack_decode_dense_prepare_v2(qkd_dense_call_v2 const*,qk_recipe_v1 const*,void**);
// v2 prepared handles retain the existing run_v1/destroy_v1 lifecycle. A
// BF16-compute module rejects v1 query/prepare instead of changing v1 math.
#ifdef __cplusplus
}
#endif
