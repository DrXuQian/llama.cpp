#pragma once
// C ABI of the Quactlize packer and dense runtime in one library, installed through ncp_flash_lib. Only the subset
// quactlize-buft.cu dlsym's is mirrored here, so llama.cpp needs no include path into ncp_flash_lib -- the same model
// as ncp-moe-lib.h. Source of truth: quactlize/packing/api.h, quactlize/dense/api.h and
// quactlize/include/quactlize_ppu_config.h.
//
// These must stay byte-identical to what the library exports: the struct layouts are passed across the dlopen
// boundary and the symbol names are dlsym'd as written. Both sides move together or not at all.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID    UINT64_C(0x51344b5034540001)
#define QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID UINT64_C(0x514b504b54000001)
#define QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID    UINT64_C(0x51384b5032540001)

typedef struct quactlize_ppu_placed_arrangement_v2 {
    int32_t  version;
    int32_t  layout;
    int32_t  bits;
    int32_t  high_bits;
    int32_t  artifact_tile_k;
    int32_t  transport_tile_k;
    int32_t  group_size;
    int32_t  reserved;
    uint64_t mapping_id;
} quactlize_ppu_placed_arrangement_v2;

typedef struct {
    uint64_t raw_bytes, low_bytes, high_bytes, units_bytes;
} quactlize_ppu_kpack_sizes_v1;

int quactlize_ppu_kpack_canonical_arrangement_v1(int qtype, quactlize_ppu_placed_arrangement_v2 * arrangement);

// Dense is experts=1. Each plane has contiguous, equal-sized expert slices.
int quactlize_ppu_kpack_sizes_for_arrangement_v1(int                                         n,
                                                 int                                         k,
                                                 int                                         experts,
                                                 int                                         qtype,
                                                 const quactlize_ppu_placed_arrangement_v2 * arrangement,
                                                 quactlize_ppu_kpack_sizes_v1 *              sizes);

// All pointers are on the current device; high is NULL when high_bytes=0. Success means enqueued on `stream`.
int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
    const uint8_t *                             blocks,
    uint8_t *                                   low,
    uint8_t *                                   high,
    uint8_t *                                   units,
    int                                         n,
    int                                         k,
    int                                         experts,
    int                                         qtype,
    const quactlize_ppu_placed_arrangement_v2 * arrangement,
    void *                                      stream);

// Public consumer ABI in libncp_quactlize.so, built by Quactlize and installed by ncp_flash_lib.
// A[M,K] and C[M,N] use F32 storage; strides count elements. W[N,K] uses the
// canonical planes below. No activation quantization to int8 or clipping.
typedef struct {
    uint32_t                            version, size;
    int32_t                             qtype, n, k, m;
    int64_t                             input_stride, output_stride;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1        sizes;
} quactlize_ppu_dense_params_v1;

typedef struct {
    uint32_t version, size;
    uint64_t workspace_bytes;
    // route: 0=FQ TC, 1=SF TC, 2=SIMT, 3=full dequant + BF16 GEMM.
    // policy identifies the library's selector; compute_type: 0=F16, 1=BF16.
    int32_t  route, policy, split, compute_type;
    char     config[192];
} quactlize_ppu_dense_plan_info_v1;

// Pure host admission including canonical layout/sizes: 1=supported, 0=unsupported, negative=invalid ABI.
// Must not allocate, JIT, query the current device or dereference device data.
// m=0 asks whether this weight supports every positive M (including decode and prefill), not an empty GEMM.
// Automatic placement requires this guarantee: no raw-GGUF copy remains after packing.
int quactlize_ppu_dense_supported_v1(const quactlize_ppu_dense_params_v1 * params);
// Device is the process-local CUDA ordinal. Returns 0 and a non-NULL context on success.
int quactlize_ppu_dense_create_v1(int device, void ** context);

// Outside capture. Selects the admitted kernel and prepares an immutable plan.
// Context owns plans/modules, caller owns weights and per-launch workspace.
// prepare/run return 0 on success; no raw-GGUF fallback exists after packing.
int quactlize_ppu_dense_prepare_v1(void *                                context,
                                   const quactlize_ppu_dense_params_v1 * params,
                                   const void *                          low,
                                   const void *                          high,
                                   const void *                          units,
                                   void **                               plan,
                                   quactlize_ppu_dense_plan_info_v1 *    info);

// Enqueue only: no allocation, JIT, host wait or device-to-host transfer.
// Workspace is 256-byte aligned, private to this stream until completion.
// Plans must accept different input/output/workspace addresses on each call.
// Includes adapters, per-call SF/full dequant and the selected Split-K reducer. Propagates launch errors.
int quactlize_ppu_dense_run_v1(void *        plan,
                               const float * input,
                               float *       output,
                               void *        workspace,
                               uint64_t      workspace_bytes,
                               void *        stream);

// Caller completes all uses before destroying the context or weight planes.
void         quactlize_ppu_dense_destroy_v1(void * context);
// Non-NULL, thread-local message for the last error; valid until the next API call on this thread.
const char * quactlize_ppu_dense_error_v1(void);

#ifdef __cplusplus
}

static_assert(sizeof(quactlize_ppu_dense_params_v1) == 112, "dense parameter ABI drift");
static_assert(sizeof(quactlize_ppu_dense_plan_info_v1) == 224, "dense plan ABI drift");
#endif
