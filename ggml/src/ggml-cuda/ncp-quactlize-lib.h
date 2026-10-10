#pragma once
// C ABI of libquactlize_ppu_pack.so (Quactlize K-pack producer), built in the ncp_flash_lib repo. Only the subset
// quactlize-buft.cu dlsym's is mirrored here, so llama.cpp needs no include path into ncp_flash_lib -- the same model
// as ncp-moe-lib.h. Source of truth: third_party/quactlize/quactlize/packing/api.h and
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

#ifdef __cplusplus
}
#endif
