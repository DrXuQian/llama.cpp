#include "quactlize/quactlize_ppu_pack.h"
#include <stdlib.h>

int quactlize_ppu_kpack_sizes_for_arrangement_v1(
        int n, int k, int experts, int qtype,
        const quactlize_ppu_placed_arrangement_v2 * arr, quactlize_ppu_kpack_sizes_v1 * out) {
    static const int unit_bytes[] = {20, 14, 16, 16, 18};
    if (qtype < 10 || qtype > 14 || !arr || !out || n <= 0 || k <= 0 || experts <= 0) return 20;
    if (n % 256 || k % 256 || ((qtype == 11 || qtype == 14) && k % 512)) return 24;
    if (getenv("QZ_PACK_QUERY_FAIL")) return 38;
    const uint64_t codes = (uint64_t) n * k * experts;
    out->low_bytes = codes * arr->bits / 8;
    out->high_bytes = codes * arr->high_bits / 8;
    out->units_bytes = (uint64_t) n * (k / 256) * experts * unit_bytes[qtype - 10];
    out->raw_bytes = out->low_bytes + out->high_bytes + out->units_bytes;
    if (getenv("QZ_PACK_BAD_SIZES")) out->low_bytes++;
    return 0;
}

#ifndef QZ_PACK_OMIT_PREPARE
int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
        const uint8_t * raw, uint8_t * low, uint8_t * high, uint8_t * units,
        int n, int k, int experts, int qtype,
        const quactlize_ppu_placed_arrangement_v2 * arr, void * stream) {
    if (getenv("QZ_PACK_LAUNCH_FAIL")) return 41;
    if (raw != (const uint8_t *) 1 || low != (uint8_t *) 2 || units != (uint8_t *) 4 ||
        stream != (void *) 5 || high != (arr->high_bits ? (uint8_t *) 3 : NULL) ||
        n != 256 || k != 512 || experts != 3 || qtype < 10 || qtype > 14) return 20;
    return 0;
}
#endif
