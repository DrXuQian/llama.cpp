#include "quactlize/quactlize_ppu_pack.h"
#include "quactlize/kpack_dispatch.h"
#include <stdlib.h>

int quactlize_ppu_kpack_canonical_arrangement_v1(int q, quactlize_ppu_placed_arrangement_v2 * a) {
    if (q!=8 || !a) return 20;
    *a=(quactlize_ppu_placed_arrangement_v2){2,4,8,0,0,32,32,0,UINT64_C(0x51384b5032540001)};
    if (getenv("QZ_Q8_BAD_DESCRIPTOR")) a->group_size=16;
    return 0;
}
int quactlize_kpack_dispatch_q8_weight_supported_v1(int n,int k,int e,int route,uint64_t map) {
    return !getenv("QZ_Q8_NO_CAPABILITY") && n>0 && n%256==0 && k>0 && k%256==0 && e>0 &&
        (route==1 || route==3) && map==UINT64_C(0x51384b5032540001);
}

int quactlize_ppu_kpack_sizes_for_arrangement_v1(
        int n, int k, int experts, int qtype,
        const quactlize_ppu_placed_arrangement_v2 * arr, quactlize_ppu_kpack_sizes_v1 * out) {
    static const int unit_bytes[] = {20, 14, 16, 16, 18};
    if (qtype==8 && arr && out && n>0 && n%256==0 && k>0 && k%256==0 && experts>0) {
        uint64_t codes=(uint64_t)n*k*experts;
        *out=(quactlize_ppu_kpack_sizes_v1){codes/32*34,codes,0,codes/32*2};
        return 0;
    }
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
        n != 256 || k != 512 || experts != 3 || (qtype!=8 && (qtype < 10 || qtype > 14))) return 20;
    return 0;
}
int quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1(
        const uint8_t * gate,const uint8_t * up,uint8_t * low,uint8_t * high,uint8_t * units,
        int n,int k,int experts,int q,const quactlize_ppu_placed_arrangement_v2 * a,void * stream) {
    if (up!=(const uint8_t*)6) return 20;
    return quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(gate,low,high,units,n,k,experts,q,a,stream);
}
#endif
