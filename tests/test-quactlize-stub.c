// Test double for one quactlize K-pack library. Exports the entries the llama.cpp loader dlsym's; every value is
// controlled by the environment so a single .so can play the positive case and each negative control.
//
//   QZ_STUB_FMT        what build_packed_format_v1 returns          (default: from QZ_STUB_QTYPE's registry row)
//   QZ_STUB_QTYPE      the qtype this library claims to serve       (default: 12 = Q4_K)
//   QZ_STUB_NO_IDENTITY   omit nothing, but report -1 (default lib)
//   QZ_STUB_BAD_GS     add 1 to the group_size it reports           (negative control 1)
//   QZ_STUB_BAD_ATK    report artifact_tile_k = 256                 (negative control 2)
//   QZ_STUB_NO_ARRANGEMENT  canonical_arrangement_v2 returns non-zero
//   QZ_STUB_NO_CONVERSION   prepare/recover/units_bytes report failure
//   QZ_STUB_BAD_UNITS  units_bytes one superblock too large     (negative control 3)
//   QZ_STUB_NO_TACTIC  the inventories return 0 rows
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct arrangement_v2 {
    int32_t version, layout, bits, high_bits, artifact_tile_k, transport_tile_k, group_size, reserved;
    uint64_t mapping_id;
};

#define LAYOUT_Q4_KPACK4 1
#define LAYOUT_KQUANT    2
#define MAP_Q4_KPACK4    UINT64_C(0x51344b5034540001)
#define MAP_KQUANT       UINT64_C(0x514b504b54000001)

// qtype, low_bits, high_bits, group_size, transport_tile_k, packed_format
static const int reg[5][6] = {
    { 10, 2, 0, 16, 128, 2 },
    { 11, 2, 1, 16, 256, 3 },
    { 12, 4, 0, 32,  64, 0 },
    { 13, 4, 1, 32, 256, 1 },
    { 14, 4, 2, 16, 128, 4 },
};

static int env_on(const char * k) { const char * v = getenv(k); return v && *v && strcmp(v, "0") != 0; }

// Which format this copy of the stub plays. Compile-time, because all five copies are loaded into ONE process and
// they would otherwise all read the same environment variable and all claim the same qtype -- which is exactly the
// confusion (five libraries, identical symbol names) the RTLD_LOCAL handling exists to prevent, so the double must
// not reintroduce it. QZ_STUB_QTYPE remains as a fallback for a single-library build.
#ifdef QZ_STUB_BUILD_QTYPE
static int my_qtype(void) { return QZ_STUB_BUILD_QTYPE; }
#else
static int my_qtype(void) { const char * v = getenv("QZ_STUB_QTYPE"); return v ? atoi(v) : 12; }
#endif
static const int * row(int qtype) {
    for (int i = 0; i < 5; ++i) if (reg[i][0] == qtype) return reg[i];
    return 0;
}

int32_t quactlize_ppu_build_packed_format_v1(void) {
    if (env_on("QZ_STUB_NO_IDENTITY")) return -1;
    const char * v = getenv("QZ_STUB_FMT");
    if (v) return atoi(v);
    const int * r = row(my_qtype());
    return r ? r[5] : -1;
}

int quactlize_ppu_canonical_arrangement_v2(int qtype, struct arrangement_v2 * out) {
    if (!out || env_on("QZ_STUB_NO_ARRANGEMENT")) return 1;
    if (qtype != my_qtype()) return 1;
    const int * r = row(qtype);
    if (!r) return 1;
    memset(out, 0, sizeof(*out));
    out->version          = 2;
    out->layout           = qtype == 12 ? LAYOUT_Q4_KPACK4 : LAYOUT_KQUANT;
    out->bits             = r[1];
    out->high_bits        = r[2];
    out->group_size       = r[3] + (env_on("QZ_STUB_BAD_GS") ? 1 : 0);
    out->transport_tile_k = r[4];
    out->artifact_tile_k  = env_on("QZ_STUB_BAD_ATK") ? 256 : 0;
    out->mapping_id       = qtype == 12 ? MAP_Q4_KPACK4 : MAP_KQUANT;
    return 0;
}

int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2(
        void * c, int32_t cap, int tr, int n, int k, int gs, int e, int mr, int qt, const void * a) {
    (void)c;(void)cap;(void)tr;(void)n;(void)k;(void)gs;(void)e;(void)mr;(void)a;
    if (env_on("QZ_STUB_NO_TACTIC") || qt != my_qtype()) return 0;
    return 1;
}
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2(
        void * c, int32_t cap, int m, int n, int k, int gs, int qt, const void * a) {
    (void)c;(void)cap;(void)m;(void)n;(void)k;(void)gs;(void)a;
    if (env_on("QZ_STUB_NO_TACTIC") || qt != my_qtype()) return 0;
    return 1;
}
int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
        int tr,int mr,int n,int k,int e,int qt,const void*a){(void)tr;(void)mr;(void)n;(void)k;(void)e;(void)qt;(void)a;return 0;}
int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2(
        int m,int n,int k,int qt,const void*a){(void)m;(void)n;(void)k;(void)qt;(void)a;return 0;}
int quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2(
        const void*x,const void*l,const void*h,const void*u,const int*o,void*out,
        int tr,int n,int k,int e,int mr,int qt,void*w,int64_t wb,void*s,const char*cn,const void*a){
    (void)x;(void)l;(void)h;(void)u;(void)o;(void)out;(void)tr;(void)n;(void)k;(void)e;(void)mr;(void)qt;
    (void)w;(void)wb;(void)s;(void)cn;(void)a; return 0;}
int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2(
        const void*x,const void*l,const void*h,const void*u,void*out,
        int m,int n,int k,int qt,void*w,int64_t wb,void*s,const char*cn,const void*a){
    (void)x;(void)l;(void)h;(void)u;(void)out;(void)m;(void)n;(void)k;(void)qt;
    (void)w;(void)wb;(void)s;(void)cn;(void)a; return 0;}

// Per EXPERT, and therefore including N: the ABI says units holds experts*quactlize_ppu_units_bytes(N,K,qtype)
// bytes. A double that drops the N factor would let a caller size the metadata plane N times too small and still
// pass every test here, so it is modelled the way the contract reads.
int64_t quactlize_ppu_units_bytes(int n, int k, int qtype) {
    if (env_on("QZ_STUB_NO_CONVERSION")) return -1;
    // k-quant metadata bytes per 256-code superblock: Q2 20, Q3 14, Q4 16, Q5 16, Q6 18
    int per_sb = 0;
    switch (qtype) { case 10: per_sb=20; break; case 11: per_sb=14; break; case 12: per_sb=16; break;
                     case 13: per_sb=16; break; case 14: per_sb=18; break; default: return -1; }
    if (n <= 0 || k <= 0 || k % 256 != 0) return -1;
    // QZ_STUB_BAD_UNITS: one superblock's worth of metadata too much. The descriptor still matches the registry,
    // so nothing upstream fires -- only the byte-neutrality identity can catch this, which is the point.
    return (int64_t) n * (k / 256) * per_sb + (env_on("QZ_STUB_BAD_UNITS") ? per_sb : 0);
}
int quactlize_ppu_prepare_fully_quantized_for_arrangement_v2(
        const void*b,void*l,void*h,void*u,int n,int k,int e,int qt,const void*a){
    (void)b;(void)l;(void)h;(void)u;(void)n;(void)k;(void)e;(void)qt;(void)a;
    return env_on("QZ_STUB_NO_CONVERSION") ? 1 : 0;}
int quactlize_ppu_recover_fully_quantized_for_arrangement_v2(
        const void*l,const void*h,const void*u,void*r,int n,int k,int e,int qt,const void*a){
    (void)l;(void)h;(void)u;(void)r;(void)n;(void)k;(void)e;(void)qt;(void)a;
    return env_on("QZ_STUB_NO_CONVERSION") ? 1 : 0;}
