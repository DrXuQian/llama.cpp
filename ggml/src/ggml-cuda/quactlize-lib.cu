// dlopen loader for the quactlize PPU K-pack libraries. See quactlize-lib.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu) but inert unless -DGGML_NCP_QUACTLIZE=ON, in which case
// every entry reports unavailable and returns a decline so callers stay on the path they had.

#include "quactlize-lib.h"

#include "ggml.h"
#include "ggml-impl.h"   // GGML_LOG_*: goes through ggml's log callback, which an embedder can redirect

#include "quactlize/ppu_format_config.inc"   // quactlize's own per-format registry, for cross-checking only

#ifdef GGML_NCP_QUACTLIZE

#include <dlfcn.h>
#include <pthread.h>
#include <cstring>

// quactlize numbers its formats with the same integers ggml does. Neither project derives the other's, so the
// identity is asserted here: if either renumbers, this stops the build instead of decoding Q4_K as Q5_K.
static_assert(GGML_TYPE_Q2_K == 10, "quactlize qtype 10 is Q2_K");
static_assert(GGML_TYPE_Q3_K == 11, "quactlize qtype 11 is Q3_K");
static_assert(GGML_TYPE_Q4_K == 12, "quactlize qtype 12 is Q4_K");
static_assert(GGML_TYPE_Q5_K == 13, "quactlize qtype 13 is Q5_K");
static_assert(GGML_TYPE_Q6_K == 14, "quactlize qtype 14 is Q6_K");

#define QZ_QTYPE_MIN 10
#define QZ_QTYPE_MAX 14
#define QZ_NFMT (QZ_QTYPE_MAX - QZ_QTYPE_MIN + 1)

typedef int32_t (*qz_list_grouped_fn)(quactlize_ppu_config_v3 *, int32_t, int, int, int, int, int, int, int,
                                      const quactlize_ppu_placed_arrangement_v2 *);
typedef int64_t (*qz_ws_grouped_fn)(int, int, int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *);
typedef int     (*qz_dev_grouped_fn)(const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *,
                                     const int *, uint16_t *, int, int, int, int, int, int,
                                     void *, int64_t, void *, const char *,
                                     const quactlize_ppu_placed_arrangement_v2 *);
typedef int32_t (*qz_list_dense_fn)(quactlize_ppu_config_v3 *, int32_t, int, int, int, int, int,
                                    const quactlize_ppu_placed_arrangement_v2 *);
typedef int64_t (*qz_ws_dense_fn)(int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *);
typedef int     (*qz_dev_dense_fn)(const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *, uint16_t *,
                                   int, int, int, int, void *, int64_t, void *, const char *,
                                   const quactlize_ppu_placed_arrangement_v2 *);
typedef int32_t (*qz_identity_fn)(void);
typedef int     (*qz_arrangement_fn)(int, quactlize_ppu_placed_arrangement_v2 *);
typedef int     (*qz_prepare_fn)(const uint8_t *, uint8_t *, uint8_t *, uint8_t *, int, int, int, int,
                                 const quactlize_ppu_placed_arrangement_v2 *);
typedef int     (*qz_recover_fn)(const uint8_t *, const uint8_t *, const uint8_t *, uint8_t *, int, int, int, int,
                                 const quactlize_ppu_placed_arrangement_v2 *);
typedef int64_t (*qz_units_bytes_fn)(int, int, int);

struct qz_lib {
    void * handle;
    int32_t packed_format;          // what the library says it is; -2 when nothing is loaded
    qz_list_grouped_fn list_grouped;
    qz_ws_grouped_fn   ws_grouped;
    qz_dev_grouped_fn  dev_grouped;
    qz_list_dense_fn   list_dense;
    qz_ws_dense_fn     ws_dense;
    qz_dev_dense_fn    dev_dense;
    // Optional: absent in the currently shipping bundle. Absence is a decline, never a fallback to _v1 (Xplane).
    qz_arrangement_fn  arrangement;
    qz_prepare_fn      prepare;
    qz_recover_fn      recover;
    qz_units_bytes_fn  units_bytes;
};

static qz_lib g_libs[QZ_NFMT];

// The bundle's format table. Not derivable from the qtype -- fmt0 is Q4_K, not Q2_K -- and each library is asked to
// confirm it after load, so a reshuffled bundle fails to arm instead of decoding with the wrong reader.
static const struct { int qtype; int fmt; const char * soname; } g_fmt_table[QZ_NFMT] = {
    { GGML_TYPE_Q2_K, 2, "libquactlize_ppu_fmt2.so" },
    { GGML_TYPE_Q3_K, 3, "libquactlize_ppu_fmt3.so" },
    { GGML_TYPE_Q4_K, 0, "libquactlize_ppu_fmt0.so" },
    { GGML_TYPE_Q5_K, 1, "libquactlize_ppu_fmt1.so" },
    { GGML_TYPE_Q6_K, 4, "libquactlize_ppu_fmt4.so" },
};

static void qz_load_one(qz_lib * L, int qtype, int want_fmt, const char * soname) {
    L->handle = NULL;
    L->packed_format = -2;

    // RTLD_LOCAL is load-bearing, not hygiene: all five libraries export the same symbol names, so a global open
    // would let whichever came first answer every dlsym and silently decode one format with another's reader.
    void * h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        GGML_LOG_INFO("[quactlize] %s not loaded (%s) -> %s stays on the existing path\n",
                      soname, dlerror(), ggml_type_name((enum ggml_type) qtype));
        return;
    }

    qz_identity_fn identity = (qz_identity_fn) dlsym(h, "quactlize_ppu_build_packed_format_v1");
    if (!identity) {
        GGML_LOG_WARN("[quactlize] %s has no quactlize_ppu_build_packed_format_v1 -- cannot prove which format it "
                      "holds, refusing to arm it\n", soname);
        dlclose(h);
        return;
    }
    const int32_t got = identity();
    if (got != want_fmt) {
        GGML_LOG_WARN("[quactlize] %s reports packed format %d, the table asked for %d (%s) -- refusing to arm it\n",
                      soname, (int) got, want_fmt, ggml_type_name((enum ggml_type) qtype));
        dlclose(h);
        return;
    }

    L->list_grouped = (qz_list_grouped_fn) dlsym(h, "quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2");
    L->ws_grouped   = (qz_ws_grouped_fn)   dlsym(h, "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2");
    L->dev_grouped  = (qz_dev_grouped_fn)  dlsym(h, "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2");
    L->list_dense   = (qz_list_dense_fn)   dlsym(h, "quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2");
    L->ws_dense     = (qz_ws_dense_fn)     dlsym(h, "quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2");
    L->dev_dense    = (qz_dev_dense_fn)    dlsym(h, "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2");

    L->arrangement = (qz_arrangement_fn) dlsym(h, "quactlize_ppu_canonical_arrangement_v2");
    L->prepare     = (qz_prepare_fn)     dlsym(h, "quactlize_ppu_prepare_fully_quantized_for_arrangement_v2");
    L->recover     = (qz_recover_fn)     dlsym(h, "quactlize_ppu_recover_fully_quantized_for_arrangement_v2");
    L->units_bytes = (qz_units_bytes_fn) dlsym(h, "quactlize_ppu_units_bytes");

    // All six or none. A library with the launch entry but no inventory would make supports_op unanswerable, and one
    // with the inventory but no launch would advertise tactics it cannot run -- and the tensor that took this buffer
    // type has no un-K-packed copy left to fall back to.
    if (!L->list_grouped || !L->ws_grouped || !L->dev_grouped ||
        !L->list_dense   || !L->ws_dense   || !L->dev_dense) {
        GGML_LOG_WARN("[quactlize] %s is missing one of the arrangement-v2 entries -- refusing to arm it\n", soname);
        dlclose(h);
        memset(L, 0, sizeof(*L));
        L->packed_format = -2;
        return;
    }

    L->handle = h;
    L->packed_format = got;
    GGML_LOG_INFO("[quactlize] loaded %s -> %s (packed format %d)\n",
                  soname, ggml_type_name((enum ggml_type) qtype), (int) got);
}

static void qz_init(void) {
    for (int i = 0; i < QZ_NFMT; ++i) {
        qz_load_one(&g_libs[i], g_fmt_table[i].qtype, g_fmt_table[i].fmt, g_fmt_table[i].soname);
    }
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static qz_lib * qz_get(int qtype) {
    if (qtype < QZ_QTYPE_MIN || qtype > QZ_QTYPE_MAX) {
        return NULL;
    }
    pthread_once(&g_once, qz_init);
    qz_lib * L = &g_libs[qtype - QZ_QTYPE_MIN];
    GGML_ASSERT(g_fmt_table[qtype - QZ_QTYPE_MIN].qtype == qtype);
    return L->handle ? L : NULL;
}

extern "C" bool ggml_quactlize_available(int qtype) {
    return qz_get(qtype) != NULL;
}

extern "C" int32_t ggml_quactlize_build_packed_format(int qtype) {
    if (qtype < QZ_QTYPE_MIN || qtype > QZ_QTYPE_MAX) {
        return -2;
    }
    pthread_once(&g_once, qz_init);
    return g_libs[qtype - QZ_QTYPE_MIN].packed_format;
}

extern "C" int32_t ggml_quactlize_list_grouped_configs(
        int qtype, quactlize_ppu_config_v3 * configs, int32_t capacity,
        int total_rows, int n, int k, int group_size, int experts, int max_rows,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->list_grouped(configs, capacity, total_rows, n, k, group_size, experts, max_rows, qtype, arrangement);
}

extern "C" int64_t ggml_quactlize_grouped_workspace_bytes(
        int qtype, int total_rows, int max_rows, int n, int k, int experts,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->ws_grouped(total_rows, max_rows, n, k, experts, qtype, arrangement);
}

extern "C" int ggml_quactlize_grouped_dev(
        int qtype,
        const uint16_t * act, const uint8_t * low, const uint8_t * high, const uint8_t * units,
        const int * offsets, uint16_t * out,
        int total_rows, int n, int k, int experts, int max_rows,
        void * workspace, int64_t workspace_bytes, void * stream,
        const char * config_name,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->dev_grouped(act, low, high, units, offsets, out,
                          total_rows, n, k, experts, max_rows, qtype,
                          workspace, workspace_bytes, stream, config_name, arrangement);
}

extern "C" int32_t ggml_quactlize_list_dense_configs(
        int qtype, quactlize_ppu_config_v3 * configs, int32_t capacity,
        int m, int n, int k, int group_size,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->list_dense(configs, capacity, m, n, k, group_size, qtype, arrangement);
}

extern "C" int64_t ggml_quactlize_dense_workspace_bytes(
        int qtype, int m, int n, int k,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->ws_dense(m, n, k, qtype, arrangement);
}

extern "C" int ggml_quactlize_dense_dev(
        int qtype,
        const uint16_t * act, const uint8_t * low, const uint8_t * high, const uint8_t * units,
        uint16_t * out, int m, int n, int k,
        void * workspace, int64_t workspace_bytes, void * stream,
        const char * config_name,
        const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->dev_dense(act, low, high, units, out, m, n, k, qtype,
                        workspace, workspace_bytes, stream, config_name, arrangement);
}

// quactlize's own per-format registry, copied verbatim beside the ABI headers. Used ONLY to contradict a library
// that hands back an arrangement disagreeing with it -- llama.cpp never builds a descriptor from these rows.
#define X(Id, Name, QType, LowBits, HighBits, GroupSize, ScaleFirstTileK, FullyQuantizedTileK, PackedFormat) \
    { QType, LowBits, HighBits, GroupSize, PackedFormat },
static const struct { int qtype, low_bits, high_bits, group_size, packed_format; } g_registry[] = {
    QUACTLIZE_PPU_FORMAT_CONFIGS(X)
};
#undef X

static const int qz_registry_rows = (int) (sizeof(g_registry) / sizeof(g_registry[0]));

extern "C" bool ggml_quactlize_arrangement_for(int qtype, quactlize_ppu_placed_arrangement_v2 * out) {
    qz_lib * L = qz_get(qtype);
    if (!L || !L->arrangement || !out) {
        return false;
    }
    quactlize_ppu_placed_arrangement_v2 a;
    memset(&a, 0, sizeof(a));
    if (L->arrangement(qtype, &a) != 0) {
        return false;
    }
    if (a.version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2) {
        GGML_LOG_WARN("[quactlize] %s returned arrangement version %d, expected %d -- refusing it\n",
                      ggml_type_name((enum ggml_type) qtype), (int) a.version,
                      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2);
        return false;
    }
    // A K-pack artifact has no artifact-TileK axis; a non-zero one means an Xplane descriptor arrived by this door.
    if (a.artifact_tile_k != 0) {
        GGML_LOG_WARN("[quactlize] %s returned artifact_tile_k=%d; K-pack has no such axis -- refusing it\n",
                      ggml_type_name((enum ggml_type) qtype), (int) a.artifact_tile_k);
        return false;
    }
    for (int i = 0; i < qz_registry_rows; ++i) {
        if (g_registry[i].qtype != qtype) {
            continue;
        }
        if (a.bits       != g_registry[i].low_bits  ||
            a.high_bits  != g_registry[i].high_bits ||
            a.group_size != g_registry[i].group_size) {
            GGML_LOG_WARN("[quactlize] %s arrangement (bits=%d high=%d gs=%d) contradicts the format registry "
                          "(bits=%d high=%d gs=%d) -- refusing it\n",
                          ggml_type_name((enum ggml_type) qtype),
                          (int) a.bits, (int) a.high_bits, (int) a.group_size,
                          g_registry[i].low_bits, g_registry[i].high_bits, g_registry[i].group_size);
            return false;
        }
        if (L->packed_format != g_registry[i].packed_format) {
            GGML_LOG_WARN("[quactlize] %s library reports packed format %d, registry says %d -- refusing it\n",
                          ggml_type_name((enum ggml_type) qtype),
                          (int) L->packed_format, g_registry[i].packed_format);
            return false;
        }
        *out = a;
        return true;
    }
    return false;   // a qtype the registry does not know is not one this path serves
}

extern "C" bool ggml_quactlize_conversion_available(int qtype) {
    qz_lib * L = qz_get(qtype);
    if (!L || !L->prepare || !L->recover || !L->units_bytes) {
        return false;
    }
    quactlize_ppu_placed_arrangement_v2 a;
    if (!ggml_quactlize_arrangement_for(qtype, &a)) {
        return false;
    }
    // An exported symbol is not an answer. One 256-code superblock is the smallest shape every k-quant format is
    // defined on, so a library that cannot size its own metadata plane for it cannot convert anything -- and it is
    // far better to learn that here, where the tensor simply does not take the buffer type, than in set_tensor,
    // where the model is already half loaded and the only honest move left is to abort.
    return L->units_bytes(1, 256, qtype) >= 0;
}

extern "C" int ggml_quactlize_prepare(
        int qtype, const uint8_t * blocks, uint8_t * low, uint8_t * high, uint8_t * units,
        int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L || !L->prepare) return -1;
    return L->prepare(blocks, low, high, units, n, k, experts, qtype, arrangement);
}

extern "C" int ggml_quactlize_recover(
        int qtype, const uint8_t * low, const uint8_t * high, const uint8_t * units, uint8_t * recovered,
        int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L || !L->recover) return -1;
    return L->recover(low, high, units, recovered, n, k, experts, qtype, arrangement);
}

extern "C" int64_t ggml_quactlize_units_bytes(int qtype, int n, int k) {
    qz_lib * L = qz_get(qtype);
    if (!L || !L->units_bytes) return -1;
    return L->units_bytes(n, k, qtype);
}

#else  // quactlize off: inert stubs

extern "C" bool    ggml_quactlize_available(int)            { return false; }
extern "C" int32_t ggml_quactlize_build_packed_format(int)  { return -2; }

extern "C" int32_t ggml_quactlize_list_grouped_configs(
        int, quactlize_ppu_config_v3 *, int32_t, int, int, int, int, int, int,
        const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int64_t ggml_quactlize_grouped_workspace_bytes(
        int, int, int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int ggml_quactlize_grouped_dev(
        int, const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        const int *, uint16_t *, int, int, int, int, int,
        void *, int64_t, void *, const char *,
        const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int32_t ggml_quactlize_list_dense_configs(
        int, quactlize_ppu_config_v3 *, int32_t, int, int, int, int,
        const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int64_t ggml_quactlize_dense_workspace_bytes(
        int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int ggml_quactlize_dense_dev(
        int, const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        uint16_t *, int, int, int, void *, int64_t, void *, const char *,
        const quactlize_ppu_placed_arrangement_v2 *) { return -1; }

extern "C" bool ggml_quactlize_arrangement_for(int, quactlize_ppu_placed_arrangement_v2 *) { return false; }
extern "C" bool ggml_quactlize_conversion_available(int) { return false; }
extern "C" int  ggml_quactlize_prepare(int, const uint8_t *, uint8_t *, uint8_t *, uint8_t *, int, int, int,
                                       const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int  ggml_quactlize_recover(int, const uint8_t *, const uint8_t *, const uint8_t *, uint8_t *,
                                       int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int64_t ggml_quactlize_units_bytes(int, int, int) { return -1; }

#endif // GGML_NCP_QUACTLIZE
