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
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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

typedef int64_t (*qz_ws_grouped_fn)(int, int, int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *);
typedef int     (*qz_dev_grouped_fn)(const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *,
                                     const int *, uint16_t *, int, int, int, int, int, int,
                                     void *, int64_t, void *, const char *,
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

typedef int32_t (*qz_any_m_dense_fn)(int, int, int, const quactlize_ppu_placed_arrangement_v2 *);
typedef int32_t (*qz_any_m_grouped_fn)(int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *);

struct qz_lib {
    void * handle;
    int32_t packed_format;          // what the library says it is; -2 when nothing is loaded
    qz_any_m_grouped_fn any_m_grouped;
    qz_ws_grouped_fn   ws_grouped;
    qz_dev_grouped_fn  dev_grouped;
    qz_any_m_dense_fn  any_m_dense;
    qz_ws_dense_fn     ws_dense;
    qz_dev_dense_fn    dev_dense;
    // Present from bundle 2826cf1 on. Absence is a decline, never a fallback to _v1 (Xplane).
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

// The PPU SDK wrapper, once and RTLD_GLOBAL, before any format library: the handoff's load order. On a PPU build
// it is already a dependency of the CUDA backend, so this is normally a refcount bump; on a host without the SDK
// it is skipped and the format libraries then fail to open on their own terms, which is the right outcome.
static void qz_preload_sdk_wrapper(void) {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const char * sdk = getenv("PPU_SDK");
    if (!sdk || !*sdk) {
        return;
    }
    const std::string wrapper = std::string(sdk) + "/lib/libhggc_wrapper.so";
    if (!dlopen(wrapper.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
        GGML_LOG_INFO("[quactlize] SDK wrapper %s not preloaded (%s)\n", wrapper.c_str(), dlerror());
    }
}

static void qz_load_one(qz_lib * L, int qtype, int want_fmt, const char * soname) {
    L->handle = NULL;
    L->packed_format = -2;

    // The deployment contract (quactlize docs/LLAMA_CPP_KPACK_HANDOFF.md): the SDK wrapper first and RTLD_GLOBAL,
    // then each format library by ABSOLUTE PATH from the bundle directory, RTLD_LOCAL. QUACTLIZE_PPU_BUNDLE names
    // that directory -- the same variable quactlize's own packer takes -- and without it the SONAME goes to the
    // dynamic loader as before, which is how the stub-driven tests find their doubles.
    //
    // RTLD_LOCAL is load-bearing, not hygiene: all five libraries export the same symbol names, so a global open
    // would let whichever came first answer every dlsym and silently decode one format with another's reader.
    qz_preload_sdk_wrapper();
    std::string path = soname;
    if (const char * dir = getenv("QUACTLIZE_PPU_BUNDLE")) {
        if (*dir) {
            path = std::string(dir) + "/" + soname;
        }
    }
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        GGML_LOG_INFO("[quactlize] %s not loaded (%s) -> %s stays on the existing path\n",
                      path.c_str(), dlerror(), ggml_type_name((enum ggml_type) qtype));
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

    L->any_m_grouped = (qz_any_m_grouped_fn) dlsym(h, "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2");
    L->ws_grouped   = (qz_ws_grouped_fn)   dlsym(h, "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2");
    L->dev_grouped  = (qz_dev_grouped_fn)  dlsym(h, "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2");
    L->any_m_dense   = (qz_any_m_dense_fn)   dlsym(h, "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2");
    L->ws_dense     = (qz_ws_dense_fn)     dlsym(h, "quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2");
    L->dev_dense    = (qz_dev_dense_fn)    dlsym(h, "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2");

    L->arrangement = (qz_arrangement_fn) dlsym(h, "quactlize_ppu_canonical_arrangement_v2");
    L->prepare     = (qz_prepare_fn)     dlsym(h, "quactlize_ppu_prepare_fully_quantized_for_arrangement_v2");
    L->recover     = (qz_recover_fn)     dlsym(h, "quactlize_ppu_recover_fully_quantized_for_arrangement_v2");
    L->units_bytes = (qz_units_bytes_fn) dlsym(h, "quactlize_ppu_units_bytes");

    // All six or none. A library with the launch entry but no any-M admission query would make supports_op
    // unanswerable -- it runs before any M exists -- and one with the query but no launch would admit tensors it
    // cannot run; either way the tensor that took this buffer type has no un-K-packed copy left to fall back to.
    if (!L->any_m_grouped || !L->ws_grouped || !L->dev_grouped ||
        !L->any_m_dense   || !L->ws_dense   || !L->dev_dense) {
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

extern "C" int32_t ggml_quactlize_grouped_any_m_valid(
        int qtype, int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->any_m_grouped(n, k, experts, qtype, arrangement);
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

extern "C" int32_t ggml_quactlize_dense_any_m_valid(
        int qtype, int n, int k, const quactlize_ppu_placed_arrangement_v2 * arrangement) {
    qz_lib * L = qz_get(qtype);
    if (!L) return -1;
    return L->any_m_dense(n, k, qtype, arrangement);
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
    // An exported symbol is not an answer: a library that cannot size its own metadata plane cannot convert
    // anything, and it is far better to learn that here, where the tensor simply does not take the buffer type,
    // than in set_tensor, where the model is half loaded and the only honest move left is to abort.
    //
    // The probe shape has to be inside EVERY format's domain. The library takes N and K in multiples of 256, and
    // Q3_K/Q6_K want K in multiples of 512 (their unit packs two superblocks). (256, 512) satisfies all five;
    // (1, 256) did not, and the first run against the real bundle said so for all five formats at once.
    return L->units_bytes(256, 512, qtype) >= 0;
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

extern "C" bool ggml_quactlize_plane_sizes(
        int qtype, int64_t n, int64_t k, int64_t experts,
        const quactlize_ppu_placed_arrangement_v2 * arrangement,
        int64_t * low_bytes, int64_t * high_bytes, int64_t * units_bytes) {
    if (!arrangement || !low_bytes || !high_bytes || !units_bytes) {
        return false;
    }
    if (n <= 0 || k <= 0 || experts <= 0) {
        return false;
    }

    const int64_t unit_per_expert = ggml_quactlize_units_bytes(qtype, (int) n, (int) k);
    if (unit_per_expert < 0) {
        return false;
    }

    const int64_t codes = experts * n * k;
    *low_bytes   = codes * arrangement->bits      / 8;
    *high_bytes  = codes * arrangement->high_bits / 8;
    *units_bytes = experts * unit_per_expert;

    return true;
}

extern "C" int ggml_quactlize_convert_threads(int64_t experts) {
    if (experts <= 0) {
        return 1;
    }
    if (const char * v = getenv("GGML_QUACTLIZE_CONVERT_THREADS")) {
        const int n = atoi(v);
        if (n >= 1) {
            return (int) (n < experts ? n : experts);
        }
    }
    const unsigned hw = std::thread::hardware_concurrency();
    const int64_t  t  = hw ? (int64_t) hw : 1;
    return (int) (t < experts ? t : experts);
}

// One pass: convert (optionally split across the expert axis) and round-trip. Returns 0 when the recovered bytes
// equal the input exactly.
static int qz_convert_once(
        int qtype, const unsigned char * blocks,
        unsigned char * low, unsigned char * high, unsigned char * units, unsigned char * recovered,
        int64_t nbytes, int64_t n, int64_t k, int64_t experts,
        const quactlize_ppu_placed_arrangement_v2 * arr,
        int64_t low_bytes, int64_t high_bytes, int64_t units_bytes, int nthreads) {
    unsigned char * high_arg = high_bytes ? high : nullptr;

    int rc = 0;
    if (nthreads <= 1 || experts <= 1) {
        rc = ggml_quactlize_prepare(qtype, blocks, low, high_arg, units, (int) n, (int) k, (int) experts, arr);
    } else {
        // Every plane has to divide evenly by expert or the slice arithmetic is meaningless.
        if (nbytes % experts || low_bytes % experts || high_bytes % experts || units_bytes % experts) {
            return -2;
        }
        const int64_t blk_e   = nbytes      / experts;
        const int64_t low_e   = low_bytes   / experts;
        const int64_t high_e  = high_bytes  / experts;
        const int64_t units_e = units_bytes / experts;

        std::vector<int>         rcs((size_t) nthreads, 0);
        std::vector<std::thread> workers;
        workers.reserve((size_t) nthreads);

        for (int t = 0; t < nthreads; ++t) {
            const int64_t first = experts * t       / nthreads;
            const int64_t last  = experts * (t + 1) / nthreads;
            if (first >= last) {
                continue;
            }
            workers.emplace_back([&, t, first, last]() {
                rcs[(size_t) t] = ggml_quactlize_prepare(
                    qtype,
                    blocks + first*blk_e,
                    low    + first*low_e,
                    high_arg ? high + first*high_e : nullptr,
                    units  + first*units_e,
                    (int) n, (int) k, (int) (last - first), arr);
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        for (const int r : rcs) {
            if (r != 0) {
                rc = r;
                break;
            }
        }
    }
    if (rc != 0) {
        return rc;
    }

    const int rrc = ggml_quactlize_recover(qtype, low, high_arg, units, recovered,
                                           (int) n, (int) k, (int) experts, arr);
    if (rrc != 0) {
        return rrc;
    }
    return memcmp(recovered, blocks, (size_t) nbytes) == 0 ? 0 : -1;
}

extern "C" int ggml_quactlize_convert_verified(
        int qtype, const unsigned char * blocks,
        unsigned char * low, unsigned char * high, unsigned char * units, unsigned char * recovered,
        int64_t nbytes, int64_t n, int64_t k, int64_t experts,
        const quactlize_ppu_placed_arrangement_v2 * arrangement,
        int64_t low_bytes, int64_t high_bytes, int64_t units_bytes,
        int * threads_used) {
    // Learned once per process, not per tensor: paying a failed parallel attempt on every expert tensor of a model
    // would cost more than the threading saves.
    static bool split_by_expert_rejected = false;

    int nthreads = split_by_expert_rejected ? 1 : ggml_quactlize_convert_threads(experts);
    int rc = qz_convert_once(qtype, blocks, low, high, units, recovered, nbytes, n, k, experts, arrangement,
                             low_bytes, high_bytes, units_bytes, nthreads);

    if (rc != 0 && nthreads > 1) {
        GGML_LOG_WARN("[quactlize] conversion split across %d threads did not round-trip (rc=%d) -- retrying in "
                      "one thread to tell a bad split from a bad library\n", nthreads, rc);
        nthreads = 1;
        rc = qz_convert_once(qtype, blocks, low, high, units, recovered, nbytes, n, k, experts, arrangement,
                             low_bytes, high_bytes, units_bytes, 1);
        if (rc == 0) {
            split_by_expert_rejected = true;
            GGML_LOG_WARN("[quactlize] the serial conversion round-trips, so this format's artifact is not a "
                          "per-expert concatenation -- staying single-threaded for the rest of this load\n");
        }
    }

    if (threads_used) {
        *threads_used = nthreads;
    }
    return rc;
}

#else  // quactlize off: inert stubs

extern "C" bool    ggml_quactlize_available(int)            { return false; }
extern "C" int32_t ggml_quactlize_build_packed_format(int)  { return -2; }

extern "C" int32_t ggml_quactlize_grouped_any_m_valid(
        int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int64_t ggml_quactlize_grouped_workspace_bytes(
        int, int, int, int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int ggml_quactlize_grouped_dev(
        int, const uint16_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        const int *, uint16_t *, int, int, int, int, int,
        void *, int64_t, void *, const char *,
        const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
extern "C" int32_t ggml_quactlize_dense_any_m_valid(
        int, int, int, const quactlize_ppu_placed_arrangement_v2 *) { return -1; }
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
extern "C" bool ggml_quactlize_plane_sizes(int, int64_t, int64_t, int64_t,
        const quactlize_ppu_placed_arrangement_v2 *, int64_t *, int64_t *, int64_t *) { return false; }
extern "C" int ggml_quactlize_convert_threads(int64_t) { return 1; }
extern "C" int ggml_quactlize_convert_verified(int, const unsigned char *, unsigned char *, unsigned char *,
        unsigned char *, unsigned char *, int64_t, int64_t, int64_t, int64_t,
        const quactlize_ppu_placed_arrangement_v2 *, int64_t, int64_t, int64_t, int *) { return -1; }

#endif // GGML_NCP_QUACTLIZE
