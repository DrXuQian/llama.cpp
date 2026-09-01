// dlopen loader for the external PPU kernel libraries (libncp_fa.so, libncp_moe.so, libncp_gdn.so). See
// ncp-lib.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu), but each of the three features is inert unless its
// own build flag is set: cmake -DGGML_NCP_FA=ON / -DGGML_NCP_MOE=ON / -DGGML_NCP_GDN=ON. A feature that is off
// compiles to stubs -- *_available() returns false, every fwd call returns -1 -- so its callers fall straight
// through to the inline ggml path.

#include "ncp-lib.h"
#include "ncp-moe-lib.h"

#include "ggml-impl.h"   // GGML_LOG_*: goes through ggml's log callback, which an embedder can redirect

#if defined(GGML_NCP_FA) || defined(GGML_NCP_MOE) || defined(GGML_NCP_GDN)

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cstring>

// The .so live next to the binaries that load them (build/bin, beside libggml-*.so). ggml-cuda is linked with an
// $ORIGIN runpath, so the bare soname resolves there; LD_LIBRARY_PATH still wins if you want another copy.
static void * open_lib(const char * soname) {
    void * h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (h) {
        GGML_LOG_INFO("[ncp-lib] loaded %s\n", soname);
    } else {
        GGML_LOG_WARN("[ncp-lib] %s not loaded (%s) -> inline ggml fallback\n", soname, dlerror());
    }
    return h;
}

#endif // any feature live

// ---- FlashAttention-3 ----

#ifdef GGML_NCP_FA

typedef int (*ncp_fa_fn)(const void *, const void *, const void *, void *,
                         int, int, int, int, int,
                         int, int,
                         long long, long long, long long, long long, long long, long long,
                         long long, long long, long long, long long, long long, long long,
                         float, float,
                         int, int, int,
                         int, void *);

static ncp_fa_fn g_fa_fn = NULL;

static void ncp_fa_init(void) {
    void * fa = open_lib("libncp_fa.so");
    // _v3, deliberately: C does not mangle, so a v2 .so would take this call and read the trailing arguments as
    // garbage. The rename makes a stale .so fail the dlsym and fall back, instead of silently computing garbage.
    if (fa) {
        g_fa_fn = (ncp_fa_fn) dlsym(fa, "ncp_flash_attn_fwd_v3");
    }
}

static pthread_once_t g_fa_once = PTHREAD_ONCE_INIT;
static void ensure_fa_init(void) { pthread_once(&g_fa_once, ncp_fa_init); }

extern "C" bool ggml_ncp_lib_fa_available(void) { ensure_fa_init(); return g_fa_fn != NULL; }

extern "C" int ggml_ncp_lib_flash_attn_fwd(
        const void * q, const void * k, const void * v, void * o,
        int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv,
        int head_dim, int head_dim_v,
        long long qbs, long long qhs, long long qrs, long long kbs, long long khs, long long krs,
        long long vbs, long long vhs, long long vrs, long long obs, long long ohs, long long ors,
        float scale, float softcap, int is_causal, int wl, int wr, int dtype, void * stream) {
    ensure_fa_init();
    if (!g_fa_fn) return -1;
    return g_fa_fn(q, k, v, o, batch, seqlen_q, seqlen_k, n_heads_q, n_heads_kv, head_dim, head_dim_v,
                   qbs, qhs, qrs, kbs, khs, krs, vbs, vhs, vrs, obs, ohs, ors,
                   scale, softcap, is_causal, wl, wr, dtype, stream);
}

#else  // FA off: inert stubs

extern "C" bool ggml_ncp_lib_fa_available(void) { return false; }
extern "C" int  ggml_ncp_lib_flash_attn_fwd(
        const void *, const void *, const void *, void *,
        int, int, int, int, int, int, int,
        long long, long long, long long, long long, long long, long long,
        long long, long long, long long, long long, long long, long long,
        float, float, int, int, int, int, void *) { return -1; }

#endif // GGML_NCP_FA

// ---- MoE grouped-GEMM (DeepGemm) ----

#ifdef GGML_NCP_MOE

typedef int (*ncp_moe_fn)(const void *, const void *, void *, const int *,
                          int, int, int, int, int, void *);
typedef int (*ncp_moe_masked_fn)(const void *, const void *, void *, const int *,
                                 int, int, int, int, int, int, void *);
typedef int (*ncp_moe_nopad_fn)(const void *, const void *, void *, const int *, const int *,
                                int, int, int, int, int, void *);
typedef int (*ncp_moe_align_fn)(void);

static ncp_moe_fn        g_moe_fn        = NULL;
static ncp_moe_masked_fn g_moe_masked_fn = NULL;
static ncp_moe_nopad_fn  g_moe_nopad_fn  = NULL;
static ncp_moe_align_fn  g_moe_align_fn  = NULL;

static void ncp_moe_init(void) {
    void * moe = open_lib("libncp_moe.so");
    if (moe) {
        g_moe_fn            = (ncp_moe_fn)            dlsym(moe, "ncp_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous");
        g_moe_masked_fn     = (ncp_moe_masked_fn)     dlsym(moe, "ncp_m_grouped_gemm_bf16_bf16_bf16_nt_masked");
        g_moe_nopad_fn = (ncp_moe_nopad_fn) dlsym(moe, "ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad");
        g_moe_align_fn      = (ncp_moe_align_fn)      dlsym(moe, "ncp_get_m_alignment_for_contiguous_layout");
        // Say so when the library loaded but has no NoPad entry: the hook is gated on that symbol alone, so this is
        // the difference between "the .so is doing the MoE" and "the .so loaded and changed nothing", which are
        // otherwise indistinguishable -- the model produces correct output either way.
        if (!g_moe_nopad_fn) {
            GGML_LOG_WARN("[ncp-lib] libncp_moe.so loaded but exports no "
                          "ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad -> MoE routing OFF, inline path\n");
        }
    }
}

static pthread_once_t g_moe_once = PTHREAD_ONCE_INIT;
static void ensure_moe_init(void) { pthread_once(&g_moe_once, ncp_moe_init); }

extern "C" bool ggml_ncp_lib_moe_nopad_available(void) { ensure_moe_init(); return g_moe_nopad_fn != NULL; }

extern "C" int ggml_ncp_lib_get_m_alignment_for_contiguous_layout(void) {
    ensure_moe_init();
    return g_moe_align_fn ? g_moe_align_fn() : 0;
}

extern "C" int ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
        const void * A, const void * B, void * out, const int * m_indices,
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream) {
    ensure_moe_init();
    if (!g_moe_fn) return -1;
    return g_moe_fn(A, B, out, m_indices, total_rows, N, K, n_experts, expected_m, stream);
}

extern "C" int ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_masked(
        const void * A, const void * B, void * out, const int * masked_m,
        int total_rows, int N, int K, int n_experts, int expected_m, int max_block_n, void * stream) {
    ensure_moe_init();
    if (!g_moe_masked_fn) return -1;
    return g_moe_masked_fn(A, B, out, masked_m, total_rows, N, K, n_experts, expected_m, max_block_n, stream);
}

extern "C" int ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        const void * A, const void * B, void * out, const int * m_indices, const int * m_rows,
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream) {
    ensure_moe_init();
    if (!g_moe_nopad_fn) return -1;
    return g_moe_nopad_fn(A, B, out, m_indices, m_rows, total_rows, N, K, n_experts, expected_m, stream);
}

#else  // MoE off: inert stubs

extern "C" bool ggml_ncp_lib_moe_nopad_available(void) { return false; }
extern "C" int  ggml_ncp_lib_get_m_alignment_for_contiguous_layout(void) { return 0; }
extern "C" int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
        const void *, const void *, void *, const int *,
        int, int, int, int, int, void *) { return -1; }
extern "C" int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_masked(
        const void *, const void *, void *, const int *,
        int, int, int, int, int, int, void *) { return -1; }
extern "C" int  ggml_ncp_lib_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        const void *, const void *, void *, const int *, const int *,
        int, int, int, int, int, void *) { return -1; }

#endif // GGML_NCP_MOE

// ---- GDN (flash-linear-attention gated delta net) ----

#ifdef GGML_NCP_GDN

typedef int    (*ncp_gdn_rec_fn)(const float *, const float *, const float *,
                               const float *, const float *, const float *,
                               float *, float *,
                               int, int, int, int, int, float, void *);
typedef int    (*ncp_gdn_chunk_fn)(const void *, const void *, const void *,
                                 const float *, const float *, const float *,
                                 void *, float *,
                                 int, int, int, int, int, float,
                                 void *, size_t, void *);
typedef size_t (*ncp_gdn_ws_fn)(int, int, int, int, int);

static ncp_gdn_rec_fn   g_gdn_rec_fn   = NULL;
static ncp_gdn_chunk_fn g_gdn_chunk_fn = NULL;
static ncp_gdn_ws_fn    g_gdn_ws_fn    = NULL;

static void ncp_gdn_init(void) {
    void * gdn = open_lib("libncp_gdn.so");
    if (gdn) {
        g_gdn_rec_fn   = (ncp_gdn_rec_fn)   dlsym(gdn, "ncp_gdn_recurrent");
        g_gdn_chunk_fn = (ncp_gdn_chunk_fn) dlsym(gdn, "ncp_gdn_chunked_bf16");
        g_gdn_ws_fn    = (ncp_gdn_ws_fn)    dlsym(gdn, "ncp_gdn_chunked_bf16_workspace_size");
    }
}

static pthread_once_t g_gdn_once = PTHREAD_ONCE_INIT;
static void ensure_gdn_init(void) { pthread_once(&g_gdn_once, ncp_gdn_init); }

extern "C" bool ggml_ncp_lib_gdn_available(void) {
    ensure_gdn_init();
    return g_gdn_rec_fn != NULL;
}

extern "C" bool ggml_ncp_lib_gdn_chunked_available(void) {
    ensure_gdn_init();
    return g_gdn_chunk_fn != NULL;
}

extern "C" int ggml_ncp_lib_gdn_recurrent(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * h0,
        float * o, float * ht,
        int n_seqs, int T, int H, int HV, int S,
        float scale, void * stream) {
    ensure_gdn_init();
    if (!g_gdn_rec_fn) return -1;
    return g_gdn_rec_fn(q, k, v, g, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, stream);
}

extern "C" int ggml_ncp_lib_gdn_chunked_bf16(
        const void * q, const void * k, const void * v,
        const float * g_raw, const float * beta, const float * h0,
        void * o, float * ht,
        int n_seqs, int T, int H, int HV, int S,
        float scale, void * ws, size_t ws_bytes, void * stream) {
    ensure_gdn_init();
    if (!g_gdn_chunk_fn) return -1;
    return g_gdn_chunk_fn(q, k, v, g_raw, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, ws, ws_bytes, stream);
}

extern "C" size_t ggml_ncp_lib_gdn_chunked_bf16_workspace_size(
        int n_seqs, int T, int H, int HV, int S) {
    ensure_gdn_init();
    if (!g_gdn_ws_fn) return 0;
    return g_gdn_ws_fn(n_seqs, T, H, HV, S);
}

#else  // GDN off: inert stubs

extern "C" bool   ggml_ncp_lib_gdn_available(void) { return false; }
extern "C" bool   ggml_ncp_lib_gdn_chunked_available(void) { return false; }
extern "C" int    ggml_ncp_lib_gdn_recurrent(
        const float *, const float *, const float *, const float *, const float *, const float *,
        float *, float *, int, int, int, int, int, float, void *) { return -1; }
extern "C" int    ggml_ncp_lib_gdn_chunked_bf16(
        const void *, const void *, const void *, const float *, const float *, const float *,
        void *, float *, int, int, int, int, int, float, void *, size_t, void *) { return -1; }
extern "C" size_t ggml_ncp_lib_gdn_chunked_bf16_workspace_size(int, int, int, int, int) { return 0; }

#endif // GGML_NCP_GDN

// GDN chunked-prefill arm: ON unless explicitly disabled. Library paths are build flags, not env vars.
#if defined(GGML_NCP_FA) || defined(GGML_NCP_MOE) || defined(GGML_NCP_GDN)

extern "C" bool ggml_ncp_gdn_chunked_enabled(void) {
    const char * p = getenv("GGML_NCP_GDN_CHUNKED");
    if (!p || !p[0]) {
        return true;
    }
    return strcmp(p, "0") != 0 && strcmp(p, "false") != 0 && strcmp(p, "off") != 0;
}

#else

extern "C" bool ggml_ncp_gdn_chunked_enabled(void) { return false; }

#endif
