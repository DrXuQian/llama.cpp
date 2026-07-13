// dlopen loader for the external FlashAttention kernel .so. See ppu-so.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu), but the whole body is inert unless GGML_PPU_SO is
// defined by the build (cmake -DGGML_PPU_SO=ON). When inert, *_available() returns false and the *_fwd wrapper
// returns -1, so every caller falls straight through to the inline ggml path.

#include "ppu-so.h"

#if defined(GGML_PPU_SO) && !defined(_WIN32)

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef int (*ppu_fa_fn)(const void *, const void *, const void *, void *,
                         int, int, int, int, int, int,
                         long long, long long, long long, long long, long long, long long,
                         long long, long long, long long, long long, long long, long long,
                         float, float, int, int, void *);

static ppu_fa_fn g_fa_fn = NULL;

static void * open_lib(const char * env, const char * soname) {
    const char * path = getenv(env);
    void * h = dlopen(path && path[0] ? path : soname, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[ppu-so] %s not loaded (%s): %s -> inline fallback\n",
                soname, path ? path : "soname", dlerror());
    }
    return h;
}

static void ppu_so_init(void) {
    void * fa = open_lib("GGML_PPU_FA_SO", "libppu_fa.so");
    if (fa) { g_fa_fn = (ppu_fa_fn) dlsym(fa, "ppu_flash_attn_fwd"); }
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void ensure_init(void) { pthread_once(&g_once, ppu_so_init); }

extern "C" bool ggml_ppu_so_fa_available(void) { ensure_init(); return g_fa_fn != NULL; }

extern "C" int ggml_ppu_so_flash_attn_fwd(
        const void * q, const void * k, const void * v, void * o,
        int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
        long long qbs, long long qhs, long long qrs, long long kbs, long long khs, long long krs,
        long long vbs, long long vhs, long long vrs, long long obs, long long ohs, long long ors,
        float scale, float logit_softcap, int is_causal, int dtype, void * stream) {
    ensure_init();
    if (!g_fa_fn) return -1;
    return g_fa_fn(q, k, v, o, batch, seqlen_q, seqlen_k, n_heads_q, n_heads_kv, head_dim,
                   qbs, qhs, qrs, kbs, khs, krs, vbs, vhs, vrs, obs, ohs, ors,
                   scale, logit_softcap, is_causal, dtype, stream);
}

#else  // GGML_PPU_SO disabled (or Windows): inert stubs

extern "C" bool ggml_ppu_so_fa_available(void) { return false; }
extern "C" int  ggml_ppu_so_flash_attn_fwd(
        const void *, const void *, const void *, void *,
        int, int, int, int, int, int,
        long long, long long, long long, long long, long long, long long,
        long long, long long, long long, long long, long long, long long,
        float, float, int, int, void *) { return -1; }

#endif
