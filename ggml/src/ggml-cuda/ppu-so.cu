// dlopen loader for the external gated-delta-net kernel .so. See ppu-so.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu), but the whole body is inert unless GGML_PPU_SO is
// defined by the build (cmake -DGGML_PPU_SO=ON). When inert, *_available() return false and the entry points
// return -1, so every caller falls straight through to the inline ggml path.

#include "ppu-so.h"

#if defined(GGML_PPU_SO) && !defined(_WIN32)

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef int (*ppu_gdn_fn)(const float*,const float*,const float*,const float*,const float*,
                          const float*,float*,float*,int,int,int,int,int,float,void*);
typedef int (*ppu_gdn_chunk_fn)(const float*,const float*,const float*,const float*,const float*,
                                const float*,float*,float*,int,int,int,int,int,float,void*,size_t,void*);
typedef size_t (*ppu_gdn_ws_fn)(int,int,int,int,int);

static ppu_gdn_fn       g_gdn_fn       = NULL;
static ppu_gdn_chunk_fn g_gdn_chunk_fn = NULL;
static ppu_gdn_ws_fn    g_gdn_ws_fn    = NULL;

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
    void * gdn = open_lib("GGML_PPU_GDN_SO", "libppu_gdn.so");
    if (gdn) {
        g_gdn_fn       = (ppu_gdn_fn) dlsym(gdn, "ppu_gdn_recurrent");
        g_gdn_chunk_fn = (ppu_gdn_chunk_fn) dlsym(gdn, "ppu_gdn_chunked");
        g_gdn_ws_fn    = (ppu_gdn_ws_fn)    dlsym(gdn, "ppu_gdn_chunked_workspace_size");
    }
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void ensure_init(void) { pthread_once(&g_once, ppu_so_init); }

extern "C" bool ggml_ppu_so_gdn_available(void) { ensure_init(); return g_gdn_fn != NULL; }

extern "C" int ggml_ppu_so_gdn_recurrent(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * h0, float * o, float * ht,
        int n_seqs, int T, int H, int HV, int S, float scale, void * stream) {
    ensure_init();
    if (!g_gdn_fn) return -1;
    return g_gdn_fn(q, k, v, g, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, stream);
}

extern "C" bool ggml_ppu_so_gdn_chunked_available(void) {
    ensure_init();
    return g_gdn_chunk_fn != NULL && g_gdn_ws_fn != NULL;
}

extern "C" size_t ggml_ppu_so_gdn_chunked_workspace_size(int n_seqs, int T, int H, int HV, int S) {
    ensure_init();
    return g_gdn_ws_fn ? g_gdn_ws_fn(n_seqs, T, H, HV, S) : 0;
}

extern "C" int ggml_ppu_so_gdn_chunked(
        const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
        const float * h0, float * o, float * ht,
        int n_seqs, int T, int H, int HV, int S, float scale,
        void * ws, size_t ws_bytes, void * stream) {
    ensure_init();
    if (!g_gdn_chunk_fn) return -1;
    return g_gdn_chunk_fn(q, k, v, g_raw, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, ws, ws_bytes, stream);
}

#else  // GGML_PPU_SO disabled (or Windows): inert stubs

extern "C" bool ggml_ppu_so_gdn_available(void) { return false; }
extern "C" int  ggml_ppu_so_gdn_recurrent(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, int, float, void *) { return -1; }
extern "C" bool   ggml_ppu_so_gdn_chunked_available(void) { return false; }
extern "C" size_t ggml_ppu_so_gdn_chunked_workspace_size(int, int, int, int, int) { return 0; }
extern "C" int    ggml_ppu_so_gdn_chunked(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, int, float,
        void *, size_t, void *) { return -1; }

#endif
