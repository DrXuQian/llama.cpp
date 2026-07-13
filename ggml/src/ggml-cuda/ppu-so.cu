// dlopen loader for the external MoE grouped-GEMM .so. See ppu-so.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu), but the whole body is inert unless GGML_PPU_SO is
// defined by the build (cmake -DGGML_PPU_SO=ON). When inert, available() returns false and the entry points
// return 0/-1, so every caller falls straight through to the inline ggml path.

#include "ppu-so.h"

#if defined(GGML_PPU_SO) && !defined(_WIN32)

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef int (*ppu_moe_fn)(const void *, const void *, void *, const int *,
                          int, int, int, int, int, void *);
typedef int (*ppu_moe_align_fn)(void);

static ppu_moe_fn       g_moe_nopad_fn  = NULL;   // true NoPad: dense A, m = total_rows, no alignment
static ppu_moe_fn       g_moe_masked_fn = NULL;
static ppu_moe_fn       g_moe_contig_fn = NULL;   // padded contiguous (diagnostics only; the hook never calls it)
static ppu_moe_align_fn g_moe_align_fn  = NULL;

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
    void * moe = open_lib("GGML_PPU_MOE_SO", "libppu_moe.so");
    if (moe) {
        g_moe_nopad_fn  = (ppu_moe_fn)       dlsym(moe, "ppu_moe_grouped_gemm_bf16_nopad");
        g_moe_masked_fn = (ppu_moe_fn)       dlsym(moe, "ppu_moe_grouped_gemm_bf16_masked");
        g_moe_contig_fn = (ppu_moe_fn)       dlsym(moe, "ppu_moe_grouped_gemm_bf16_contiguous");
        g_moe_align_fn  = (ppu_moe_align_fn) dlsym(moe, "ppu_moe_row_alignment");
    }
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void ensure_init(void) { pthread_once(&g_once, ppu_so_init); }

extern "C" int ggml_ppu_so_moe_row_alignment(void) {
    ensure_init();
    return g_moe_align_fn ? g_moe_align_fn() : 0;
}

extern "C" bool ggml_ppu_so_moe_nopad_available(void) { ensure_init(); return g_moe_nopad_fn != NULL; }

extern "C" int ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
        const void * A, const void * B, void * out, const int * m_indices,
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream) {
    ensure_init();
    if (!g_moe_nopad_fn) return -1;
    return g_moe_nopad_fn(A, B, out, m_indices, total_rows, N, K, n_experts, expected_m, stream);
}

extern "C" bool ggml_ppu_so_moe_masked_available(void) { ensure_init(); return g_moe_masked_fn != NULL; }

extern "C" int ggml_ppu_so_moe_grouped_gemm_bf16_masked(
        const void * A, const void * B, void * out, const int * masked_m,
        int max_m, int N, int K, int n_experts, int expected_m, void * stream) {
    ensure_init();
    if (!g_moe_masked_fn) return -1;
    return g_moe_masked_fn(A, B, out, masked_m, max_m, N, K, n_experts, expected_m, stream);
}

#else  // GGML_PPU_SO disabled (or Windows): inert stubs

extern "C" int  ggml_ppu_so_moe_row_alignment(void) { return 0; }
extern "C" bool ggml_ppu_so_moe_nopad_available(void) { return false; }
extern "C" int  ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
        const void *, const void *, void *, const int *,
        int, int, int, int, int, void *) { return -1; }
extern "C" bool ggml_ppu_so_moe_masked_available(void) { return false; }
extern "C" int  ggml_ppu_so_moe_grouped_gemm_bf16_masked(
        const void *, const void *, void *, const int *,
        int, int, int, int, int, void *) { return -1; }

#endif
