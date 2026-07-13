#pragma once
// dlopen wrapper around the external gated-delta-net kernel .so (libppu_gdn.so).
//
// Why dlopen and not link-time: llama.cpp must build with ZERO torch / triton / python dependency. The kernels are
// Flash-Linear-Attention's Triton kernels, compiled out of tree into a standalone libppu_gdn.so that carries their
// cubins inside it and launches them through the CUDA driver API; only the .so binary (+ ppu-gdn-so.h) is shipped,
// never the generator. At runtime we dlopen the .so and dlsym the C ABI; if it is absent or has no kernel for the
// requested shape, the caller transparently falls back to the inline ggml path.
//
// .so location is resolved (first hit wins):
//   1. env  GGML_PPU_GDN_SO   (absolute path to the .so)
//   2. bare soname on the loader search path:  libppu_gdn.so
//
// Everything here is a no-op (available()==false) unless the build defines GGML_PPU_SO.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

// ---- gated delta net, recurrent path (mirror of ppu-gdn-so.h's ppu_gdn_recurrent) ----
bool ggml_ppu_so_gdn_available(void);
int  ggml_ppu_so_gdn_recurrent(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale, void * stream);
bool   ggml_ppu_so_gdn_chunked_available(void);
// Device scratch the chunked entry needs; the caller allocates it (we use ggml's CUDA pool). 0 if the .so is absent.
size_t ggml_ppu_so_gdn_chunked_workspace_size(int n_seqs, int T, int H, int HV, int S);
int    ggml_ppu_so_gdn_chunked(
    const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale,
    void * ws, size_t ws_bytes, void * stream);

#ifdef __cplusplus
}
#endif
