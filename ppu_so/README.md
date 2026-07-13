# `ppu_so` — external MoE grouped-GEMM kernel as a `.so` plugin

llama.cpp's CUDA backend can route **bf16-weight MoE `mul_mat_id`** to an external kernel `.so` that wraps
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s sm90 grouped GEMM, instead of using the inline ggml path.

The point is the **seam**, not the kernel: llama.cpp compiles with **zero** cutlass / DeepGEMM / torch headers and
zero link deps. The kernel is built here, out of tree, into `libppu_moe.so`; at runtime ggml-cuda `dlopen`s it and
`dlsym`s a thin C ABI. If the `.so` is missing, or the shape/dtype isn't supported, the call falls back to the
inline ggml kernel transparently. Same shape as cuBLAS: a binary behind a stable C entry point.

This is the pattern we use to reuse the kernel team's PPU DeepGEMM build rather than re-porting grouped GEMM.

---

## Layout

```
ppu_so/
  CMakeLists.txt                                standalone build (NOT part of llama.cpp's build)
  moe/deepgemm_c.cpp                            torch-free re-impl of sm90 masked + contiguous grouped GEMM
  moe/test_moe.cpp                              standalone check of both layouts vs an fp32 CPU reference
  patches/0001-deepgemm-no-torch-cublaslt.patch strips DeepGEMM's hard libtorch dep (see trap 1)
thirdparty/DeepGEMM/                            submodule
ggml/src/ggml-cuda/
  ppu-moe-so.h                                  the C ABI (the only header shipped with the .so)
  ppu-so.{h,cu}                                 dlopen loader; inert unless -DGGML_PPU_SO=ON
  ggml-cuda.cu                                  ggml_cuda_mul_mat_id_ppu_so() — the hook
```

## Build

```bash
git submodule update --init --recursive thirdparty/DeepGEMM

# 1. the kernel .so (needs torch HEADERS to compile; does NOT link libtorch)
cd ppu_so
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))')
cmake --build build -j
./build/test_moe          # standalone correctness check

# 2. llama.cpp with the seam enabled (default OFF -> fully inert)
cd ..
cmake -B build -DGGML_CUDA=ON -DGGML_PPU_SO=ON
cmake --build build -j

# 3. run
GGML_PPU_MOE_SO=$PWD/ppu_so/build/libppu_moe.so ./build/bin/test-backend-ops -o MUL_MAT_ID
```

`GGML_PPU_MOE_SO` is optional — without it the loader just `dlopen("libppu_moe.so")` off the normal search path.

## The C ABI — two layouts

```c
// ppu-moe-so.h

// MASKED (what the hook uses)
int ppu_moe_grouped_gemm_bf16_masked(
        const void * A,            // [n_experts, max_m, K] bf16
        const void * B,            // [n_experts, N, K] bf16
        void       * out,          // [n_experts, max_m, N] bf16
        const int  * masked_m,     // [n_experts] real row count per expert — NO alignment required
        int max_m, int N, int K, int n_experts, int expected_m, void * stream);

// CONTIGUOUS (kept as the documented alternative; "nopad" = the .so does not pad FOR you)
int ppu_moe_row_alignment(void);   // = 128 (DeepGEMM pins BLOCK_M to it)
int ppu_moe_grouped_gemm_bf16_nopad(
        const void * A,            // [total_rows, K] bf16, expert-grouped, each segment padded to the alignment
        const void * B,            // [n_experts, N, K] bf16
        void       * out,          // [total_rows, N] bf16
        const int  * m_indices,    // [total_rows] expert id per row
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream);
```

The hook in `ggml-cuda.cu` builds the `(token, slot) -> expert` permutation on the host, gathers the F32
activations into `A` as bf16, runs one grouped GEMM, and scatters the result back.

### Why masked and not contiguous

The contiguous layout pins `BLOCK_M` to the 128-row alignment, so an expert holding 32 rows still costs a full
128-row wgmma. That is exactly llama.cpp's regime: a few hundred tokens spread over 100+ experts leaves ~32 rows
per expert, i.e. **4× wasted FLOPs**.

Masked's scheduler builds its block queue straight from `masked_m` — group `g` contributes exactly
`ceil(masked_m[g] / BLOCK_M)` row-blocks (`scheduler/gemm.cuh:207`) — and its `BLOCK_M` candidates are `{64, 128}`.
The same expert now costs 64 rows. `masked_m` needs no alignment whatsoever; upstream's own test drives it with 20
rows per group.

Contiguous is still the better shape under heavy routing imbalance, where masked's `n_experts * max_m` buffer blows
up. The hook falls back to the inline ggml path when that buffer would exceed `8 * total_rows`.

## Which calls are routed — and where the hook sits

The hook is **not** at the top of `ggml_cuda_mul_mat_id`. It sits below ggml's `mmvq` / `mmq` / `mmf` dispatch and
directly above the sorted-cuBLAS fallback. That placement is the whole design:

```
ggml_cuda_mul_mat_id:
  mul_mat_vec_q   quantized, small batch (decode)   takes ids on device, no sync
  mul_mat_q       quantized (MMQ)                   takes ids on device, no sync
  mul_mat_f       float, small batch (MMF)          takes ids on device, no sync
  ---- ppu_so hook -----------------------------------------------------------
  sorted cuBLAS   float, large batch                host sort => D2H + hard stream sync
```

Everything above the line already stays on the device. Only the fallback below it pays a `cudaMemcpy` D2H plus a
`cudaStreamSynchronize` — and so does our hook, unavoidably (`max_m` sizes the buffers and goes into the TMA
descriptor, so it has to be a host value). Hooking any earlier would trade a sync-free device kernel for one that
drains the pipeline, and no GEMM is fast enough to pay that back.

Concretely, `ggml_cuda_should_use_mmf` accepts a bf16 `mul_mat_id` when `src0->ne[1] <= 1024 && n_tokens <= 512`
(mmf.cu:162). On a typical MoE prefill (`ubatch=512`, `moe_intermediate=768`, `hidden=2048`) that means **gate/up go
to MMF and only down_proj falls through to the sorted path**. An early hook would have stolen gate/up from a
sync-free kernel. So the `.so` competes only where ggml itself gave up on staying on-device: **large-batch float
MoE**, i.e. exactly the batched-cuBLAS regime a grouped GEMM should win.

On top of that the hook requires:

* `src0` (expert weights) **bf16**, `src1`/`dst` F32 → gathered to bf16
* sm90/sm100 (DeepGEMM's bf16 kernels; anything else returns `rc=3` → inline fallback)
* all three operands contiguous, `K % 64 == 0`

**MoE decode is never routed** — it is caught by `mul_mat_vec_q`/`mmf` long before the hook. Good: with one token per
expert, even masked's 64-row `BLOCK_M` would turn a 1-row GEMM into a 64-row GEMM. Quantized MoE is likewise never
routed: those are ggml-proprietary formats no external library has ever seen.

---

## Three traps this cost us — read before touching it

### 1. DeepGEMM is not torch-free, and `ldd` won't tell you

`DeviceRuntime`'s constructor unconditionally does `cublaslt_workspace = torch::empty(...)` and holds the result in
a `torch::Tensor` **member**. That is a hard libtorch dependency (`at::_ops::empty_memory_format`,
`UndefinedTensorImpl::_singleton`, the `AutogradMeta` vtable) which `--gc-sections` **cannot** strip, because the
ctor genuinely references it — and it makes `dlopen(RTLD_NOW)` fail at runtime.

`patches/0001-deepgemm-no-torch-cublaslt.patch` guards that member behind `#ifndef DG_NO_TORCH`; the bf16 grouped
GEMM never touches cuBLASLt. CMake applies it idempotently.

When you verify the result, **`ldd` alone lies**: an *undefined* symbol carries no `DT_NEEDED` entry, so a torch
dependency is invisible to `ldd` and only surfaces as a `dlopen` failure. Check both:

```bash
ldd     build/libppu_moe.so | grep -i torch                          # must be empty
nm -D -u build/libppu_moe.so | c++filt | grep -E 'c10::|at::|torch'  # must ALSO be empty
```

*(The right long-term fix is upstream, not here: ask whoever maintains the kernel lib for a torch-free build mode.
Every kernel library we want to consume through this seam needs one.)*

### 2. Both layouts have a 128-row alignment trap, in different places

**Contiguous:** the kernel assigns whole `BLOCK_M=128` row-blocks to one expert and reads the expert id from each
block's *first* row. If expert *e*'s segment doesn't start on a 128-row boundary, a block straddles two experts and
multiplies rows by the wrong weight matrix — **no error, just wrong output**. Every expert's segment must be padded
up to `ppu_moe_row_alignment()`.

**Masked:** `masked_m` needs no alignment — but **`max_m` does**. It is the row *pitch* between groups: the kernel
computes a global row as `group_idx * max_m + m_block_idx * BLOCK_M` (`scheduler/gemm.cuh:164`). If `max_m` weren't
a multiple of `BLOCK_M`, group *g*'s last row-block would spill past the group boundary and TMA-**store** over
group *g+1*'s real output rows. The hook rounds `max_m` up to 128, a safe superset of both `BLOCK_M` candidates
(the caller cannot know which one the heuristic picks).

Conversely, rows `[masked_m[g], max_m)` of `A` may be left **uninitialized** — garbage in compact row *r* only ever
reaches `out` row *r*, which the hook never scatters back. That is what lets the gather kernel touch only the real
rows instead of the whole `n_experts * max_m` buffer.

### 3. The JIT needs cutlass on the *runtime* box

DeepGEMM NVRTC-compiles its kernel on first use with only two `-I` flags (`deep_gemm/include` and `$CUDA_HOME/
include`), and the generated code `#include`s `<cutlass/...>` / `<cute/...>`. CMake symlinks cutlass into
`deep_gemm/include` — that's a **runtime** requirement of the `.so`, not a dev convenience. The compiled cubin is
disk-cached, so this only bites on the first call.

---

## Known gaps

* **The masked path is built and torch-free-clean but its numerics are NOT yet validated** — the box this was last
  touched on is an RTX 5090 (sm120) and DeepGEMM's bf16 kernels are sm90/sm100 only, so both entries correctly
  return `rc=3` and llama.cpp falls back inline. Run `./build/test_moe 8 100 512 256` and
  `test-backend-ops -o MUL_MAT_ID` on an sm90 box before trusting it. The contiguous path *was* validated on H800
  (`MUL_MAT_ID 790/790`).
* **The permutation is built on the host**, which costs a `cudaMemcpy` D2H + a hard `cudaStreamSynchronize` per MoE
  layer and keeps the whole path out of CUDA graphs. ggml's own inline `mul_mat_id` has exactly the same problem
  (it even says so in a comment above its sort). Moving the histogram + prefix sum + scatter onto the device would
  leave only a `n_experts`-int D2H — `max_m` has to be a host value because it sizes the buffers and goes into the
  TMA descriptor. This is worth fixing for the inline path too, i.e. it is upstreamable.
* **MoE decode must not be routed here.** DeepGEMM is a prefill kernel; with one token per expert even masked's
  64-row `BLOCK_M` is a 64× waste.
