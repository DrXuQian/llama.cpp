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

## The permutation is built on the device — no D2H, no H2D, no sync

The hook sorts the tokens by expert on the GPU: **histogram → exclusive scan → atomic scatter**, the same shape as
vLLM's `moe_align_block_size` minus its block padding (a NoPad kernel needs none). Gather and scatter are one kernel
each. **Nothing crosses to the host**, so unlike ggml's own sorted-cuBLAS fallback — which does a D2H, an
`O(n_experts × n_tokens × n_expert_used)` CPU loop, a `cudaStreamSynchronize` and three H2Ds per MoE layer — this
path stays CUDA-graph capturable.

It deliberately does **not** use ggml's `ggml_cuda_launch_mm_ids_helper`, even though that is also a device-side
sort. That kernel runs **one warp per expert** and has each warp rescan the *entire* `ids` tensor, so its work is
`O(n_experts × n_tokens × n_expert_used)` — the same complexity as the CPU loop, just spread over `n_experts` warps —
and its time grows linearly with `n_tokens`. Measured on a 5090, 128 experts / top-8:

| | 512 tokens | 2048 tokens | 4096 tokens (256 experts) |
|---|---|---|---|
| `mm_ids_helper` | 10.2 µs | 36.9 µs | 71.7 µs |
| counting sort | 9.4 µs | 12.2 µs | 18.4 µs |
| | 1.09× | **3.03×** | **3.89×** |

(The ~10 µs floor is launch overhead, which CUDA-graph capture removes; the real gap is wider.) `mm_ids_helper` also
stages 4 bytes per token in shared memory and hard-`GGML_ASSERT`s that it fits (`mmid.cu:132`), which caps the ubatch.
**Replacing it inside ggml's own `mmq`/`mmf` would speed up the native MoE paths too — that is upstreamable.**

The atomic scatter makes the order of rows *within* an expert nondeterministic. That is harmless: compact row `r`'s
output depends only on compact row `r`'s input, and the scatter maps it back to a fixed `dst` slot, so `dst` is
bit-identical run to run.

That is only possible because **no host value depends on the routing**. Whether a layout can guarantee that comes down
to one thing: *where the kernel's scheduler gets its block count from.*

```
scheduler/gemm.cuh, Scheduler ctor:
  MGroupedContiguous               num_blocks = num_m_blocks * num_n_blocks   <- from shape_m, a HOST value
  MGroupedMasked                   (num_blocks not set)                       <- walks groups off device masked_m
  MGroupedContiguousWithPsumLayout (num_blocks not set)                       <- walks groups off device psum array
```

Plain contiguous schedules from `shape_m`, so `shape_m` must be the *exact* padded total — a data-dependent host
value, i.e. a mandatory D2H. Masked (and psum) walk their device-resident layout array instead, so `shape_m` is only
a memory extent and an **upper bound suffices**. That is the whole trick, and it is what lets vLLM and the PPU stack
run this path with no host round-trip.

## Dispatch: regime first, then layout

**Decode and prefill are different kernels, not the same kernel at different sizes.** A grouped GEMM is a tile-based
compute kernel; MoE decode is memory-bound weight streaming. One token over top-k experts leaves 1–2 rows per expert,
and even a NoPad kernel pays a whole `BLOCK_M`-row tile — plus a wgmma/TMA pipeline tuned for `M >> 1` — to produce
them. So decode gets a **separate entry**, `ppu_moe_gemv_bf16`, over the same dense layout.

The regime split is checked **inside the hook** (`n_tokens < GGML_PPU_MOE_MIN_TOKENS`, default 128), not implied by
where the hook sits in ggml's dispatch chain: `mmvq`/`mmf` decline for reasons of their own
(`src0->ne[1] % rows_per_block`, `K % (warp_size*2)`, …) and a decode that slipped past them would otherwise land in
the grouped GEMM and be silently slow.

Which *entry* within a regime is then decided by **which symbol the `.so` exports** — presence is the capability query:

| regime | exported symbol | layout | host-known row count | pad flops | scratch |
|---|---|---|---|---|---|
| decode | `ppu_moe_gemv_bf16` | dense | `total_rows` | n/a (GEMV) | `total_rows × K` |
| prefill | `ppu_moe_grouped_gemm_bf16_nopad` | dense contiguous | `total_rows = n_tokens × n_expert_used` (an identity) | none | `total_rows × K` |
| prefill | `ppu_moe_grouped_gemm_bf16_masked` | masked | `max_m = align(n_tokens, 128)` (an upper bound) | `ceil(rows/64)*64` per expert | `n_experts × max_m × K` |

**Public DeepGEMM has no bf16/sm90 MoE GEMV**, so the `.so` built here does not export `ppu_moe_gemv_bf16` and decode
falls through to ggml's own `mmvf`/`mmf` — the right answer there anyway. Neither of DeepGEMM's batched entries fits:

* `einsum::bmk_bnk_mn(a[s,m,k], b[s,n,k], d[s,m,n])` indexes **B by the same batch dim as A**, so the selected
  experts' weights would have to be materialised as a contiguous `[total_rows, N, K]` — a copy that *doubles* the
  weight traffic the GEMV was meant to save.
* `einsum::bhr_hdr_bhd(A[b,h,r], B[h,d,r], D[b,h,d])` shares B across the batch but indexes it by *head*. Forcing
  `h = n_experts` computes **every** expert for **every** token — 16× the weight reads at top-8/128.

A PPU `.so` with a real batched-GEMV kernel should export `ppu_moe_gemv_bf16` and the hook picks it up automatically.

The PPU kernel team's DeepGemm has a genuine NoPad kernel — `bf16_grouped_deep_gemm_NoPad`, reached via
`deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)`, which runs a device-side
`computeBlockInfoKernel` to build the block→group map when `n_experts >= 128`. Its `.so` should export
`ppu_moe_grouped_gemm_bf16_nopad` and the hook will take the dense arm automatically.

**Public DeepGEMM has no such kernel** (grepped: `nopad` appears nowhere in the repo; the bf16 grouped APIs are only
`m_grouped_bf16_gemm_nt_contiguous` and `..._masked`). So `libppu_moe.so` built here deliberately does **not** export
that symbol — it exports `ppu_moe_grouped_gemm_bf16_contiguous` (padded, diagnostics/tests only) and
`ppu_moe_grouped_gemm_bf16_masked`, and the hook takes the masked arm. Naming the padded entry `nopad` — as an
earlier revision of this code did — is a trap: it silently promises a contract it cannot honour.

## The C ABI — two layouts

```c
// ppu-moe-so.h

// Capability query. 1 = "NoPad" (dense, take each expert's rows as they are). >1 = per-expert segments must start
// and end on a multiple of it. 0 = no .so.
int ppu_moe_row_alignment(void);

// DENSE CONTIGUOUS / "NoPad" — preferred; used when row_alignment() == 1
int ppu_moe_grouped_gemm_bf16_nopad(
        const void * A,            // [total_rows, K] bf16, expert-grouped, NO padding
        const void * B,            // [n_experts, N, K] bf16
        void       * out,          // [total_rows, N] bf16
        const int  * m_indices,    // [total_rows] expert id per compact row
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// MASKED — used when row_alignment() > 1 (i.e. against public DeepGEMM)
int ppu_moe_grouped_gemm_bf16_masked(
        const void * A,            // [n_experts, max_m, K] bf16
        const void * B,            // [n_experts, N, K] bf16
        void       * out,          // [n_experts, max_m, N] bf16
        const int  * masked_m,     // [n_experts] real row count per expert — NO alignment required
        int max_m, int N, int K, int n_experts, int expected_m, void * stream);
```

The hook gathers the F32 activations into `A` as bf16 (device kernel), runs one grouped GEMM, and scatters the
result back (device kernel). The permutation feeding both comes from `ggml_cuda_launch_mm_ids_helper`.

### Why the public build needs masked, and the PPU build does not

Public DeepGEMM's bf16 contiguous kernel pins `BLOCK_M` to its 128-row alignment (`heuristics/sm90.hpp:33`), so every
expert's segment must be padded to 128 — and **the padded total is data-dependent**, which is exactly the host value
we are trying not to need. Its fp8 kernel dodges this the way vLLM does: launch with a host-known upper-bound `m`,
mark the tail `m_indices = -1`, and let the kernel skip those blocks. But that skip does not exist for bf16 —
`is_computation_valid` has exactly one call site in the whole repo, `sm90_fp8_gemm_1d2d.cuh:274`. A bf16 contiguous
kernel would compute the tail against expert 0 and throw it away.

So on public DeepGEMM we use **masked** instead. Its scheduler enqueues exactly `ceil(masked_m[e] / BLOCK_M)`
row-blocks per expert (`scheduler/gemm.cuh:207`), so unused capacity costs *zero* flops — which means `max_m` only
has to be an **upper bound**, and a host-known one exists: `mm_ids_helper` emits at most one compact row per
(token, expert), so no expert can hold more than `n_tokens` rows. Zero sync, at the cost of an
`n_experts / n_expert_used` (16× on a 128-expert top-8 model) scratch buffer.

The PPU kernel team's `bf16_grouped_deep_gemm_NoPad` needs none of this: it takes each expert's rows as they are, so
the row count it needs is just `total_rows = n_tokens × n_expert_used` — an identity. Dense layout, zero padding,
zero wasted flops, compact scratch. That is the path the hook takes whenever `ppu_moe_row_alignment()` returns 1.

**The gap worth reporting upstream:** DeepGEMM's bf16 grouped kernel should call `is_computation_valid` like its fp8
sibling does. It looks like an omission, not a design choice, and it would let the public build use the same dense
contiguous shape.

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

Everything above the line already stays on the device; only the fallback below it sorts on the CPU. Our hook stays on
the device too (see above), so the placement is not about the sync — it is about not stealing work from a kernel that
is already the right tool. Hooking any earlier would preempt MMF on shapes it handles well.

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

**Contiguous (only when `row_alignment() > 1`):** the kernel assigns whole `BLOCK_M=128` row-blocks to one expert and
reads the expert id from each block's *first* row. If expert *e*'s segment doesn't start on a 128-row boundary, a
block straddles two experts and multiplies rows by the wrong weight matrix — **no error, just wrong output**. A kernel
that reports `row_alignment() == 1` has no such constraint, which is the whole point of the NoPad variant.

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
* **The dense/NoPad arm has never been exercised**, because public DeepGEMM has no NoPad kernel and so this `.so`
  does not export the symbol. It is written against `m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)`;
  confirm the exact entry signature with the kernel team before trusting it.
* **`use_psum_layout` is unexplored.** Public DeepGEMM's contiguous entry has a `MGroupedContiguousWithPsumLayout`
  variant whose scheduler also walks a device-resident array, so it too could run sync-free with an upper-bound
  `shape_m`, at `align(rows, 128)` flops per expert but only ~1/3 of masked's scratch. Worth trying if masked's
  capacity buffer turns out to hurt.
* **MoE decode must not be routed here.** DeepGEMM is a prefill kernel; with one token per expert even masked's
  64-row `BLOCK_M` is a 64× waste.
