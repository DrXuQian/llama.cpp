# `ppu_so/gdn` — Flash-Linear-Attention's Gated Delta Net as a torch/triton/python-free `.so`

llama.cpp's CUDA backend can route `GGML_OP_GATED_DELTA_NET` to an external kernel `.so` built from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)'s Triton kernels, instead of using the
inline ggml path.

The point is the **seam**: llama.cpp compiles and runs with **zero** torch, triton or python dependency. FLA's Triton
kernels are compiled out of tree into `libppu_gdn.so`, which carries their **cubins inside it as byte arrays** and
launches them through the **CUDA driver API** (`cuModuleLoadData` / `cuLaunchKernel`). At runtime ggml-cuda `dlopen`s
the `.so` and `dlsym`s a thin C ABI; if it is absent or has no kernel for the requested shape, the call falls back to
the inline ggml kernel transparently.

Two paths, matching ggml's own two:

| | FLA entry | kernels | when |
|---|---|---|---|
| **recurrent** (decode) | `fused_recurrent_gated_delta_rule` | 1 | always, when the `.so` is loaded |
| **chunked** (prefill) | `chunk_gated_delta_rule` | 5 | opt-in: `GGML_PPU_GDN_CHUNKED=1` **and** `n_tokens >= 128` |

---

## Layout

```
ppu_so/gdn/
  build.sh                 builds libppu_gdn.so for a list of "H,HV,S" shapes
  aot_recurrent.py         Triton-AOT the recurrent kernel (this one AOTs fine)
  gen_chunk_so.py          JIT the 5 chunked kernels, embed their cubins, emit a driver-API launcher
  gen_golden_recurrent.py  fp64 CPU golden for the recurrent path
  gen_golden_chunk.py      golden for the chunked path
  test_abi.c               standalone ABI smoke test
thirdparty/flash-linear-attention/   submodule
ggml/src/ggml-cuda/
  ppu-gdn-so.h             the C ABI (the only header shipped with the .so)
  ppu-so.{h,cu}            dlopen loader; inert unless -DGGML_PPU_SO=ON
  gated_delta_net.cu       the hook + two layout-fixup kernels
```

## Build

```bash
git submodule update --init thirdparty/flash-linear-attention

cd ppu_so/gdn
./build.sh                      # default shapes: 32,32,128  16,16,64  4,4,64  4,8,64  32,32,64  2,4,128
                                # -> libppu_gdn.so   (needs torch+triton to BUILD; the .so needs neither to RUN)

cd ../..
cmake -B build -DGGML_CUDA=ON -DGGML_PPU_SO=ON     # default OFF -> fully inert
cmake --build build -j

GGML_PPU_GDN_SO=$PWD/ppu_so/gdn/libppu_gdn.so \
GGML_PPU_GDN_CHUNKED=1 \
  ./build/bin/test-backend-ops -o GATED_DELTA_NET
```

`libppu_gdn.so` is compiled **per (H, HV, S) shape** — the Triton kernels specialise on them. A shape that was not
compiled in returns non-zero and llama.cpp falls back inline, **silently**. That is the single most common reason the
`.so` looks like it is doing nothing. Set `GGML_PPU_GDN_DEBUG=1` and it will tell you exactly which shape to add:

```
[ppu-gdn] recurrent declined rc=1 for H=16 HV=32 S=128 (T=1) -- add "16,32,128" to ppu_so/gdn/build.sh
```

The default shape list does **not** include Qwen3-Next's `16,32,128`.

## The C ABI

```c
// ppu-gdn-so.h
int ppu_gdn_recurrent(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale, void * stream);

// The .so allocates NOTHING. Size the scratch, hand it in. llama.cpp uses ggml's CUDA pool.
size_t ppu_gdn_chunked_workspace_size(int n_seqs, int T, int H, int HV, int S);   // ~176 MB at T=2048

int ppu_gdn_chunked(  // `g` is the RAW gate (see trap 3)
    const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale,
    void * ws, size_t ws_bytes, void * stream);
```

---

## Five traps this cost us — read before touching it

### 1. Triton AOT miscompiles the chunked kernels. Use JIT cubins instead.

`triton.tools.compile` (AOT) works for the **recurrent** kernel. It does **not** work for the two tensor-core kernels
in the chunked chain (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`): they either hit an
illegal memory access or produce wrong results. This is a known open Triton issue, not something you can work around
at the call site.

So `gen_chunk_so.py` does **not** use AOT. It JIT-compiles each kernel, pulls the resulting **cubin** out of Triton's
cache, embeds it in the generated C as a byte array, and launches it with `cuModuleLoadData` + `cuFuncSetAttribute` +
`cuLaunchKernel`. That path is exact.

### 2. Triton *prunes* kernel parameters that were passed `None`.

The JIT cubin's parameter list is **not** the Python signature. `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is
declared with 13 params but its cubin has **10** — the ones passed `None` (`gk`, `cu_seqlens`, `chunk_offsets`) are
gone. Pack the surviving ones only, or you write zeros and see nothing.

Count them for yourself:

```bash
cuobjdump --dump-elf kernel.cubin | grep -c EIATTR_KPARAM_INFO
```

Two related ones, both silent:
* `chunk_fwd_kernel_o`'s grid is **3-D** — `(cdiv(V,BV), NT, HV)`, not 2-D. Launching it 2-D gives you a partially
  correct result (`rel_rms ≈ 0.85`), which is the worst kind of wrong.
* On the AOT path (the recurrent kernel), the generated `.c` declares `double scale` but the kernel takes a 4-byte
  fp32 argument. `sed` it to `float scale` or you pass garbage.

### 3. Three codebases, three gate conventions.

| caller | what it wants in `g` |
|---|---|
| FlashInfer's `chunk_gated_delta_rule` | `exp(g)` — linear space |
| FLA's `chunk_gated_delta_rule` | `g` — **log** space, already cumsum'd |
| **`ppu_gdn_chunked` (this ABI)** | **RAW `g`** — pre-cumsum; the `.so` runs `chunk_local_cumsum` itself |

ggml hands us the raw gate, so the `.so` owns the cumsum. Mixing any two of these up is silently wrong, never an
error.

### 4. ggml's state is `[v][k]`; FLA's default is `[k][v]`.

ggml stores the recurrent state as `M[col][i]`, i.e. **[v][k]**. FLA's `STATE_V_FIRST` defaults to `False`, which is
[k][v]. Two consequences:

* the **recurrent** kernel is AOT'd with `STATE_V_FIRST=1`;
* the **chunked** kernel is not, so `gated_delta_net.cu` transposes the state on the way in and out
  (`ppu_gdn_state_transpose`).

Get this wrong and every number is wrong, with no error.

### 5. `v` is not contiguous in ggml.

On real models (Qwen3.5) `src_v` comes in non-contiguous, and FLA's kernels index it as if it were. `gated_delta_net.cu`
copies it with `ppu_gdn_make_contig` first. The `.so` silently reads garbage otherwise.

---

## Known gaps

* **GVA (`H != HV`) is compiled but never numerically validated.** `build.sh` compiles `4,8,64` and `2,4,128`, but
  `gen_golden_chunk.py:21` sets `B,T,H,HV,K,V = 1, 256, 4, 4, 128, 128` with the comment *"non-GVA first"* — and
  there was never a second. Qwen3-Next-80B is `HV=32 / H=16`. The two head-count parameters are positional and their
  order differs between FLA revisions (vLLM's vendored copy calls them `(H, Hg)` with `H` = value heads; ours passes
  `(H, HV)` with `H` = key heads), so a wrong order would be **silently wrong on GVA shapes only**. Extend the golden
  to `H != HV` before trusting this on an 80B.
* **The chunked path is still opt-in** (`GGML_PPU_GDN_CHUNKED`), now only because it needs L2-normalized `k` to be
  numerically stable (real models do this upstream; random test inputs do not). It no longer allocates: its scratch
  comes from ggml's CUDA pool via `ppu_gdn_chunked_workspace_size()`.
* **Benchmarks on H800 favour ggml's native kernels** on small models (pp512: 33354 native vs 26240 `.so`). The FLA
  kernels are tuned for large-batch training; the gap is the per-call `cudaMalloc`, the `v`-contiguity copy, and the
  F32↔bf16 bridge — all fixable seam overheads, not kernel quality.
