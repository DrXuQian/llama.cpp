# ppu_so — external FlashAttention kernel `.so`

Builds the FlashAttention forward kernels **outside** the llama.cpp build into a standalone shared object, which
ggml-cuda `dlopen`s at runtime.

| `.so` | source submodule | C ABI header (in ggml-cuda) | kernels |
|---|---|---|---|
| `libppu_fa.so`  | `thirdparty/flash-attention` (`csrc/flash_attn`) | `ggml/src/ggml-cuda/ppu-fa-so.h` | FA2 **sm80** fwd (`run_mha_fwd_<T, hdim, Is_causal>`) |
| `libppu_fa3.so` | `thirdparty/flash-attention` (`hopper/`)         | same ABI                         | FA3 **sm90** fwd (wgmma/TMA, `run_mha_fwd_<90, T, hd, hd, …>`) |

Both expose the **same** `ppu_flash_attn_fwd` ABI, so they are interchangeable — point `GGML_PPU_FA_SO` at whichever
matches your GPU.

**Decoupling contract.** llama.cpp compiles with **zero** cutlass / flash-attention / torch headers or link deps. It
only knows the thin `extern "C"` header (`ppu-fa-so.h`) and `dlopen`s the `.so` at runtime (`ppu-so.cu`). If the
`.so` is missing or has no kernel for a shape, the caller falls back to the inline ggml FA. Build llama.cpp with
`-DGGML_PPU_SO=ON`; the default OFF is fully inert.

**Scope.** fp16/bf16 only, head_dim ∈ {64, 96, 128, 192, 256}, GQA, softcap (FA2), causal. Quantized KV and
attention sinks fall through to the inline path.

## 1. Submodule

```sh
git submodule update --init --recursive thirdparty/flash-attention
```

## 2. Build

**FA2 (sm80 kernels; run anywhere ≥ Ampere):**
```sh
cd ppu_so
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))')
cmake --build build -j            # -> build/libppu_fa.so, build/test_fa
```

**FA3 (sm90 / Hopper — the fast one on H100/H800):**
```sh
./fa3/build_fa3.sh                # -> fa3/libppu_fa3.so
```

Torch is a **compile-time dependency only** (header include dirs — `flash.h` pulls an ATen header for the
header-only `at::PhiloxCudaState` field). Neither `.so` links libtorch — verify with **both**:

```sh
ldd libppu_fa3.so | grep -i torch                              # must be empty
nm -D -u libppu_fa3.so | c++filt | grep -iE 'c10::|at::|torch' # must be empty  (ldd alone lies: an
                                                               #  undefined symbol has no DT_NEEDED)
```

The two torch symbols the FA launch templates bottom out in (`c10::cuda::c10_cuda_check_implementation`) are
**defined by the shim itself** — they throw, the caller catches, and ggml falls back to inline.

On the PPU box pass `-DCMAKE_CUDA_COMPILER=<ppu-nvcc> -DCMAKE_CUDA_ARCHITECTURES=OFF` instead: a forced `-arch=sm_XX`
routes onto ppu0015 and rejects ppu001-only atoms (see the `ppu-ptx` skill).

## 3. Standalone test (no ggml, no model)

```sh
./build/test_fa   <causal> <head_dim>     # FA2
./fa3/test_fa3    <causal> <head_dim>     # FA3
```
Both compare against an fp32 CPU reference. FA3: 6/6 PASS (causal×{0,1} × hd{64,128,256}), err ~1e-4 (fp16).

## 4. Point llama.cpp at it

```sh
cmake -B build -DGGML_CUDA=ON -DGGML_PPU_SO=ON -DCMAKE_CUDA_ARCHITECTURES=90
export GGML_PPU_FA_SO=/abs/path/ppu_so/fa3/libppu_fa3.so   # env wins over the bare soname
```

## The three things this integration had to get right

**1. ggml's FA output tensor has head/seqlen SWAPPED vs Q.** `ggml_flash_attn_ext` builds dst as
`ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}` = `{dv, n_head, seqlen_q, batch}` — so dst is physically
`[b][s][h][dv]` (FA's own packed layout) while Q is `[b][h][s][d]`. The O strides handed to the `.so` must be
`o_head_stride = dv`, `o_row_stride = dv*n_head` — **not** Q's order. Getting this wrong writes O transposed and is
**invisible whenever seqlen_q == 1** (i.e. all of decode), so it only shows up on prefill. Safest rule: derive every
stride from `->nb[i]/elem_size`, never from the `ne` order. (Q is likewise only *contiguously allocated* — it's a
`ggml_permute` view from `build_attn_mha` — and `ggml_get_to_fp16_cuda` converts the flat element range, so the F16
copy inherits Q's physical strides.)

**2. The causal hint.** FA's sm80/sm90 forward has **no additive-mask input** — attention shape is expressed only via
`is_causal`/window. ggml passes an additive mask tensor. So the hook can only engage for full attention
(`mask == null`) or a *pure* bottom-right causal mask. `ggml_flash_attn_ext_set_causal` (new, `op_params[4]`) lets the
host say so: `llama-graph.cpp build_attn_mha` sets it when `causal_attn && no-ALiBi && !SWA && !kv_unified`, and the
hook engages causal only when the hint is set **and** `mask->ne[3] == 1` (single stream). Everything else — SWA,
padding, cross-sequence, ALiBi, sinks — falls through to inline. Also: non-causal must pass
`window_size_left = window_size_right = -1` (**not** `seqlen_k`, which flips the kernel onto the slower `Is_local`
path), and `seqlen_q == 1` downgrades to non-causal, as upstream `mha_fwd` does.

**3. FA3 (sm90) needs a `tile_count_semaphore`.** Its persistent tile scheduler always wants one for `arch >= 90`
(even at `num_splits == 1`) — an atomic counter that must be **zeroed before every launch**. The shim allocates and
zeroes it, and pins `num_splits=1, num_splits_dynamic_ptr=nullptr, pack_gqa=false, page_table=nullptr, dv=d,
arch=90, num_sm` from `cudaGetDeviceProperties`. FA3's `hopper/flash.h` has **no namespace** (global
`Flash_fwd_params` / `run_mha_fwd_` — don't write `FLASH_NAMESPACE::`), and it must be compiled with
`-gencode arch=compute_90a,code=sm_90a` against cutlass 4.
