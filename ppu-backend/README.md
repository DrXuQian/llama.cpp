# External PPU backend: native RMS_NORM override

This standalone project builds `libggml-ppu.so` against an unmodified llama.cpp checkout. `GGML_BACKEND_PATH` takes the path to this **file**, not a directory. No changes to llama.cpp, GGML, its operator enum, or its main CMake files are needed.

The plugin follows the existing BLAS accelerator pattern: it registers an `ACCEL` backend with host buffers, claims contiguous FP32 `GGML_OP_RMS_NORM`, and leaves other compute operators to the original backends. The graph still calls `ggml_rms_norm()`. Device initialization is deferred until the operator runs, so loading the plugin does not require a usable PPU. The first implementation uses synchronous host/device copies and temporary device buffers; it proves dispatch and is not a performance implementation or a full GPU backend.

The PPU implementation prints the native node name before entering the device runtime:

```text
[ggml-ppu] ggml_backend_init reached; registering native GGML_OP_RMS_NORM override
[ggml-ppu] loaded backend=PPU override=GGML_OP_RMS_NORM
[ggml-ppu] RMS_NORM node=norm-0 device=PPU0 rows=1 columns=896 eps=1e-06
```

Build the host libraries from the repository root:

```bash
cmake -S . -B build-host \
  -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build build-host --target ggml ggml-cpu llama-bench -j 8
```

Build the plugin with the PPU SDK. The host build and the headers must come from the same llama.cpp revision; the GGML backend ABI is not stable across arbitrary revisions. This prototype was built against local master `2092353c8`.

```bash
cmake -S ppu-backend -B ppu-backend/build-ppu \
  -DPPU_SDK="$PPU_SDK" \
  -DLLAMA_CPP_BUILD_DIR="$PWD/build-host"
cmake --build ppu-backend/build-ppu -j 8

export GGML_BACKEND_PATH="$PWD/ppu-backend/build-ppu/libggml-ppu.so"
./ppu-backend/build-ppu/ppu-rms-norm-demo
```

The demo uses the stock GGML loader and scheduler. It does not explicitly assign RMS_NORM to the plugin: backend capabilities and priority choose it. The following SCALE stays on CPU. Three shapes, changed inputs, and zero inputs are compared against the original CPU RMS_NORM implementation. A missing PPU runtime may fail after the `RMS_NORM` entry message; that proves dispatch, not successful PPU execution.

Use a real model with the unchanged host executable:

```bash
GGML_BACKEND_PATH="$PWD/ppu-backend/build-ppu/libggml-ppu.so" \
  ./build-host/bin/llama-bench -m "$MODEL" -ngl 0 -p 0 -n 1 -r 1 -t 2 -v
```

The ACCEL backend is added automatically by llama.cpp. This demonstration keeps model weights on CPU so native RMS_NORM can be routed to the plugin using host buffers. It does not intercept a separate CUDA backend's device-resident kernels. On a PPU host, use the SDK runtime environment and `CUDA_VISIBLE_DEVICES` to choose which physical device is visible as device 0.

For a host-only check of the loader, automatic routing, and numerical comparison:

```bash
cmake -S ppu-backend -B ppu-backend/build-host-test -DPPU_BACKEND_HOST_TEST=ON
cmake --build ppu-backend/build-host-test -j 8
ctest --test-dir ppu-backend/build-host-test --output-on-failure -V
```

That fixture builds `libggml-ppu-test.so`, uses CPU math, and identifies itself as `PPU-TEST0`. It does not test PPU kernel execution.

Local validation: the PPU SDK 2.1.1 build produced the plugin and the unchanged `llama-bench` reached the RMS_NORM message above with Qwen2.5-0.5B-Instruct-Q4_K_M. Execution then stopped at `cudaSetDevice` because this host lacks `libhggcrt.13.0.so`. The host-only scheduler/numerical test passed all three shapes with zero maximum error. PPU kernel numerics still require a PPU runtime and device.
