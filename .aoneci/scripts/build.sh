#!/bin/bash
# ============================================================
# build.sh - ncp_flash_lib + llama.cpp 联合构建
#
# How to use:
#   1. 设置仓库路径 (必填):
#        export LLAMA_CI_DIR=/path/to/llama.cpp
#        export NCP_LIB_DIR=/path/to/ncp_flash_lib
#
#   2. 设置 gitlab 凭证 (可选, 仅子模块未拉取时需要):
#        export USRNAME=user@alibaba-inc.com
#        export TOKEN=<your-gitlab-token>
#
#   3. 设置模型路径 (可选, 仅集成测试需要):
#        # 目前仅支持 Qwen3-32B-BF16.gguf
#        export LLAMA_MODEL=/path/to/Qwen3-32B-BF16.gguf
#
#   4. 构建并测试:
#        cd ppu_ci && ./build.sh && ./run_tests.sh
#
#   5. 按需调试单个算子 (可选):
#        ./run_backend_ops.sh MUL_MAT
#
# 可选环境变量 (覆盖默认值):
#   PPU_NVCC     PPU nvcc 路径   (default: /usr/local/PPU_SDK/CUDA_SDK/bin/nvcc)
#   JOBS         并行编译数     (default: $(nproc))
#   NCP_LIB_REV  ncp_flash_lib 完整 40 位 commit sha (default: .aoneci/NCP_LIB_VERSION 里记录的值)
#   LLAMA_BUILD_DIR  New build output directory (default: LLAMA_CI_DIR/build-ci)
# ============================================================

set -euo pipefail

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LLAMA_BUILD_DIR=$(realpath -m -- "${LLAMA_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}")
if [ -e "$LLAMA_BUILD_DIR" ] || [ -L "$LLAMA_BUILD_DIR" ]; then
    echo "ERROR: build output exists: $LLAMA_BUILD_DIR; set LLAMA_BUILD_DIR to a new path" >&2
    exit 1
fi

echo "=========================================="
echo "Build: ncp_flash_lib + llama.cpp"
echo "=========================================="
echo "llama.cpp   : $LLAMA_CI_DIR"
echo "build output: $LLAMA_BUILD_DIR"
echo "ncp_flash   : $NCP_LIB_DIR"
echo "PPU nvcc    : $PPU_NVCC"
echo "jobs        : $JOBS"
echo ""

# --- 1. Git URL rewrite (if USRNAME/TOKEN set) ---
if [ -n "${USRNAME:-}" ] && [ -n "${TOKEN:-}" ]; then
    echo "==> [1/5] Configuring git URL rewrite..."
    ENCODED_USR="${USRNAME/@/%40}"
    GITLAB_HTTP="http://${ENCODED_USR}:${TOKEN}@gitlab.alibaba-inc.com/"
    git config --global --add url."${GITLAB_HTTP}".insteadOf "git@code.alibaba-inc.com:"
    git config --global --add url."${GITLAB_HTTP}".insteadOf "git@gitlab.alibaba-inc.com:"
    git config --global --add url."${GITLAB_HTTP}".insteadOf "http://gitlab.alibaba-inc.com/"
    trap 'git config --global --unset-all url."${GITLAB_HTTP}".insteadOf 2>/dev/null || true' EXIT
else
    echo "==> [1/5] USRNAME/TOKEN not set -- skipping URL rewrite"
fi

# --- 2. Check out the pinned ncp_flash_lib + check submodules ---
# The .so and the ggml hook that calls it have to move together, so .aoneci/NCP_LIB_VERSION names the ncp_flash_lib
# commit this llama.cpp tree is built against: a run is reproducible from the two shas, and a kernel-side fix reaches
# CI as an explicit bump. The file is read from LLAMA_CI_DIR, not SCRIPT_DIR, so a copy of this script placed
# elsewhere still picks up the pin. NCP_LIB_REV overrides it for a one-off build.
echo "==> [2/5] Checking out the pinned ncp_flash_lib + checking submodules..."
NCP_VERSION_FILE="${LLAMA_CI_DIR}/.aoneci/NCP_LIB_VERSION"
NCP_REV="${NCP_LIB_REV:-}"
if [ -z "${NCP_REV}" ]; then
    if [ ! -f "${NCP_VERSION_FILE}" ]; then
        echo "ERROR: ${NCP_VERSION_FILE} is missing -- it names the ncp_flash_lib commit to build against" >&2
        exit 1
    fi
    NCP_REV=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "${NCP_VERSION_FILE}" | grep -m1 . || true)
fi
# One full sha and nothing else: a branch or a tag moves, and a pin that moves does not name a single build.
if ! printf '%s\n' "${NCP_REV}" | grep -qE '^[0-9a-fA-F]{40}$'; then
    echo "ERROR: need one full 40-char commit sha for ncp_flash_lib, got '${NCP_REV}'" >&2
    exit 1
fi
# Lowercase from here on -- the form git prints -- so the compare below stays a plain string compare.
NCP_REV=$(printf '%s' "${NCP_REV}" | tr 'A-F' 'a-f')

cd "${NCP_LIB_DIR}"
if [ "$(git rev-parse HEAD)" = "${NCP_REV}" ]; then
    echo "    ncp_flash_lib already at ${NCP_REV}"
else
    git cat-file -e "${NCP_REV}^{commit}" 2>/dev/null || git fetch --all --tags
    if ! git cat-file -e "${NCP_REV}^{commit}" 2>/dev/null; then
        echo "ERROR: ncp_flash_lib has no commit ${NCP_REV}, even after fetching" >&2
        exit 1
    fi
    git checkout --detach "${NCP_REV}"
    # The pin covers the submodule commits too. --force because cmake applies the DeepGemm patches into the submodule
    # work tree: a plain update keeps those edits and the patches then fail to apply onto themselves.
    git submodule update --init --recursive --force
    echo "    ncp_flash_lib -> ${NCP_REV}"
fi

if [ ! -f third_party/flash-attention/.git ] || [ ! -f third_party/DeepGemm/.git ]; then
    echo "    submodules missing -- pulling (--init --recursive)"
    git submodule update --init --recursive
else
    echo "    submodules already checked out"
fi

# --- 3. Build ncp_flash_lib (FA + MoE; GDN off) ---
# CI exercises the FA (FlashAttention-3) and MoE (DeepGemm) hooks, so both .so are built.
# MoE needs Torch HEADERS to compile but does NOT link libtorch -- without torch there is nothing to test, so fail.
# FA needs no torch at all (v3 dropped the ATen include v2 pulled in); the ldd check below is what keeps it that way.
echo "==> [3/5] Building ncp_flash_lib (FA + MoE)..."
TORCH_DIR=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))' 2>/dev/null || echo "")
if [ -z "$TORCH_DIR" ]; then
    echo "ERROR: torch not found -- MoE needs torch headers at build time" >&2
    exit 1
fi
echo "    torch found: ${TORCH_DIR}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${PPU_NVCC}" \
    -DCMAKE_PROJECT_INCLUDE="${LLAMA_CI_DIR}/.aoneci/cmake/ncp-ppu-runtime.cmake" \
    -DCMAKE_CUDA_ARCHITECTURES=OFF \
    -DNCP_BUILD_FA=ON \
    -DNCP_BUILD_MOE=ON \
    -DNCP_BUILD_GDN=OFF \
    -DTorch_DIR="${TORCH_DIR}"
cmake --build build -j"${JOBS}"

# Neither .so may link libtorch: on a box without torch the dlopen fails and ggml falls back to the inline kernels --
# green tests that exercised nothing. MoE gets there via --gc-sections, FA by not needing torch at all, so a hit on FA
# means the ATen dependency v3 dropped has come back.
for so in libncp_fa.so libncp_moe.so; do
    if [ -f "build/${so}" ] && ldd "build/${so}" 2>/dev/null | grep -qi torch; then
        echo "    WARNING: ${so} links libtorch -- will fail on boxes without torch"
    fi
done

# --- 4. Check both .so came out ---
# llama.cpp decides at COMPILE time which .so it dlopens (the -DGGML_NCP_* flags below) and looks for them next to its
# own binaries at runtime. Those flags are stated explicitly, NOT derived from which .so happen to exist: a leftover
# libncp_gdn.so from an earlier run must not switch the GDN hook on. Checked here, BEFORE llama.cpp builds, so a
# missing .so costs one line of output instead of a `mv: cannot stat` at the end of a full compile.
for so in libncp_fa.so libncp_moe.so; do
    if [ ! -f "${NCP_LIB_DIR}/build/${so}" ]; then
        echo "ERROR: ${NCP_LIB_DIR}/build/${so} was not built" >&2
        exit 1
    fi
done
echo "==> [4/5] libncp_fa.so + libncp_moe.so built -> llama.cpp gets -DGGML_NCP_FA=ON -DGGML_NCP_MOE=ON (GDN stays off)"

# --- 5. Build llama.cpp, then drop the .so next to the test binaries ---
echo "==> [5/5] Building llama.cpp..."
cd "${LLAMA_CI_DIR}"
cmake -S . -B "${LLAMA_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${PPU_NVCC}" \
    -DGGML_CUDA=ON \
    -DGGML_USE_PPU=ON \
    -DGGML_NCP_MOE=ON \
    -DGGML_NCP_FA=ON \
    -DGGML_NCP_GDN=OFF \
    -DGGML_NCP_QUACTLIZE=ON \
    -DCMAKE_CUDA_ARCHITECTURES=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF
cmake --build "${LLAMA_BUILD_DIR}" -j"${JOBS}"

# build-ci/bin holds libggml-cuda.so and the test binaries, and is where the $ORIGIN runpath looks. Moved, not
# copied: one copy of the .so, so there is never a question of which one is being loaded. A re-run relinks it.
mv -f "${NCP_LIB_DIR}/build/libncp_moe.so" "${LLAMA_BUILD_DIR}/bin/"
mv -f "${NCP_LIB_DIR}/build/libncp_fa.so" "${LLAMA_BUILD_DIR}/bin/"
echo "    installed libncp_moe.so, libncp_fa.so  -> ${LLAMA_BUILD_DIR}/bin/"


# The .so finds its own JIT include tree at bin/deep_gemm/include through dladdr, the way deep_gemm/__init__.py uses
# dirname(__file__) -- so nothing has to export DG_LIBRARY_ROOT, here or on a target box. Warm cubins are not shipped; the .so picks up
# bin/deep_gemm/cache if someone puts them there, otherwise it JITs into DG_JIT_CACHE_DIR.
DG_INC_SRC="${NCP_LIB_DIR}/third_party/DeepGemm/deep_gemm/include"
DG_INC_DST="${LLAMA_BUILD_DIR}/bin/deep_gemm/include"
mkdir -p "${DG_INC_DST}"

# deep_gemm/ is the one that must be there: DeepGemm hashes every .cuh under it into the cubin cache key, on every start
# and not just on a cache miss. Without this check the loop below would ship a tree with no headers in it.
if [ ! -d "${DG_INC_SRC}/deep_gemm" ]; then
    echo "ERROR: ${DG_INC_SRC}/deep_gemm is missing -- the MoE .so would have no JIT include tree" >&2
    exit 1
fi

# Whatever else the submodule ships comes along as-is (the nvcc -I list points into here, so a cache miss needs it), with
# -L to turn a header symlinked out to a sibling submodule into a real file. A dangling one means that submodule was
# never checked out: warn and skip, since a cache hit does not need it.
for p in "${DG_INC_SRC}"/*; do
    if [ -e "${p}" ]; then
        cp -RL "${p}" "${DG_INC_DST}/"
    else
        echo "    WARNING: include/$(basename "${p}") dangles (submodule not checked out) -- a JIT cache miss cannot compile"
    fi
done
echo "    installed deep_gemm/include -> build-ci/bin/deep_gemm/ ($(find -L "${DG_INC_DST}" -type f | wc -l | tr -d ' ') files)"

# --- Done ---
echo ""
echo "========================================"
echo "BUILD COMPLETE"
echo "========================================"
echo "  NCP hooks     : FA + MoE (libncp_fa.so, libncp_moe.so + deep_gemm/ in the output bin directory)"
echo "  K-pack hook   : ON (load the separately published Quactlize runtime at execution)"
echo "  test binaries : ${LLAMA_BUILD_DIR}/bin/"
echo "========================================"
echo ""
echo "No GGML_NCP_*_LIB to export -- the \$ORIGIN runpath finds the .so beside the binaries."
echo "Replacing it? Put the new .so there; the loader warns and falls back to the inline kernels when it is missing."
echo ""
echo "Before running tests manually (not via run_tests.sh), source the runtime env first:"
echo "  source ${SCRIPT_DIR}/config.sh"
