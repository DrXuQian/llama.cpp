# ===== llama.cpp CI 配置文件 =====
# 必填环境变量:
#   LLAMA_CI_DIR  - llama.cpp 仓库路径
#   NCP_LIB_DIR   - ncp_flash_lib 仓库路径
# 可选环境变量:
#   LLAMA_MODEL   - 模型 .gguf 路径 (仅集成测试需要)
#   USRNAME/TOKEN - gitlab 凭证 (仅子模块拉取需要)
#   PPU_NVCC      - PPU nvcc 路径
#   JOBS          - 并行数

# --- 路径配置 (必填) ---
if [ -z "$LLAMA_CI_DIR" ]; then
    echo "ERROR: LLAMA_CI_DIR is not set." >&2
    echo "  export LLAMA_CI_DIR=/path/to/llama.cpp" >&2
    exit 1
fi
export LLAMA_CI_DIR

if [ -z "$NCP_LIB_DIR" ]; then
    echo "ERROR: NCP_LIB_DIR is not set." >&2
    echo "  export NCP_LIB_DIR=/path/to/ncp_flash_lib" >&2
    exit 1
fi
export NCP_LIB_DIR

# --- 模型文件 (可选, 仅集成测试需要) ---
export LLAMA_MODEL="${LLAMA_MODEL:-}"

# --- 代理设置 ---
export https_proxy=http://11.122.78.49:3128
export http_proxy=http://11.122.78.49:3128
export no_proxy=localhost,127.0.0.1,::1,.eng.t-head.cn,.dev.t-head.cn

# --- 构建参数 ---
export PPU_NVCC="${PPU_NVCC:-/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc}"
export JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Ensure PPU nvcc is in PATH
export PATH="${PPU_NVCC%/*}:${PATH}"

# --- 运行时环境 (build 和 test 共用) ---
# .so 路径不再由环境变量给出: 由编译开关 (-DGGML_NCP_MOE) 决定加载哪个库,
# 运行时从二进制同目录 (build-ci/bin) 找 .so, build.sh 负责把 .so 拷进去。

# GDN chunked arm is ON by default now; keep this for explicitness.
export GGML_NCP_GDN_CHUNKED=ON

# MoE (DeepGemm) JIT runtime env. DeepGemm dereferences CUDA_HOME without a null
# check (unset -> crash). No DG_LIBRARY_ROOT on purpose: the .so finds the
# deep_gemm/ tree build.sh ships beside it, and setting the variable here would
# keep that lookup -- the one a target box relies on -- out of what CI runs. The
# .so says so itself if it comes up without a root; a JIT that cannot init throws
# on every call and ggml falls back to inline MoE, so the answer stays correct and
# nothing else tells you the accelerator never ran. DG_JIT_CACHE_DIR holds the JIT
# cubins; kept under ncp build/, not build-ci which every build wipes, so
# first-run auto-warm survives a rebuild.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-${NCP_LIB_DIR}/build/ncp_moe_cache}"
mkdir -p "$DG_JIT_CACHE_DIR"

# --- 测试参数 ---
export TEST_TIMEOUT=5400
export TEST_THREADS=8

# 输出目录和报告文件 (放在 ppu_ci 目录下, 不污染 llama.cpp 仓库)
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"
export REPORT_FILE="$SCRIPT_DIR/REPORT.md"
