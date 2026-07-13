# Standalone build for the two external kernel .so files. NOT part of the llama.cpp build tree.
#
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 \
#         -DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))')
#   cmake --build build -j
#
# On the PPU box (ppu001) instead pass -DCMAKE_CUDA_COMPILER=<ppu-nvcc> -DCMAKE_CUDA_ARCHITECTURES=OFF: a forced
# -arch=sm_XX routes onto ppu0015 and rejects ppu001-only atoms. See the ppu-ptx skill.
cmake_minimum_required(VERSION 3.18)
project(ppu_so LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")

find_package(CUDAToolkit REQUIRED)

# Submodules live under <llama.cpp>/thirdparty (see .gitmodules).
set(FA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/flash-attention")
set(DG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/DeepGEMM")

# Torch is a COMPILE-time dependency only (headers): flash.h includes <ATen/cuda/CUDAGeneratorImpl.h> for the
# header-only `at::PhiloxCudaState philox_args`. We never link ${TORCH_LIBRARIES}.
# Verify after building:  ldd libppu_fa.so | grep -i torch   -> must be EMPTY.
find_package(Torch REQUIRED)

# ---------------- libppu_fa.so ----------------
# FA forward kernels (per-headdim template instances) + our torch-free shim. NO libtorch linked.
# Only the NON-SPLIT forward kernels: the shim pins num_splits=1, so run_mha_fwd_splitkv_dispatch is never called.
file(GLOB FA_KERNEL_SRCS "${FA_ROOT}/csrc/flash_attn/src/flash_fwd_hdim*_sm80.cu")
if (NOT FA_KERNEL_SRCS)
    message(FATAL_ERROR "no FA kernel .cu found under ${FA_ROOT} -- did you `git submodule update --init`?")
endif()
list(LENGTH FA_KERNEL_SRCS FA_N)
message(STATUS "ppu_fa: ${FA_N} flash_fwd kernel translation units")

add_library(ppu_fa SHARED fa/flash_api_c.cpp ${FA_KERNEL_SRCS})
target_include_directories(ppu_fa PRIVATE
    "${FA_ROOT}/csrc/flash_attn"
    "${FA_ROOT}/csrc/flash_attn/src"
    "${FA_ROOT}/csrc/cutlass/include"     # cutlass stays INSIDE the .so
    ${TORCH_INCLUDE_DIRS})                # headers only (PhiloxCudaState); not linked
target_compile_definitions(ppu_fa PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_ALIBI)
target_compile_options(ppu_fa PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections -Xcompiler=-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>)
target_link_options(ppu_fa PRIVATE "LINKER:--gc-sections")
target_link_libraries(ppu_fa PRIVATE CUDA::cudart)
set_target_properties(ppu_fa PROPERTIES OUTPUT_NAME ppu_fa)

# test_fa: standalone correctness check (packed layout, vs fp32 CPU ref). Links libppu_fa.
add_executable(test_fa fa/test_fa.cpp)
target_link_libraries(test_fa PRIVATE ppu_fa CUDA::cudart)
