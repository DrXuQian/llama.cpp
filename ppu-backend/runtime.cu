#include "runtime.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace ppu {
static thread_local const char * last_error = "success";
static bool check(cudaError_t status) {
    last_error = cudaGetErrorString(status);
    return status == cudaSuccess;
}
const char * name() { return "PPU"; }
const char * error() { return last_error; }
bool host_test() { return false; }

static __global__ void rms_norm_f32(const float * src, float * dst, int64_t columns, int64_t rows, float eps) {
    __shared__ float sums[256];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = 0.0f;
        for (int64_t col = threadIdx.x; col < columns; col += blockDim.x) {
            const float x = src[row*columns + col];
            sum += x*x;
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        const float scale = rsqrtf(sums[0] / columns + eps);
        for (int64_t col = threadIdx.x; col < columns; col += blockDim.x) {
            dst[row*columns + col] = src[row*columns + col]*scale;
        }
        __syncthreads();
    }
}
bool rms_norm(int device, const float * src, float * dst, int64_t columns, int64_t rows, float eps) {
    if (rows == 0) return true;
    if (!check(cudaSetDevice(device))) return false;
    const size_t bytes = size_t(columns*rows)*sizeof(float);
    float * input = nullptr;
    float * output = nullptr;
    if (!check(cudaMalloc(reinterpret_cast<void **>(&input), bytes))) return false;
    if (!check(cudaMalloc(reinterpret_cast<void **>(&output), bytes))) {
        cudaFree(input);
        return false;
    }
    bool ok = check(cudaMemcpy(input, src, bytes, cudaMemcpyHostToDevice));
    if (ok) {
        rms_norm_f32<<<unsigned(std::min(rows, int64_t(65535))), 256>>>(input, output, columns, rows, eps);
        ok = check(cudaGetLastError()) && check(cudaMemcpy(dst, output, bytes, cudaMemcpyDeviceToHost));
    }
    const cudaError_t free_input = cudaFree(input), free_output = cudaFree(output);
    return ok && check(free_input) && check(free_output);
}
}
