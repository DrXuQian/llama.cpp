#pragma once

// Sail W4FA16 layout, derived from DeepGEMM-for-sail tests/math_utils.py.
// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdint>
#include <climits>

#if defined(__CUDACC__) || defined(__HGGCCC__) || defined(__HGGC__)
#    define NCP_MX_HD __host__ __device__ __forceinline__
#else
#    define NCP_MX_HD inline
#endif

namespace ggml_ncp_mxfp4 {

inline bool shape(int64_t n, int64_t k, int64_t experts) {
    return n > 0 && k > 0 && n <= INT_MAX && k <= INT_MAX && n % 64 == 0 && k % 64 == 0 && experts > 0 &&
           experts <= 1024 && n <= INT_MAX / k && uint64_t(experts) * n * k / 8 <= INT_MAX;
}

// Returns a nibble index in int32 [E,K/16,2N]. No numeric conversion.
NCP_MX_HD uint64_t code_index(int n, int k, uint64_t expert, unsigned row, unsigned col) {
    const unsigned r = col % 16, c = row % 16;
    const unsigned lane        = (c % 8) * 4 + (r % 8) / 2;
    const unsigned interleaved = r % 2 + 2 * (r / 8) + 4 * (c / 8);
    const unsigned slot        = (interleaved % 2) * 4 + interleaved / 2;
    return expert * n * k + uint64_t(col / 16) * n * 16 + uint64_t(row / 64) * 1024 + lane * 32 +
           ((row % 64) / 16) * 8 + slot;
}

// Returns a byte index in E8M0 [E,K/32,N], after the [0,2,1,3] swap.
NCP_MX_HD uint64_t scale_index(int n, int k, uint64_t expert, unsigned row, unsigned block) {
    const uint64_t logical = uint64_t(block) * n + row;
    const unsigned source  = logical % 64;
    const unsigned i       = (source % 8) * 8 + source / 8;
    const unsigned out     = (i / 4) * 4 + ((i % 4) % 2) * 2 + (i % 4) / 2;
    return expert * n * (k / 32) + (logical / 64) * 64 + out;
}

// One output word owns all eight codes. Input is one contiguous 64-row tile.
NCP_MX_HD uint32_t tile_word(const uint8_t * raw, int k, uint64_t word) {
    uint32_t result = 0;
    for (unsigned s = 0; s < 8; ++s) {
        const uint64_t index = word * 8 + s;
        const unsigned out = index % 1024, lane = out / 32, tile = (out / 8) % 4;
        const unsigned v     = (s % 4) * 2 + s / 4;
        const unsigned col   = unsigned(index / 1024) * 16 + 2 * (lane % 4) + v % 2 + 8 * ((v % 4) / 2);
        const unsigned row   = tile * 16 + lane / 4 + 8 * (v / 4);
        const uint64_t block = uint64_t(row) * (k / 32) + col / 32;
        const unsigned code  = (raw[block * 17 + 1 + col % 16] >> (col % 32 >= 16 ? 4 : 0)) & 15;
        result |= code << (4 * s);
    }
    return result;
}

NCP_MX_HD uint8_t raw_byte(const uint8_t * packed, int n, int k, int experts, uint64_t index) {
    const uint64_t block = index / 17;
    const unsigned byte = index % 17, kb = block % (k / 32), row = (block / (k / 32)) % n;
    const uint64_t expert = block / (uint64_t(n) * (k / 32));
    if (byte == 0) {
        return packed[uint64_t(experts) * n * k / 2 + scale_index(n, k, expert, row, kb)];
    }
    const auto *   words = reinterpret_cast<const uint32_t *>(packed);
    const uint64_t lo    = code_index(n, k, expert, row, kb * 32 + byte - 1);
    const uint64_t hi    = code_index(n, k, expert, row, kb * 32 + byte - 1 + 16);
    return ((words[lo / 8] >> (4 * (lo % 8))) & 15) | (((words[hi / 8] >> (4 * (hi % 8))) & 15) << 4);
}

}  // namespace ggml_ncp_mxfp4

#undef NCP_MX_HD
