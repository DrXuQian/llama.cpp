// Host-only fixture for testing the plugin ABI and ggml scheduler without a PPU.
#include "runtime.h"

#include <cmath>

namespace ppu {
const char * name() { return "PPU-TEST"; }
const char * error() { return "host test runtime"; }
bool host_test() { return true; }
bool rms_norm(int, const float * src, float * dst, int64_t columns, int64_t rows, float eps) {
    for (int64_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (int64_t col = 0; col < columns; ++col) {
            const double x = src[row*columns + col];
            sum += x*x;
        }
        const float scale = 1.0f / std::sqrt(float(sum / columns) + eps);
        for (int64_t col = 0; col < columns; ++col) dst[row*columns + col] = src[row*columns + col]*scale;
    }
    return true;
}
}
