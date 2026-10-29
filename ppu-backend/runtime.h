#pragma once

#include <cstdint>

namespace ppu {
const char * name();
const char * error();
bool host_test();
// Synchronous operation on host tensors; the PPU runtime owns device staging.
bool rms_norm(int device, const float * src, float * dst, int64_t columns, int64_t rows, float eps);
}
