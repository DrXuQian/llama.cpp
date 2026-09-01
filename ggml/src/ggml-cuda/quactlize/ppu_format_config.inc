#pragma once
// ONE shipping-path source for the PPU arrangement constants that are fixed per GGUF format.
//
// Consumers define X as:
//   X(Id, "name", qtype, low_bits, high_bits, group_size,
//     scale_first_tile_k, fully_quantized_tile_k, packed_format)
//
// This is an X-macro rather than a C++ array so the C++ library/bench can include it and the offline Python packer
// can parse the same rows without loading a device library. Tests may independently recompute these values; shipping
// code must consume this registry or make divergence fail loudly.
//
// SCALE_FIRST TileK is the canonical minimum 32-byte run for the narrowest code plane: 256/bits. Two-plane formats
// use the narrower/high plane. FULLY_QUANTIZED is deliberately distinct: Q2/Q4's packed metadata unit covers a whole
// 256-code superblock and measured best when one k-tile consumes it; Q3/Q5 already need 256 for their int1 plane;
// Q6 keeps the independently validated 128-K high-plane map because its 256-K inverse is incomplete.
#define QUACTLIZE_PPU_FORMAT_CONFIGS(X) \
  X(Q2_K, "Q2_K", 10, 2, 0, 16, 128, 256, 2) \
  X(Q3_K, "Q3_K", 11, 2, 1, 16, 256, 256, 3) \
  X(Q4_K, "Q4_K", 12, 4, 0, 32,  64, 256, 0) \
  X(Q5_K, "Q5_K", 13, 4, 1, 32, 256, 256, 1) \
  X(Q6_K, "Q6_K", 14, 4, 2, 16, 128, 128, 4)
