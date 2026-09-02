#pragma once
// Runtime inventory of tactics compiled into libquactlize_ppu.so. The array and its strings have static lifetime;
// callers neither allocate nor free them.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quactlize_ppu_config_v1 {
  // Compare the family discriminator first. When true, the tile fields are meaningless and must be ignored; this
  // lets a CUDA-core GEMV live in the same candidate array and profiling loop as tensor-core tile configurations.
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
} quactlize_ppu_config_v1;

// Per-problem tactic description. Unlike the legacy geometry-only v1 inventory, this record carries TileK: TileK
// selects the resident weight arrangement and therefore must agree with the offline artifact. The scheme-specific
// filtered queries below are the authoritative inventory for new tuners. name has static lifetime. As in v1, every
// tile field (including tile_k) is meaningless and zero for a CUDA-core family record.
typedef struct quactlize_ppu_config_v2 {
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t tile_k;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
} quactlize_ppu_config_v2;

// Versioned identity of the bytes produced by the dense offline placer.  `bits` and `high_bits` describe the
// physical low/high code planes; artifact_tile_k selects their delivery fold.  This descriptor belongs to the
// artifact, not to a tactic.  New readers require the pointer and fail closed on a null/unknown version instead of
// guessing the registry default.  The established v1/v2 reader entries below remain the compatibility surface for
// registry-default artifacts.
#define QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1 1
typedef struct quactlize_ppu_placed_arrangement_v1 {
  int32_t version;
  int32_t bits;
  int32_t artifact_tile_k;
  int32_t high_bits;
} quactlize_ppu_placed_arrangement_v1;

// v2 makes the physical byte map explicit.  v1 remains the immutable xplane
// compatibility ABI; a v2 K-pack4 artifact must never be passed through a v1
// pointer and inferred from ArtifactTileK.
#define QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2 2
#define QUACTLIZE_PPU_LAYOUT_XPLANE_V1 0
#define QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 1
#define QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1 2
#define QUACTLIZE_PPU_LAYOUT_Q4_N16K64_DIRECT_V1 3
#define QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID UINT64_C(0x51344b5034540001)
#define QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID UINT64_C(0x514b504b54000001)
#define QUACTLIZE_PPU_Q4_N16K64_DIRECT_MAPPING_ID UINT64_C(0x51344e3136440001)

// Host-only identity of the loaded binary. The default/ScaleFirst library returns -1; format-selected fully
// quantized builds return FMT0..FMT4. Loaders must compare this value with the filename/manifest role before using
// any arrangement-aware producer, inventory or launch entry.
int32_t quactlize_ppu_build_packed_format_v1(void);

typedef struct quactlize_ppu_placed_arrangement_v2 {
  int32_t version;
  int32_t layout;
  int32_t bits;
  int32_t high_bits;
  // Xplane retains its resident copy quantum here.  K-pack4 has no artifact
  // TileK axis and must set this field to zero.
  int32_t artifact_tile_k;
  int32_t transport_tile_k;
  int32_t group_size;
  int32_t reserved;
  uint64_t mapping_id;
} quactlize_ppu_placed_arrangement_v2;

// Return the one canonical resident K-pack descriptor owned by this loaded
// library for qtype. This query is host-only and does not create a PPU context
// or launch work. It succeeds only in a format-selected PPU_PACKED_FORMAT
// build for that format's one qtype; the default/ScaleFirst build and every
// other qtype return a nonzero status. `out` is zeroed before every such
// failure so a caller cannot accidentally reuse a descriptor from a previous
// library. A null `out` returns 23 without a write; an unknown qtype returns
// 22; a known qtype not owned by the loaded format returns 29.
int quactlize_ppu_canonical_arrangement_v2(
    int qtype, quactlize_ppu_placed_arrangement_v2* out);

// Host-only offline placement/recovery.  The v2 descriptor is checked by the
// same library that produces the bytes; null/unknown/mismatched descriptors
// fail closed.  Existing fixed/tile-aware symbols remain source and ABI
// compatible and continue to produce xplane bytes.
int quactlize_ppu_prepare_dense_for_arrangement_v2(
    uint8_t const* low_native, uint8_t const* high_native,
    uint8_t* low_layout, uint8_t* high_layout,
    int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int quactlize_ppu_recover_dense_for_arrangement_v2(
    uint8_t const* low_layout, uint8_t const* high_layout,
    uint8_t* low_native, uint8_t* high_native,
    int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);

// Arrangement-aware inventory row.  v2 could use one TileK field only because its artifact and tactic were the
// same registry value.  A folded artifact may be consumed by a wider tactic, so v3 names both quantities and makes
// accidental substitution visible to callers.
typedef struct quactlize_ppu_config_v3 {
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t tactic_tile_k;
  int32_t artifact_tile_k;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
} quactlize_ppu_config_v3;

// Scheduler-aware successor.  K-pack4 decode uses fixed Split-K while prefill uses S1; encoding S only in a
// human-readable config name would make a deployment registry unable to prove which measured product it selected.
// Existing v3 queries remain ABI compatible and expose the same rows without this final field.
typedef struct quactlize_ppu_config_v4 {
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t tactic_tile_k;
  int32_t artifact_tile_k;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
  int32_t split_k_slices;
} quactlize_ppu_config_v4;

// Stores a static config-array address in *configs when configs is non-null and returns its element count.
// No CUDA/PPU context is required. Dense and grouped are separate operators and therefore separate inventories;
// each array also contains its CUDA-core GEMV tactic, discriminated before its meaningless tile fields.
int32_t quactlize_ppu_list_configs(quactlize_ppu_config_v1 const** configs);
int32_t quactlize_ppu_list_grouped_configs(quactlize_ppu_config_v1 const** configs);

// Resolve the exact tactic that an arrangement-v2 launch would use for the
// requested name. A null/empty name applies the library's automatic policy;
// a known nonempty name is returned unchanged; an unknown or invalid name
// fails closed. These host-only queries launch no work. They return 1 after
// writing a record whose name has static library lifetime, otherwise 0 after
// zeroing a non-null output record. Dense uses v4 because Q4 K-pack4 carries
// fixed Split-K; grouped currently resolves only explicit/default policy.
int32_t quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2(
    quactlize_ppu_config_v4* config,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* requested_config_name);
int32_t quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2(
    quactlize_ppu_config_v3* config,
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* requested_config_name);

// Host-only admission for loaders that replace the source GGUF bytes with one
// resident K-pack artifact before runtime M is known.  These queries describe
// the null-config device path, not merely whether an unrelated explicit row is
// present in the inventory.  They launch no work and return 1 only when that
// path has a compiled tactic for every positive runtime M in its public domain.
//
// Dense covers every exact measured-M selector point and both unmeasured
// fallback regions.  Grouped covers every legal ragged distribution with
// total_rows > 0 and 0 < max_rows <= total_rows; zero-row experts remain legal.
// This is a tactic-capability query, not a promise that an arbitrary-size
// workspace allocation will succeed.
int32_t quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2(
    int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int32_t quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2(
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);

// Writes up to capacity valid records and returns the full valid-record count, so a caller may first pass
// (NULL, 0), allocate exactly that many records, and query again. A negative capacity writes nothing. These are
// host-only queries and require no CUDA/PPU context. Dense/grouped tensor records report the scheme-specific TileK
// selected by ppu_format_config.inc; CUDA records report their family with zero/meaningless tile fields.
int32_t quactlize_ppu_list_valid_dense_lowbit_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype);
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype);
// Arrangement-aware successor.  The same descriptor-to-tactic predicate is used here and by the corresponding
// launch entry; a missing/unknown/mismatched descriptor returns no rows rather than decoding with a default fold.
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v1(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement);
// v2 descriptor successor. Xplane-v2 delegates to the immutable v1 reader
// classes; Q4 K-pack4 returns rows whose artifact_tile_k is zero because the
// physical bytes have no tactic-TileK identity.
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
    quactlize_ppu_config_v4* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement);
int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype);
// Physical-layout-aware grouped inventory. K-pack4 has no artifact TileK, so
// returned v3 rows carry tactic_tile_k=256 and artifact_tile_k=0. The same
// predicate is used by host/device launches; malformed descriptors yield zero
// rows rather than inheriting the legacy Xplane map.
int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement);
int32_t quactlize_ppu_list_valid_vecdot_moe_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype);

// Host-only per-problem validity predicates for the inventories above. They return 1 only when config_name names a
// compiled tactic that the corresponding shipping entry can run for this problem, and 0 otherwise. For dense, a
// null/empty name on canonical K-quant K-pack first selects an exact measured
// `(qtype,m,n,k)` point and otherwise asks about the M<8 decode default or the
// M>=8 legacy default. Xplane and legacy dense APIs retain the old shape
// default; grouped has one compiled default. No CUDA/PPU context is required,
// and the launch entries enforce the same
// exact-type shared-memory/compact-A checks even when a caller neglects to query first.
//
// Both inventories contain tensor-core and CUDA-core families. Call the predicate matching enable_cuda_kernel;
// using a family with the other predicate returns 0. max_rows is the largest expert extent (the only distribution
// property that can affect a compiled kernel); total_rows and experts validate the grouped domain.
//
// "Any shape" for the compiled tensor fallback means every positive M (or every positive total_rows/experts/max_rows
// grouped problem) with N and K positive multiples of 256, the qtype's GGUF group size (16 for Q2/Q3/Q6, 32 for
// Q4/Q5), and any extra format-selected constraint reported by this binary (currently K%512 for paired Q3/Q6 packed
// units). Tile extents do not narrow that domain: M/N tails are predicated. In particular N=32 is outside the current
// resident-artifact ABI, not rejected because a default TileN is wider. Fully-quantized entries are outside the
// default build and return invalid unless PPU_PACKED_SCALE and that qtype's PPU_PACKED_FORMAT were compiled.
int32_t quactlize_ppu_dense_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const* config_name);
int32_t quactlize_ppu_gemv_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_dense_fully_quantized_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v1(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, char const* config_name);
int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const* config_name);
int32_t quactlize_ppu_grouped_lowbit_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name);
int32_t quactlize_ppu_grouped_fully_quantized_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name);
int32_t quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name);
int32_t quactlize_ppu_vecdot_moe_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name);

// Config-selecting host-pointer operator entries. config_name comes from the corresponding dense or grouped
// inventory above. For non-arrangement dense entries, a null/empty name requests the shape-selected default (M<8
// decode, M>=8 legacy); an unknown non-empty name reports the decline and retains the legacy fallback. Arrangement
// entries remain strict: an unknown non-empty config name returns 39. Null/empty K-quant K-pack first selects an
// exact measured point and otherwise uses the shape default; Xplane keeps the shape default directly. Each tensor
// default is instantiated with compile-time assertions that its
// exact shared storage fits ppu001 and that it uses the unrestricted ordinary-A path throughout the admitted domain.
int quactlize_ppu_dense_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int quactlize_ppu_dense_lowbit_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const* config_name);
int quactlize_ppu_gemv_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int quactlize_ppu_dense_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, char const* config_name);
// v1 above is the legacy reader-default ABI: its tactic and resident bytes both keep the historical registry
// fully_quantized_tile_k. An artifact carrying any explicit arrangement uses this successor instead; an unknown
// non-empty config name returns 39 rather than falling back.
int quactlize_ppu_dense_fully_quantized_for_arrangement_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, char const* config_name);
int quactlize_ppu_dense_fully_quantized_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const* config_name);
int quactlize_ppu_grouped_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int qtype, char const* config_name);
int quactlize_ppu_grouped_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int experts, int qtype, char const* config_name);
int quactlize_ppu_grouped_fully_quantized_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name);
int quactlize_ppu_vecdot_moe_config_v1(
    uint8_t const* blocks, int64_t block_bytes, uint16_t const* x,
    int const* offsets, float* out, int n, int blocks_per_row, int experts,
    int total_rows, int max_rows, int qtype, char const* config_name);

#ifdef __cplusplus
}
#endif
