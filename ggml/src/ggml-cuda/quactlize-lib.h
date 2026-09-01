#pragma once
// dlopen wrapper around the quactlize PPU K-pack libraries.
//
// Same shape and the same reasons as ncp-lib.h: llama.cpp builds with zero cutlass and zero quactlize sources, the
// kernels ship as prebuilt .so, and a library that is absent or has no tactic for a shape reads as "unavailable" so
// the caller falls through to the existing path with correct (just slower) output.
//
// ONE LIBRARY PER QUANTISATION FORMAT. Unlike ncp, quactlize compiles a separate .so per GGUF k-quant, each built
// with QUACTLIZE_DENSE_ONLY=<qtype> and PPU_PACKED_FORMAT=<fmt>, and all five export the SAME symbol names:
//
//     Q2_K (10) -> libquactlize_ppu_fmt2.so      Q5_K (13) -> libquactlize_ppu_fmt1.so
//     Q3_K (11) -> libquactlize_ppu_fmt3.so      Q6_K (14) -> libquactlize_ppu_fmt4.so
//     Q4_K (12) -> libquactlize_ppu_fmt0.so
//
// The format index is NOT the qtype and the table is not derivable -- it is the bundle manifest's mapping, checked
// after load against each library's own quactlize_ppu_build_packed_format_v1(). Because the five export identical
// names they are opened RTLD_LOCAL and every entry is called through that library's own handle: with RTLD_GLOBAL the
// first one opened would answer for all five and a Q5_K tensor would be decoded by the Q2_K reader.
//
// libquactlize_ppu.so, the sixth file in the bundle, is the default Q4 ScaleFirst build (build_packed_format_v1()
// returns -1). It is deliberately NOT part of this table: it is a different resident format, not a fallback for it.
//
// qtype IS ggml_type. quactlize numbers its formats 10..14 and so does ggml (GGML_TYPE_Q2_K..GGML_TYPE_Q6_K). That
// is an identity two projects have to keep, not one either derives, so quactlize-lib.cu static_asserts it rather
// than converting -- a silent renumbering on either side would otherwise decode Q4_K weights as Q5_K.
//
// Inert unless built with -DGGML_NCP_QUACTLIZE=ON: every entry below then reports unavailable and returns a decline.

#include "quactlize/quactlize_ppu_device.h"   // brings quactlize_ppu_config.h: the arrangement and config structs

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Is a K-pack library loaded for this format, and did it prove its identity? False for every qtype outside 10..14,
// for a missing .so, for a .so missing any entry this header wraps, and for one whose build_packed_format_v1 does
// not match the format the table asked for.
bool ggml_quactlize_available(int qtype);

// What the loaded library says it is: 0..4 for a packed build, -1 for the default (non-K-pack) build, and -2 when
// nothing is loaded for this qtype. Reported so a log line can name the actual artifact rather than the intent.
int32_t ggml_quactlize_build_packed_format(int qtype);

// ---- grouped (MoE): mirrors quactlize_ppu_*_grouped_fully_quantized_*_for_arrangement_v2 ----

// Valid-tactic inventory. Pass (NULL, 0) to learn the count, then query again with room for that many. Returns a
// negative count if no library is loaded. THIS is the capability oracle: a supports_op answering from a hand-written
// table beside it would be a second source that can disagree with the kernels.
int32_t ggml_quactlize_list_grouped_configs(
    int qtype, quactlize_ppu_config_v3 * configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);

// Device scratch the chosen tactic needs; < 0 if unavailable.
int64_t ggml_quactlize_grouped_workspace_bytes(
    int qtype, int total_rows, int max_rows, int n, int k, int experts,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);

// act/out are device FP16, low/high/units are the device-resident artifact planes, offsets is the per-expert row
// partition, workspace is a device allocation of at least the size above, stream is the current CUDA/PPU stream.
// 0 on success; any non-zero is a decline and the caller must fall back rather than trust out.
int ggml_quactlize_grouped_dev(
    int qtype,
    const uint16_t * act, const uint8_t * low, const uint8_t * high, const uint8_t * units,
    const int * offsets, uint16_t * out,
    int total_rows, int n, int k, int experts, int max_rows,
    void * workspace, int64_t workspace_bytes, void * stream,
    const char * config_name,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);

// ---- dense: mirrors quactlize_ppu_*_dense_fully_quantized_*_for_arrangement_v2 ----

int32_t ggml_quactlize_list_dense_configs(
    int qtype, quactlize_ppu_config_v3 * configs, int32_t capacity,
    int m, int n, int k, int group_size,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);

int64_t ggml_quactlize_dense_workspace_bytes(
    int qtype, int m, int n, int k,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);

int ggml_quactlize_dense_dev(
    int qtype,
    const uint16_t * act, const uint8_t * low, const uint8_t * high, const uint8_t * units,
    uint16_t * out, int m, int n, int k,
    void * workspace, int64_t workspace_bytes, void * stream,
    const char * config_name,
    const quactlize_ppu_placed_arrangement_v2 * arrangement);


// ---- host-side artifact production: descriptor + GGUF blocks -> resident planes ----
//
// NEITHER OF THESE IS IN THE CURRENTLY SHIPPING BUNDLE. Both are dlsym'd optionally and every caller treats their
// absence as "this format cannot be served", so a build against today's .so is completely inert rather than wrong.
//
// (1) THE DESCRIPTOR. quactlize defines the canonical arrangement per format as constexpr (ppu_placed_arrangement
//     .hpp: q4_kpack4_transpose_v1() and kquant_kpack_transpose_v1(qtype)), but exports no C entry returning it, so
//     a loader can only get it by rebuilding the policy from ppu_format_config.inc -- a second source of the same
//     decision, which is exactly what that registry's own header forbids. Wrapped here as one proposed entry:
//
//         int quactlize_ppu_canonical_arrangement_v2(int qtype, quactlize_ppu_placed_arrangement_v2 * out);
//
//     returning 0 on success, non-zero for a qtype this library was not built for. What it returns is checked
//     against the registry copied beside these headers: not to derive the descriptor, but so that a library and a
//     registry that disagree stop the load instead of decoding with one of the two.
//
// (2) THE CONVERSION. quactlize_ppu_prepare/recover_fully_quantized_for_arrangement_v2, whose signatures are in
//     quactlize/quactlize_ppu_packed.h. The legacy _v1 producer is NOT a fallback for these -- it emits Xplane
//     bytes, a different resident format -- so it is deliberately not wrapped.

// The arrangement this library's kernels are compiled for. false when the entry is absent, when the library was not
// built for this qtype, or when what it returned contradicts the format registry.
bool ggml_quactlize_arrangement_for(int qtype, quactlize_ppu_placed_arrangement_v2 * out);

// Can this build turn GGUF blocks into resident planes in process? False today for every format.
bool ggml_quactlize_conversion_available(int qtype);

// blocks is the raw GGUF byte image of the whole tensor (experts * n * k/256 records); low/high/units are host
// buffers of the sizes documented in quactlize_ppu_packed.h. 0 on success.
int ggml_quactlize_prepare(
    int qtype, const uint8_t * blocks, uint8_t * low, uint8_t * high, uint8_t * units,
    int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arrangement);

// The inverse, for the byte-exact round-trip self-check: recover must reproduce blocks exactly. Having it separate
// from dequantisation is the point -- it tells a packing bug from an arithmetic one.
int ggml_quactlize_recover(
    int qtype, const uint8_t * low, const uint8_t * high, const uint8_t * units, uint8_t * recovered,
    int n, int k, int experts, const quactlize_ppu_placed_arrangement_v2 * arrangement);

// Bytes of the metadata (units) plane for one expert; < 0 when unavailable.
int64_t ggml_quactlize_units_bytes(int qtype, int n, int k);

// The three resident planes, laid out [low][high][units] inside the tensor's own allocation.
//
// K-pack is BYTE-NEUTRAL, and that is the whole reason this path can own the weight buffer without growing the
// resident footprint -- so the three sizes must add up to exactly what ggml already allocates for the tensor. The
// caller checks that; this only computes. Returns false for a shape or descriptor it cannot size at all.
//
// The identity is a cross-check between three independent sources: quactlize's per-format registry (bits,
// high_bits), the library's own units_bytes, and ggml's block size for the type. Two of them agreeing means
// nothing; all three is the claim.
bool ggml_quactlize_plane_sizes(
    int qtype, int64_t n, int64_t k, int64_t experts,
    const quactlize_ppu_placed_arrangement_v2 * arrangement,
    int64_t * low_bytes, int64_t * high_bytes, int64_t * units_bytes);

// How many threads the host-side conversion may use for a tensor with this many experts.
//
// The conversion is the ONLY new cost the K-pack path adds to a model load -- the GGUF bytes are already on the
// host and the upload was going to happen anyway -- and it is a scatter, so the difference between one thread and
// all of them is the difference between a load nobody notices and one everybody does. GGML_QUACTLIZE_CONVERT_
// THREADS=1 forces the serial path, which is what makes that delta measurable rather than asserted.
int ggml_quactlize_convert_threads(int64_t experts);

// Convert blocks into the three resident planes and PROVE the result reproduces its input, byte for byte.
//
// The proof is not belt-and-braces, it is what makes threading safe. Splitting the work by expert assumes each
// expert's share of every plane is a contiguous slice at the same index -- which the ABI's sizing implies but does
// not state -- and a wrong decomposition cannot survive the round trip. So the split is attempted, and a failure
// is retried in one thread: if the serial run then succeeds, the split was the problem and this says so and stays
// serial; if it fails too, the library is the problem. Collapsing the two into one abort would have thrown away
// the distinction that tells you which side to go fix.
//
// recovered must hold nbytes. Returns 0 on success; *threads_used reports what produced the accepted result.
int ggml_quactlize_convert_verified(
    int qtype, const unsigned char * blocks,
    unsigned char * low, unsigned char * high, unsigned char * units, unsigned char * recovered,
    int64_t nbytes, int64_t n, int64_t k, int64_t experts,
    const quactlize_ppu_placed_arrangement_v2 * arrangement,
    int64_t low_bytes, int64_t high_bytes, int64_t units_bytes,
    int * threads_used);

#ifdef __cplusplus
}
#endif
