#pragma once
#include <stdint.h>

// Versioned, device-pointer-only interface for one compiled K-pack parent.
// No GGUF repacking, allocation, profiling or implicit configuration choice is
// performed by prepare/run. Callers own all buffers and the execution stream.
#ifdef __cplusplus
extern "C" {
#endif

enum { QK_OK = 0, QK_UNSUPPORTED = 1, QK_INVALID = 2,
       QK_RUNTIME_ERROR = 3, QK_INITIALIZE_ERROR = 4 };
enum { QK_DENSE_FQ = 0, QK_DENSE_SF = 1, QK_GROUPED_FQ = 2, QK_GROUPED_SF = 3 };
enum { QK_ORDINARY = 0, QK_PERSISTENT = 1 };

typedef struct {
  uint32_t version, size;
  int32_t qtype, route, tm, tn, tk, wm, wn, stages, ap, delivery_n;
  uint64_t mapping_id;
  char const* parent;
  char const* build_key;
} qk_identity_v1;

typedef struct {
  uint32_t version, size;
  int32_t m, n, k, experts, group_size, device, compute_units;
  uint64_t mapping_id;
  void const* a;
  void const* low;
  void const* high;
  // FQ: packed units, zero=NULL. SF: resident FP16 scale/zero planes.
  void const* metadata;
  void const* zero;
  void* output;
  // Grouped rows_host[experts] and matching device rows/offsets. m is the
  // concatenated row count. Empty experts are legal. No hidden D2H occurs.
  int32_t const* rows_host;
  int32_t const* rows_device;
  int32_t const* offsets_device;
  void* workspace;
  uint64_t workspace_bytes;
  void* stream;
} qk_call_v1;

// Additive device-only grouped successor. The embedded call keeps its v1
// layout/version; rows_host and rows_device must be NULL. offsets_device is
// a nondecreasing device array [experts+1], from 0 to call.m. Every expert's
// row count must be <= max_rows. Host query sizes from that bound only.
// prepare owns no asynchronous host copies. run rebuilds device metadata
// from the current offsets, so graph replay can change expert routing.
typedef struct {
  uint32_t version, size;
  qk_call_v1 call;
  int32_t max_rows, reserved;
} qk_device_call_v2;

typedef struct {
  uint32_t version, size;
  int32_t algorithm, split, grid;
} qk_recipe_v1;

typedef struct {
  uint32_t version, size;
  uint64_t workspace_bytes, shared_bytes;
  int32_t occupancy, runtime_status, cutlass_status;
} qk_resources_v1;

qk_identity_v1 const* quactlize_kpack_identity_v1(void);
int quactlize_kpack_device_v1(char* name, int capacity, int32_t* ordinal, int32_t* compute_units);
// Query never dereferences device data. Exact occupancy can use the runtime.
int quactlize_kpack_query_v1(qk_call_v1 const*, qk_recipe_v1 const*, qk_resources_v1*);
// Prepare may enqueue small grouped metadata copies. The handle owns their
// host backing until destroy. All workspace/device data remain caller-owned.
int quactlize_kpack_prepare_v1(qk_call_v1 const*, qk_recipe_v1 const*, void**);
// Includes the grouped directory build or dense Split-K reducer when needed.
int quactlize_kpack_run_v1(void*, void* stream);
// Caller must complete work using the handle before destroy/reusing workspace.
void quactlize_kpack_destroy_v1(void*);
int quactlize_kpack_measure_v1(void*, void* stream, int repeats, double* microseconds);

int quactlize_kpack_grouped_query_v2(qk_device_call_v2 const*, qk_recipe_v1 const*, qk_resources_v1*);
int quactlize_kpack_grouped_prepare_v2(qk_device_call_v2 const*, qk_recipe_v1 const*, void**);
// v2 handles use the existing run_v1/destroy_v1 lifecycle. This does not
// grant unmeasured any-M or performance admission to a particular parent.

#ifdef __cplusplus
}
#endif
