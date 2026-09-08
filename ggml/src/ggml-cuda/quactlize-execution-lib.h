#pragma once
#include "quactlize/kpack_dispatch.h"
#include "quactlize/kpack_execution.h"

struct ggml_quactlize_execution_api {
    const char * root;
    decltype(&quactlize_kpack_dispatch_open_v1) open;
    decltype(&quactlize_kpack_dispatch_close_v1) close;
    decltype(&quactlize_kpack_dispatch_query_v1) query;
    decltype(&quactlize_kpack_dispatch_prepare_v1) prepare;
    decltype(&quactlize_kpack_dispatch_run_v1) run;
    decltype(&quactlize_kpack_dispatch_destroy_v1) destroy;
    decltype(&quactlize_kpack_dispatch_error_v1) error;
    decltype(&quactlize_kpack_gemv_query_v1) gemv_query;
    decltype(&quactlize_kpack_gemv_run_v1) gemv_run;
    decltype(&quactlize_kpack_sf_prepare_v1) sf_prepare;
};

// QUACTLIZE_KPACK_EXECUTION names the complete native package. An explicit
// but broken package is an error, not an unnoticed return to the old route.
const ggml_quactlize_execution_api * ggml_quactlize_execution_library();
// Exact measured GEMV-pool choice, loaded once from QUACTLIZE_KPACK_GEMV_POLICY.
// Missing shapes decline; they do not inherit an arbitrary launch recipe.
bool ggml_quactlize_gemv_config(const qkg_call_v1 & call, qkg_config_v1 * config);
// Exact FQ/SF comparison: -1 missing, 0 FQ, 1 resident SF. Unknown contexts
// retain selected FQ; they do not infer an SF win from token count alone.
int ggml_quactlize_prefill_route(const qks_request_v1 & request);
