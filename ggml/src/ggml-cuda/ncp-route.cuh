#pragma once

// Trace which kernel each MoE/MatMul node routed to (.so vs inline). A declined route looks identical to one never
// compiled in -- same output, ggml's kernel takes over -- so this line is the only cheap way to tell them apart.
// Emitted at DEBUG level through ggml's log callback; enable ggml debug logging to see it.

#include "common.cuh"
#include "ncp-lib.h"
#include <cstdio>
#include <cstring>

// `declined` is the REASON the .so path was not taken, or nullptr if it was.
static void ggml_ncp_route_log(const ggml_tensor * dst, const char * kernel, const char * declined) {
    const ggml_tensor * src0 = dst->src[0];
    const bool          mmid = dst->op == GGML_OP_MUL_MAT_ID;
    const ggml_tensor * ids  = mmid ? dst->src[2] : nullptr;
    const char *        name = src0->name[0] ? src0->name : "(unnamed)";
    const int           T    = (int) (mmid ? dst->ne[2] : dst->ne[1]);

    char line[320];
    char shape[64];
    if (mmid) {
        snprintf(shape, sizeof(shape), "T=%-5d E=%-4d used=%d", T, (int) src0->ne[2], (int) ids->ne[0]);
    } else {
        snprintf(shape, sizeof(shape), "T=%-5d K=%-5d N=%d", T, (int) src0->ne[0], (int) src0->ne[1]);
    }
    if (declined) {
        snprintf(line, sizeof(line), "[ncp-route] %-11s %-30s %-24s -> GENERIC (%s)\n",
                 mmid ? "MUL_MAT_ID" : "MUL_MAT", name, shape, declined);
    } else {
        snprintf(line, sizeof(line), "[ncp-route] %-11s %-30s %-24s -> %s\n",
                 mmid ? "MUL_MAT_ID" : "MUL_MAT", name, shape, kernel);
    }

    // One line per distinct (tensor, shape, decision); a long generation would otherwise repeat it thousands of times.
    static char seen[64][320];
    static int  n_seen = 0;
    for (int i = 0; i < n_seen; i++) {
        if (strcmp(seen[i], line) == 0) {
            return;
        }
    }
    if (n_seen < 64) {
        snprintf(seen[n_seen++], sizeof(seen[0]), "%s", line);
    }
    GGML_LOG_DEBUG("%s", line);
}
