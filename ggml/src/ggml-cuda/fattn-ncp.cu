#include "fattn-ncp.cuh"

#ifdef GGML_NCP_FA

// Named explicitly rather than leaned on transitively: fattn.cu reached ggml_get_to_fp32_cuda through
// fattn-common.cuh's own includes, which would have made this file's correctness depend on an unrelated header.
#include "convert.cuh"
#include "ncp-lib.h"

// Try the external PPU FlashAttention .so (libncp_fa.so). Returns true if it handled dst; false -> inline fallback.
// ggml gives F32 Q + F32 O and (usually) F16 K/V; the .so is half in/out and consumes ggml's native layout via
// strides (ncp-fa-lib.h). This hook: converts Q F32->F16 (pool scratch), passes F16 K/V directly with their strides,
// runs the .so into an F16 O scratch, then converts O F16->F32 into dst. It engages only for patterns the sm80 FA
// forward can represent exactly (no additive mask -> full or pure-causal); everything else falls through to inline.
bool ggml_cuda_flash_attn_ext_ncp_lib(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_ncp_lib_fa_available()) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // --- coverage gates: only what the sm80 FA forward can represent exactly ---
    if (sinks) {
        return false;                                                   // attention sinks not supported here
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;                                                   // require f16 K/V (no dequant on this path)
    }
    const int head_dim = (int) Q->ne[0];
    // FA3 takes V's head dim separately (params.dv) -- FA2 could not express dv != d at all, so this used to be
    // an unstated assumption. ggml already carries it: V->ne[0] (== dst->ne[0]) is V's head dim, not Q's.
    const int head_dim_v = (int) V->ne[0];
    if (Q->ne[0] != K->ne[0]) {
        return false;                                                   // Q*K^T needs a shared d_qk
    }
    if (!(head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 192 || head_dim == 256)) {
        return false;                                                   // instantiated head dims only
    }
    // dv != d exists only for the (d, dv) pairs FA3 instantiates -- mha_fwd's own check: "Q/K headdim in
    // (128, 192] and V headdim in (96, 128], or (Q/K <= 64 and V <= 512)". The .so re-checks this and returns
    // non-zero, but rejecting up front keeps the pool allocations below out of shapes that can never run.
    if (head_dim_v != head_dim &&
        !((head_dim > 128 && head_dim <= 192 && head_dim_v > 96 && head_dim_v <= 128) ||
          (head_dim <= 64 && head_dim_v <= 512))) {
        return false;
    }
    const int n_head_q  = (int) Q->ne[2];
    const int n_head_kv = (int) K->ne[2];
    if (n_head_kv == 0 || n_head_q % n_head_kv != 0) {
        return false;
    }
    if (!ggml_is_contiguously_allocated(Q) || !ggml_is_contiguous(dst)) {
        return false;                                                   // we convert Q / write O contiguously
    }
    if (K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) {
        return false;                                                   // FA needs a contiguous head_dim
    }

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) {
        return false;                                                   // ALiBi not represented here
    }

    // Mask -> causal expressibility. The sm80 forward has no additive-mask input, so a non-null mask is only safe
    // when it is exactly a pure bottom-right causal mask. The host sets that hint via ggml_flash_attn_ext_set_causal
    // (causal_attn && no ALiBi && !SWA && !kv_unified). The ne[3]==1 guard rejects multi-stream masks that pack
    // several sequences (kv_unified=false), which the positional causal bound cannot represent.
    int is_causal = 0;
    if (mask) {
        if (!(ggml_flash_attn_ext_get_causal(dst) && mask->ne[3] == 1)) {
            return false;
        }
        is_causal = 1;
    }

    const int batch    = (int) Q->ne[3];
    const int seqlen_q = (int) Q->ne[1];
    const int seqlen_k = (int) K->ne[1];
    const int dv       = (int) dst->ne[0];

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream   = ctx.stream();

    // Q: F32 -> F16, contiguous in ggml ne order {d, seqlen_q, n_head_q, batch}.
    ggml_cuda_pool_alloc<half> Qh(pool, ggml_nelements(Q));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(Q->data, Qh.ptr, ggml_nelements(Q), stream);

    // O: F16 scratch, contiguous in dst ne order {dv, seqlen_q, n_head_q, batch}.
    ggml_cuda_pool_alloc<half> Oh(pool, ggml_nelements(dst));

    const long long es  = (long long) sizeof(half);
    const long long esq = (long long) sizeof(float);
    // Q is only contiguously *allocated*, not contiguous: build_attn_mha hands us a ggml_permute view, so its ne
    // order {d,seqlen_q,n_head_q,batch} does not match its memory order. ggml_get_to_fp16_cuda converts the flat
    // element range, i.e. Qh preserves Q's physical layout -- so Qh's element strides are exactly Q->nb/sizeof(f32),
    // NOT the strides a contiguous {d,sq,h,b} tensor would have. (Deriving them from the ne order silently reads Q
    // transposed whenever seqlen_q > 1 and n_head_q > 1.) K/V strides likewise come from their byte nb.
    const long long qrs = Q->nb[1] / esq, qhs = Q->nb[2] / esq, qbs = Q->nb[3] / esq;
    const long long krs = K->nb[1] / es,  khs = K->nb[2] / es,  kbs = K->nb[3] / es;
    const long long vrs = V->nb[1] / es,  vhs = V->nb[2] / es,  vbs = V->nb[3] / es;
    // O: dst's ne order is {dv, n_head_q, seqlen_q, batch} -- ggml_flash_attn_ext builds it as
    // {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}, i.e. head and seqlen are SWAPPED relative to Q. So dst is physically
    // [batch][seqlen][head][dv] (which is exactly FA's own packed layout), and the O scratch we hand the .so must
    // use the SAME strides: head step = dv, seqlen("row") step = dv*n_head_q. Deriving them from Q's order instead
    // writes O transposed -- invisible when seqlen_q == 1, wrong for every longer prompt.
    const long long ohs = dst->nb[1] / esq, ors = dst->nb[2] / esq, obs = dst->nb[3] / esq;

    // Window: (-1,-1) is full attention, (-1,0) is causal. FA3 also does LOCAL attention, so a real window here would
    // let SWA models use the .so instead of falling back to the inline path -- the hook does not derive one yet,
    // because ggml hands us an additive mask rather than a window. FOLLOW-UP: read hparams.n_swa in the graph.
    const int rc = ggml_ncp_lib_flash_attn_fwd(
        Qh.ptr, K->data, V->data, Oh.ptr,
        batch, seqlen_q, seqlen_k, n_head_q, n_head_kv, head_dim, head_dim_v,
        qbs, qhs, qrs, kbs, khs, krs, vbs, vhs, vrs, obs, ohs, ors,
        scale, logit_softcap, is_causal, /*window_left=*/-1, /*window_right=*/is_causal ? 0 : -1,
        /*dtype=*/0, stream);
    if (rc != 0) {
        return false;
    }

    // O: F16 -> F32 into dst.
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(Oh.ptr, (float *) dst->data, ggml_nelements(dst), stream);
    return true;
}

#endif // GGML_NCP_FA
