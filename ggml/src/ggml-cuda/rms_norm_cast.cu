#include "rms_norm_cast.cuh"

#include <cstring>

// Translation unit for the rms_norm_cast kernels. The template lives in the header so that tests can
// instantiate it directly; this file exists so that the instantiations are part of the normal
// ggml-cuda build and compile errors are caught even while nothing calls the op yet.

static rms_norm_cast_args rms_norm_cast_args_from_src(const ggml_tensor * src0, ggml_tensor * out) {
    rms_norm_cast_args a = {};

    a.src      = src0->data;
    a.src_type = src0->type;
    a.dst      = out->data;
    a.dst_type = out->type;

    a.ncols     = src0->ne[0];
    a.nrows     = src0->ne[1];
    a.nchannels = src0->ne[2];
    a.nsamples  = src0->ne[3];

    const size_t ts = ggml_type_size(src0->type);
    GGML_ASSERT(src0->nb[0] == ts);
    a.stride_row     = src0->nb[1] / ts;
    a.stride_channel = src0->nb[2] / ts;
    a.stride_sample  = src0->nb[3] / ts;

    return a;
}

// The fused nodes may hold the rms_norm result in either src slot.
static const ggml_tensor * rms_norm_cast_other_src(const ggml_tensor * node, const ggml_tensor * self) {
    if (node->src[0] == self) {
        return node->src[1];
    }
    if (node->src[1] == self) {
        return node->src[0];
    }
    GGML_ABORT("rms_norm_cast: %s is not a source of %s", self->name, node->name);
}

void ggml_cuda_op_rms_norm_cast(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_cast_args a = rms_norm_cast_args_from_src(src0, dst);
    a.eps = eps;

    rms_norm_cast_launch(a, ctx.stream());
}

void ggml_cuda_op_rms_norm_cast_fused(ggml_backend_cuda_context & ctx,
                                      ggml_tensor *               dst,
                                      ggml_tensor *               mul_tensor) {
    const ggml_tensor * src0    = dst->src[0];
    const ggml_tensor * mul_src = rms_norm_cast_other_src(mul_tensor, dst);

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_cast_args a = rms_norm_cast_args_from_src(src0, mul_tensor);
    a.eps = eps;

    a.mul      = mul_src->data;
    a.mul_type = mul_src->type;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    a.mul_stride_row     = mul_src->nb[1] / ts_mul;
    a.mul_stride_channel = mul_src->nb[2] / ts_mul;
    a.mul_stride_sample  = mul_src->nb[3] / ts_mul;

    a.mul_ncols     = mul_src->ne[0];
    a.mul_nrows     = mul_src->ne[1];
    a.mul_nchannels = mul_src->ne[2];
    a.mul_nsamples  = mul_src->ne[3];

    rms_norm_cast_launch(a, ctx.stream());
}

void ggml_cuda_op_rms_norm_cast_fused_add(ggml_backend_cuda_context & ctx,
                                          ggml_tensor *               dst,
                                          ggml_tensor *               mul_tensor,
                                          ggml_tensor *               add_tensor) {
    const ggml_tensor * src0    = dst->src[0];
    const ggml_tensor * mul_src = rms_norm_cast_other_src(mul_tensor, dst);
    const ggml_tensor * add_src = rms_norm_cast_other_src(add_tensor, mul_tensor);

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_cast_args a = rms_norm_cast_args_from_src(src0, add_tensor);
    a.eps = eps;

    a.mul      = mul_src->data;
    a.mul_type = mul_src->type;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    a.mul_stride_row     = mul_src->nb[1] / ts_mul;
    a.mul_stride_channel = mul_src->nb[2] / ts_mul;
    a.mul_stride_sample  = mul_src->nb[3] / ts_mul;

    a.mul_ncols     = mul_src->ne[0];
    a.mul_nrows     = mul_src->ne[1];
    a.mul_nchannels = mul_src->ne[2];
    a.mul_nsamples  = mul_src->ne[3];

    a.add      = add_src->data;
    a.add_type = add_src->type;

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    a.add_stride_row     = add_src->nb[1] / ts_add;
    a.add_stride_channel = add_src->nb[2] / ts_add;
    a.add_stride_sample  = add_src->nb[3] / ts_add;

    a.add_ncols     = add_src->ne[0];
    a.add_nrows     = add_src->ne[1];
    a.add_nchannels = add_src->ne[2];
    a.add_nsamples  = add_src->ne[3];

    rms_norm_cast_launch(a, ctx.stream());
}
