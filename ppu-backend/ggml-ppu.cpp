#include "ggml-backend-impl.h"
#include "runtime.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
struct device_context {
    int index;
    std::string name;
    ggml_backend_device device = {};
};
device_context * device_ctx(ggml_backend_dev_t dev) {
    return static_cast<device_context *>(dev->context);
}
const char * device_name(ggml_backend_dev_t dev) { return device_ctx(dev)->name.c_str(); }
const char * device_description(ggml_backend_dev_t) {
    return ppu::host_test() ? "Host validation of the PPU plugin (no PPU hardware)" : "PPU RMS_NORM accelerator (lazy device initialization)";
}
enum ggml_backend_dev_type device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
void device_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 0; }
void device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name = device_name(dev);
    props->description = device_description(dev);
    props->type = device_type(dev);
    device_memory(dev, &props->memory_free, &props->memory_total);
}
bool noop(const ggml_tensor * op) {
    return op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
           op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE;
}
bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (noop(op)) return true;
    if (op->op != GGML_OP_RMS_NORM || !op->src[0] || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || op->ne[0] <= 0 ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_are_same_shape(op, op->src[0])) {
        return false;
    }
    float eps;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    return std::isfinite(eps) && eps >= 0.0f;
}
bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
}
bool offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return op->op == GGML_OP_RMS_NORM && supports_op(dev, op);
}

ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }

const char * backend_name(ggml_backend_t backend) { return device_name(backend->device); }
void backend_free(ggml_backend_t backend) { delete backend; }
ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) && !supports_op(backend->device, node)) return GGML_STATUS_FAILED;
    }
    auto * ctx = device_ctx(backend->device);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (noop(node) || ggml_is_empty(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        float eps;
        std::memcpy(&eps, node->op_params, sizeof(eps));
        const int64_t columns = node->ne[0], rows = ggml_nelements(node) / columns;
        std::fprintf(stderr, "[ggml-ppu] RMS_NORM node=%s device=%s rows=%" PRId64 " columns=%" PRId64 " eps=%g\n",
                     node->name, ctx->name.c_str(), rows, columns, eps);
        if (!ppu::rms_norm(ctx->index, static_cast<const float *>(node->src[0]->data),
                           static_cast<float *>(node->data), columns, rows, eps)) {
            std::fprintf(stderr, "[ggml-ppu] RMS_NORM failed: %s\n", ppu::error());
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}
ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    static ggml_guid guid = {0x47, 0x20, 0x8c, 0x51, 0x17, 0x39, 0x4d, 0x7e, 0x9b, 0xc1, 0x51, 0x88, 0xfe, 0x24, 0x31, 0xd0};
    auto * backend = new ggml_backend{};
    backend->guid = &guid;
    backend->device = dev;
    backend->iface.get_name = backend_name;
    backend->iface.free = backend_free;
    backend->iface.graph_compute = graph_compute;
    return backend;
}

const char * reg_name(ggml_backend_reg_t) { return ppu::name(); }
size_t reg_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t reg_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    return &static_cast<device_context *>(reg->context)->device;
}
ggml_backend_reg_t ppu_reg() {
    static device_context ctx;
    static ggml_backend_reg reg = {};
    static bool initialized = false;
    if (initialized) return &reg;
    std::fprintf(stderr, "[ggml-ppu] ggml_backend_init reached; registering native GGML_OP_RMS_NORM override\n");
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface = {reg_name, reg_count, reg_device, nullptr};
    reg.context = &ctx;
    ctx.index = 0;
    ctx.name = std::string(ppu::name()) + "0";
    ctx.device.reg = &reg;
    ctx.device.context = &ctx;
    auto & iface = ctx.device.iface;
    iface.get_name = device_name;
    iface.get_description = device_description;
    iface.get_memory = device_memory;
    iface.get_type = device_type;
    iface.get_props = device_props;
    iface.init_backend = init_backend;
    iface.get_buffer_type = device_buft;
    iface.supports_op = supports_op;
    iface.supports_buft = supports_buft;
    iface.offload_op = offload_op;
    initialized = true;
    std::fprintf(stderr, "[ggml-ppu] loaded backend=%s override=GGML_OP_RMS_NORM%s\n",
                 ppu::name(), ppu::host_test() ? " (HOST TEST, no PPU execution)" : "");
    return &reg;
}
}

GGML_BACKEND_DL_IMPL(ppu_reg)
