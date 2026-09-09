// Production adapters and graph replay with a device tag oracle. The external
// GEMM/GEMV stubs test pointer, router and lifetime contracts, not arithmetic.
#define ggml_quactlize_execution_library test_execution_library
#define ggml_quactlize_gemv_config test_gemv_config
#define ggml_quactlize_prefill_route test_prefill_route
#define ggml_quactlize_artifact_for test_artifact_for
#define ggml_quactlize_execution_prepare_graph test_prepare_graph
#define ggml_quactlize_execution_run test_execution_run
#include "../ggml/src/ggml-cuda/quactlize-execution.cu"
#undef ggml_quactlize_execution_library
#undef ggml_quactlize_gemv_config
#undef ggml_quactlize_prefill_route
#undef ggml_quactlize_artifact_for
#undef ggml_quactlize_execution_prepare_graph
#undef ggml_quactlize_execution_run

static ggml_tensor * test_weight;
static ggml_quactlize_artifact test_art;
static int prepares, queries, scale_prepares;
static int gemv_lookups, prefill_lookups, expected_route;

static void require(bool value) {
    if (!value) GGML_ABORT("KPACK_ADAPTER_DEVICE invariant failed");
}
bool test_artifact_for(const ggml_tensor * t, ggml_quactlize_artifact * out) {
    if (t != test_weight) return false;
    *out = test_art;
    return true;
}
bool test_gemv_config(const qkg_call_v1 &, qkg_config_v1 * out) {
    ++gemv_lookups;
    *out = {1,sizeof(*out),32,4,1};
    return true;
}
int test_prefill_route(const qks_request_v1 &) {
    ++prefill_lookups;
    return 1;
}
static int fake_open(const char *, void ** p) { *p = (void *) 1; return 0; }
static void fake_close(void *) {}
static const char * fake_error() { return "device tag stub"; }
static int fake_query(void *, const qks_request_v1 * r, qks_choice_v1 * c) {
    ++queries;
    require(r->m > 0 && r->max_rows > 0);
    require(r->route == expected_route);
    *c = {}; c->version=1; c->size=sizeof(*c); c->ticket=1;
    c->compute_units=72; c->split=1;
    snprintf(c->parent,sizeof(c->parent),"device_tag_stub");
    return 0;
}
static int fake_prepare(void *, const qks_choice_v1 *, const qk_call_v1 * c, void ** out) {
    require(!c->rows_host && !c->rows_device && c->a && c->output);
    require(bool(c->zero) == (expected_route == QK_DENSE_SF || expected_route == QK_GROUPED_SF));
    ++prepares;
    *out = new qk_call_v1(*c);
    return 0;
}
static void fake_destroy(void * p) { delete (qk_call_v1 *) p; }

static __global__ void tag_gemm(qk_call_v1 c) {
    int r = blockIdx.x;
    int expert = 0;
    if (c.offsets_device) {
        while (expert+1 < c.experts && r >= c.offsets_device[expert+1]) ++expert;
    }
    for (int n=threadIdx.x; n<c.n; n+=blockDim.x) {
        ((half *) c.output)[int64_t(r)*c.n+n] = __float2half(
            __half2float(((const half *) c.a)[int64_t(r)*c.k]) + 8*(expert+1) + n%8 +
            (c.zero ? __half2float(*(const half *) c.metadata) + __half2float(*(const half *) c.zero) : 0));
    }
}
static int fake_run(void * p, void * stream) {
    auto c = *(qk_call_v1 *) p;
    tag_gemm<<<c.m,128,0,(cudaStream_t) stream>>>(c);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
static int fake_gemv_query(const qkg_call_v1 * c, const qkg_config_v1 *,
                         const quactlize_ppu_placed_arrangement_v2 *, qkg_sizes_v1 * s) {
    *s = {};
    s->sf_plane_bytes = uint64_t(c->experts)*c->n*c->k/32*2;
    s->units_bytes = 16;
    return 0;
}
static __global__ void tag_gemv(qkg_call_v1 c) {
    int r = blockIdx.x, t = r/c.topk, slot = r%c.topk;
    int e = c.mode == QKG_DENSE ? 0 : c.ids[int64_t(t)*c.ids_stride+slot];
    int64_t a = c.mode == QKG_DENSE ? int64_t(r)*c.a_row_stride :
        int64_t(t)*c.a_token_stride + (slot%c.channels)*c.a_row_stride;
    for (int n=threadIdx.x; n<c.n; n+=blockDim.x)
        c.output[int64_t(r)*c.out_row_stride+n] = ((const float *) c.a)[a] + 8*(e+1) + n%8;
}
static int fake_gemv_run(const qkg_call_v1 * c, const qkg_config_v1 *,
                       const quactlize_ppu_placed_arrangement_v2 *) {
    tag_gemv<<<c->rows,128,0,(cudaStream_t) c->stream>>>(*c);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
static __global__ void tag_prepass(const float * units, half * scale, half * zero) {
    *scale = __float2half(*units);
    *zero = __float2half(2 * *units);
}
static int fake_sf_prepare(int, int, int, int, const uint8_t * units, uint64_t,
                          uint16_t * scale, uint16_t * zero, uint64_t,
                          const quactlize_ppu_placed_arrangement_v2 *, void * stream) {
    ++scale_prepares;
    tag_prepass<<<1,1,0,(cudaStream_t) stream>>>((const float *) units, (half *) scale, (half *) zero);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
const ggml_quactlize_execution_api * test_execution_library() {
    static const ggml_quactlize_execution_api api = {"stub",fake_open,fake_close,fake_query,
        fake_prepare,fake_run,fake_destroy,fake_error,fake_gemv_query,fake_gemv_run,fake_sf_prepare,nullptr};
    return &api;
}

static void run_case(bool grouped, int tokens, int channels) {
    constexpr int k=512, n=256, experts=4;
    const int topk=grouped ? 2 : 1, rows=tokens*topk;
    const bool gemv = !strcmp(getenv("QUACTLIZE_KPACK_ROUTE"),"gemv");
    const bool auto_prefill = !strcmp(getenv("QUACTLIZE_KPACK_ROUTE"),"auto") && tokens > 1;
    const bool sf = !strcmp(getenv("QUACTLIZE_KPACK_ROUTE"),"sf") || auto_prefill;
    expected_route = grouped ? (sf ? QK_GROUPED_SF : QK_GROUPED_FQ) : (sf ? QK_DENSE_SF : QK_DENSE_FQ);
    ggml_context * tensors = ggml_init({1<<20,nullptr,true});
    require(tensors);
    test_weight = ggml_new_tensor_3d(tensors,GGML_TYPE_Q4_K,k,n,grouped ? experts : 1);
    ggml_set_name(test_weight,"adapter-weight");
    auto * a=grouped ? ggml_new_tensor_3d(tensors,GGML_TYPE_F32,k,channels,tokens) :
        ggml_new_tensor_2d(tensors,GGML_TYPE_F32,k,tokens);
    auto * ids=grouped ? ggml_new_tensor_2d(tensors,GGML_TYPE_I32,topk,tokens) : nullptr;
    auto * out=grouped ? ggml_new_tensor_3d(tensors,GGML_TYPE_F32,n,topk,tokens) :
        ggml_new_tensor_2d(tensors,GGML_TYPE_F32,n,tokens);
    out->op=grouped ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT;
    out->src[0]=test_weight; out->src[1]=a; out->src[2]=ids;
    CUDA_CHECK(cudaMalloc(&a->data,ggml_nbytes(a)));
    CUDA_CHECK(cudaMalloc(&out->data,ggml_nbytes(out)));
    if (ids) CUDA_CHECK(cudaMalloc(&ids->data,ggml_nbytes(ids)));
    test_art={}; test_art.qtype=12; test_art.n=n; test_art.k=k; test_art.experts=grouped ? experts : 1;
    test_art.low=(const uint8_t *) a->data; test_art.units=test_art.low;
    test_art.arrangement.group_size=32; test_art.arrangement.mapping_id=UINT64_C(0x51344b5034540001);
    CUDA_CHECK(cudaEventCreateWithFlags(&test_art.ready,cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(test_art.ready));
    CUDA_CHECK(cudaEventSynchronize(test_art.ready));
    prepares=queries=scale_prepares=gemv_lookups=prefill_lookups=0;
    {
        ggml_backend_cuda_context ctx(0);
        auto * graph=ggml_new_graph_custom(tensors,16,false);
        graph->n_nodes=1; graph->nodes[0]=out;
        test_prepare_graph(ctx,graph);
        test_prepare_graph(ctx,graph);
        require(prepares==(gemv ? 0 : 1) && queries==prepares && scale_prepares==0);
        require(gemv_lookups==(gemv ? 1 : 0) && prefill_lookups==(auto_prefill ? 1 : 0));
        cudaGraph_t capture;
        cudaGraphExec_t instance;
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(),cudaStreamCaptureModeRelaxed));
        require(test_execution_run(ctx,test_weight,a,ids,out));
        CUDA_CHECK(cudaStreamEndCapture(ctx.stream(),&capture));
        CUDA_CHECK(cudaGraphInstantiate(&instance,capture,nullptr,nullptr,0));
        for (int replay=0; replay<4; ++replay) {
            std::vector<float> input(ggml_nelements(a)), output(size_t(rows)*n);
            std::vector<int32_t> router(rows);
            for (size_t i=0; i<input.size(); ++i) input[i]=float(i/k+replay);
            for (int r=0; r<rows; ++r) router[r]=(r%topk+replay)%experts;
            CUDA_CHECK(cudaMemcpyAsync(a->data,input.data(),ggml_nbytes(a),cudaMemcpyHostToDevice,ctx.stream()));
            if (ids) CUDA_CHECK(cudaMemcpyAsync(ids->data,router.data(),ggml_nbytes(ids),cudaMemcpyHostToDevice,ctx.stream()));
            CUDA_CHECK(cudaMemsetAsync(out->data,0xa5,ggml_nbytes(out),ctx.stream()));
            if (sf) {
                auto & plan = prepare(ctx,test_weight,a,ids,out);
                CUDA_CHECK(cudaMemsetAsync(plan.scale,0x7e,plan.sf_plane_bytes,ctx.stream()));
                CUDA_CHECK(cudaMemsetAsync(plan.zero,0x7e,plan.sf_plane_bytes,ctx.stream()));
            }
            if (replay%2) require(test_execution_run(ctx,test_weight,a,ids,out));
            else CUDA_CHECK(cudaGraphLaunch(instance,ctx.stream()));
            CUDA_CHECK(cudaMemcpyAsync(output.data(),out->data,ggml_nbytes(out),cudaMemcpyDeviceToHost,ctx.stream()));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            size_t bad=0;
            for (int r=0; r<rows; ++r) for (int col=0; col<n; ++col) {
                int ar=grouped ? r/topk*channels+r%topk%channels : r;
                float want=input[size_t(ar)*k]+8*(grouped ? router[r]+1 : 1)+col%8+(sf ? 3*replay : 0);
                bad+=output[size_t(r)*n+col]!=want;
            }
            require(bad==0);
        }
        require(prepares==(gemv ? 0 : 1) && queries==prepares && scale_prepares==(sf ? 3 : 0));
        require(gemv_lookups==(gemv ? 1 : 0) && prefill_lookups==(auto_prefill ? 1 : 0));
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        CUDA_CHECK(cudaGraphDestroy(capture));
    }
    CUDA_CHECK(cudaEventDestroy(test_art.ready));
    CUDA_CHECK(cudaFree(a->data)); CUDA_CHECK(cudaFree(out->data));
    if (ids) CUDA_CHECK(cudaFree(ids->data));
    ggml_free(tensors);
    printf("KPACK_ADAPTER_DEVICE PASS op=%s route=%s tokens=%d channels=%d eager=2 replay=2 changing_ids=1 host_rows=0\n",
           grouped ? "grouped" : "dense",gemv ? "gemv" : sf ? "sf" : "fq",tokens,channels);
}
int main() {
    int devices=0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) return 77;
    require(getenv("QUACTLIZE_KPACK_ROUTE"));
    for (int tokens : {1,4,16}) {
        run_case(false,tokens,1);
        run_case(true,tokens,1);
        run_case(true,tokens,2);
    }
    printf("KPACK_ADAPTER_DEVICE_ALL PASS external_arithmetic=STUB routing_and_capture=PRODUCTION\n");
}
