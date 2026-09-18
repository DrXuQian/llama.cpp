// Exercise the real backend and scheduler. Stub libraries answer inventory queries;
// no pack or GEMM entry is called, and weight contents are never read.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <cmath>

#define CHECK(expr) do { \
    if (!(expr)) { fprintf(stderr, "scheduler check failed: %s\n", #expr); exit(1); } \
} while (0)

static ggml_backend_meta_split_state tp_weight_split;

static ggml_backend_meta_split_state tp_split(const ggml_tensor * t, void *) {
    if (!strcmp(t->name, "tp-down")) return {GGML_BACKEND_SPLIT_AXIS_0, {512,512}, {1}, 1};
    if (!strcmp(t->name, "tp-weight") ||
            (!strcmp(t->name, "tp-input") && tp_weight_split.axis == GGML_BACKEND_SPLIT_AXIS_0)) {
        return tp_weight_split;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void tp_chain(ggml_backend_t backend, ggml_backend_buffer_type_t packed, int q, int tokens) {
    constexpr int k=1024, hidden=1024, n=512, experts=4, topk=2;
    tp_weight_split={GGML_BACKEND_SPLIT_AXIS_1,{hidden/2,hidden/2},{2},1};
    auto * wc=ggml_init({2<<20,nullptr,true});
    auto * ctx=ggml_init({2<<20,nullptr,true});
    auto * gate=ggml_new_tensor_3d(wc,(ggml_type)q,k,2*hidden,experts);
    auto * down=ggml_new_tensor_3d(wc,q==8?GGML_TYPE_Q8_0:GGML_TYPE_Q5_K,hidden,n,experts);
    ggml_set_name(gate,"tp-weight"); ggml_set_name(down,"tp-down");
    auto * wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,packed); CHECK(wb);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> gold[2];
    int index=0;
    for (auto * w : {gate,down}) {
        std::vector<float> source(ggml_nelements(w)),importance(w->ne[0],1.f);
        std::vector<uint8_t> raw(ggml_nbytes(w)); gold[index].resize(source.size());
        for (size_t i=0;i<source.size();++i) source[i]=float(int((i*19+i/23)%71)-35)/256.f;
        CHECK(ggml_quantize_chunk(w->type,source.data(),raw.data(),0,w->ne[1]*experts,w->ne[0],importance.data())==raw.size());
        ggml_get_type_traits(w->type)->to_float(raw.data(),gold[index].data(),gold[index].size());
        if (w==gate) {
            const size_t half=w->nb[2]/2;
            std::vector<uint8_t> part(raw.size()/2);
            for (int side=0;side<2;++side) {
                for (int e=0;e<experts;++e) memcpy(part.data()+e*half,raw.data()+e*half*2+side*half,half);
                ggml_backend_tensor_set_2d(w,part.data(),side*half,half,experts,half*2,half);
            }
        } else ggml_backend_tensor_set(w,raw.data(),0,raw.size());
        ++index;
    }
    auto * a=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,1,tokens); ggml_set_name(a,"tp-input");
    auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,topk,tokens);
    auto * pair=ggml_mul_mat_id(ctx,gate,a,ids);
    auto * activation=ggml_swiglu(ctx,pair);
    auto * result=ggml_sqr(ctx,ggml_mul_mat_id(ctx,down,activation,ids));
    auto * graph=ggml_new_graph_custom(ctx,64,false); ggml_build_forward_expand(graph,result);
    auto * cb=ggml_backend_alloc_ctx_tensors(ctx,backend); CHECK(cb);
    std::vector<float> input(ggml_nelements(a)),output(ggml_nelements(result)),mid(hidden);
    std::vector<int32_t> router(topk*tokens);
    double maximum=0;
    for (int replay=0;replay<3;++replay) {
        for (size_t i=0;i<input.size();++i) input[i]=float(int((i*13+i/31+replay*11)%61)-30)/128.f;
        for (size_t i=0;i<router.size();++i) router[i]=(i+replay)%experts;
        ggml_backend_tensor_set(a,input.data(),0,ggml_nbytes(a));
        ggml_backend_tensor_set(ids,router.data(),0,ggml_nbytes(ids));
        CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend); ggml_backend_tensor_get(result,output.data(),0,ggml_nbytes(result));
        double err=0,norm=0;
        for (int r=0;r<tokens*topk;++r) {
            const int e=router[r],t=r/topk;
            for (int h=0;h<hidden;++h) {
                double g=0,u=0;
                for (int c=0;c<k;++c) {
                    g+=double(input[t*k+c])*gold[0][(size_t(e)*hidden*2+h)*k+c];
                    u+=double(input[t*k+c])*gold[0][(size_t(e)*hidden*2+hidden+h)*k+c];
                }
                mid[h]=g/(1+std::exp(-g))*u;
            }
            for (int col=0;col<n;++col) {
                double want=0;
                for (int h=0;h<hidden;++h) want+=double(mid[h])*gold[1][(size_t(e)*n+col)*hidden+h];
                want*=want; const double got=output[size_t(r)*n+col]; CHECK(std::isfinite(got));
                err+=(got-want)*(got-want); norm+=want*want;
            }
        }
        maximum=std::max(maximum,std::sqrt(err/std::max(norm,1.e-30)));
        CHECK(maximum<.02);
    }
    printf("KPACK_TP2_CHAIN q=%d tokens=%d experts=4 topk=2 replays=3 error=%.8g status=PASS\n",q,tokens,maximum);
    fflush(stdout);
    ggml_backend_buffer_free(cb); ggml_backend_buffer_free(wb); ggml_free(ctx); ggml_free(wc);
}

static void tp_numerical(ggml_backend_t backend, ggml_backend_buffer_type_t packed,
                         int qtype, int experts, int tokens, int axis) {
    constexpr int k = 1024, n = 512;
    const int topk = experts > 1 ? 2 : 1;
    tp_weight_split = {(ggml_backend_meta_split_axis) axis, {0}, {1}, 1};
    tp_weight_split.ne[0] = tp_weight_split.ne[1] = (axis == 0 ? k : n)/2;
    auto * weights = ggml_init({2 << 20, nullptr, true});
    auto * ctx = ggml_init({2 << 20, nullptr, true});
    CHECK(weights && ctx);
    auto * w = ggml_new_tensor_3d(weights, (ggml_type) qtype, k, n, experts);
    ggml_set_name(w, "tp-weight");
    auto * wb = ggml_backend_alloc_ctx_tensors_from_buft(weights, packed);
    CHECK(wb);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> source(size_t(k)*n*experts), dequant(source.size()), importance(k, 1.f);
    std::vector<uint8_t> raw(ggml_nbytes(w));
    for (size_t i = 0; i < source.size(); ++i) source[i] = float(int((i*17+i/29)%97)-48)/128.f;
    CHECK(ggml_quantize_chunk(w->type, source.data(), raw.data(), 0, n*experts, k, importance.data()) == raw.size());
    ggml_get_type_traits(w->type)->to_float(raw.data(), dequant.data(), dequant.size());
    ggml_backend_tensor_set(w, raw.data(), 0, raw.size());
    auto * a = experts > 1 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, topk, tokens) :
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    ggml_set_name(a, "tp-input");
    auto * ids = experts > 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens) : nullptr;
    auto * product = ids ? ggml_mul_mat_id(ctx, w, a, ids) : ggml_mul_mat(ctx, w, a);
    // A nonlinear consumer requires the existing Meta all-reduce for a K split.
    auto * out = ggml_sqr(ctx, product);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    auto * cb = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(cb && ggml_backend_supports_op(backend, product));
    std::vector<float> input(ggml_nelements(a)), output(ggml_nelements(out));
    std::vector<int32_t> router(tokens*topk);
    double max_error = 0;
    for (int replay = 0; replay < 3; ++replay) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i*13+i/37+replay*5)%61)-30)/128.f;
        for (int r = 0; r < tokens*topk; ++r) router[r] = (r+replay)%experts;
        ggml_backend_tensor_set(a, input.data(), 0, ggml_nbytes(a));
        if (ids) ggml_backend_tensor_set(ids, router.data(), 0, ggml_nbytes(ids));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
        double err = 0, norm = 0;
        for (int r = 0; r < tokens*topk; ++r) for (int col = 0; col < n; ++col) {
            double dot = 0;
            for (int c = 0; c < k; ++c) dot += double(input[size_t(r)*k+c])*dequant[(size_t(router[r])*n+col)*k+c];
            const double want = dot*dot, got = output[size_t(r)*n+col];
            CHECK(std::isfinite(got));
            err += (got-want)*(got-want); norm += want*want;
        }
        const double relative = std::sqrt(err/std::max(norm, 1.e-30));
        max_error = std::max(max_error, relative);
        CHECK(relative < .02);
    }
    printf("KPACK_TP2_CELL q=%d experts=%d tokens=%d split=%c replays=3 error=%.8g status=PASS\n",
           qtype, experts, tokens, axis == 0 ? 'K' : 'N', max_error);
    fflush(stdout);
    ggml_backend_buffer_free(cb); ggml_backend_buffer_free(wb);
    ggml_free(ctx); ggml_free(weights);
}

static int run_tp2() {
    CHECK(ggml_backend_cuda_get_device_count() == 2);
    ggml_backend_dev_t devs[2];
    for (int i = 0; i < 2; ++i) devs[i] = ggml_backend_buft_get_device(ggml_backend_cuda_buffer_type(i));
    auto * device = ggml_backend_meta_device(devs, 2, tp_split, nullptr);
    auto extras = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(devs[0]), "ggml_backend_dev_get_extra_bufts");
    CHECK(extras && extras(devs[0]) && extras(devs[0])[0]);
    auto * packed = ggml_backend_meta_buffer_type(device, extras(devs[0])[0]);
    auto * backend = ggml_backend_dev_init(device, nullptr);
    CHECK(packed && backend);
    for (int q : {8,10,11,12,13,14}) for (int e : {1,4}) for (int m : {1,8,32}) for (int axis : {0,1}) {
        tp_numerical(backend, packed, q, e, m, axis);
    }
    for (int q : {8,12}) for (int m : {1,8,32}) tp_chain(backend, packed, q, m);
    ggml_backend_free(backend);
    puts("KPACK_TP2_DEVICE PASS formats=6 cases=72 chains=6 replays=3 oracle=GGUF_FP32_DOT_SQUARE allreduce=K_SPLIT");
    return 0;
}

static void run_case(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_buffer_type_t buft,
                     int qtype, int experts, int tokens) {
    ggml_init_params init = { 2 << 20, nullptr, true };
    ggml_context * weights = ggml_init(init);
    ggml_context * ctx = ggml_init(init);
    CHECK(weights && ctx);
    ggml_tensor * weight = ggml_new_tensor_3d(weights, (ggml_type) qtype, 512, 256, experts);
    ggml_set_name(weight, "blk.0.ffn_down_exps.weight");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
    CHECK(buffer && weight->data && weight->op == GGML_OP_NONE);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(ggml_backend_supports_op(gpu, weight));

    ggml_tensor * input = experts == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens)
                                     : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, tokens);
    ggml_tensor * out;
    if (experts == 1) {
        out = ggml_mul_mat(ctx, weight, input);
    } else {
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens);
        ggml_set_input(ids);
        out = ggml_mul_mat_id(ctx, weight, input, ids);
    }
    ggml_set_input(input);
    ggml_set_output(out);
    CHECK(ggml_backend_supports_op(gpu, out));

    ggml_tensor unsupported = *out;
    unsupported.op = GGML_OP_DUP;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));
    unsupported.op = GGML_OP_VIEW;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));
    unsupported = *out;
    unsupported.src[0] = input;
    unsupported.src[1] = weight;
    CHECK(!ggml_backend_supports_op(gpu, &unsupported));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);
    CHECK(sched);
    printf("KPACK_SCHEDULER_RESERVE q=%d experts=%d tokens=%d buffer=%s op=%s\n",
           qtype, experts, tokens, ggml_backend_buft_name(buft), ggml_op_name(weight->op));
    fflush(stdout);
    CHECK(ggml_backend_sched_reserve(sched, graph));
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    CHECK(ggml_backend_sched_get_tensor_backend(sched, weight) == gpu);
    CHECK(ggml_backend_sched_get_tensor_backend(sched, out) == gpu);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(weights);
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "--tp2")) return run_tp2();
    const bool reserve_only = argc == 2 && strcmp(argv[1], "--reserve-only") == 0;
    CHECK(argc == 1 || reserve_only);
    if (ggml_backend_cuda_get_device_count() == 0) {
        fprintf(stderr, "KPACK_SCHEDULER SKIP: no CUDA/PPU device\n");
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(gpu && cpu);
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    auto extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    CHECK(extra_bufts);
    ggml_backend_buffer_type_t * extras = extra_bufts(dev);
    CHECK(extras && extras[0] && !extras[1]);
    ggml_backend_buffer_type_t kpack = extras[0];
    CHECK(kpack->device == dev);
    if (!reserve_only) {
        CHECK(ggml_backend_supports_buft(gpu, kpack));
        CHECK(!ggml_backend_supports_buft(cpu, kpack));
        CHECK(ggml_backend_supports_buft(gpu, ggml_backend_cuda_buffer_type(0)));
        auto split_buffer_type = (ggml_backend_split_buffer_type_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_split_buffer_type");
        if (split_buffer_type) {
            ggml_backend_buffer_type_t split = split_buffer_type(0, nullptr);
            CHECK(split && ggml_backend_supports_buft(gpu, split));
        }
        ggml_backend_buffer_type foreign = *kpack;
        foreign.device = ggml_backend_get_device(cpu);
        CHECK(!ggml_backend_supports_buft(gpu, &foreign));
        ggml_backend_buffer_type impostor = *kpack;
        impostor.iface.get_name = [](ggml_backend_buffer_type_t) { return "CUDA0_KPACK"; };
        CHECK(!ggml_backend_supports_buft(gpu, &impostor));
    }
    for (int qtype = 10; qtype <= 14; ++qtype) {
        for (int tokens : { 1, 16 }) {
            run_case(gpu, cpu, kpack, qtype, 1, tokens);
            run_case(gpu, cpu, kpack, qtype, 3, tokens);
        }
    }
    run_case(gpu, cpu, kpack, 12, 256, 4);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("KPACK_SCHEDULER PASS formats=5 cases=21 real_backend=1 real_scheduler=1 compute=NOT_RUN\n");
    return 0;
}
