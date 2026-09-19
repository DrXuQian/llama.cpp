// Default mode checks scheduler admission with stub inventory. TP2 modes also
// execute packing, arithmetic and cache reloads with real device libraries.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "llama-kpack-cache.h"
#include "llama-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <cmath>
#include <cstdarg>
#include <string>
#include <fstream>
#include <set>
#ifndef QUACTLIZE_TP2_HOST_TEST
#include <dlfcn.h>
#endif

#define CHECK(expr) do { \
    if (!(expr)) { fprintf(stderr, "scheduler check failed: %s\n", #expr); exit(1); } \
} while (0)

static ggml_backend_meta_split_state tp_weight_split;
static std::string tp_cache_root;
static bool tp_cache_hot = false;
static std::vector<std::string> tp_cache_manifests;
static std::string tp_chain_dump;
static bool tp_chain_retain = false;

static void tp_chain_write(const std::string & name, const void * data, size_t bytes) {
    if (tp_chain_dump.empty()) { return; }
    const std::string path = tp_chain_dump + "/" + name;
    CHECK(!std::ifstream(path).good());
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), bytes);
    out.close();
    CHECK(out.good());
}

static void tp_chain_write_shards(const char * name, ggml_tensor * tensor, int replay) {
    for (int rank = 0; rank < 2; ++rank) {
        auto * local = ggml_backend_meta_tensor_shard(tensor, rank, nullptr);
        CHECK(local && local->type == GGML_TYPE_F32 && ggml_is_contiguous(local));
        std::vector<float> values(ggml_nelements(local));
        ggml_backend_tensor_get(local, values.data(), 0, ggml_nbytes(local));
        tp_chain_write(std::string(name) + "-r" + std::to_string(rank) + "-i" + std::to_string(replay) + ".f32",
                       values.data(), values.size() * sizeof(float));
    }
}

void llama_log_internal(ggml_log_level, const char * format, ...) {
    va_list args;
    va_start(args, format); vfprintf(stderr, format, args); va_end(args);
}

static std::unique_ptr<llama_kpack_cache> weight_cache(
        ggml_tensor * w, const std::vector<uint8_t> & raw, const std::string & key) {
    if (tp_cache_root.empty()) { return {}; }
    const std::string path = tp_cache_root + "/" + key + ".gguf";
    auto * meta = gguf_init_empty();
    // GGUF reads through buffer callbacks when buffer is set, even with new data.
    ggml_tensor source = *w;
    source.buffer = nullptr;
    source.view_src = nullptr;
    source.view_offs = 0;
    source.extra = nullptr;
    source.data = const_cast<uint8_t *>(raw.data());
    gguf_add_tensor(meta, &source);
    if (!tp_cache_hot) {
        CHECK(!std::ifstream(path).good());
        CHECK(gguf_write_to_file(meta, path.c_str(), false));
    }
    tp_cache_manifests.push_back(path + ".cache/manifest.json");
    llama_kpack_source_tensor src;
    src.name = w->name; src.gguf_index = 0; src.data_offset = gguf_get_meta_size(meta);
    src.ggml_type = w->type; src.rank = ggml_n_dims(w); src.size_bytes = raw.size();
    src.n = w->ne[1]; src.k = w->ne[0]; src.experts = src.rank == 3 ? w->ne[2] : 0;
    gguf_free(meta);
    auto cache = std::make_unique<llama_kpack_cache>(path + ".cache", path,
        std::vector<llama_kpack_source_tensor>{src});
    CHECK(cache->load(w) == tp_cache_hot);
    return cache;
}

static ggml_backend_meta_split_state tp_split(const ggml_tensor * t, void *) {
    if (!strcmp(t->name, "tp-down")) return {GGML_BACKEND_SPLIT_AXIS_0, {512,512}, {1}, 1};
    if (!strcmp(t->name, "tp-weight") ||
            (!strcmp(t->name, "tp-input") && tp_weight_split.axis == GGML_BACKEND_SPLIT_AXIS_0)) {
        return tp_weight_split;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void check_tp_compute_split(ggml_tensor * tensor, ggml_backend_meta_split_axis axis) {
    CHECK(ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_meta_split_state state{};
    CHECK(ggml_backend_meta_tensor_shard(tensor, 0, &state));
    CHECK(state.axis == axis);
}

static void tp_chain(ggml_backend_t backend, ggml_backend_buffer_type_t packed, int q, int tokens) {
    printf("KPACK_TP2_BEGIN chain q=%d tokens=%d cache=%s\n", q, tokens,
           tp_cache_root.empty() ? "disabled" : tp_cache_hot ? "hot" : "cold");
    fflush(stdout);
    constexpr int k=1024, hidden=1024, n=512, experts=4, topk=2;
    tp_weight_split={GGML_BACKEND_SPLIT_AXIS_1,{hidden/2,hidden/2},{2},1};
    auto * wc=ggml_init({2<<20,nullptr,true});
    auto * ctx=ggml_init({2<<20,nullptr,true});
    auto * inputs=ggml_init({2<<20,nullptr,true});
    CHECK(wc && ctx && inputs);
    auto * gate=ggml_new_tensor_3d(wc,(ggml_type)q,k,2*hidden,experts);
    auto * down=ggml_new_tensor_3d(wc,q==0?GGML_TYPE_F32:q==8?GGML_TYPE_Q8_0:GGML_TYPE_Q5_K,hidden,n,experts);
    ggml_set_name(gate,"tp-weight"); ggml_set_name(down,"tp-down");
    auto * wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,packed); CHECK(wb);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> gold[2];
    std::unique_ptr<llama_kpack_cache> caches[2];
    int index=0;
    for (auto * w : {gate,down}) {
        std::vector<float> source(ggml_nelements(w)),importance(w->ne[0],1.f);
        std::vector<uint8_t> raw(ggml_nbytes(w)); gold[index].resize(source.size());
        for (size_t i=0;i<source.size();++i) source[i]=float(int((i*19+i/23)%71)-35)/256.f;
        CHECK(ggml_quantize_chunk(w->type,source.data(),raw.data(),0,w->ne[1]*experts,w->ne[0],importance.data())==raw.size());
        if (w->type == GGML_TYPE_F32) memcpy(gold[index].data(),raw.data(),raw.size());
        else ggml_get_type_traits(w->type)->to_float(raw.data(),gold[index].data(),gold[index].size());
        tp_chain_write("weight-" + std::to_string(index) + ".bin", raw.data(), raw.size());
        caches[index] = weight_cache(w, raw, "chain-" + std::to_string(q) + "-" +
            std::to_string(tokens) + "-" + std::to_string(index));
        if (!tp_cache_hot && w==gate) {
            const size_t half=w->nb[2]/2;
            std::vector<uint8_t> part(raw.size()/2);
            for (int side=0;side<2;++side) {
                for (int e=0;e<experts;++e) memcpy(part.data()+e*half,raw.data()+e*half*2+side*half,half);
                ggml_backend_tensor_set_2d(w,part.data(),side*half,half,experts,half*2,half);
            }
        } else if (!tp_cache_hot) ggml_backend_tensor_set(w,raw.data(),0,raw.size());
        if (caches[index]) { caches[index]->capture(w); caches[index]->start(); }
        ++index;
    }
    auto * a=ggml_new_tensor_3d(inputs,GGML_TYPE_F32,k,1,tokens); ggml_set_name(a,"tp-input");
    auto * ids=ggml_new_tensor_2d(inputs,GGML_TYPE_I32,topk,tokens);
    auto * ib=ggml_backend_alloc_ctx_tensors(inputs,backend); CHECK(ib);
    auto * pair=ggml_mul_mat_id(ctx,gate,a,ids);
    auto * activation=ggml_swiglu(ctx,pair);
    auto * product=ggml_mul_mat_id(ctx,down,activation,ids);
    auto * result=ggml_sqr(ctx,product);
    ggml_set_output(result);
    if (tp_chain_retain) {
        ggml_set_output(pair);
        ggml_set_output(activation);
        ggml_set_output(product);
    }
    auto * graph=ggml_new_graph_custom(ctx,64,false); ggml_build_forward_expand(graph,result);
    auto * ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(ga && ggml_gallocr_alloc_graph(ga,graph));
    check_tp_compute_split(pair,GGML_BACKEND_SPLIT_AXIS_0);
    check_tp_compute_split(activation,GGML_BACKEND_SPLIT_AXIS_0);
    check_tp_compute_split(product,GGML_BACKEND_SPLIT_AXIS_PARTIAL);
    check_tp_compute_split(result,GGML_BACKEND_SPLIT_AXIS_MIRRORED);
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
        if (!tp_chain_dump.empty()) {
            const std::string suffix = "-i" + std::to_string(replay);
            tp_chain_write("input" + suffix + ".f32", input.data(), input.size() * sizeof(float));
            tp_chain_write("ids" + suffix + ".i32", router.data(), router.size() * sizeof(int32_t));
            tp_chain_write("result" + suffix + ".f32", output.data(), output.size() * sizeof(float));
            if (tp_chain_retain) {
                tp_chain_write_shards("pair", pair, replay);
                tp_chain_write_shards("activation", activation, replay);
                // Meta reduces product in place before its nonlinear consumer.
                tp_chain_write_shards("reduced", product, replay);
            }
        }
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
        const double relative = std::sqrt(err/std::max(norm,1.e-30));
        maximum=std::max(maximum,relative);
        if (!tp_chain_dump.empty()) {
            printf("KPACK_TP2_CHAIN_CAPTURE replay=%d retain=%d relative=%.9g threshold=0.02 admitted=0\n",
                   replay, int(tp_chain_retain), relative);
            fflush(stdout);
        }
        if (!(maximum < .02)) {
            fprintf(stderr, "KPACK_TP2_CHAIN_MISMATCH q=%d tokens=%d replay=%d relative=%.9g\n", q, tokens, replay, maximum);
        }
        if (tp_chain_dump.empty()) { CHECK(maximum<.02); }
    }
    if (tp_chain_dump.empty()) {
        printf("KPACK_TP2_CHAIN q=%d tokens=%d experts=4 topk=2 replays=3 error=%.8g status=PASS\n",q,tokens,maximum);
    } else {
        printf("KPACK_TP2_CHAIN_CAPTURE_COMPLETE q=%d tokens=%d replays=3 retain=%d maximum=%.9g gate=%s admitted=0\n",
               q, tokens, int(tp_chain_retain), maximum, maximum < .02 ? "PASS" : "FAIL");
    }
    fflush(stdout);
    for (auto & cache : caches) { cache.reset(); }
    ggml_gallocr_free(ga); ggml_backend_buffer_free(ib); ggml_backend_buffer_free(wb);
    ggml_free(inputs); ggml_free(ctx); ggml_free(wc);
}

static void tp_numerical(ggml_backend_t backend, ggml_backend_buffer_type_t packed,
                         int qtype, int experts, int tokens, int axis, bool missing_reduce_negative = false) {
    CHECK(!missing_reduce_negative || (axis == 0 && experts == 1 && tokens == 1));
    printf("KPACK_TP2_BEGIN cell q=%d experts=%d tokens=%d split=%c cache=%s\n",
           qtype, experts, tokens, axis == 0 ? 'K' : 'N',
           tp_cache_root.empty() ? "disabled" : tp_cache_hot ? "hot" : "cold");
    fflush(stdout);
    constexpr int k = 1024, n = 512;
    const int topk = experts > 1 ? 2 : 1;
    tp_weight_split = {(ggml_backend_meta_split_axis) axis, {0}, {1}, 1};
    tp_weight_split.ne[0] = tp_weight_split.ne[1] = (axis == 0 ? k : n)/2;
    auto * weights = ggml_init({2 << 20, nullptr, true});
    auto * ctx = ggml_init({2 << 20, nullptr, true});
    auto * inputs = ggml_init({2 << 20, nullptr, true});
    CHECK(weights && ctx && inputs);
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
    auto cache = weight_cache(w, raw, "cell-" + std::to_string(qtype) + "-" +
        std::to_string(experts) + "-" + std::to_string(tokens) + "-" + std::to_string(axis));
    if (!tp_cache_hot) { ggml_backend_tensor_set(w, raw.data(), 0, raw.size()); }
    if (cache) { cache->capture(w); cache->start(); }
    auto * a = experts > 1 ? ggml_new_tensor_3d(inputs, GGML_TYPE_F32, k, topk, tokens) :
                            ggml_new_tensor_2d(inputs, GGML_TYPE_F32, k, tokens);
    ggml_set_name(a, "tp-input");
    auto * ids = experts > 1 ? ggml_new_tensor_2d(inputs, GGML_TYPE_I32, topk, tokens) : nullptr;
    auto * ib = ggml_backend_alloc_ctx_tensors(inputs, backend);
    CHECK(ib);
    auto * product = ids ? ggml_mul_mat_id(ctx, w, a, ids) : ggml_mul_mat(ctx, w, a);
    // A nonlinear consumer requires the existing Meta all-reduce for a K split.
    auto * out = ggml_sqr(ctx, product);
    ggml_set_output(out);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t ga = nullptr;
    ggml_backend_buffer_t legacy = nullptr;
    if (missing_reduce_negative) {
        legacy = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(legacy);
        ggml_backend_meta_split_state state{};
        CHECK(ggml_backend_meta_tensor_shard(product, 0, &state));
        CHECK(state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    } else {
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        CHECK(ga && ggml_gallocr_alloc_graph(ga, graph));
        check_tp_compute_split(product, axis == 0 ? GGML_BACKEND_SPLIT_AXIS_PARTIAL : GGML_BACKEND_SPLIT_AXIS_0);
        check_tp_compute_split(out, axis == 0 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_0);
    }
    CHECK(ggml_backend_supports_op(backend, product));
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
        double err = 0, norm = 0, partial_err = 0, partial_norm = 0;
        for (int r = 0; r < tokens*topk; ++r) for (int col = 0; col < n; ++col) {
            double dot = 0, partial = 0;
            for (int c = 0; c < k; ++c) {
                const double term = double(input[size_t(r)*k+c])*dequant[(size_t(router[r])*n+col)*k+c];
                dot += term;
                if (c < k/2) partial += term;
            }
            const double want = dot*dot, got = output[size_t(r)*n+col];
            CHECK(std::isfinite(got));
            err += (got-want)*(got-want); norm += want*want;
            partial *= partial;
            partial_err += (got-partial)*(got-partial); partial_norm += partial*partial;
        }
        const double relative = std::sqrt(err/std::max(norm, 1.e-30));
        max_error = std::max(max_error, relative);
        if (missing_reduce_negative) {
            const double partial_relative = std::sqrt(partial_err/std::max(partial_norm,1.e-30));
            CHECK(relative > .02 && partial_relative < .02);
            printf("KPACK_TP2_GRAPH_NEGATIVE replay=%d full_error=%.9g rank0_only_error=%.9g EXPECTED_RED\n",
                   replay,relative,partial_relative);
        } else if (!(relative < .02)) {
            fprintf(stderr, "KPACK_TP2_MISMATCH q=%d experts=%d tokens=%d split=%c replay=%d relative=%.9g\n",
                    qtype, experts, tokens, axis == 0 ? 'K' : 'N', replay, relative);
        }
        if (!missing_reduce_negative) CHECK(relative < .02);
    }
    if (!missing_reduce_negative) {
        printf("KPACK_TP2_CELL q=%d experts=%d tokens=%d split=%c replays=3 error=%.8g status=PASS\n",
               qtype, experts, tokens, axis == 0 ? 'K' : 'N', max_error);
    }
    fflush(stdout);
    cache.reset();
    ggml_gallocr_free(ga); ggml_backend_buffer_free(legacy);
    ggml_backend_buffer_free(ib); ggml_backend_buffer_free(wb);
    ggml_free(inputs); ggml_free(ctx); ggml_free(weights);
}

#ifdef QUACTLIZE_TP2_HOST_TEST
int main() {
    auto * cpu = ggml_backend_cpu_init();
    CHECK(cpu);
    auto * dev = ggml_backend_get_device(cpu);
    ggml_backend_dev_t devs[] = {dev,dev};
    auto * meta = ggml_backend_meta_device(devs,2,tp_split,nullptr);
    auto * backend = ggml_backend_dev_init(meta,nullptr);
    auto * buft = ggml_backend_dev_buffer_type(meta);
    CHECK(backend && buft);
    tp_numerical(backend,buft,8,1,1,0,true);
    for (int q : {8,10,11,12,13,14}) for (int e : {1,4}) for (int m : {1,8,32}) for (int axis : {0,1}) {
        tp_numerical(backend,buft,q,e,m,axis);
    }
    // CPU quantized GEMM requantizes A; use F32 to isolate chain split semantics.
    for (int m : {1,8,32}) tp_chain(backend,buft,GGML_TYPE_F32,m);
    ggml_backend_free(backend); ggml_backend_free(cpu);
    puts("KPACK_TP2_HOST PASS cells=72 f32_chains=3 missing_reduce=EXPECTED_RED device_admission=PENDING");
}
#else
static void tp_comm_libraries() {
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::set<std::string> libraries;
    std::string line;
    while (std::getline(maps,line)) {
        const auto start=line.find('/');
        if (start==std::string::npos) continue;
        const auto path=line.substr(start);
        for (const char * name : {"libnccl", "libpccl", "libhggc", "libhgrtc", "libcuda", "libggml", "libquactlize"}) {
            if (path.find(name)!=std::string::npos) libraries.insert(path);
        }
    }
    for (const auto & path : libraries) printf("KPACK_TP2_COMM_LIBRARY path=%s\n",path.c_str());
    fflush(stdout);
#endif
}

static void tp_comm_symbols(const char * phase) {
    for (const char * name : {"hggcLaunchKernel", "hggcGetFuncBySymbol", "__hggcRegisterFatBinary"}) {
        Dl_info info{};
        void * address=dlsym(RTLD_DEFAULT,name);
        const char * library=address && dladdr(address,&info) && info.dli_fname?info.dli_fname:"NOT_GLOBAL";
        printf("KPACK_TP2_COMM_SYMBOL phase=%s name=%s library=%s\n",phase,name,library);
    }
    fflush(stdout);
}

static double tp_comm_error(const std::vector<float> & got, const std::vector<float> & want) {
    CHECK(got.size()==want.size());
    double error=0, norm=0;
    for (size_t i=0;i<got.size();++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(want[i]));
        const double delta=double(got[i])-want[i];
        error+=delta*delta; norm+=double(want[i])*want[i];
    }
    return std::sqrt(error/std::max(norm,1.e-30));
}

// Use the caller's unchanged communication entry, with synchronized local oracles.
static uint64_t tp_bytes_hash(const void * data, size_t size) {
    uint64_t hash=UINT64_C(14695981039346656037);
    const auto * bytes=static_cast<const uint8_t *>(data);
    for (size_t i=0;i<size;++i) hash=(hash^bytes[i])*UINT64_C(1099511628211);
    return hash;
}

static int run_tp2_comm(const char * arm, int count, const char * scope="none", int qtype=8, int experts=1) {
    const bool copy=!strcmp(arm,"copy"), packed=!strcmp(arm,"kpack");
    CHECK(copy || packed || !strcmp(arm,"raw"));
    CHECK((qtype==8 && experts==1) || (!copy && qtype==12 && experts==4));
    CHECK(count==512 || (copy && (count==3072 || count==32768)));
    const int topk=experts>1?2:1, elements=count*topk;
    CHECK(!strcmp(scope,"none") || !strcmp(scope,"local") || !strcmp(scope,"global"));
    CHECK(ggml_backend_cuda_get_device_count()==2);
    printf("KPACK_TP2_COMM_BEGIN arm=%s count=%d bytes=%zu\n",arm,elements,size_t(elements)*sizeof(float));
    printf("KPACK_TP2_COMM_SHAPE q=%d n=%d global_k=1024 local_k=512 experts=%d tokens=1 topk=%d\n",qtype,count,experts,topk);
    for (const char * name : {"CUDA_VISIBLE_DEVICES", "GGML_CUDA_ALLREDUCE", "PCCL_ENABLE_EXT_KERNEL", "PCCL_EXT_KERNEL_PLUGIN", "PCCL_ALGO", "PCCL_PROTO"}) {
        const char * value=std::getenv(name);
        printf("KPACK_TP2_COMM_ENV %s=%s\n",name,value?value:"UNSET");
    }
    fflush(stdout);
    ggml_backend_t backends[]={ggml_backend_cuda_init(0),ggml_backend_cuda_init(1)};
    CHECK(backends[0] && backends[1]);
    auto * reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[0]));
    auto init=(ggml_backend_comm_init_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_init");
    auto reduce=(ggml_backend_comm_allreduce_tensor_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_allreduce_tensor");
    auto release=(ggml_backend_comm_free_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_free");
    CHECK(init && reduce && release);
    auto * comm=init(backends,2); CHECK(comm);
    tp_comm_symbols("before-wrapper");
    if (strcmp(scope,"none")) {
        const char * sdk=std::getenv("PPU_SDK"); CHECK(sdk && *sdk);
        const std::string path=std::string(sdk)+"/lib/libhggc_wrapper.so";
        void * wrapper=dlopen(path.c_str(),RTLD_NOW|(!strcmp(scope,"global")?RTLD_GLOBAL:RTLD_LOCAL));
        if (!wrapper) fprintf(stderr,"KPACK_TP2_COMM_WRAPPER load failed: %s\n",dlerror());
        CHECK(wrapper);
        printf("KPACK_TP2_COMM_WRAPPER scope=%s path=%s\n",scope,path.c_str());
    }
    tp_comm_symbols("after-wrapper");
    tp_comm_libraries();

    constexpr int global_k=1024, local_k=512;
    std::vector<uint8_t> raw;
    std::vector<float> gold;
    if (!copy) {
        std::vector<float> source(size_t(global_k)*count*experts), importance(global_k,1.f);
        raw.resize(ggml_row_size((ggml_type)qtype,global_k)*count*experts); gold.resize(source.size());
        for (size_t i=0;i<source.size();++i) source[i]=float(int((i*17+i/29)%97)-48)/128.f;
        CHECK(ggml_quantize_chunk((ggml_type)qtype,source.data(),raw.data(),0,count*experts,global_k,importance.data())==raw.size());
        ggml_get_type_traits((ggml_type)qtype)->to_float(raw.data(),gold.data(),gold.size());
    }
    ggml_context * weights[2]{}, * inputs[2]{}, * ctx[2]{};
    ggml_backend_buffer_t wb[2]{}, ib[2]{};
    ggml_gallocr_t ga[2]{};
    ggml_tensor * a[2]{}, * ids[2]{}, * output[2]{};
    ggml_cgraph * graph[2]{};
    std::vector<float> expected[2]={std::vector<float>(elements),std::vector<float>(elements)};
    for (int rank=0;rank<2;++rank) {
        inputs[rank]=ggml_init({2<<20,nullptr,true});
        ctx[rank]=ggml_init({2<<20,nullptr,true}); CHECK(inputs[rank] && ctx[rank]);
        if (copy) {
            output[rank]=ggml_new_tensor_1d(inputs[rank],GGML_TYPE_F32,count);
        } else {
            weights[rank]=ggml_init({2<<20,nullptr,true}); CHECK(weights[rank]);
            auto * w=ggml_new_tensor_3d(weights[rank],(ggml_type)qtype,local_k,count,experts);
            ggml_set_name(w,"tp-weight");
            auto * buft=ggml_backend_get_default_buffer_type(backends[rank]);
            if (packed) {
                auto extras=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");
                CHECK(extras);
                auto ** list=extras(ggml_backend_get_device(backends[rank])); CHECK(list && list[0]);
                buft=list[0];
            }
            wb[rank]=ggml_backend_alloc_ctx_tensors_from_buft(weights[rank],buft); CHECK(wb[rank]);
            ggml_backend_buffer_set_usage(wb[rank],GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t row=ggml_row_size(w->type,local_k);
            std::vector<uint8_t> shard(row*count*experts);
            for (int col=0;col<count*experts;++col) memcpy(shard.data()+col*row,raw.data()+(2*col+rank)*row,row);
            printf("KPACK_TP2_COMM_WEIGHT rank=%d q=%d bytes=%zu hash=%016llx\n",rank,qtype,shard.size(),
                   (unsigned long long)tp_bytes_hash(shard.data(),shard.size()));
            ggml_backend_tensor_set(w,shard.data(),0,shard.size());
            a[rank]=ggml_new_tensor_3d(inputs[rank],GGML_TYPE_F32,local_k,topk,1);
            if (experts>1) ids[rank]=ggml_new_tensor_2d(inputs[rank],GGML_TYPE_I32,topk,1);
            output[rank]=ids[rank]?ggml_mul_mat_id(ctx[rank],w,a[rank],ids[rank]):ggml_mul_mat(ctx[rank],w,a[rank]);
            ggml_set_output(output[rank]);
            graph[rank]=ggml_new_graph_custom(ctx[rank],16,false);
            ggml_build_forward_expand(graph[rank],output[rank]);
        }
        ib[rank]=ggml_backend_alloc_ctx_tensors(inputs[rank],backends[rank]); CHECK(ib[rank]);
        if (!copy) {
            ga[rank]=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[rank]));
            CHECK(ga[rank] && ggml_gallocr_alloc_graph(ga[rank],graph[rank]));
        }
        output[rank]->flags|=GGML_TENSOR_FLAG_COMPUTE;
    }
    bool numerical_ok=true;
    for (int replay=0;replay<3;++replay) {
        std::vector<float> input(global_k*topk), got(elements), sum(elements);
        std::vector<int32_t> router(topk);
        for (size_t i=0;i<input.size();++i) input[i]=float(int((i*13+i/37+replay*5)%61)-30)/128.f;
        for (int r=0;r<topk;++r) router[r]=(r+replay)%experts;
        for (int rank=0;rank<2;++rank) {
            printf("KPACK_TP2_COMM_LOCAL_BEGIN arm=%s rank=%d replay=%d\n",arm,rank,replay); fflush(stdout);
            for (int row=0;row<topk;++row) for (int col=0;col<count;++col) {
                double dot=0;
                if (copy) dot=float((col*7+rank*5+replay*3)%61-30)/8.f;
                else for (int c=rank*local_k;c<(rank+1)*local_k;++c)
                    dot+=double(input[row*global_k+c])*gold[(size_t(router[row])*count+col)*global_k+c];
                expected[rank][row*count+col]=float(dot);
            }
            printf("KPACK_TP2_COMM_ORACLE rank=%d replay=%d input_hash=%016llx ids_hash=%016llx golden_hash=%016llx\n",rank,replay,
                   (unsigned long long)tp_bytes_hash(input.data(),input.size()*sizeof(float)),
                   (unsigned long long)tp_bytes_hash(router.data(),router.size()*sizeof(int32_t)),
                   (unsigned long long)tp_bytes_hash(expected[rank].data(),expected[rank].size()*sizeof(float)));
            if (copy) ggml_backend_tensor_set(output[rank],expected[rank].data(),0,count*sizeof(float));
            else {
                for (int row=0;row<topk;++row)
                    ggml_backend_tensor_set(a[rank],input.data()+row*global_k+rank*local_k,row*local_k*sizeof(float),local_k*sizeof(float));
                if (ids[rank]) ggml_backend_tensor_set(ids[rank],router.data(),0,topk*sizeof(int32_t));
                CHECK(ggml_backend_graph_compute(backends[rank],graph[rank])==GGML_STATUS_SUCCESS);
            }
            ggml_backend_synchronize(backends[rank]);
            ggml_backend_tensor_get(output[rank],got.data(),0,elements*sizeof(float));
            const double error=tp_comm_error(got,expected[rank]);
            const bool valid=copy?error==0:error<.02;
            printf("KPACK_TP2_COMM_LOCAL arm=%s rank=%d replay=%d error=%.9g synchronized=1 status=%s\n",
                   arm,rank,replay,error,valid?"PASS":"FAIL"); fflush(stdout);
            if (!valid) {
                for (int i=0;i<elements && i<8;++i)
                    printf("KPACK_TP2_COMM_VALUE rank=%d replay=%d index=%d want=%.9g got=%.9g\n",rank,replay,i,expected[rank][i],got[i]);
                numerical_ok=false;
            }
        }
        if (!numerical_ok) break;
        for (int col=0;col<elements;++col) sum[col]=expected[0][col]+expected[1][col];
        tp_comm_libraries();
        tp_comm_symbols("before-reduce");
        printf("KPACK_TP2_COMM_REDUCE_BEGIN arm=%s count=%d replay=%d\n",arm,elements,replay); fflush(stdout);
        CHECK(reduce(comm,output));
        for (int rank=0;rank<2;++rank) {
            ggml_backend_synchronize(backends[rank]);
            ggml_backend_tensor_get(output[rank],got.data(),0,elements*sizeof(float));
            const double error=tp_comm_error(got,sum);
            const bool valid=copy?error==0:error<.02;
            printf("KPACK_TP2_COMM_SUM arm=%s rank=%d replay=%d error=%.9g status=%s\n",
                   arm,rank,replay,error,valid?"PASS":"FAIL"); fflush(stdout);
            CHECK(valid);
        }
    }
    release(comm);
    for (int rank=0;rank<2;++rank) {
        ggml_gallocr_free(ga[rank]); ggml_backend_buffer_free(ib[rank]); ggml_backend_buffer_free(wb[rank]);
        ggml_free(inputs[rank]); ggml_free(ctx[rank]); ggml_free(weights[rank]); ggml_backend_free(backends[rank]);
    }
    if (!numerical_ok) {
        puts("KPACK_TP2_COMM_STOP reason=LOCAL_NUMERICAL_FAILURE collective_not_launched=1");
        return 1;
    }
    printf("KPACK_TP2_COMM PASS arm=%s count=%d replays=3\n",arm,elements);
    return 0;
}

static int run_tp2(bool single=false) {
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
    if (!tp_chain_dump.empty()) {
        tp_chain(backend, packed, 12, 32);
        ggml_backend_free(backend);
        return 0;
    }
    if (single) {
        tp_numerical(backend,packed,12,4,1,0);
        ggml_backend_free(backend);
        puts("KPACK_TP2_SINGLE PASS q=12 experts=4 tokens=1 split=K replays=3");
        return 0;
    }
    for (int q : {8,10,11,12,13,14}) for (int e : {1,4}) for (int m : {1,8,32}) for (int axis : {0,1}) {
        tp_numerical(backend, packed, q, e, m, axis);
    }
    for (int q : {8,12}) for (int m : {1,8,32}) tp_chain(backend, packed, q, m);
    ggml_backend_free(backend);
    for (const auto & manifest : tp_cache_manifests) {
        CHECK(std::ifstream(manifest).peek() != std::char_traits<char>::eof());
    }
    puts("KPACK_TP2_DEVICE PASS formats=6 cases=72 chains=6 replays=3 oracle=GGUF_FP32_DOT_SQUARE allreduce=K_SPLIT");
    if (!tp_cache_root.empty()) printf("KPACK_TP2_CACHE PASS mode=%s cases=72 chains=6\n", tp_cache_hot ? "hot" : "cold");
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
    if (argc == 4 && !strcmp(argv[1], "--tp2-chain-dump")) {
        CHECK(!strcmp(argv[3], "ordinary") || !strcmp(argv[3], "retained"));
        tp_chain_dump = argv[2];
        CHECK(!tp_chain_dump.empty());
        tp_chain_retain = !strcmp(argv[3], "retained");
        return run_tp2();
    }
    if (argc==3 && !strcmp(argv[1],"--tp2-q4-local")) return run_tp2_comm(argv[2],512,"none",12,4);
    if (argc==2 && !strcmp(argv[1],"--tp2-q4-meta")) return run_tp2(true);
    if ((argc==4 || argc==5) && !strcmp(argv[1],"--tp2-comm"))
        return run_tp2_comm(argv[2],std::stoi(argv[3]),argc==5?argv[4]:"none");
    if (argc == 2 && !strcmp(argv[1], "--tp2")) return run_tp2();
    if (argc == 3 && (!strcmp(argv[1], "--tp2-cache-write") || !strcmp(argv[1], "--tp2-cache-read"))) {
        tp_cache_root = argv[2];
        tp_cache_hot = !strcmp(argv[1], "--tp2-cache-read");
        return run_tp2();
    }
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
#endif
