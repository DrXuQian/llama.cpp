#include "quactlize-execution-lib.h"
#include "ggml-impl.h"

#ifdef GGML_NCP_QUACTLIZE
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>

template<typename T> static T qz_entry(void * library, const char * name) {
    auto entry = reinterpret_cast<T>(dlsym(library, name));
    if (!entry) { GGML_ABORT("[quactlize] native package missing %s", name); }
    return entry;
}

const ggml_quactlize_execution_api * ggml_quactlize_execution_library() {
    static const ggml_quactlize_execution_api api = [] {
        ggml_quactlize_execution_api result{};
        const char * root = getenv("QUACTLIZE_KPACK_EXECUTION");
        if (!root || !*root) { return result; }
        static const std::string directory(root);
        void * host = dlopen((directory + "/libquactlize_kpack_dispatch.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!host) { GGML_ABORT("[quactlize] native dispatch load: %s", dlerror()); }
        void * device = dlopen((directory + "/libquactlize_ppu_execution.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!device) { GGML_ABORT("[quactlize] execution load: %s", dlerror()); }
        result.root = directory.c_str();
#define QZ_BIND(field, library, name) result.field = qz_entry<decltype(result.field)>(library, name)
        QZ_BIND(open, host, "quactlize_kpack_dispatch_open_v1");
        QZ_BIND(close, host, "quactlize_kpack_dispatch_close_v1");
        QZ_BIND(query, host, "quactlize_kpack_dispatch_query_v1");
        QZ_BIND(prepare, host, "quactlize_kpack_dispatch_prepare_v1");
        QZ_BIND(run, host, "quactlize_kpack_dispatch_run_v1");
        QZ_BIND(destroy, host, "quactlize_kpack_dispatch_destroy_v1");
        QZ_BIND(error, host, "quactlize_kpack_dispatch_error_v1");
        QZ_BIND(gemv_query, device, "quactlize_kpack_gemv_query_v1");
        QZ_BIND(gemv_run, device, "quactlize_kpack_gemv_run_v1");
        QZ_BIND(sf_prepare, device, "quactlize_kpack_sf_prepare_v1");
        if (getenv("QUACTLIZE_KPACK_JIT_HELPER")) {
            QZ_BIND(enable_jit, host, "quactlize_kpack_dispatch_enable_jit_v1");
        }
#undef QZ_BIND
        GGML_LOG_INFO("[quactlize] native execution package: %s\n", root);
        return result;
    }();
    return api.root ? &api : nullptr;
}

bool ggml_quactlize_gemv_config(const qkg_call_v1 & call, qkg_config_v1 * config) {
    using Key = std::tuple<int,int,int,int,int,int,int,int,int>;
    static const std::map<Key, qkg_config_v1> choices = [] {
        std::map<Key, qkg_config_v1> result;
        const char * path = getenv("QUACTLIZE_KPACK_GEMV_POLICY");
        if (!path || !*path) return result;
        std::ifstream input(path);
        std::string line;
        if (!std::getline(input, line) || line != "KPACK_GEMV_POLICY_V1")
            GGML_ABORT("[quactlize] invalid GEMV policy: %s", path);
        while (std::getline(input, line)) {
            std::istringstream row(line);
            int q,n,k,e,mode,rows,channels,topk,type,columns,warps,split;
            std::string extra;
            if (!(row >> q >> n >> k >> e >> mode >> rows >> channels >> topk >> type >> columns >> warps >> split) ||
                (row >> extra) || q < 10 || q > 14 || n <= 0 || k <= 0 || e <= 0 || rows <= 0 ||
                channels <= 0 || topk <= 0 || type != QKG_F32 || mode < 0 || mode > 2 ||
                (columns != 16 && columns != 32) || (warps != 4 && warps != 8) || (split != 1 && split != 4))
                GGML_ABORT("[quactlize] malformed GEMV policy row: %s", path);
            if (!result.emplace(Key{q,n,k,e,mode,rows,channels,topk,type},
                    qkg_config_v1{1,sizeof(qkg_config_v1),columns,warps,split}).second)
                GGML_ABORT("[quactlize] duplicate GEMV policy row: %s", path);
        }
        if (input.bad()) GGML_ABORT("[quactlize] unreadable GEMV policy: %s", path);
        GGML_LOG_INFO("[quactlize] measured GEMV recipes: %zu (%s)\n", result.size(), path);
        return result;
    }();
    auto it = choices.find(Key{call.qtype,call.n,call.k,call.experts,call.mode,call.rows,call.channels,call.topk,call.input_type});
    if (it == choices.end()) return false;
    *config = it->second;
    return true;
}
int ggml_quactlize_prefill_route(const qks_request_v1 & request) {
    using Key = std::tuple<int,int,int,int,int,int,int>;
    static const std::map<Key,int> choices = [] {
        std::map<Key,int> result;
        const char * path = getenv("QUACTLIZE_KPACK_PREFILL_POLICY");
        if (!path || !*path) return result;
        std::ifstream input(path);
        std::string line;
        if (!std::getline(input,line) || line != "KPACK_PREFILL_POLICY_V2_PER_CALL")
            GGML_ABORT("[quactlize] invalid prefill policy: %s",path);
        while (std::getline(input,line)) {
            std::istringstream row(line);
            int q,n,k,e,m,maximum,op,selected;
            std::string extra;
            if (!(row >> q >> n >> k >> e >> m >> maximum >> op >> selected) || (row >> extra) ||
                q < 10 || q > 14 || n <= 0 || k <= 0 || e <= 0 || m <= 0 || maximum <= 0 ||
                (op != QK_DENSE_FQ && op != QK_GROUPED_FQ) || (selected != 0 && selected != 1))
                GGML_ABORT("[quactlize] malformed prefill policy row: %s",path);
            if (!result.emplace(Key{q,n,k,e,m,maximum,op},selected).second)
                GGML_ABORT("[quactlize] duplicate prefill policy row: %s",path);
        }
        if (input.bad()) GGML_ABORT("[quactlize] unreadable prefill policy: %s",path);
        GGML_LOG_INFO("[quactlize] measured FQ/SF comparisons: %zu (%s)\n",result.size(),path);
        return result;
    }();
    auto it=choices.find(Key{request.qtype,request.n,request.k,request.experts,request.m,request.max_rows,request.route});
    return it == choices.end() ? -1 : it->second;
}
#else
const ggml_quactlize_execution_api * ggml_quactlize_execution_library() { return nullptr; }
bool ggml_quactlize_gemv_config(const qkg_call_v1 &, qkg_config_v1 *) { return false; }
int ggml_quactlize_prefill_route(const qks_request_v1 &) { return -1; }
#endif
