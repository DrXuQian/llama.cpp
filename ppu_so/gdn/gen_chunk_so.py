#!/usr/bin/env python3
# Generate a self-contained C source for the chunked GDN forward chain, from FLA's Triton kernels via JIT cubins
# (NOT triton.tools.compile). Embeds the 5 cubins as byte arrays and emits ppu_gdn_chunk_<H>_<HV>_<S>(...) which
# loads them once and runs cumsum->kkt->wu->fwd_h->fwd_o, allocating the intermediates (g,A,w,u,h,v_new). State uses
# STATE_V_FIRST=1 so h0/ht are [v][k], matching ggml's recurrent-state layout (see the recurrent transpose note).
#
# Usage: FLA_ROOT=... python3 gen_chunk_so.py "H,HV,S" ["H,HV,S" ...]   -> writes aot_chunk_so/ppu_gdn_chunk.c
import os, sys, json
import numpy as np, torch

FLA = os.environ["FLA_ROOT"]; sys.path.insert(0, FLA)
import triton
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd_kernel
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.common.chunk_o import chunk_fwd_kernel_o

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "aot_chunk_so"); os.makedirs(OUT, exist_ok=True)

def unwrap(k):
    while not isinstance(k, triton.runtime.JITFunction) and hasattr(k, "fn"): k = k.fn
    return k

def carr(name, b):
    out = f"static const unsigned char {name}[{len(b)}] = {{"
    out += ",".join(str(x) for x in b)
    out += "};\n"
    return out

def compile_config(H, HV, S):
    K = V = S; BT = 64; dev = "cuda"; T = 256; NT = T // BT; B = 1
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(B,T,H,K,device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B,T,H,K,device=dev), dim=-1)
    v = torch.randn(B,T,HV,V,device=dev); beta = torch.rand(B,T,HV,device=dev)
    g_raw = -torch.rand(B,T,HV,device=dev)*0.5; h0 = torch.randn(B,HV,K,V,device=dev)*0.1
    g = chunk_local_cumsum(g_raw, chunk_size=BT, scale=RCP_LN2, output_dtype=torch.float32)
    BK = 1
    while BK < K: BK <<= 1
    BV = 1
    while BV < V: BV <<= 1
    A = torch.zeros(B,T,HV,BT,device=dev); w = torch.empty(B,T,HV,K,device=dev); u = torch.empty(B,T,HV,V,device=dev)
    h = torch.empty(B,NT,HV,V,K,device=dev); vnew = torch.empty_like(u); ht = torch.zeros(B,HV,V,K,device=dev)
    o = torch.empty_like(u); scale = 1.0/(K**0.5)
    BVh = 32; BKo = 64 if K % 64 == 0 else BK; BVo = 32

    # Kernel faults are ASYNCHRONOUS: without this, a bad launch here surfaces at the next synchronisation point --
    # which is the NEXT shape's first kernel -- and the traceback points at the wrong kernel entirely. Sync after each
    # one so a fault is attributed to the launch that caused it. Costs nothing; this is a code generator.
    def sync(tag):
        torch.cuda.synchronize()

    K_ = {}
    c = unwrap(chunk_local_cumsum_scalar_kernel)[(triton.cdiv(T,BT), B*HV)](g_raw, g, RCP_LN2, None, None, T, B, HV, BT, False, True, False, False)
    K_["cumsum"] = c; sync("cumsum")
    c = unwrap(chunk_gated_delta_rule_fwd_kkt_solve_kernel)[(triton.cdiv(T,BT), B*HV)](k, g, beta, A, None, None, T, H, HV, K, BT, 16, BK, True, False)
    K_["kkt"] = c; sync("kkt")
    c = unwrap(recompute_w_u_fwd_kernel)[(triton.cdiv(T,BT), B*HV)](k, v, beta, w, u, A, g, None, None, T, H, HV, K, V, BT, BK, BV, True, False)
    K_["wu"] = c; sync("wu")
    # STATE_V_FIRST=False here: the chunked kernel keeps FLA's [k][v] state and gated_delta_net.cu transposes
    # ggml's [v][k] on the way in and out. (The RECURRENT kernel is the opposite -- AOT'd STATE_V_FIRST=1, no
    # transpose. Mixing these up is not an error, just ~100% wrong numbers.)
    c = unwrap(chunk_gated_delta_rule_fwd_kernel_h_blockdim64)[(triton.cdiv(V,BVh), B*HV)](k, u, w, vnew, g, None, h, h0, ht, None, None, T, H, HV, K, V, BT, BVh, True, False, True, True, True, False, False, num_warps=4, num_stages=1)
    K_["h"] = c; sync("h")
    # NOTE: chunk_fwd_kernel_o's grid is 3-D -- (i_v, i_t, i_bh) = program_id(0,1,2). Launching it 2-D lets i_t range
    # over B*HV instead of NT, and `h += (i_tg*H + i_h)*V*K` is a RAW pointer bump that make_block_ptr's
    # boundary_check cannot save (it only guards the (V,K) dims relative to that base). With the defaults that reads
    # ~8x past the end of `h`. On NVIDIA the overrun lands inside torch's allocator pool -- garbage, but no fault, and
    # invisible because we only want the cubin here, not the numbers. On other backends it is an unmapped page and
    # the launch dies. The generated C launcher has always used the 3-D grid; only this warm-up did not.
    c = unwrap(chunk_fwd_kernel_o)[(triton.cdiv(V,BVo), triton.cdiv(T,BT), B*HV)](q, k, vnew, h, g, None, o, None, None, scale, T, H, HV, K, V, BT, BKo, BVo, True, False, False, False, num_warps=4, num_stages=1)
    K_["o"] = c; sync("o")
    meta = {kk: dict(name=cc.metadata.name, smem=int(cc.metadata.shared), block=cc.metadata.num_warps*32,
                     cubin=cc.asm["cubin"]) for kk, cc in K_.items()}
    return dict(H=H, HV=HV, S=S, BT=BT, BVh=BVh, meta=meta)


def emit(cfgs):
    src = ['// AUTO-GENERATED by gen_chunk_so.py -- FLA chunked GDN forward via embedded JIT cubins (torch/python-free).',
           '#include <cuda.h>', '#include <cuda_runtime.h>', '#include <stddef.h>', '#include <stdio.h>', '#include <pthread.h>', '']
    dispatch_arms = []
    for cfg in cfgs:
        H, HV, S = cfg["H"], cfg["HV"], cfg["S"]
        tag = f"{H}_{HV}_{S}"
        for kk in ["cumsum","kkt","wu","h","o"]:
            src.append(carr(f"cb_{tag}_{kk}", cfg["meta"][kk]["cubin"]))
        # per-config loader + launcher
        m = cfg["meta"]
        src.append(f"""
static CUfunction fn_{tag}[5];
static int load_ok_{tag} = 0;
// EVERY driver call is checked. cuModuleLoadData failing (arch mismatch, no current context, OOM) used to leave
// `mod` as an UNINITIALISED STACK VALUE, which cuModuleGetFunction then handed to the driver -- a host SEGFAULT,
// not an error return. That is the single nastiest failure mode of the whole dlopen approach: it looks like a bug
// in the caller.
static void load_{tag}(void) {{
    const unsigned char* cbs[5] = {{cb_{tag}_cumsum, cb_{tag}_kkt, cb_{tag}_wu, cb_{tag}_h, cb_{tag}_o}};
    const char* nm[5] = {{"{m['cumsum']['name']}","{m['kkt']['name']}","{m['wu']['name']}","{m['h']['name']}","{m['o']['name']}"}};
    int smem[5] = {{{m['cumsum']['smem']},{m['kkt']['smem']},{m['wu']['smem']},{m['h']['smem']},{m['o']['smem']}}};
    for (int i=0;i<5;i++) {{
        CUmodule mod = NULL;
        CUresult r = cuModuleLoadData(&mod, cbs[i]);
        if (r != CUDA_SUCCESS || mod == NULL) {{
            const char * e = NULL; cuGetErrorString(r, &e);
            fprintf(stderr, "[ppu-gdn] cuModuleLoadData(%s) failed: %d (%s) -- the cubin does not match this device, "
                            "or there is no current CUDA context on this thread\\n", nm[i], (int) r, e ? e : "?");
            fn_{tag}[i] = NULL;
            return;
        }}
        r = cuModuleGetFunction(&fn_{tag}[i], mod, nm[i]);
        if (r != CUDA_SUCCESS) {{
            fprintf(stderr, "[ppu-gdn] cuModuleGetFunction(%s) failed: %d\\n", nm[i], (int) r);
            fn_{tag}[i] = NULL;
            return;
        }}
        if (smem[i] > 49152) {{
            r = cuFuncSetAttribute(fn_{tag}[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem[i]);
            if (r != CUDA_SUCCESS) {{
                fprintf(stderr, "[ppu-gdn] cuFuncSetAttribute(%s, smem=%d) failed: %d -- device cannot give that "
                                "much dynamic shared memory\\n", nm[i], smem[i], (int) r);
                fn_{tag}[i] = NULL;
                return;
            }}
        }}
    }}
    load_ok_{tag} = 1;
}}
// q,k[n,T,H,S] v[n,T,HV,S] g_cumsum? no: g_raw[n,T,HV] beta[n,T,HV] h0,ht[n,HV,S,S] o[n,T,HV,S]. one sequence at a time.
static int run_{tag}(const float* q,const float* k,const float* v,const float* g_raw,const float* beta,
                     const float* h0,float* o,float* ht,int T,float scale,CUstream st,
                     float* g,float* A,float* w,float* u,float* h,float* vnew) {{
    if (!load_ok_{tag}) return 2;   // module failed to load; see the message from load_{tag}()
    const int H={H},HV={HV},S={S},BT={cfg['BT']},BVh={cfg['BVh']}; const int NT=(T+BT-1)/BT; const float RCP=1.4426950408889634f;
    CUdeviceptr gsc=0; int Tv=T;
    CUdeviceptr dq=(CUdeviceptr)q,dk=(CUdeviceptr)k,dv=(CUdeviceptr)v,dgr=(CUdeviceptr)g_raw,db=(CUdeviceptr)beta,dh0=(CUdeviceptr)h0,dO=(CUdeviceptr)o,dht=(CUdeviceptr)ht;
    CUdeviceptr dg=(CUdeviceptr)g,dA=(CUdeviceptr)A,dw=(CUdeviceptr)w,du=(CUdeviceptr)u,dh=(CUdeviceptr)h,dvn=(CUdeviceptr)vnew;
    float rcp=RCP;
    // A must be zeroed (kkt accumulates into it)
    cudaMemsetAsync((void*)dA, 0, (size_t)T*HV*BT*4, st);
    {{ void* a[]={{&dgr,&dg,&rcp,&Tv,&gsc}};            if(cuLaunchKernel(fn_{tag}[0],NT,HV,1, {m['cumsum']['block']},1,1, {m['cumsum']['smem']}, st, a, 0)) return 1; }}
    {{ void* a[]={{&dk,&dg,&db,&dA,&Tv,&gsc}};          if(cuLaunchKernel(fn_{tag}[1],NT,HV,1, {m['kkt']['block']},1,1, {m['kkt']['smem']}, st, a, 0)) return 1; }}
    {{ void* a[]={{&dk,&dv,&db,&dw,&du,&dA,&dg,&Tv,&gsc}}; if(cuLaunchKernel(fn_{tag}[2],NT,HV,1, {m['wu']['block']},1,1, {m['wu']['smem']}, st, a, 0)) return 1; }}
    {{ void* a[]={{&dk,&du,&dw,&dvn,&dg,&dh,&dh0,&dht,&Tv,&gsc}}; if(cuLaunchKernel(fn_{tag}[3],(S+BVh-1)/BVh,HV,1, {m['h']['block']},1,1, {m['h']['smem']}, st, a, 0)) return 1; }}
    {{ void* a[]={{&dq,&dk,&dvn,&dh,&dg,&dO,&scale,&Tv,&gsc}};   if(cuLaunchKernel(fn_{tag}[4],(S+BVh-1)/BVh,NT,HV, {m['o']['block']},1,1, {m['o']['smem']}, st, a, 0)) return 1; }}
    return 0;
}}""")
        dispatch_arms.append(f"""    if (H=={H} && HV=={HV} && S=={S}) {{
        pthread_once(&once_{tag}, load_{tag});
        if (!load_ok_{tag}) return 2;
        const size_t es=4; const int NT=(T+63)/64;
        // Per-call scratch. NOTE this is cudaMalloc'd OUTSIDE ggml's pool, ~220 MB at T=2048 -- and llama.cpp has
        // already filled the device with the model and the KV cache, so OOM here is entirely plausible. An
        // unchecked cudaMalloc leaves the pointer UNINITIALISED, which then goes to the kernel. Check all six.
        // (Caching this scratch across calls is the obvious fix, and is also what would make the path
        // CUDA-graph-capturable.)
        float *g=NULL,*A=NULL,*w=NULL,*u=NULL,*h=NULL,*vn=NULL;
        int oom = 0;
        oom |= cudaMalloc((void**)&g, (size_t)T*HV*es)      != cudaSuccess;
        oom |= cudaMalloc((void**)&A, (size_t)T*HV*64*es)   != cudaSuccess;
        oom |= cudaMalloc((void**)&w, (size_t)T*HV*S*es)    != cudaSuccess;
        oom |= cudaMalloc((void**)&u, (size_t)T*HV*S*es)    != cudaSuccess;
        oom |= cudaMalloc((void**)&h, (size_t)NT*HV*S*S*es) != cudaSuccess;
        oom |= cudaMalloc((void**)&vn,(size_t)T*HV*S*es)    != cudaSuccess;
        if (oom) {{
            fprintf(stderr, "[ppu-gdn] chunked scratch cudaMalloc failed (T=%d, ~%zu MB needed outside ggml's pool)"
                            " -- falling back\\n", T,
                    (size_t)((size_t)T*HV*es + (size_t)T*HV*64*es + 3*(size_t)T*HV*S*es
                             + (size_t)NT*HV*S*S*es) >> 20);
            cudaFree(g);cudaFree(A);cudaFree(w);cudaFree(u);cudaFree(h);cudaFree(vn);
            return 3;
        }}
        int rc=0;
        for (int n=0;n<n_seqs;n++) {{
            rc = run_{tag}(q+(size_t)n*T*H*S, k+(size_t)n*T*H*S, v+(size_t)n*T*HV*S,
                           g_raw+(size_t)n*T*HV, beta+(size_t)n*T*HV, h0+(size_t)n*HV*S*S,
                           o+(size_t)n*T*HV*S, ht+(size_t)n*HV*S*S, T, scale, (CUstream)stream, g,A,w,u,h,vn);
            if (rc) break;
        }}
        cudaFree(g);cudaFree(A);cudaFree(w);cudaFree(u);cudaFree(h);cudaFree(vn);
        return rc;
    }}""")
    # once flags
    for cfg in cfgs:
        tag = f"{cfg['H']}_{cfg['HV']}_{cfg['S']}"
        src.append(f"static pthread_once_t once_{tag} = PTHREAD_ONCE_INIT;")
    src.append(f"""
// ggml GDN chunked prefill (non-KDA, K_snapshot==1, contiguous, F32). g_raw is the RAW gate (pre-cumsum);
// the chain does the cumsum. Returns 0 ok, -1 unsupported (H,HV,S).
int ppu_gdn_chunked(const float* q,const float* k,const float* v,const float* g_raw,const float* beta,
                    const float* h0,float* o,float* ht,int n_seqs,int T,int H,int HV,int S,float scale,void* stream) {{
{chr(10).join(dispatch_arms)}
    return -1;
}}""")
    open(os.path.join(OUT, "ppu_gdn_chunk.c"), "w").write("\n".join(src))
    print("wrote", os.path.join(OUT, "ppu_gdn_chunk.c"), " configs:", [f"{c['H']}_{c['HV']}_{c['S']}" for c in cfgs])

if __name__ == "__main__":
    specs = sys.argv[1:] or ["4,4,128"]
    cfgs = [compile_config(*(int(x) for x in s.split(","))) for s in specs]
    emit(cfgs)
