// Compare ppu_gdn_chunked (the emitted .so's C launcher) against FLA's own python chain, at whatever shape
// gen_golden_chunk.py was run with. The GVA case (H != HV) is the one that was never covered.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

int ppu_gdn_chunked(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,
                    int,int,int,int,int,float,void*);

static float * rd(const char * nm, size_t n) {
    char p[512]; snprintf(p, sizeof p, "golden_chunk/%s.bin", nm);
    FILE * f = fopen(p, "rb"); if (!f) { printf("missing %s\n", p); exit(1); }
    float * b = malloc(n*4); if (fread(b, 4, n, f) != n) { printf("short read %s\n", p); exit(1); } fclose(f);
    return b;
}
static double relrms(const float * a, const float * b, size_t n) {
    double e = 0, r = 0;
    for (size_t i = 0; i < n; ++i) { const double d = (double)a[i]-b[i]; e += d*d; r += (double)b[i]*b[i]; }
    return sqrt(e / (r > 1e-30 ? r : 1e-30));
}
int main(void) {
    int B=1,T=256,H=0,HV=0,S=0; float scale=0;
    FILE * c = fopen("golden_chunk/config.txt","rb"); if(!c){puts("no config.txt");return 1;}
    char ln[128];
    while (fgets(ln,sizeof ln,c)) {
        if (!strncmp(ln,"H=",2))  H  = atoi(ln+2);
        if (!strncmp(ln,"HV=",3)) HV = atoi(ln+3);
        if (!strncmp(ln,"K=",2))  S  = atoi(ln+2);
        if (!strncmp(ln,"T=",2))  T  = atoi(ln+2);
        if (!strncmp(ln,"scale=",6)) scale = atof(ln+6);
    }
    fclose(c);
    printf("shape H=%d HV=%d S=%d T=%d  scale=%.6f   %s\n", H, HV, S, T, scale, H==HV?"(non-GVA)":"(GVA)");

    const size_t nq=(size_t)B*T*H*S, nv=(size_t)B*T*HV*S, ng=(size_t)B*T*HV, nh=(size_t)B*HV*S*S;
    float *q=rd("q",nq), *k=rd("k",nq), *v=rd("v",nv), *gr=rd("g_raw",ng), *be=rd("beta",ng),
          *h0=rd("h0",nh), *o_ref=rd("o",nv), *ht_ref=rd("final_state",nh);

    float *dq,*dk,*dv,*dg,*db,*dh0,*d_o,*d_ht;
    cudaMalloc((void**)&dq,nq*4); cudaMalloc((void**)&dk,nq*4); cudaMalloc((void**)&dv,nv*4); cudaMalloc((void**)&dg,ng*4);
    cudaMalloc((void**)&db,ng*4); cudaMalloc((void**)&dh0,nh*4); cudaMalloc((void**)&d_o,nv*4); cudaMalloc((void**)&d_ht,nh*4);
    cudaMemcpy(dq,q,nq*4,cudaMemcpyHostToDevice); cudaMemcpy(dk,k,nq*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,v,nv*4,cudaMemcpyHostToDevice); cudaMemcpy(dg,gr,ng*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,be,ng*4,cudaMemcpyHostToDevice); cudaMemcpy(dh0,h0,nh*4,cudaMemcpyHostToDevice);

    const int rc = ppu_gdn_chunked(dq,dk,dv,dg,db,dh0,d_o,d_ht, B,T,H,HV,S, scale, (void*)0);
    if (rc) { printf("ppu_gdn_chunked rc=%d (shape not compiled in?)\n", rc); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }

    float *o=malloc(nv*4), *ht=malloc(nh*4);
    cudaMemcpy(o,d_o,nv*4,cudaMemcpyDeviceToHost); cudaMemcpy(ht,d_ht,nh*4,cudaMemcpyDeviceToHost);
    const double ro = relrms(o,o_ref,nv), rh = relrms(ht,ht_ref,nh);
    const int ok = ro < 1e-4 && rh < 1e-4;
    printf("o rel_rms=%.3e   ht rel_rms=%.3e   -> %s\n", ro, rh, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
