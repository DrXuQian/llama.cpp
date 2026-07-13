// test_moe — standalone correctness check for libppu_moe.so, independent of ggml. Exercises both layouts:
//
//   masked      out[g, i, :] = A[g, i, :] @ B[g]^T   for i < masked_m[g]   A:[G,max_m,k]  out:[G,max_m,n]
//   contiguous  out[i, :]    = A[i, :]    @ B[mi[i]]^T                     A:[m,k]        out:[m,n]
//
// NOTE: public DeepGEMM has no NoPad kernel, so this .so exports no ppu_moe_grouped_gemm_bf16_nopad and there is
// nothing here to test for it. The PPU kernel team's bf16_grouped_deep_gemm_NoPad is what would provide it.
//
//   ./test_moe                 # G=4, rows_per_expert=128, n=256, k=128
//   ./test_moe 8 100 512 256   # G rows_per_expert n k
//
// The masked case deliberately gives each expert a DIFFERENT, unaligned row count (rows_per_expert +/- a few) --
// masked_m needs no alignment, and that raggedness is the whole reason we prefer it over the contiguous layout.
// Both are compared against an fp32 CPU reference computed from the same bf16 inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" int ppu_moe_grouped_gemm_bf16_masked(
    const void * A, const void * B, void * out, const int * masked_m,
    int max_m, int N, int K, int n_experts, int expected_m, void * stream);
extern "C" int ppu_moe_grouped_gemm_bf16_contiguous(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);
extern "C" int ppu_moe_row_alignment(void);

static float rnd(int i) { return std::sin(0.037f * i) * 0.5f; }

// Round-trip fp32 -> bf16 -> fp32 so the CPU reference sees exactly the values the kernel sees.
static std::vector<__nv_bfloat16> quantize(std::vector<float> & f) {
    std::vector<__nv_bfloat16> b(f.size());
    for (size_t i = 0; i < f.size(); ++i) { b[i] = __float2bfloat16(f[i]); f[i] = __bfloat162float(b[i]); }
    return b;
}

// rel_rms over the rows the caller marks real; pad rows are garbage by design in both layouts.
static bool check(const char * name, const std::vector<__nv_bfloat16> & got, const std::vector<float> & ref,
                  int N, const std::vector<char> & row_is_real) {
    double maxrel = 0, l2 = 0, refl2 = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!row_is_real[i / N]) continue;
        const double g = __bfloat162float(got[i]), w = ref[i], e = std::fabs(g - w);
        maxrel = std::fmax(maxrel, e / std::fmax(1.0, std::fabs(w)));
        l2 += e*e; refl2 += w*w;
    }
    const double rel_rms = std::sqrt(l2 / std::fmax(refl2, 1e-12));
    const bool ok = rel_rms < 1e-2;
    printf("  %-11s max_rel=%.4g  rel_rms=%.4g  -> %s\n", name, maxrel, rel_rms, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    const int G   = argc > 1 ? atoi(argv[1]) : 4;
    const int RPE = argc > 2 ? atoi(argv[2]) : 128;   // nominal rows per expert
    const int N   = argc > 3 ? atoi(argv[3]) : 256;
    const int K   = argc > 4 ? atoi(argv[4]) : 128;

    std::vector<float> hB((size_t) G * N * K);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = rnd((int) i + 11);
    auto bB = quantize(hB);
    void * dB; cudaMalloc(&dB, bB.size()*2);
    cudaMemcpy(dB, bB.data(), bB.size()*2, cudaMemcpyHostToDevice);

    bool all_ok = true;

    // ---------------- masked: ragged, unaligned per-expert row counts ----------------
    {
        std::vector<int> masked_m(G);
        int max_rows = 0;
        for (int g = 0; g < G; ++g) {                       // deliberately uneven and not a multiple of anything
            masked_m[g] = RPE > 8 ? RPE - 7 + (g * 5) % 15 : RPE;
            max_rows = std::max(max_rows, masked_m[g]);
        }
        // max_m must be a multiple of DeepGEMM's BLOCK_M (64 or 128) -- it is the row pitch between groups, and a
        // group's last row-block would otherwise spill into the next group's rows. 128 is the safe superset.
        const int max_m = (max_rows + 127) / 128 * 128;
        const size_t rows = (size_t) G * max_m;

        std::vector<float> hA(rows * K);
        for (size_t i = 0; i < hA.size(); ++i) hA[i] = rnd((int) i);
        auto bA = quantize(hA);

        std::vector<float> ref(rows * (size_t) N, 0.f);
        std::vector<char>  real(rows, 0);
        for (int g = 0; g < G; ++g) {
            for (int i = 0; i < masked_m[g]; ++i) {
                const size_t r = (size_t) g*max_m + i;
                real[r] = 1;
                for (int j = 0; j < N; ++j) {
                    float acc = 0.f;
                    for (int kk = 0; kk < K; ++kk) acc += hA[r*K + kk] * hB[((size_t) g*N + j)*K + kk];
                    ref[r*N + j] = acc;
                }
            }
        }

        void * dA, * dO; int * dM;
        cudaMalloc(&dA, bA.size()*2); cudaMalloc(&dO, rows*N*2); cudaMalloc(&dM, G*sizeof(int));
        cudaMemcpy(dA, bA.data(), bA.size()*2, cudaMemcpyHostToDevice);
        cudaMemcpy(dM, masked_m.data(), G*sizeof(int), cudaMemcpyHostToDevice);

        int total = 0; for (int g = 0; g < G; ++g) total += masked_m[g];
        printf("masked     G=%d max_m=%d (max rows/expert=%d) N=%d K=%d\n", G, max_m, max_rows, N, K);
        const int rc = ppu_moe_grouped_gemm_bf16_masked(dA, dB, dO, dM, max_m, N, K, G, total / G, nullptr);
        if (rc != 0) {
            printf("  ppu_moe_grouped_gemm_bf16_masked rc=%d (unsupported arch / JIT failed)\n", rc);
            all_ok = false;
        } else if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("  launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            all_ok = false;
        } else {
            std::vector<__nv_bfloat16> ho(rows * (size_t) N);
            cudaMemcpy(ho.data(), dO, rows*N*2, cudaMemcpyDeviceToHost);
            all_ok &= check("masked", ho, ref, N, real);
        }
        cudaFree(dA); cudaFree(dO); cudaFree(dM);
    }

    // ---------------- contiguous: each expert segment padded up to row_alignment() ----------------
    {
        const int ALIGN = ppu_moe_row_alignment();
        const int RPE_P = (RPE + ALIGN - 1) / ALIGN * ALIGN;
        const size_t M  = (size_t) G * RPE_P;

        std::vector<float> hA(M * K);
        for (size_t i = 0; i < hA.size(); ++i) hA[i] = rnd((int) i);
        auto bA = quantize(hA);

        std::vector<int>  mi(M);
        std::vector<char> real(M, 0);
        for (size_t i = 0; i < M; ++i) { mi[i] = (int) (i / RPE_P); real[i] = (i % RPE_P) < (size_t) RPE; }

        std::vector<float> ref(M * (size_t) N, 0.f);
        for (size_t i = 0; i < M; ++i) {
            if (!real[i]) continue;
            for (int j = 0; j < N; ++j) {
                float acc = 0.f;
                for (int kk = 0; kk < K; ++kk) acc += hA[i*K + kk] * hB[((size_t) mi[i]*N + j)*K + kk];
                ref[i*N + j] = acc;
            }
        }

        void * dA, * dO; int * dMI;
        cudaMalloc(&dA, bA.size()*2); cudaMalloc(&dO, M*N*2); cudaMalloc(&dMI, M*sizeof(int));
        cudaMemcpy(dA, bA.data(), bA.size()*2, cudaMemcpyHostToDevice);
        cudaMemcpy(dMI, mi.data(), M*sizeof(int), cudaMemcpyHostToDevice);

        printf("contiguous G=%d rows/expert=%d(->%d) M=%zu N=%d K=%d align=%d\n", G, RPE, RPE_P, M, N, K, ALIGN);
        const int rc = ppu_moe_grouped_gemm_bf16_contiguous(dA, dB, dO, dMI, (int) M, N, K, G, (int) M / G, nullptr);
        if (rc != 0) {
            printf("  ppu_moe_grouped_gemm_bf16_contiguous rc=%d (unsupported arch / JIT failed)\n", rc);
            all_ok = false;
        } else if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("  launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            all_ok = false;
        } else {
            std::vector<__nv_bfloat16> ho(M * (size_t) N);
            cudaMemcpy(ho.data(), dO, M*N*2, cudaMemcpyDeviceToHost);
            all_ok &= check("contiguous", ho, ref, N, real);
        }
        cudaFree(dA); cudaFree(dO); cudaFree(dMI);
    }

    cudaFree(dB);
    return all_ok ? 0 : 1;
}
