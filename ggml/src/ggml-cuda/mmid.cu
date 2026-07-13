#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

// ---------------------------------------------------------------------------------------------------------------
// Counting-sort implementation of the same mapping, for large token counts.
//
// mm_ids_helper above gives each expert one warp and has that warp scan the WHOLE ids tensor, so its work is
// O(n_experts * n_tokens * n_expert_used) and its runtime grows linearly with n_tokens. That is cheap when n_tokens
// is small -- it is a single kernel launch, which nothing here can beat -- but it degrades badly on long prompts.
//
// The counting sort below is O(n_tokens * n_expert_used) and essentially flat in n_tokens, at the cost of three
// kernel launches. Measured on an RTX 5090 (launch floor 2.46 us), producing BIT-IDENTICAL ids_src1 / ids_dst /
// expert_bounds:
//
//   n_experts=128 n_expert_used=8    n_tokens=   1     32    128    512   2048   4096
//     scan (above)                              2.5    2.5    4.4   10.3   36.9   71.7  us
//     counting sort (below)                     9.9   10.5   10.4   10.4   10.4   10.4  us
//   n_experts=8   n_expert_used=2    n_tokens= 512   2048
//     scan                                      6.3   18.5  us
//     counting sort                            10.3   10.4  us
//
// The crossover sits at n_tokens*n_expert_used ~= 4096, which is the switch used below. Above it the counting sort
// is 1.8x - 6.9x faster; below it the scan wins on launch count alone.
//
// It is a STABLE sort -- rows keep their (expert, token, slot) order -- so the outputs are bit-identical to the scan
// path, not merely equivalent. Getting that without atomics ordering the rows takes the standard three passes:
// per-block partial histogram, then an exclusive scan of each expert's counts over the blocks, then a scatter where
// each row's slot is expert_base + block_base + its rank among same-expert rows earlier in its own block.
// ---------------------------------------------------------------------------------------------------------------

#define MM_IDS_SORT_BLOCK 256
#define MM_IDS_SORT_MIN_ROWS 4096   // n_tokens*n_expert_used; see the table above

// Rows per (block, expert). Written expert-major per block so the scan below reads it with a stride.
static __global__ void mm_ids_sort_hist(
        const int32_t * __restrict__ ids, int32_t * __restrict__ block_counts,
        const int n_rows, const int n_experts, const int n_expert_used, const int si1) {
    extern __shared__ int32_t shist[];
    for (int e = threadIdx.x; e < n_experts; e += MM_IDS_SORT_BLOCK) {
        shist[e] = 0;
    }
    __syncthreads();

    const int i = blockIdx.x*MM_IDS_SORT_BLOCK + threadIdx.x;
    if (i < n_rows) {
        atomicAdd(&shist[ids[(i/n_expert_used)*si1 + i % n_expert_used]], 1);
    }
    __syncthreads();

    for (int e = threadIdx.x; e < n_experts; e += MM_IDS_SORT_BLOCK) {
        block_counts[(int64_t) blockIdx.x*n_experts + e] = shist[e];
    }
}

// One block per expert: turn that expert's per-block counts into exclusive prefix sums over the blocks, and record
// the expert's total.
static __global__ void mm_ids_sort_scan_blocks(
        int32_t * __restrict__ block_counts, int32_t * __restrict__ expert_counts,
        const int n_blocks, const int n_experts) {
    extern __shared__ int32_t sscan[];
    const int e = blockIdx.x;

    __shared__ int32_t running;
    if (threadIdx.x == 0) {
        running = 0;
    }
    __syncthreads();

    for (int b0 = 0; b0 < n_blocks; b0 += MM_IDS_SORT_BLOCK) {
        const int     b = b0 + threadIdx.x;
        const int32_t v = b < n_blocks ? block_counts[(int64_t) b*n_experts + e] : 0;

        sscan[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < MM_IDS_SORT_BLOCK; d <<= 1) {                 // Hillis-Steele, inclusive
            const int32_t x = threadIdx.x >= (unsigned) d ? sscan[threadIdx.x - d] : 0;
            __syncthreads();
            sscan[threadIdx.x] += x;
            __syncthreads();
        }

        if (b < n_blocks) {
            block_counts[(int64_t) b*n_experts + e] = running + sscan[threadIdx.x] - v;   // exclusive
        }
        __syncthreads();
        if (threadIdx.x == MM_IDS_SORT_BLOCK - 1) {
            running += sscan[MM_IDS_SORT_BLOCK - 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        expert_counts[e] = running;
    }
}

// One block: exclusive prefix sum of the per-expert totals -> expert_bounds[0..n_experts].
static __global__ void mm_ids_sort_scan_experts(
        const int32_t * __restrict__ expert_counts, int32_t * __restrict__ expert_bounds, const int n_experts) {
    extern __shared__ int32_t sscan[];
    const int e = threadIdx.x;

    sscan[e] = e < n_experts ? expert_counts[e] : 0;
    __syncthreads();
    for (int d = 1; d < (int) blockDim.x; d <<= 1) {                      // Hillis-Steele, inclusive
        const int32_t x = e >= d ? sscan[e - d] : 0;
        __syncthreads();
        sscan[e] += x;
        __syncthreads();
    }

    if (e < n_experts) {
        expert_bounds[e] = sscan[e] - expert_counts[e];                   // exclusive
    }
    if (e == n_experts - 1) {
        expert_bounds[n_experts] = sscan[e];
    }
}

// Scatter. Row i of the (token, slot) grid goes to
//     expert_bounds[e] + block_counts[block][e] + (its rank among same-expert rows earlier in its own block).
// The rank loop is what makes the sort stable, i.e. what makes the output bit-identical to the scan path.
static __global__ void mm_ids_sort_scatter(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ block_counts,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_rows, const int n_experts, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1) {
    __shared__ int32_t srow[MM_IDS_SORT_BLOCK];

    const int i = blockIdx.x*MM_IDS_SORT_BLOCK + threadIdx.x;
    const int e = i < n_rows ? ids[(i/n_expert_used)*si1 + i % n_expert_used] : -1;
    srow[threadIdx.x] = e;
    __syncthreads();

    if (i >= n_rows) {
        return;
    }

    int rank = 0;
    for (int j = 0; j < (int) threadIdx.x; ++j) {
        rank += srow[j] == e;
    }

    const int c  = expert_bounds[e] + block_counts[(int64_t) blockIdx.x*n_experts + e] + rank;
    const int it = i / n_expert_used;
    const int ix = i % n_expert_used;
    ids_src1[c] = it*sis1 + ix % nchannels_y;
    ids_dst [c] = i;                                                      // == it*n_expert_used + ix
}

static void launch_mm_ids_sort(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, ggml_cuda_pool & pool, cudaStream_t stream) {
    const int n_rows   = n_tokens*n_expert_used;
    const int n_blocks = (n_rows + MM_IDS_SORT_BLOCK - 1) / MM_IDS_SORT_BLOCK;

    int scan_threads = 1;
    while (scan_threads < n_experts) {
        scan_threads <<= 1;
    }

    ggml_cuda_pool_alloc<int32_t> block_counts (pool, (size_t) n_blocks*n_experts);
    ggml_cuda_pool_alloc<int32_t> expert_counts(pool, n_experts);

    mm_ids_sort_hist<<<n_blocks, MM_IDS_SORT_BLOCK, n_experts*sizeof(int32_t), stream>>>
        (ids, block_counts.get(), n_rows, n_experts, n_expert_used, si1);

    mm_ids_sort_scan_blocks<<<n_experts, MM_IDS_SORT_BLOCK, MM_IDS_SORT_BLOCK*sizeof(int32_t), stream>>>
        (block_counts.get(), expert_counts.get(), n_blocks, n_experts);

    mm_ids_sort_scan_experts<<<1, scan_threads, scan_threads*sizeof(int32_t), stream>>>
        (expert_counts.get(), expert_bounds, n_experts);

    mm_ids_sort_scatter<<<n_blocks, MM_IDS_SORT_BLOCK, 0, stream>>>
        (ids, expert_bounds, block_counts.get(), ids_src1, ids_dst,
         n_rows, n_experts, n_expert_used, nchannels_y, si1, sis1);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        ggml_cuda_pool & pool, cudaStream_t stream) {
    // See the comment above launch_mm_ids_sort: the scan path is a single launch and unbeatable while n_tokens is
    // small; the counting sort is flat in n_tokens and wins from ~4096 rows up. Both produce identical output.
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    if ((int64_t) n_tokens*n_expert_used >= MM_IDS_SORT_MIN_ROWS && n_experts <= 1024 &&
        2*n_experts*sizeof(int32_t) <= smpbo) {   // mm_ids_sort_hist stages one int per expert in shared
        launch_mm_ids_sort(ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, pool, stream);
        return;
    }

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
    }
}
