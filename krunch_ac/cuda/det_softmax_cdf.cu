// Deterministic batched softmax + CDF kernel.
//
// Per-row computation is bit-identical regardless of how many rows
// are in the batch, so encoder calling on [T, V] and decoder calling
// on [1, V] produce byte-identical CDFs.
//
// Replaces the per-row Python loop in cpp_path.softmax_cdfs_per_row,
// which was the bottleneck of the encoder (~8K iterations per 32 KB
// chunk).
//
// Pipeline per row (one block):
//   1. block-reduce max(logits)
//   2. compute exp(x - max), block-reduce sum
//   3. p = exp / sum
//   4. count = floor(p * (CDF_T - V)) + 1   (V is vocab size, MIN_PROB=1)
//   5. block-reduce argmax(p) and sum(count)
//   6. count[argmax] += (CDF_T - sum)        (deficit fix)
//   7. cumsum across V → CDF[1..V]; CDF[0] = 0
//
// Each thread handles ELEMS_PER_THREAD = ceil(V / BLOCK) elements.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BLOCK 256

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, off);
        v = fmaxf(v, other);
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}
__device__ __forceinline__ int warp_reduce_argmax(float v, int idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float other_v = __shfl_xor_sync(0xffffffffu, v, off);
        int   other_i = __shfl_xor_sync(0xffffffffu, idx, off);
        if (other_v > v || (other_v == v && other_i < idx)) {
            v = other_v;
            idx = other_i;
        }
    }
    return idx;
}
__device__ __forceinline__ int warp_reduce_sum_int(int v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// One block per row.
// logits: [T, V] fp16
// cdf:    [T, V+1] int32  (cdf[t,0]=0, cdf[t,V]=CDF_T)
extern "C" __global__ void det_softmax_cdf_kernel(
    const __half* __restrict__ logits,
    int32_t* __restrict__ cdf,
    int V,
    int CDF_T)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = BLOCK >> 5;

    const __half* row_logits = logits + (size_t)row * V;
    int32_t* row_cdf = cdf + (size_t)row * (V + 1);

    // Per-block scratch
    __shared__ float s_warp_max[8];     // up to 8 warps
    __shared__ float s_warp_sum[8];
    __shared__ int   s_warp_amax[8];
    __shared__ float s_warp_amax_v[8];
    __shared__ int   s_warp_csum[8];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ int   s_argmax;
    __shared__ int   s_count_sum;

    // ---------- 1) max ----------
    float my_max = -INFINITY;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row_logits[i]);
        if (v > my_max) my_max = v;
    }
    float w_max = warp_reduce_max(my_max);
    if (lane == 0) s_warp_max[wid] = w_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_max[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) s_max = v;
    }
    __syncthreads();
    const float row_max = s_max;

    // ---------- 2) sum of exp ----------
    float my_sum = 0.0f;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row_logits[i]);
        my_sum += expf(v - row_max);
    }
    float w_sum = warp_reduce_sum(my_sum);
    if (lane == 0) s_warp_sum[wid] = w_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_sum;

    // ---------- 3) compute counts and reduce argmax + sum ----------
    // We need argmax of p (== argmax of logits, but tie-breaking matches
    // the python probs_to_cdf_gpu which uses torch.argmax — we'll use
    // smallest-index tiebreak which torch usually does).
    // Also accumulate sum(count) so we can apply deficit.
    int my_argmax = -1;
    float my_argmax_v = -INFINITY;
    int my_count_sum = 0;

    // Pass A: write the *count* values to CDF[t, i+1] (we'll cumsum later).
    // count[i] = floor( p[i] * (CDF_T - V) ) + 1     where p[i] = exp(x-m) * inv_sum
    const float scale = (float)(CDF_T - V);
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row_logits[i]);
        float p = expf(v - row_max) * inv_sum;
        int   c = (int)floorf(p * scale) + 1;  // MIN_PROB=1
        row_cdf[i + 1] = c;
        my_count_sum += c;
        if (p > my_argmax_v || (p == my_argmax_v && i < my_argmax)) {
            my_argmax_v = p;
            my_argmax = i;
        }
    }
    int w_amax = warp_reduce_argmax(my_argmax_v, my_argmax);
    int w_csum = warp_reduce_sum_int(my_count_sum);
    if (lane == 0) {
        s_warp_amax[wid] = w_amax;
        // Carry the value too so the final reduce can break ties.
        // Reload value at the chosen idx (cheap — single thread).
        // Use idx<V to avoid reading garbage when V is small.
        int idx = w_amax;
        s_warp_amax_v[wid] = (idx >= 0 && idx < V)
            ? expf(__half2float(row_logits[idx]) - row_max) * inv_sum
            : -INFINITY;
        s_warp_csum[wid] = w_csum;
    }
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_amax_v[lane] : -INFINITY;
        int idx = (lane < n_warps) ? s_warp_amax[lane]   : -1;
        idx = warp_reduce_argmax(v, idx);
        int csum = (lane < n_warps) ? s_warp_csum[lane] : 0;
        csum = warp_reduce_sum_int(csum);
        if (lane == 0) {
            s_argmax = idx;
            s_count_sum = csum;
        }
    }
    __syncthreads();

    // ---------- 4) deficit fix ----------
    if (tid == 0) {
        int deficit = CDF_T - s_count_sum;
        row_cdf[s_argmax + 1] += deficit;
        row_cdf[0] = 0;
    }
    __syncthreads();

    // NOTE: cumsum is done by the caller (torch.cumsum on the output
    // slice). Earlier we did a serial cumsum-by-thread-0 here, but
    // that was dramatically slower than torch's tuned scan
    // (1.9 ms → 0.1 ms per row at V=50277 on T4). The kernel writes
    // counts; caller scans.
}

extern "C" void launch_det_softmax_cdf(
    const void* logits, void* cdf,
    int T, int V, int cdf_T_value, cudaStream_t stream)
{
    det_softmax_cdf_kernel<<<T, BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(logits),
        reinterpret_cast<int32_t*>(cdf),
        V, cdf_T_value);
}
