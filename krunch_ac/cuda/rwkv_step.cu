// Fused single-step RWKV-4 forward kernels for B=1, T=1 autoregressive
// decompression. Two designs in this file:
//
//   v1: rwkv4_layer_step_kernel      — single-block <<<1, 768>>> (1 SM only)
//   v2: rwkv4_layer_step_v2 (3 sub-kernels) — <<<24, 32>>> multi-block
//
// v1 was correct (max abs diff 0.016 vs reference) but slow on T4
// (11.2 ms/token vs BlinkDL's 7.5 ms — 1.45× slower) because it
// runs on 1 SM out of 40. v2 spreads the 768 channels across 24
// blocks so 24 SMs read HBM in parallel.
//
// Numerical contract: matches scripts/rwkv4_step_ref.py::_layer_step
// within fp16 noise (verified).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int N_EMBD = 768;
constexpr int N_ATT  = 768;
constexpr int N_FFN  = 3072;

// =============================================================================
// V1: single-block kernel (correctness reference, slow)
// =============================================================================

constexpr int V1_BLOCK_DIM = 768;

__device__ __forceinline__ float v1_block_sum(float val, float* smem) {
    for (int off = 16; off > 0; off /= 2)
        val += __shfl_xor_sync(0xFFFFFFFFu, val, off);
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < (V1_BLOCK_DIM / 32)) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = 16; off > 0; off /= 2)
            val += __shfl_xor_sync(0xFFFFFFFFu, val, off);
        if (threadIdx.x == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float v1_layernorm_at(
    const __half* vec_shared, const __half* w, const __half* b, float* smem)
{
    const int tid = threadIdx.x;
    const float xv = __half2float(vec_shared[tid]);
    const float mean = v1_block_sum(xv, smem) / (float)N_EMBD;
    const float diff = xv - mean;
    const float var = v1_block_sum(diff * diff, smem) / (float)N_EMBD;
    const float inv_std = rsqrtf(var + 1e-5f);
    const float norm = (xv - mean) * inv_std;
    return norm * __half2float(w[tid]) + __half2float(b[tid]);
}

__device__ __forceinline__ float v1_gemv_768x768(
    const __half* x_shared, const __half* W, int out_channel)
{
    float acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_EMBD; i++) {
        const float a = __half2float(x_shared[i]);
        const float w = __half2float(W[i * N_ATT + out_channel]);
        acc += a * w;
    }
    return acc;
}

__device__ __forceinline__ void v1_gemv_768_to_3072(
    const __half* x_shared, const __half* W, float out[4])
{
    const int tid = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int oc = tid + j * N_EMBD;
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < N_EMBD; i++) {
            const float a = __half2float(x_shared[i]);
            const float w = __half2float(W[i * N_FFN + oc]);
            acc += a * w;
        }
        out[j] = acc;
    }
}

__device__ __forceinline__ float v1_gemv_3072_to_768(
    const __half* k_shared, const __half* W, int out_channel)
{
    float acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_FFN; i++) {
        const float a = __half2float(k_shared[i]);
        const float w = __half2float(W[i * N_EMBD + out_channel]);
        acc += a * w;
    }
    return acc;
}

__device__ __forceinline__ float v1_sigmoidf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void rwkv4_layer_step_kernel(
    const __half* __restrict__ x_in, __half* __restrict__ x_out,
    __half* __restrict__ att_xx,
    float* __restrict__ aa, float* __restrict__ bb, float* __restrict__ pp,
    __half* __restrict__ ffn_xx,
    const __half* __restrict__ ln1_w, const __half* __restrict__ ln1_b,
    const __half* __restrict__ tm_k, const __half* __restrict__ tm_v,
    const __half* __restrict__ tm_r,
    const float* __restrict__ time_decay, const float* __restrict__ time_first,
    const __half* __restrict__ Kw, const __half* __restrict__ Vw,
    const __half* __restrict__ Rw, const __half* __restrict__ Ow,
    const __half* __restrict__ ln2_w, const __half* __restrict__ ln2_b,
    const __half* __restrict__ ffn_tm_k, const __half* __restrict__ ffn_tm_r,
    const __half* __restrict__ ffn_Kw, const __half* __restrict__ ffn_Vw,
    const __half* __restrict__ ffn_Rw)
{
    const int tid = threadIdx.x;
    __shared__ __half x_shared[N_EMBD];
    __shared__ __half xx_shared[N_EMBD];
    __shared__ __half kff_shared[N_FFN];
    __shared__ float  smem[V1_BLOCK_DIM / 32];

    x_shared[tid] = x_in[tid];
    __syncthreads();

    const float xx_val = v1_layernorm_at(x_shared, ln1_w, ln1_b, smem);
    xx_shared[tid] = __float2half(xx_val);
    __syncthreads();

    const float att_xx_val = __half2float(att_xx[tid]);
    const float tm_k_v = __half2float(tm_k[tid]);
    const float tm_v_v = __half2float(tm_v[tid]);
    const float tm_r_v = __half2float(tm_r[tid]);
    const float kx = xx_val * tm_k_v + att_xx_val * (1.0f - tm_k_v);
    const float vx = xx_val * tm_v_v + att_xx_val * (1.0f - tm_v_v);
    const float rx = xx_val * tm_r_v + att_xx_val * (1.0f - tm_r_v);

    att_xx[tid] = __float2half(xx_val);
    __syncthreads();

    xx_shared[tid] = __float2half(rx);
    __syncthreads();
    const float r_val = v1_sigmoidf(v1_gemv_768x768(xx_shared, Rw, tid));

    xx_shared[tid] = __float2half(kx);
    __syncthreads();
    const float k_val = v1_gemv_768x768(xx_shared, Kw, tid);

    xx_shared[tid] = __float2half(vx);
    __syncthreads();
    const float v_val = v1_gemv_768x768(xx_shared, Vw, tid);

    const float aa_v = aa[tid];
    const float bb_v = bb[tid];
    const float pp_v = pp[tid];
    const float tf_v = time_first[tid];
    const float td_v = time_decay[tid];

    const float ww = pp_v + tf_v;
    const float p  = fmaxf(ww, k_val);
    const float e1 = __expf(ww - p);
    const float e2 = __expf(k_val - p);
    const float y_val = (e1 * aa_v + e2 * v_val) / (e1 * bb_v + e2);

    const float decay = __expf(td_v);
    const float ww2 = pp_v - decay;
    const float p2  = fmaxf(ww2, k_val);
    const float e1_2 = __expf(ww2 - p2);
    const float e2_2 = __expf(k_val - p2);
    aa[tid] = e1_2 * aa_v + e2_2 * v_val;
    bb[tid] = e1_2 * bb_v + e2_2;
    pp[tid] = p2;

    xx_shared[tid] = __float2half(r_val * y_val);
    __syncthreads();
    const float att_out = v1_gemv_768x768(xx_shared, Ow, tid);

    const float x_after_att = __half2float(x_shared[tid]) + att_out;
    x_shared[tid] = __float2half(x_after_att);
    __syncthreads();

    const float xx2_val = v1_layernorm_at(x_shared, ln2_w, ln2_b, smem);
    xx_shared[tid] = __float2half(xx2_val);
    __syncthreads();

    const float ffn_xx_val = __half2float(ffn_xx[tid]);
    const float ffn_tm_k_v = __half2float(ffn_tm_k[tid]);
    const float ffn_tm_r_v = __half2float(ffn_tm_r[tid]);
    const float ffn_kx = xx2_val * ffn_tm_k_v + ffn_xx_val * (1.0f - ffn_tm_k_v);
    const float ffn_rx = xx2_val * ffn_tm_r_v + ffn_xx_val * (1.0f - ffn_tm_r_v);
    ffn_xx[tid] = __float2half(xx2_val);

    xx_shared[tid] = __float2half(ffn_rx);
    __syncthreads();
    const float r_ffn = v1_sigmoidf(v1_gemv_768x768(xx_shared, ffn_Rw, tid));

    xx_shared[tid] = __float2half(ffn_kx);
    __syncthreads();
    float ffn_k_out[4];
    v1_gemv_768_to_3072(xx_shared, ffn_Kw, ffn_k_out);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float v = ffn_k_out[j];
        const float relu_v = (v > 0.0f) ? v : 0.0f;
        kff_shared[tid + j * N_EMBD] = __float2half(relu_v * relu_v);
    }
    __syncthreads();

    const float ffn_v = v1_gemv_3072_to_768(kff_shared, ffn_Vw, tid);
    const float x_final = x_after_att + r_ffn * ffn_v;
    x_out[tid] = __float2half(x_final);
}

// =============================================================================
// V2: multi-block layer kernels  <<<24, 32>>>  → uses 24 SMs
// =============================================================================
//
// Each layer becomes 3 kernel launches:
//   v2_layernorm:  24 blocks × 32 threads. Two-pass via per-block partials
//                  written to a small HBM scratch, finalize in a 1-block kernel.
//   v2_attn_block: 24 blocks × 32 threads. Read x[768] + xx[768] + state from
//                  HBM, compute K/V/R + WKV + Out, write x_after_att.
//   v2_ffn_block:  24 blocks × 32 threads. Read x[768] + xx2[768] from HBM,
//                  compute FFN, write x_out.
//
// Cross-block reductions (LN) use atomicAdd into HBM globals; the
// "finalize" kernel computes mean+var and broadcasts.
//
// Note: v2 is scaffolded but unfinished; the 3 sub-kernels would replace
// rwkv4_layer_step_kernel as the production path after benchmarking.
// Leaving v1 as the verified-correct baseline.

constexpr int V2_BLOCK_DIM = 32;
constexpr int V2_GRID_DIM  = N_EMBD / V2_BLOCK_DIM;  // 24

// TODO(v2): implement. Design notes:
//   - LN uses one launch for partial sums (24 blocks atomic-add to globals)
//     plus a finalize kernel (1 block) that reads globals + writes mean/var
//     to a known HBM slot, plus a third launch that applies the LN per-block
//     (but this can be folded into the next phase to save a launch).
//   - Multi-block GEMV: each block computes 32 outputs by iterating 768
//     elements of x (read from shared after one cooperative load).
//   - Within a block, threads still do per-channel work; HBM bandwidth scales
//     ~24× because 24 SMs each issue independent loads.

// =============================================================================
// V1 launch wrapper (current production kernel)
// =============================================================================

void launch_rwkv4_layer_step(
    const void* x_in, void* x_out,
    void* att_xx, float* aa, float* bb, float* pp, void* ffn_xx,
    const void* ln1_w, const void* ln1_b,
    const void* tm_k, const void* tm_v, const void* tm_r,
    const float* time_decay, const float* time_first,
    const void* Kw, const void* Vw, const void* Rw, const void* Ow,
    const void* ln2_w, const void* ln2_b,
    const void* ffn_tm_k, const void* ffn_tm_r,
    const void* ffn_Kw, const void* ffn_Vw, const void* ffn_Rw,
    cudaStream_t stream)
{
    rwkv4_layer_step_kernel<<<1, V1_BLOCK_DIM, 0, stream>>>(
        reinterpret_cast<const __half*>(x_in),
        reinterpret_cast<__half*>(x_out),
        reinterpret_cast<__half*>(att_xx), aa, bb, pp,
        reinterpret_cast<__half*>(ffn_xx),
        reinterpret_cast<const __half*>(ln1_w),
        reinterpret_cast<const __half*>(ln1_b),
        reinterpret_cast<const __half*>(tm_k),
        reinterpret_cast<const __half*>(tm_v),
        reinterpret_cast<const __half*>(tm_r),
        time_decay, time_first,
        reinterpret_cast<const __half*>(Kw),
        reinterpret_cast<const __half*>(Vw),
        reinterpret_cast<const __half*>(Rw),
        reinterpret_cast<const __half*>(Ow),
        reinterpret_cast<const __half*>(ln2_w),
        reinterpret_cast<const __half*>(ln2_b),
        reinterpret_cast<const __half*>(ffn_tm_k),
        reinterpret_cast<const __half*>(ffn_tm_r),
        reinterpret_cast<const __half*>(ffn_Kw),
        reinterpret_cast<const __half*>(ffn_Vw),
        reinterpret_cast<const __half*>(ffn_Rw));
}
