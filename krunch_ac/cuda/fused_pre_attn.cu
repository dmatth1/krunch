// Fused LN1 + premix3 for the attention block entry — T=1 only.
//
// Replaces this chain in rwkv4_layer_step_cpp_t1 (decompress / B-batched
// stepped):
//   xx = at::layer_norm(x, {C}, ln1_w, ln1_b)
//   launch_premix_3(xx, att_xx, tm_k, tm_v, tm_r) -> kx, vx, rx
//
// One block per batch row (B blocks total at T=1). Each block:
//   1. Computes layer_norm of x[batch] in fp32 with per-row reduction
//   2. Computes kx/vx/rx = xx*tm_* + att_xx*(1-tm_*) elementwise
//   3. Writes xx (so caller can copy to att_xx for next step),
//      kx, vx, rx (inputs to next matmul).
//
// T>1 (compress packed) NOT supported here because the premix step
// reads xx[batch][t-1] which would cross-block race with this same
// kernel writing xx[batch][t]. Compress keeps using LN + premix3 as
// separate launches (matmul dominates compress, not these small ops).
//
// Bit-stability: per-row independent + fixed reduction order. Same
// bits regardless of B. Bit-pattern DIFFERS from the original
// at::layer_norm + launch_premix_3 chain because LN reduction order
// differs — both compress + decompress must use the SAME chain
// version. To keep AC roundtrip valid, this kernel can only be
// enabled if compress also runs LN+premix3 with bit-identical
// arithmetic (which the original separated kernels happen to do
// already if we implement compress's LN to match this kernel's
// reduction order). For now: enable on decompress only AND verify
// against the existing chain via a bit-equality unit test before
// production use.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BLOCK 256

__device__ __forceinline__ float warp_red_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template<int C_DIM>
__global__ void fused_pre_attn_kernel(
    const __half* __restrict__ x,        // [B, T, C_DIM]  (or flat [B*T, C])
    const __half* __restrict__ att_xx,   // [B, C_DIM]  prev-call last-token xx
    const __half* __restrict__ ln1_w,    // [C_DIM]
    const __half* __restrict__ ln1_b,    // [C_DIM]
    const __half* __restrict__ tm_k,     // [C_DIM]
    const __half* __restrict__ tm_v,     // [C_DIM]
    const __half* __restrict__ tm_r,     // [C_DIM]
    __half*       __restrict__ xx_out,   // [B, T, C_DIM]  layer-norm output
    __half*       __restrict__ kx_out,   // [B, T, C_DIM]
    __half*       __restrict__ vx_out,   // [B, T, C_DIM]
    __half*       __restrict__ rx_out,   // [B, T, C_DIM]
    int B, int T, float eps)
{
    const int row = blockIdx.x;          // 0..B*T-1
    if (row >= B * T) return;
    const int b = row / T;
    const int t = row % T;

    const __half* x_row = x + (size_t)row * C_DIM;
    __half* xx_row = xx_out + (size_t)row * C_DIM;
    __half* kx_row = kx_out + (size_t)row * C_DIM;
    __half* vx_row = vx_out + (size_t)row * C_DIM;
    __half* rx_row = rx_out + (size_t)row * C_DIM;

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = BLOCK >> 5;

    __shared__ float s_warp[8];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // ---- LN pass 1: row sum ----
    float my_sum = 0.0f;
    for (int i = tid; i < C_DIM; i += BLOCK)
        my_sum += __half2float(x_row[i]);
    float w = warp_red_sum(my_sum);
    if (lane == 0) s_warp[wid] = w;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp[lane] : 0.0f;
        v = warp_red_sum(v);
        if (lane == 0) s_mean = v / (float)C_DIM;
    }
    __syncthreads();
    const float mean = s_mean;

    // ---- LN pass 2: variance ----
    float my_sq = 0.0f;
    for (int i = tid; i < C_DIM; i += BLOCK) {
        const float d = __half2float(x_row[i]) - mean;
        my_sq += d * d;
    }
    w = warp_red_sum(my_sq);
    if (lane == 0) s_warp[wid] = w;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp[lane] : 0.0f;
        v = warp_red_sum(v);
        if (lane == 0) s_inv_std = rsqrtf(v / (float)C_DIM + eps);
    }
    __syncthreads();
    const float inv_std = s_inv_std;

    // ---- LN apply + write xx_out (consumed by next-token sx lookup) ----
    for (int i = tid; i < C_DIM; i += BLOCK) {
        const float xv = __half2float(x_row[i]);
        const float g  = __half2float(ln1_w[i]);
        const float bv = __half2float(ln1_b[i]);
        const float yv = (xv - mean) * inv_std * g + bv;
        xx_row[i] = __float2half(yv);
    }
    __syncthreads();

    // ---- Premix: kx, vx, rx = mix(xx, sx_full, tm_*) ----
    // sx_full = att_xx (shared per-batch state) when t==0, else xx[t-1]
    // from THIS forward pass's xx_out.
    for (int i = tid; i < C_DIM; i += BLOCK) {
        const float xx_v = __half2float(xx_row[i]);
        float sx_v;
        if (t == 0) {
            sx_v = __half2float(att_xx[b * C_DIM + i]);
        } else {
            // xx_out[(b * T + t-1) * C_DIM + i]
            sx_v = __half2float(xx_out[((size_t)b * T + (t - 1)) * C_DIM + i]);
        }
        const float kk = __half2float(tm_k[i]);
        const float vv = __half2float(tm_v[i]);
        const float rr = __half2float(tm_r[i]);
        kx_row[i] = __float2half(xx_v * kk + sx_v * (1.0f - kk));
        vx_row[i] = __float2half(xx_v * vv + sx_v * (1.0f - vv));
        rx_row[i] = __float2half(xx_v * rr + sx_v * (1.0f - rr));
    }
}

extern "C" void launch_fused_pre_attn(
    const void* x, const void* att_xx,
    const void* ln1_w, const void* ln1_b,
    const void* tm_k, const void* tm_v, const void* tm_r,
    void* xx_out, void* kx_out, void* vx_out, void* rx_out,
    int B, int T, int C, float eps, cudaStream_t stream)
{
    if (C == 768) {
        fused_pre_attn_kernel<768><<<B * T, BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(att_xx),
            reinterpret_cast<const __half*>(ln1_w),
            reinterpret_cast<const __half*>(ln1_b),
            reinterpret_cast<const __half*>(tm_k),
            reinterpret_cast<const __half*>(tm_v),
            reinterpret_cast<const __half*>(tm_r),
            reinterpret_cast<__half*>(xx_out),
            reinterpret_cast<__half*>(kx_out),
            reinterpret_cast<__half*>(vx_out),
            reinterpret_cast<__half*>(rx_out),
            B, T, eps);
    } else {
        // Generic-C kernel TBD; for RWKV-4-Pile-169M we only ever
        // call with C=768.
    }
}
