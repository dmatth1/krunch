// Custom row-wise LayerNorm. Bit-stable across batch (each row's
// computation is independent + uses a fixed parallel reduction
// pattern). Replaces at::layer_norm in the layer-step hot path:
// at::layer_norm goes through the PyTorch dispatcher + may pick
// Welford / shape-dependent reduction algos internally — slow + a
// shape-stability risk for AC roundtrip.
//
// Per row of size C:
//   m = sum(x) / C
//   v = sum((x - m)^2) / C
//   y = (x - m) / sqrt(v + eps) * gamma + beta
//
// Inputs are fp16; reductions in fp32; output fp16. Same as ATen's
// fp16 layer_norm semantics. BLOCK = 256 threads (one row per block).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BLOCK 256

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

template<int C_DIM>
__global__ void layer_norm_kernel(
    const __half* __restrict__ x,      // [N, C_DIM]
    const __half* __restrict__ gamma,  // [C_DIM]
    const __half* __restrict__ beta,   // [C_DIM]
    __half*       __restrict__ y,      // [N, C_DIM]
    int N, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const __half* x_row = x + (size_t)row * C_DIM;
    __half* y_row = y + (size_t)row * C_DIM;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = BLOCK >> 5;

    __shared__ float s_warp_sum[8];
    __shared__ float s_warp_sumsq[8];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // ---- pass 1: row sum ----
    float my_sum = 0.0f;
    for (int i = tid; i < C_DIM; i += BLOCK) {
        my_sum += __half2float(x_row[i]);
    }
    float w_sum = warp_reduce_sum(my_sum);
    if (lane == 0) s_warp_sum[wid] = w_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_mean = v / (float)C_DIM;
    }
    __syncthreads();
    const float mean = s_mean;

    // ---- pass 2: row variance ----
    float my_sumsq = 0.0f;
    for (int i = tid; i < C_DIM; i += BLOCK) {
        const float d = __half2float(x_row[i]) - mean;
        my_sumsq += d * d;
    }
    float w_sumsq = warp_reduce_sum(my_sumsq);
    if (lane == 0) s_warp_sumsq[wid] = w_sumsq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_sumsq[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_inv_std = rsqrtf(v / (float)C_DIM + eps);
    }
    __syncthreads();
    const float inv_std = s_inv_std;

    // ---- pass 3: normalize + affine ----
    for (int i = tid; i < C_DIM; i += BLOCK) {
        const float xv = __half2float(x_row[i]);
        const float g  = __half2float(gamma[i]);
        const float b  = __half2float(beta[i]);
        const float yv = (xv - mean) * inv_std * g + b;
        y_row[i] = __float2half(yv);
    }
}

// Generic-C fallback for unusual shapes.
__global__ void layer_norm_kernel_generic(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half*       __restrict__ y,
    int N, int C, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const __half* x_row = x + (size_t)row * C;
    __half* y_row = y + (size_t)row * C;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = BLOCK >> 5;

    __shared__ float s_warp_sum[8];
    __shared__ float s_warp_sumsq[8];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    float my_sum = 0.0f;
    for (int i = tid; i < C; i += BLOCK) my_sum += __half2float(x_row[i]);
    float w_sum = warp_reduce_sum(my_sum);
    if (lane == 0) s_warp_sum[wid] = w_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_mean = v / (float)C;
    }
    __syncthreads();
    const float mean = s_mean;

    float my_sumsq = 0.0f;
    for (int i = tid; i < C; i += BLOCK) {
        const float d = __half2float(x_row[i]) - mean;
        my_sumsq += d * d;
    }
    float w_sumsq = warp_reduce_sum(my_sumsq);
    if (lane == 0) s_warp_sumsq[wid] = w_sumsq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_sumsq[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_inv_std = rsqrtf(v / (float)C + eps);
    }
    __syncthreads();
    const float inv_std = s_inv_std;

    for (int i = tid; i < C; i += BLOCK) {
        const float xv = __half2float(x_row[i]);
        const float g  = __half2float(gamma[i]);
        const float b  = __half2float(beta[i]);
        const float yv = (xv - mean) * inv_std * g + b;
        y_row[i] = __float2half(yv);
    }
}

extern "C" void launch_layer_norm(
    const void* x, const void* gamma, const void* beta, void* y,
    int N, int C, float eps, cudaStream_t stream)
{
    auto a = reinterpret_cast<const __half*>(x);
    auto g = reinterpret_cast<const __half*>(gamma);
    auto b = reinterpret_cast<const __half*>(beta);
    auto o = reinterpret_cast<__half*>(y);
    if (C == 768) {
        layer_norm_kernel<768><<<N, BLOCK, 0, stream>>>(a, g, b, o, N, eps);
    } else if (C == 3072) {
        layer_norm_kernel<3072><<<N, BLOCK, 0, stream>>>(a, g, b, o, N, eps);
    } else {
        layer_norm_kernel_generic<<<N, BLOCK, 0, stream>>>(a, g, b, o, N, C, eps);
    }
}
