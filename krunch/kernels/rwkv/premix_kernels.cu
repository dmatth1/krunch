// Fused elementwise kernels for the C++ packed forward path. These
// replace 3-7 separate ATen ops per layer that each pay launch overhead.
// Goal: close the 1.55× compress regression of C++ packed vs BlinkDL
// packed (which uses torchscript-fused elementwise ops between gemms).
//
// Numerical contract: bit-equivalent to the unfused ATen sequence at
// fp16 precision (verified via test_premix_kernels.py).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// premix_3: computes kx, vx, rx for time-mix in ONE kernel pass.
// kx = xx * tm_k + sx_full * (1 - tm_k)   — same for vx (with tm_v) and rx (with tm_r)
// where sx_full[t=0] = prev_state, sx_full[t>0] = xx[t-1].
// Replaces: at::cat + 3 × (mul + mul + add) = ~7 ATen ops.
// =============================================================================

__global__ void premix_3_kernel(
    const __half* __restrict__ xx,          // [B*T, C]
    const __half* __restrict__ prev_state,  // [B, C]
    const __half* __restrict__ tm_k,        // [C]
    const __half* __restrict__ tm_v,        // [C]
    const __half* __restrict__ tm_r,        // [C]
    __half* __restrict__ kx,                // [B*T, C]
    __half* __restrict__ vx,                // [B*T, C]
    __half* __restrict__ rx,                // [B*T, C]
    int B, int T, int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * C;
    if (idx >= total) return;

    const int bt = idx / C;
    const int c = idx % C;
    const int b = bt / T;
    const int t = bt % T;

    const float xx_v = __half2float(xx[idx]);
    float sx_v;
    if (t == 0) {
        sx_v = __half2float(prev_state[b * C + c]);
    } else {
        sx_v = __half2float(xx[(b * T + (t - 1)) * C + c]);
    }

    const float kk = __half2float(tm_k[c]);
    const float vv = __half2float(tm_v[c]);
    const float rr = __half2float(tm_r[c]);

    kx[idx] = __float2half(xx_v * kk + sx_v * (1.0f - kk));
    vx[idx] = __float2half(xx_v * vv + sx_v * (1.0f - vv));
    rx[idx] = __float2half(xx_v * rr + sx_v * (1.0f - rr));
}

// =============================================================================
// premix_2: same pattern but 2 outputs (FFN premix: kx and rx only).
// =============================================================================

__global__ void premix_2_kernel(
    const __half* __restrict__ xx,
    const __half* __restrict__ prev_state,
    const __half* __restrict__ tm_k,
    const __half* __restrict__ tm_r,
    __half* __restrict__ kx,
    __half* __restrict__ rx,
    int B, int T, int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * C;
    if (idx >= total) return;

    const int bt = idx / C;
    const int c = idx % C;
    const int b = bt / T;
    const int t = bt % T;

    const float xx_v = __half2float(xx[idx]);
    float sx_v;
    if (t == 0) {
        sx_v = __half2float(prev_state[b * C + c]);
    } else {
        sx_v = __half2float(xx[(b * T + (t - 1)) * C + c]);
    }

    const float kk = __half2float(tm_k[c]);
    const float rr = __half2float(tm_r[c]);

    kx[idx] = __float2half(xx_v * kk + sx_v * (1.0f - kk));
    rx[idx] = __float2half(xx_v * rr + sx_v * (1.0f - rr));
}

// =============================================================================
// relu_sq: in-place x[i] = relu(x[i])^2. Fuses at::relu + .pow(2).
// =============================================================================

__global__ void relu_sq_kernel(__half* __restrict__ x, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = __half2float(x[idx]);
    const float r = (v > 0.0f) ? v : 0.0f;
    x[idx] = __float2half(r * r);
}

// =============================================================================
// Launch wrappers (called from main.cpp; void* lets us avoid pulling
// cuda_fp16.h into the .cpp side).
// =============================================================================

extern "C" {

void launch_premix_3(
    const void* xx, const void* prev_state,
    const void* tm_k, const void* tm_v, const void* tm_r,
    void* kx, void* vx, void* rx,
    int B, int T, int C, cudaStream_t stream)
{
    const int total = B * T * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    premix_3_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(xx),
        reinterpret_cast<const __half*>(prev_state),
        reinterpret_cast<const __half*>(tm_k),
        reinterpret_cast<const __half*>(tm_v),
        reinterpret_cast<const __half*>(tm_r),
        reinterpret_cast<__half*>(kx),
        reinterpret_cast<__half*>(vx),
        reinterpret_cast<__half*>(rx),
        B, T, C);
}

void launch_premix_2(
    const void* xx, const void* prev_state,
    const void* tm_k, const void* tm_r,
    void* kx, void* rx,
    int B, int T, int C, cudaStream_t stream)
{
    const int total = B * T * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    premix_2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(xx),
        reinterpret_cast<const __half*>(prev_state),
        reinterpret_cast<const __half*>(tm_k),
        reinterpret_cast<const __half*>(tm_r),
        reinterpret_cast<__half*>(kx),
        reinterpret_cast<__half*>(rx),
        B, T, C);
}

void launch_relu_sq(void* x, int n, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    relu_sq_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(x), n);
}

}  // extern "C"
