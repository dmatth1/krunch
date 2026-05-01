// Multi-block GEMV experiment: y[N] = x[K] @ W[K, N] (W stored row-major).
// Launch <<<N/32, 32>>> so each block handles 32 contiguous output channels.
// Goal: beat V1's single-block GEMV by spreading across SMs.
//
// W[i, j] = W[i * N + j], row-major. Each warp's threads read 32
// contiguous half values per row → 1 cache line per row read =
// perfect coalescing. Multi-block means 24 SMs each issue independent
// memory transactions, increasing HBM utilization.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<int K_DIM, int N_DIM>
__global__ void mb_gemv_kernel(
    const __half* __restrict__ x,    // [K_DIM]
    const __half* __restrict__ W,    // [K_DIM, N_DIM]
    __half*       __restrict__ y)    // [N_DIM]
{
    const int chunk = blockIdx.x;
    const int local_tid = threadIdx.x;
    const int out_c = chunk * 32 + local_tid;
    if (out_c >= N_DIM) return;

    __shared__ __half x_shared[K_DIM];
    // Cooperative load of x into shared (32 threads, K_DIM elements).
    #pragma unroll 4
    for (int i = local_tid; i < K_DIM; i += 32) {
        x_shared[i] = x[i];
    }
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < K_DIM; i++) {
        const float a = __half2float(x_shared[i]);
        const float w = __half2float(W[i * N_DIM + out_c]);
        acc += a * w;
    }
    y[out_c] = __float2half(acc);
}

extern "C" {

// Specializations for the four shapes used in RWKV-4-Pile-169M:
//   768 → 768   (Kw, Vw, Rw, Ow, ffn_Rw)
//   768 → 3072  (ffn_Kw)
//   3072 → 768  (ffn_Vw)

void launch_mb_gemv_768x768(const void* x, const void* W, void* y, cudaStream_t s) {
    mb_gemv_kernel<768, 768><<<768/32, 32, 0, s>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(W),
        reinterpret_cast<__half*>(y));
}

void launch_mb_gemv_768x3072(const void* x, const void* W, void* y, cudaStream_t s) {
    mb_gemv_kernel<768, 3072><<<3072/32, 32, 0, s>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(W),
        reinterpret_cast<__half*>(y));
}

void launch_mb_gemv_3072x768(const void* x, const void* W, void* y, cudaStream_t s) {
    mb_gemv_kernel<3072, 768><<<768/32, 32, 0, s>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(W),
        reinterpret_cast<__half*>(y));
}

}  // extern "C"
