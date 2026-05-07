// Deterministic shape-invariant matmul.
//
// Replaces `rwkv::gemm_fp16_cublas` for the AC-bit-exact path: cuBLAS
// chooses different algorithms at different M (batch dim), producing
// fp16 outputs that drift by ~0.19 abs between M=1 and M=1024. That
// drift is fundamental to cuBLAS's shape-dependent algorithm choice and
// destroys AC roundtrip when encoder runs packed (M=N) and decoder runs
// stepped (M=1) — even tiny drift flips integer CDF bins at boundaries.
//
// This kernel computes y = x @ W with one thread per output element,
// each doing a sequential fp32 accumulation over the K dimension. The
// per-element arithmetic is identical regardless of M → output[m, n]
// bit-identical between M=1 and M=N. Bit-exact AC roundtrip restored.
//
// Speed: probably 0.1-0.5× cuBLAS for these shapes (no tensor cores,
// no shared-memory tiling). Correctness over speed for now.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// y[M, N] = x[M, K] @ W[K, N]
// Block grid: <<<dim3(M, ceil(N/BLOCK_N)), BLOCK_N>>>.
// blockIdx.x = m, blockIdx.y = n_chunk.
// threadIdx.x = n_within_chunk.
// Each thread computes y[m, n] independently with identical accumulation
// order regardless of M. Outputs are fp16 (default) or fp32 (if write_fp32).

template<int K_DIM>
__global__ void det_matmul_kernel(
    const __half* __restrict__ x,    // [M, K_DIM]
    const __half* __restrict__ W,    // [K_DIM, N]
    __half*       __restrict__ y_fp16,  // [M, N], used if write_fp32 == 0
    float*        __restrict__ y_fp32,  // [M, N], used if write_fp32 == 1
    int M, int N, int write_fp32)
{
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K_DIM; k++) {
        const float a = __half2float(x[m * K_DIM + k]);
        const float w = __half2float(W[k * N + n]);
        acc += a * w;
    }

    if (write_fp32) {
        y_fp32[m * N + n] = acc;
    } else {
        y_fp16[m * N + n] = __float2half(acc);
    }
}

// Generic-K version when K isn't one of our specialized values.
__global__ void det_matmul_kernel_generic(
    const __half* __restrict__ x,
    const __half* __restrict__ W,
    __half*       __restrict__ y_fp16,
    float*        __restrict__ y_fp32,
    int M, int K, int N, int write_fp32)
{
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const float a = __half2float(x[m * K + k]);
        const float w = __half2float(W[k * N + n]);
        acc += a * w;
    }

    if (write_fp32) {
        y_fp32[m * N + n] = acc;
    } else {
        y_fp16[m * N + n] = __float2half(acc);
    }
}

// Public launcher: y = x @ W, fp16 (or fp32) output.
extern "C" void launch_det_matmul(
    const void* x, const void* W,
    void* y, int write_fp32,
    int M, int K, int N, cudaStream_t stream)
{
    const int BLOCK_N = 32;
    dim3 grid(M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);

    if (K == 768) {
        det_matmul_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(W),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(y),
            write_fp32 ? reinterpret_cast<float*>(y) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(W),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(y),
            write_fp32 ? reinterpret_cast<float*>(y) : nullptr,
            M, N, write_fp32);
    } else {
        det_matmul_kernel_generic<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(W),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(y),
            write_fp32 ? reinterpret_cast<float*>(y) : nullptr,
            M, K, N, write_fp32);
    }
}
