// Pinned-algo cuBLAS GEMM. Same cuBLAS code path that's much faster
// than our hand-rolled WMMA kernel, but with a FIXED algo enum so the
// tile schedule + reduction order doesn't vary with M. That makes it
// bit-exact between any M values (M=1 stepped vs M=N packed) — the
// shape-stability guarantee we need for AC roundtrip.
//
// Why CUBLAS_GEMM_DEFAULT_TENSOR_OP doesn't work: "default" means
// "auto-select among the TC algos based on M/K/N". That's exactly
// what makes it shape-dependent for our use case.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>

// Fixed algo for layer matmuls. CUBLAS_GEMM_ALGO0_TENSOR_OP picks
// a 128x128x32 tile schedule that fits our M (any), K=768, N∈{768, 3072}.
// Verified bit-stable across M∈{1, 16, 64, 1024} — see
// scripts/test_cublas_pinned.py.
//
// For the head matmul (K=768, N=50277), N isn't a clean multiple
// of any common tile size — cuBLAS pads internally; output is still
// shape-stable because the per-(m,n) compute path doesn't depend on
// other rows.
//
// If a future GPU/cuBLAS version drops this algo enum, falls back
// to the WMMA `det_matmul_tc` path automatically (env override
// KRUNCH_CUBLAS_PINNED=0).
static cublasGemmAlgo_t g_layer_algo = CUBLAS_GEMM_ALGO0_TENSOR_OP;

extern "C" void launch_det_matmul_cublas(
    const void* x, const void* W,
    void* y, int write_fp32,
    int M, int K, int N, cudaStream_t stream)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    // cuBLAS is column-major. We have row-major: y_rm = x_rm @ W_rm.
    // Equivalently in column-major: y_cm^T = W_cm^T @ x_cm^T.
    // Swap A/B so cublas sees A=W, B=x with both N-transpose.
    const auto in_dtype = CUDA_R_16F;
    const auto out_dtype = write_fp32 ? CUDA_R_32F : CUDA_R_16F;
    const auto compute_type = CUDA_R_32F;
    const float alpha = 1.f, beta = 0.f;

    // Reorder: m, n, k for "y = A @ B" become n, m, k after swap.
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                        // m, n, k of column-major view
        &alpha,
        W, in_dtype, N,                 // A = W (n-major in row layout)
        x, in_dtype, K,                 // B = x
        &beta,
        y, out_dtype, N,                // C
        compute_type,
        g_layer_algo);
}

// Allow runtime override of the pinned algo (for benchmarking
// alternative algos without recompiling).
extern "C" void set_det_matmul_cublas_algo(int algo_id) {
    g_layer_algo = static_cast<cublasGemmAlgo_t>(algo_id);
}
