// W8A8 int8-Tensor-Core matmul, per-output-channel weight scale +
// per-row activation scale. 64×64 output tile, 4 warps, mirroring
// det_matmul_tc_mw schedule for M-invariance.
//
// Inputs:
//   X_q:     int8 [M, K]            row-quantized activations
//   scale_x: fp16 [M]               per-row activation scale
//   W_q:     int8 [K, N]            per-output-channel weight (signed)
//   scale_w: fp16 [N]               per-output-channel weight scale
//
// Output (fp16 or fp32):
//   Y[m, n] = scale_x[m] * scale_w[n] * sum_k(X_q[m, k] * W_q[k, n])
//
// The inner WMMA is pure int8 × int8 → int32 accumulator. K-loop is
// fixed-order, output tile is fixed-position → bit-stable across M
// (encoder M=large packed, decoder M=B stepped both produce identical
// output rows). Per-row scale_x is computed online by the upstream
// `quantize_per_row_int8` kernel, so it depends ONLY on the row's
// fp16 values — same encoder/decoder fp16 input → same scale → same
// quantized values → same WMMA result → same output. Bit-exact AC
// roundtrip preserved (verified by integration tests).
//
// Compile guards: int8 WMMA available since sm_72 (Volta tensor cores
// 2nd gen / Turing). Both T4 (sm_75) and A10G (sm_86) qualify. No
// cp.async path here; this is the basic int8 kernel. A future
// _async variant can be added if benchmarks show launch latency room.
//
// Shared mem per block: As[64*16] int8 + Bs[16*64] int8 +
// scale_x_s[64] fp16 + scale_w_s[64] fp16 + Cs[64*64] int32
// = 1KB + 1KB + 0.13KB + 0.13KB + 16KB ≈ 18 KB / block. Fits 2 active
// blocks/SM on T4 (64 KB shared) and on A10G (100 KB shared).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;


template<int K_DIM>
__global__ void det_matmul_w8a8_tc_kernel(
    const int8_t* __restrict__ X_q,         // [M, K_DIM] row-major int8
    const __half* __restrict__ scale_x,     // [M] fp16
    const int8_t* __restrict__ W_q,         // [K_DIM, N] row-major int8
    const __half* __restrict__ scale_w,     // [N] fp16
    __half*       __restrict__ Y16,         // [M, N], used if write_fp32 == 0
    float*        __restrict__ Y32,         // [M, N], used if write_fp32 == 1
    int M, int N, int write_fp32)
{
    const int m_tile_start = blockIdx.x * 64;
    const int n_tile_start = blockIdx.y * 64;

    const int tid = threadIdx.x;            // 0..127
    const int warp_id = tid >> 5;           // 0..3
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;        // 0 or 1
    const int warp_n = warp_id & 1;
    const int warp_m_off = warp_m * 32;
    const int warp_n_off = warp_n * 32;

    // 4 int32 accumulator frags per warp (2×2 grid of 16×16 = 32×32 sub-tile).
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    __shared__ int8_t As[64 * 16];     // 1 KB
    __shared__ int8_t Bs[16 * 64];     // 1 KB
    __shared__ int32_t Cs[64 * 64];    // 16 KB

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += 16) {
        // Load A tile [64, 16] int8 — 1024 bytes, 8/thread.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 4;
            const int col = idx & 15;
            const int gm = m_tile_start + row;
            int8_t v = (gm < M) ? X_q[gm * K_DIM + (k + col)] : (int8_t)0;
            As[row * 16 + col] = v;
        }
        // Load B tile [16, 64] int8 — 1024 bytes, 8/thread.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 6;
            const int col = idx & 63;
            const int gn = n_tile_start + col;
            int8_t v = (gn < N) ? W_q[(k + row) * N + gn] : (int8_t)0;
            Bs[row * 64 + col] = v;
        }
        __syncthreads();

        // 4 WMMA mma_sync per warp.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &As[(warp_m_off + i * 16) * 16], 16);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &Bs[(warp_n_off + j * 16)], 64);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Store int32 frags to shared mem at sub-tile location.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int32_t* dst = &Cs[(warp_m_off + i * 16) * 64 + (warp_n_off + j * 16)];
            wmma::store_matrix_sync(dst, acc[i][j], 64, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Coalesced global write with per-row × per-col scale applied.
    // Y[m, n] = scale_x[m] * scale_w[n] * Cs[m, n]
    // 64*64 = 4096 / 128 = 32 elem/thread.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int idx = i * 128 + tid;
        const int row = idx >> 6;
        const int col = idx & 63;
        const int gm = m_tile_start + row;
        const int gn = n_tile_start + col;
        if (gm < M && gn < N) {
            const float acc_f = (float)Cs[row * 64 + col];
            const float sx = __half2float(scale_x[gm]);
            const float sw = __half2float(scale_w[gn]);
            const float v = acc_f * sx * sw;
            if (write_fp32) {
                Y32[gm * N + gn] = v;
            } else {
                Y16[gm * N + gn] = __float2half(v);
            }
        }
    }
}


extern "C" void launch_det_matmul_w8a8_tc(
    const void* X_q, const void* scale_x,
    const void* W_q, const void* scale_w,
    void* Y, int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 63) / 64;
    const int N_blocks = (N + 63) / 64;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_w8a8_tc_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(X_q),
            reinterpret_cast<const __half*>(scale_x),
            reinterpret_cast<const int8_t*>(W_q),
            reinterpret_cast<const __half*>(scale_w),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(Y),
            write_fp32 ? reinterpret_cast<float*>(Y) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_w8a8_tc_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(X_q),
            reinterpret_cast<const __half*>(scale_x),
            reinterpret_cast<const int8_t*>(W_q),
            reinterpret_cast<const __half*>(scale_w),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(Y),
            write_fp32 ? reinterpret_cast<float*>(Y) : nullptr,
            M, N, write_fp32);
    } else {
        fprintf(stderr, "FATAL: launch_det_matmul_w8a8_tc requires "
                "K in {768, 3072}, got K=%d\n", K);
        std::abort();
    }
}


// =============================================================================
// quantize_per_row_int8: per-row symmetric int8 quantization of fp16 [M, K].
// Outputs:
//   X_q:     int8 [M, K]
//   scale_x: fp16 [M]
// One block per row; threads cooperatively find row max(|X|) then quantize.
//
// Symmetric per-row quant pairs with the W8A8 matmul kernel's per-row
// dequant: Y[m, n] = scale_x[m] * scale_w[n] * sum_k(X_q[m,k] * W_q[k,n]).
// =============================================================================

__device__ __forceinline__ float warp_max_abs(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, off);
        v = fmaxf(fabsf(v), fabsf(other));
    }
    return v;
}

template<int K_DIM>
__global__ void quantize_per_row_int8_kernel(
    const __half* __restrict__ X,    // [M, K_DIM] fp16
    int8_t*       __restrict__ X_q,  // [M, K_DIM] int8
    __half*       __restrict__ scale_x,  // [M] fp16
    int M)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    constexpr int BLOCK = 256;
    constexpr int n_warps = BLOCK / 32;
    const __half* X_row = X + (size_t)row * K_DIM;
    int8_t* Xq_row = X_q + (size_t)row * K_DIM;

    // Pass 1: max(|x|) over the row.
    float my_max = 0.0f;
    for (int i = tid; i < K_DIM; i += BLOCK) {
        const float v = __half2float(X_row[i]);
        my_max = fmaxf(my_max, fabsf(v));
    }
    // Reduce within warp, then across warps via shared.
    float wm = warp_max_abs(my_max);
    __shared__ float s_warp_max[BLOCK / 32];
    if (lane == 0) s_warp_max[wid] = wm;
    __syncthreads();
    __shared__ float s_max;
    if (wid == 0) {
        float v = (lane < n_warps) ? s_warp_max[lane] : 0.0f;
        v = warp_max_abs(v);
        if (lane == 0) {
            s_max = v;
            // Compute scale: max_abs / 127. Clamp tiny rows to avoid /0.
            const float scale = fmaxf(v / 127.0f, 1e-8f);
            scale_x[row] = __float2half(scale);
            // Stash inverse scale for pass 2 via s_max reuse.
            s_max = 1.0f / scale;  // overload: now holds inv_scale
        }
    }
    __syncthreads();
    const float inv_scale = s_max;

    // Pass 2: write quantized int8.
    for (int i = tid; i < K_DIM; i += BLOCK) {
        const float v = __half2float(X_row[i]);
        const float q = roundf(v * inv_scale);
        const int qi = (int)fmaxf(-127.0f, fminf(127.0f, q));
        Xq_row[i] = (int8_t)qi;
    }
}


extern "C" void launch_quantize_per_row_int8(
    const void* X, void* X_q, void* scale_x,
    int M, int K, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    if (K == 768) {
        quantize_per_row_int8_kernel<768><<<M, BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(X),
            reinterpret_cast<int8_t*>(X_q),
            reinterpret_cast<__half*>(scale_x),
            M);
    } else if (K == 3072) {
        quantize_per_row_int8_kernel<3072><<<M, BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(X),
            reinterpret_cast<int8_t*>(X_q),
            reinterpret_cast<__half*>(scale_x),
            M);
    } else {
        fprintf(stderr, "FATAL: launch_quantize_per_row_int8 requires "
                "K in {768, 3072}, got K=%d\n", K);
        std::abort();
    }
}
