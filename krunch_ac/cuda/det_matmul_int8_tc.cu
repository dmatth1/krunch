// int8 weight × fp16 activation TC matmul, with inline per-input-channel
// dequant. 64×64 output tile per block, 4 warps (128 threads), 2×2
// WMMA-frag layout per warp. Same scheduling as det_matmul_tc_mw —
// bit-stable across M by construction.
//
// Storage:
//   A: fp16 [M, K]               activations
//   B_q: uint8 [K, N]            quantized weights (per-input-channel uint8)
//   scale: fp16 [K]              per-row scale
//   offset: fp16 [K]             per-row offset
//   C: fp16 or fp32 [M, N]       output
//
// Math:
//   W_dequant[i, j] = B_q[i, j] * scale[i] + offset[i]
//   Y[m, n] = sum_i A[m, i] * W_dequant[i, n]
//
// Bandwidth:
//   fp16 W: K*N*2 bytes per matmul
//   int8 W: K*N + 2*K*2 bytes per matmul ≈ K*N (for N >> 4) → ~2× less HBM
//
// Compile guards split sm_75 (sync K-loop) from sm_80+ (cp.async double-
// buffered K-loop). Single source file, two SASS targets via nvcc gencode
// list — caller doesn't need to dispatch on arch.
//
// Bit-stability: encoder M=large packed and decoder M=B stepped both
// produce identical bytes per output row because (a) tile schedule is
// fixed, (b) K-loop order is fixed, (c) dequant happens in fp16 with
// the same scale/offset broadcast pattern regardless of M.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;


#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_4B(__half* smem_dst, const __half* gmem_src) {
    unsigned int smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_int_ptr), "l"(gmem_src)
    );
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N> __device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#endif


// Inner-loop dequant helper. Loads uint8 weight, casts to fp16, applies
// scale + offset (per-input-channel = per K-row).
__device__ __forceinline__ __half dequant_one(uint8_t q, __half scale, __half offset) {
    // Spell out: ((float)q * (float)scale + (float)offset) as fp16. Keep all
    // arithmetic deterministic. fp32 intermediates so encoder + decoder
    // produce same bits regardless of fp16 rounding mode subtleties.
    float v = (float)q * __half2float(scale) + __half2float(offset);
    return __float2half(v);
}


template<int K_DIM>
__global__ void det_matmul_int8_tc_kernel(
    const __half*  __restrict__ A,        // [M, K_DIM] fp16
    const uint8_t* __restrict__ B_q,      // [K_DIM, N] uint8
    const __half*  __restrict__ scale,    // [K_DIM] fp16
    const __half*  __restrict__ offset,   // [K_DIM] fp16
    __half*        __restrict__ C16,
    float*         __restrict__ C32,
    int M, int N, int write_fp32)
{
    const int m_tile_start = blockIdx.x * 64;
    const int n_tile_start = blockIdx.y * 64;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;
    const int warp_m_off = warp_m * 32;
    const int warp_n_off = warp_n * 32;

    // 4 fp32 accumulator frags per warp (2×2 grid of 16×16 = 32×32 sub-tile).
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    __shared__ __half As[64 * 16];      // 2 KB
    __shared__ __half Bs[16 * 64];      // 2 KB (post-dequant)
    __shared__ __half scale_s[16];      // 32 B
    __shared__ __half offset_s[16];     // 32 B
    __shared__ float  Cs[64 * 64];      // 16 KB

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += 16) {
        // Load A tile [64, 16] — 1024 elements, 8/thread.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 4;
            const int col = idx & 15;
            const int gm = m_tile_start + row;
            __half v;
            if (gm < M) {
                v = A[gm * K_DIM + (k + col)];
            } else {
                v = __float2half(0.0f);
            }
            As[row * 16 + col] = v;
        }

        // Load per-K-row scale + offset for this 16-row tile (32 elem total,
        // first 16 threads handle scale, next 16 handle offset).
        if (tid < 16) {
            scale_s[tid] = scale[k + tid];
        } else if (tid < 32) {
            offset_s[tid - 16] = offset[k + (tid - 16)];
        }
        __syncthreads();

        // Load B_q tile [16, 64] uint8, dequant to fp16, write into Bs.
        // 1024 elements / 128 threads = 8/thread.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 6;       // / 64
            const int col = idx & 63;       // % 64
            const int gn = n_tile_start + col;
            __half v;
            if (gn < N) {
                uint8_t q = B_q[(k + row) * N + gn];
                v = dequant_one(q, scale_s[row], offset_s[row]);
            } else {
                v = __float2half(0.0f);
            }
            Bs[row * 64 + col] = v;
        }
        __syncthreads();

        // 4 WMMA mma_sync ops per warp using As/Bs (same as det_matmul_tc_mw).
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &As[(warp_m_off + i * 16) * 16], 16);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &Bs[(warp_n_off + j * 16)], 64);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Store fragments → shared mem at sub-tile location, then coalesced
    // global write. Same as det_matmul_tc_mw.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* dst = &Cs[(warp_m_off + i * 16) * 64 + (warp_n_off + j * 16)];
            wmma::store_matrix_sync(dst, acc[i][j], 64, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int idx = i * 128 + tid;
        const int row = idx >> 6;
        const int col = idx & 63;
        const int gm = m_tile_start + row;
        const int gn = n_tile_start + col;
        if (gm < M && gn < N) {
            const float v = Cs[row * 64 + col];
            if (write_fp32) {
                C32[gm * N + gn] = v;
            } else {
                C16[gm * N + gn] = __float2half(v);
            }
        }
    }
}


extern "C" void launch_det_matmul_int8_tc(
    const void* A, const void* B_q,
    const void* scale, const void* offset,
    void* C, int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 63) / 64;
    const int N_blocks = (N + 63) / 64;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_int8_tc_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const uint8_t*>(B_q),
            reinterpret_cast<const __half*>(scale),
            reinterpret_cast<const __half*>(offset),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_int8_tc_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const uint8_t*>(B_q),
            reinterpret_cast<const __half*>(scale),
            reinterpret_cast<const __half*>(offset),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
    // Other K values intentionally unsupported — RWKV-4-Pile-169M only uses
    // K∈{768, 3072} for the matmul shapes that quantize.
}
