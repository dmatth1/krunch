// Multi-warp TC matmul. 64×64 output tile per block, 4 warps (128 threads),
// 2×2 WMMA-frag layout per warp. Designed for the head matmul shape
// (M = window size, K = 768, N = 50277) where the existing 1-warp-per-tile
// kernel under-uses SMs.
//
// Bit-stable across M by construction: tile schedule is m_tile = blockIdx.x*64,
// n_tile = blockIdx.y*64, K loop iterates k = 0..K-16 in fixed order. Each
// output element (m, n) is owned by exactly one warp and accumulated by the
// same WMMA mma_sync sequence regardless of M. Bit-pattern DIFFERS from the
// 1-warp-per-tile kernel (different reduction order across warps' accumulators,
// even though each output's accumulation chain is fixed) — encoder + decoder
// must use the same code path.
//
// Shared mem: A_stage[64,16] + B_stage[16,64] + result[64,64] = 20 KB/block.
// T4 SM has 64 KB shared, A10G has 100 KB → comfortable for 2 active blocks/SM.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template<int K_DIM>
__global__ void det_matmul_tc_mw_kernel(
    const __half* __restrict__ A,    // [M, K_DIM] row-major
    const __half* __restrict__ B,    // [K_DIM, N] row-major
    __half*       __restrict__ C16,  // [M, N], used if write_fp32 == 0
    float*        __restrict__ C32,  // [M, N], used if write_fp32 == 1
    int M, int N, int write_fp32)
{
    const int m_tile_start = blockIdx.x * 64;
    const int n_tile_start = blockIdx.y * 64;

    const int tid = threadIdx.x;            // 0..127
    const int warp_id = tid >> 5;           // 0..3
    const int lane = tid & 31;              // 0..31

    // 2x2 warp layout: warp 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    const int warp_m = warp_id >> 1;        // 0 or 1
    const int warp_n = warp_id & 1;         // 0 or 1
    const int warp_m_off = warp_m * 32;     // 0 or 32 within block tile
    const int warp_n_off = warp_n * 32;

    // 4 accumulators per warp (2x2 grid of 16x16 WMMA frags = 32x32 sub-tile).
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    __shared__ __half As[64 * 16];      // 2 KB
    __shared__ __half Bs[16 * 64];      // 2 KB
    __shared__ float  Cs[64 * 64];      // 16 KB

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += 16) {
        // Load A[m_tile..+64, k..k+16] into As (1024 elements / 128 threads = 8/thread).
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 4;       // / 16
            const int col = idx & 15;       // % 16
            const int gm = m_tile_start + row;
            __half v;
            if (gm < M) {
                v = A[gm * K_DIM + (k + col)];
            } else {
                v = __float2half(0.0f);
            }
            As[row * 16 + col] = v;
        }
        // Load B[k..k+16, n_tile..+64] into Bs (1024 elements / 128 threads = 8/thread).
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 128 + tid;
            const int row = idx >> 6;       // / 64
            const int col = idx & 63;       // % 64
            const int gn = n_tile_start + col;
            __half v;
            if (gn < N) {
                v = B[(k + row) * N + gn];
            } else {
                v = __float2half(0.0f);
            }
            Bs[row * 64 + col] = v;
        }
        __syncthreads();

        // Each warp computes its 2x2 = 4 WMMA frags using As/Bs.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &As[(warp_m_off + i*16) * 16], 16);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &Bs[(warp_n_off + j*16)], 64);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Each warp stores its 4 frags into Cs at the right sub-tile location.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* dst = &Cs[(warp_m_off + i*16) * 64 + (warp_n_off + j*16)];
            wmma::store_matrix_sync(dst, acc[i][j], 64, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Cooperative coalesced write to global. 64*64 = 4096 / 128 = 32 elem/thread.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int idx = i * 128 + tid;
        const int row = idx >> 6;          // / 64
        const int col = idx & 63;          // % 64
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

extern "C" void launch_det_matmul_tc_mw(
    const void* A, const void* B, void* C,
    int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 63) / 64;
    const int N_blocks = (N + 63) / 64;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);  // 4 warps

    if (K == 768) {
        det_matmul_tc_mw_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_mw_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
    // Generic-K fallback intentionally absent — only the head matmul benefits
    // and that's K=768. Caller must pre-check shape.
}
