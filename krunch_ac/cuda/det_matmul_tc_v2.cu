// Tuned det_matmul_tc — 32×32 output tile per block (4 WMMA frags)
// instead of the 16×16 (1 frag) of v1. Same shape-stability
// guarantee (per-output reduction order is fixed by the K loop,
// independent of M), much higher per-block arithmetic intensity:
//
//   v1: 1 mma_sync × 48 K steps = 48 mmas per 16×16 output
//   v2: 4 mma_sync × 48 K steps = 192 mmas per 32×32 output
//
// Same total math per output element (each output is one 16×16 mma
// regardless of which kernel computes it), but 4× more outputs per
// block → 4× fewer block launches → less scheduling overhead, better
// register reuse, more chances to overlap math + memory.
//
// Per-row output depends ONLY on that row's A values + the K loop
// reduction. Zero-padding M<32 rows (out-of-bounds reads → 0) gives
// the same row 0..M-1 outputs as if M had been bigger. So this kernel
// is bit-identical to itself across any M value.
//
// What's NOT bit-identical: this kernel vs the v1 16×16 kernel. They
// produce different (but each shape-stable) bits because the WMMA
// reduction order inside the fragment differs by tile size. Compress
// + decompress must both use v2 (or both v1) for AC roundtrip.
//
// Per-arch behavior: WMMA m16n16k16 is sm_70+. Same template specializes
// per arch via the WMMA backend. No arch-specific code required.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template<int K_DIM>
__global__ void det_matmul_tc_v2_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C16,
    float*        __restrict__ C32,
    int M, int N, int write_fp32)
{
    const int m_tile = blockIdx.x * 32;
    const int n_tile = blockIdx.y * 32;

    // Two halves of A (rows [m_tile, m_tile+16) and [m_tile+16, m_tile+32))
    // and B (cols [n_tile, n_tile+16) and [n_tile+16, n_tile+32))
    // stage cooperatively into shared memory each K step.
    __shared__ __half a_stage[2 * 16 * 16];   // 2 row-blocks × 16K cols
    __shared__ __half b_stage[2 * 16 * 16];   // 16K rows × 2 col-blocks

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0_frag, a1_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0_frag, b1_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc00, acc01, acc10, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    const int tid = threadIdx.x;

    #pragma unroll
    for (int k = 0; k < K_DIM; k += 16) {
        // Stage A[m_tile..m_tile+32, k..k+16] (32×16 = 512 elements)
        // 32 threads × 16 elements each.
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            const int idx = e * 32 + tid;
            const int dm = idx >> 4;       // 0..31
            const int dk = idx & 15;
            const int gm = m_tile + dm;
            const int gk = k + dk;
            __half v = __float2half(0.0f);
            if (gm < M) v = A[gm * K_DIM + gk];
            a_stage[idx] = v;
        }
        // Stage B[k..k+16, n_tile..n_tile+32] (16×32 = 512 elements)
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            const int idx = e * 32 + tid;
            const int dk = idx >> 5;       // 0..15
            const int dn = idx & 31;
            const int gn = n_tile + dn;
            __half v = __float2half(0.0f);
            if (gn < N) v = B[(k + dk) * N + gn];
            b_stage[idx] = v;
        }
        __syncthreads();

        // Load fragments from shared memory.
        // a_stage layout: 32 rows × 16 cols, row-major. So rows [0,16) and
        // [16,32) live in a_stage[0..256) and a_stage[256..512).
        wmma::load_matrix_sync(a0_frag, a_stage,           16);
        wmma::load_matrix_sync(a1_frag, a_stage + 16 * 16, 16);
        // b_stage layout: 16 rows × 32 cols, row-major. Cols [0,16) and
        // [16,32) need stride=32 for both fragments.
        wmma::load_matrix_sync(b0_frag, b_stage,      32);
        wmma::load_matrix_sync(b1_frag, b_stage + 16, 32);

        wmma::mma_sync(acc00, a0_frag, b0_frag, acc00);
        wmma::mma_sync(acc01, a0_frag, b1_frag, acc01);
        wmma::mma_sync(acc10, a1_frag, b0_frag, acc10);
        wmma::mma_sync(acc11, a1_frag, b1_frag, acc11);

        __syncthreads();
    }

    // Store all 4 sub-tiles back to global memory via shared staging.
    __shared__ float tile_f32[32 * 32];
    wmma::store_matrix_sync(tile_f32 +              0,      acc00, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(tile_f32 +             16,      acc01, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(tile_f32 + 16 * 32 +    0,      acc10, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(tile_f32 + 16 * 32 +   16,      acc11, 32, wmma::mem_row_major);
    __syncthreads();

    // 32 threads write 1024 elements → 32 each.
    #pragma unroll
    for (int e = 0; e < 32; e++) {
        const int idx = e * 32 + tid;
        const int dm = idx >> 5;     // 0..31
        const int dn = idx & 31;
        const int gm = m_tile + dm;
        const int gn = n_tile + dn;
        if (gm < M && gn < N) {
            const float v = tile_f32[dm * 32 + dn];
            if (write_fp32) C32[gm * N + gn] = v;
            else            C16[gm * N + gn] = __float2half(v);
        }
    }
}

extern "C" void launch_det_matmul_tc_v2(
    const void* A, const void* B,
    void* C, int write_fp32,
    int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 31) / 32;
    const int N_blocks = (N + 31) / 32;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(32);  // one warp per block

    auto a = reinterpret_cast<const __half*>(A);
    auto b = reinterpret_cast<const __half*>(B);
    auto c16 = write_fp32 ? nullptr : reinterpret_cast<__half*>(C);
    auto c32 = write_fp32 ? reinterpret_cast<float*>(C) : nullptr;

    if (K == 768) {
        det_matmul_tc_v2_kernel<768><<<grid, block, 0, stream>>>(
            a, b, c16, c32, M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_v2_kernel<3072><<<grid, block, 0, stream>>>(
            a, b, c16, c32, M, N, write_fp32);
    } else {
        // No generic-K v2 fallback yet — caller must use launch_det_matmul_tc
        // (16×16) for non-{768, 3072} K.
    }
}
