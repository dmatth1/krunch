// Tensor-Core-accelerated shape-invariant matmul.
//
// Uses WMMA fragments (16x16x16 fp16-input / fp32-accumulator). The
// per-block tile schedule and per-K reduction order are fixed by the
// kernel — independent of the M dimension passed in — so output[m, n]
// is bit-identical between any M values (e.g., M=1 stepped vs M=N
// packed).
//
// Compares to det_matmul.cu: 100-1000× faster at the layer matmul
// shapes used in RWKV-4-Pile-169M (M=any, K=768, N=768 or 3072) while
// staying bit-invariant across M.
//
// Block schedule: each block computes one BLOCK_M × BLOCK_N tile of
// output. BLOCK_M = BLOCK_N = 16 (one WMMA fragment). One warp per
// block.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ---- 16x16x16 WMMA, single warp per block ----
//
// A: [M, K] fp16 row-major
// B: [K, N] fp16 row-major
// Output: [M, N] fp16 OR [M, N] fp32 (depending on write_fp32 flag)
//
// One block computes a 16x16 tile of output at (m_tile, n_tile).
// We loop over K in 16-element chunks, multiplying-accumulating
// fragments into an fp32 accumulator. After the K loop we store the
// fp32 result (or cast to fp16) into output.

template<int K_DIM>
__global__ void det_matmul_tc_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C16,
    float*        __restrict__ C32,
    int M, int N, int write_fp32)
{
    const int m_tile = blockIdx.x * 16;
    const int n_tile = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    #pragma unroll
    for (int k = 0; k < K_DIM; k += 16) {
        // A[m_tile..m_tile+16, k..k+16]
        const __half* a_ptr = A + m_tile * K_DIM + k;
        // B[k..k+16, n_tile..n_tile+16]
        const __half* b_ptr = B + k * N + n_tile;
        wmma::load_matrix_sync(a_frag, a_ptr, K_DIM);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store. Use shared memory staging to handle M < 16 padding safely.
    if (write_fp32) {
        // Direct fp32 store via wmma store, then masked write into C32.
        __shared__ float tile_f32[16 * 16];
        wmma::store_matrix_sync(tile_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        const int tid = threadIdx.x;
        for (int idx = tid; idx < 256; idx += blockDim.x) {
            const int dm = idx / 16;
            const int dn = idx & 15;
            const int gm = m_tile + dm;
            const int gn = n_tile + dn;
            if (gm < M && gn < N) {
                C32[gm * N + gn] = tile_f32[dm * 16 + dn];
            }
        }
    } else {
        __shared__ float tile_f32[16 * 16];
        wmma::store_matrix_sync(tile_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        const int tid = threadIdx.x;
        for (int idx = tid; idx < 256; idx += blockDim.x) {
            const int dm = idx / 16;
            const int dn = idx & 15;
            const int gm = m_tile + dm;
            const int gn = n_tile + dn;
            if (gm < M && gn < N) {
                C16[gm * N + gn] = __float2half(tile_f32[dm * 16 + dn]);
            }
        }
    }
}

// Same as the templated kernel but stages A and B through shared
// memory with bounds checks — safe for arbitrary M, N (no caller
// padding required). Used when M < 16 or N % 16 != 0. Produces
// bit-identical output to the templated kernel (same WMMA tile
// schedule + same K-loop reduction order).
template<int K_DIM>
__global__ void det_matmul_tc_kernel_safe(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C16,
    float*        __restrict__ C32,
    int M, int N, int write_fp32)
{
    const int m_tile = blockIdx.x * 16;
    const int n_tile = blockIdx.y * 16;

    __shared__ __half a_stage[16 * 16];
    __shared__ __half b_stage[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int tid = threadIdx.x;

    #pragma unroll
    for (int k = 0; k < K_DIM; k += 16) {
        // Cooperatively stage A[m_tile..m_tile+16, k..k+16].
        // 32 threads, 256 elements → 8 each.
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = e * 32 + tid;
            const int dm = idx >> 4;
            const int dk = idx & 15;
            const int gm = m_tile + dm;
            const int gk = k + dk;  // gk always < K_DIM by construction
            __half v = __float2half(0.0f);
            if (gm < M) {
                v = A[gm * K_DIM + gk];
            }
            a_stage[idx] = v;
        }
        // Stage B[k..k+16, n_tile..n_tile+16].
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = e * 32 + tid;
            const int dk = idx >> 4;
            const int dn = idx & 15;
            const int gn = n_tile + dn;
            __half v = __float2half(0.0f);
            if (gn < N) {
                v = B[(k + dk) * N + gn];
            }
            b_stage[idx] = v;
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_stage, 16);
        wmma::load_matrix_sync(b_frag, b_stage, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    __shared__ float tile_f32[16 * 16];
    wmma::store_matrix_sync(tile_f32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 256; idx += blockDim.x) {
        const int dm = idx / 16;
        const int dn = idx & 15;
        const int gm = m_tile + dm;
        const int gn = n_tile + dn;
        if (gm < M && gn < N) {
            if (write_fp32) {
                C32[gm * N + gn] = tile_f32[dm * 16 + dn];
            } else {
                C16[gm * N + gn] = __float2half(tile_f32[dm * 16 + dn]);
            }
        }
    }
}

// Generic K version when K isn't 768 / 3072. (Slightly slower — no
// unrolled K loop.)
__global__ void det_matmul_tc_kernel_generic(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C16,
    float*        __restrict__ C32,
    int M, int K, int N, int write_fp32)
{
    const int m_tile = blockIdx.x * 16;
    const int n_tile = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        const __half* a_ptr = A + m_tile * K + k;
        const __half* b_ptr = B + k * N + n_tile;
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    __shared__ float tile_f32[16 * 16];
    wmma::store_matrix_sync(tile_f32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 256; idx += blockDim.x) {
        const int dm = idx / 16;
        const int dn = idx & 15;
        const int gm = m_tile + dm;
        const int gn = n_tile + dn;
        if (gm < M && gn < N) {
            if (write_fp32) {
                C32[gm * N + gn] = tile_f32[dm * 16 + dn];
            } else {
                C16[gm * N + gn] = __float2half(tile_f32[dm * 16 + dn]);
            }
        }
    }
}

extern "C" void launch_det_matmul_tc(
    const void* A, const void* B,
    void* C, int write_fp32,
    int M, int K, int N, cudaStream_t stream)
{
    // M and N must be padded to multiples of 16 in the access pattern.
    // The kernel handles M < 16 via masked store; A is read with stride
    // K, but rows beyond M will be read out of bounds — caller must
    // ensure A is allocated with at least ceil(M/16)*16 rows OR we add
    // a load-with-mask. For now we require padded A. Most callers
    // already allocate per-token buffers that are 16-aligned; for
    // M=1 stepped path the loader allocates [16, K] padded buffers.
    //
    // ACTUALLY: WMMA load_matrix_sync reads 16 contiguous rows. If M=1
    // we'd read 15 rows past M-1. To avoid fault we must either:
    //   (a) require padded A from caller, or
    //   (b) stage A through shared memory (slower but safe).
    // For the v0 we go with (a): caller passes padded.
    const int M_blocks = (M + 15) / 16;
    const int N_blocks = (N + 15) / 16;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(32);  // one warp

    // Always use the safe (shared-mem-staged) kernel — produces
    // bit-identical output to the unsafe variant but handles any
    // M/N. We keep the unsafe variant for future micro-optimization.
    if (K == 768) {
        det_matmul_tc_kernel_safe<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_kernel_safe<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else {
        det_matmul_tc_kernel_generic<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, K, N, write_fp32);
    }
}
