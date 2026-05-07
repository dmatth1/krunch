// det_matmul_tc_async.cu — cp.async double-buffered WMMA matmul (sm_80+).
//
// Closes (most of) the cuBLAS gap measured 2026-05-01:
//   M=1024 K=N=768 on A10G: det_matmul_tc 0.092 ms vs cuBLAS 0.034 ms (2.7×)
//   M=1024 K=768 N=50277:  det_matmul_tc 5.00 ms vs cuBLAS 1.77 ms  (2.8×)
// cuBLAS's win is mostly cp.async + double-buffered K — replicated here.
//
// Layout: 64×64 output tile per block, 4 warps, 2×2 frags per warp.
// Same as det_matmul_tc_mw, but the K loop pre-issues async loads for tile
// N+1 while computing tile N. Hides global-load latency behind WMMA mma_sync.
//
// Bit-stable across M by construction (tile schedule fixed, K reduction
// order fixed). Bit-pattern DIFFERS from det_matmul_tc / _mw because the
// async load order is identical but the WMMA schedule is the same.
//
// sm_75 (T4) doesn't support cp.async hardware. Compile guarded; on
// sm_75 the kernel is empty and routing should never select it.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#if __CUDA_ARCH__ >= 800

// 16-byte cp.async (eight fp16 elements per call, sm_80+ hardware).
__device__ __forceinline__ void cp_async_16B(__half* smem_dst, const __half* gmem_src) {
    unsigned int smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int_ptr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N pending groups remain.
template<int N>
__device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif

template<int K_DIM>
__global__ void det_matmul_tc_async_kernel(
    const __half* __restrict__ A,    // [M, K_DIM] row-major
    const __half* __restrict__ B,    // [K_DIM, N] row-major
    __half*       __restrict__ C16,
    float*        __restrict__ C32,
    int M, int N, int write_fp32)
{
#if __CUDA_ARCH__ >= 800
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;

    const int m_tile = blockIdx.x * TILE_M;
    const int n_tile = blockIdx.y * TILE_N;

    const int tid = threadIdx.x;          // 0..127
    const int wid = tid >> 5;             // 0..3
    const int lane = tid & 31;
    const int warp_m = wid >> 1;          // 0 or 1
    const int warp_n = wid & 1;           // 0 or 1
    const int wm_off = warp_m * 32;
    const int wn_off = warp_n * 32;

    // Accumulators: 2x2 frags per warp (32x32 sub-tile)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Double-buffered shared mem.
    // A_smem[buf][TILE_M][TILE_K] = 64×16 = 1024 halves = 2 KB per buf
    // B_smem[buf][TILE_K][TILE_N] = 16×64 = 1024 halves = 2 KB per buf
    // Total: 8 KB shared.
    __shared__ __half A_smem[2][TILE_M * TILE_K];
    __shared__ __half B_smem[2][TILE_K * TILE_N];

    // 128 threads load 1024 halves per tile = 8 halves per thread = 16 bytes
    // = 1 cp.async.16B per thread. Layout:
    //   A: thread tid loads A_smem[buf][row*TILE_K + col0..col0+7]
    //      where row = tid/2, col0 = (tid&1)*8.
    //   B: thread tid loads B_smem[buf][row*TILE_N + col0..col0+7]
    //      where row = tid/8 (range 0..15), col0 = (tid&7)*8.
    auto issue_A = [&](int buf, int k_off) {
        const int row = tid >> 1;
        const int col0 = (tid & 1) * 8;
        const int gm = m_tile + row;
        if (gm < M && (k_off + col0 + 7) < K_DIM) {
            cp_async_16B(&A_smem[buf][row * TILE_K + col0],
                          &A[gm * K_DIM + (k_off + col0)]);
        } else {
            // OOB — fill with zeros (rare for K-aligned cases; M alignment
            // is caller's responsibility per routing M ≥ 64).
            #pragma unroll
            for (int j = 0; j < 8; j++) A_smem[buf][row * TILE_K + col0 + j] = __float2half(0.0f);
        }
    };
    auto issue_B = [&](int buf, int k_off) {
        const int row = tid >> 3;        // 0..15
        const int col0 = (tid & 7) * 8;  // 0,8,16,...,56
        const int gn = n_tile + col0;
        if ((k_off + row) < K_DIM && gn < N) {
            cp_async_16B(&B_smem[buf][row * TILE_N + col0],
                          &B[(k_off + row) * N + gn]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) B_smem[buf][row * TILE_N + col0 + j] = __float2half(0.0f);
        }
    };

    // Prologue: prefetch tile 0
    issue_A(0, 0);
    issue_B(0, 0);
    cp_async_commit();

    // Main loop
    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += TILE_K) {
        const int curr = (k / TILE_K) & 1;
        const int next = curr ^ 1;
        const int k_next = k + TILE_K;

        // Issue prefetch for next tile (if any)
        if (k_next < K_DIM) {
            issue_A(next, k_next);
            issue_B(next, k_next);
            cp_async_commit();
            // Wait for the CURRENT tile (prev commit), allow the NEXT to be in flight.
            cp_async_wait_n<1>();
        } else {
            // Last iteration — wait for everything
            cp_async_wait_n<0>();
        }
        __syncthreads();

        // Compute on curr buffer
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &A_smem[curr][(wm_off + i*16) * TILE_K], TILE_K);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &B_smem[curr][wn_off + j*16], TILE_N);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Store fragments. Use shared scratch for output staging (matches
    // det_matmul_tc_mw for coalesced writes + safe partial-tile masking).
    __shared__ float Cs[TILE_M * TILE_N];  // 16 KB

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* dst = &Cs[(wm_off + i*16) * TILE_N + (wn_off + j*16)];
            wmma::store_matrix_sync(dst, acc[i][j], TILE_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Cooperative coalesced write to global. 64*64 / 128 = 32 elem/thread.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int idx = i * 128 + tid;
        const int row = idx / TILE_N;
        const int col = idx % TILE_N;
        const int gm = m_tile + row;
        const int gn = n_tile + col;
        if (gm < M && gn < N) {
            const float v = Cs[row * TILE_N + col];
            if (write_fp32) {
                C32[gm * N + gn] = v;
            } else {
                C16[gm * N + gn] = __float2half(v);
            }
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

extern "C" void launch_det_matmul_tc_async(
    const void* A, const void* B, void* C,
    int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int M_blocks = (M + TILE_M - 1) / TILE_M;
    const int N_blocks = (N + TILE_N - 1) / TILE_N;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_tc_async_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_async_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
}
