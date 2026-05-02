// det_matmul_tc_uint8.cu — uint8 weights + inline dequant + fp16 WMMA (sm_80+).
//
// Per-input-channel asymmetric quantization (matches rwkv-cpp-accelerated
// scheme). Validated on RWKV-4-Pile-169M: 0.75-1.25% rel-RMSE per matrix,
// 2× storage reduction. See INT8_WEIGHTS_DESIGN.md.
//
// Compute: y[m, n] = sum_k A_fp16[m, k] * (uint8(W[k, n]) * scale[k] + offset[k])
//
// Layout: 64×64 output tile per block, 4 warps, 2×2 frags per warp — same
// as det_matmul_tc_async. cp.async on A; B loaded uint8 → dequanted to fp16
// in smem before WMMA.
//
// Bytes diverge from fp16 codec by definition → v2 model_id territory.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_16B_u8(__half* smem_dst, const __half* gmem_src) {
    unsigned int smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int_ptr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_u8() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_n_u8() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif

template<int K_DIM>
__global__ void det_matmul_tc_uint8_kernel(
    const __half*   __restrict__ A,           // [M, K_DIM] fp16
    const uint8_t*  __restrict__ W_u8,        // [K_DIM, N] uint8
    const __half*   __restrict__ scale,       // [K_DIM] fp16, per-input-channel
    const __half*   __restrict__ offset,      // [K_DIM] fp16
    __half*         __restrict__ C16,
    float*          __restrict__ C32,
    int M, int N, int write_fp32)
{
#if __CUDA_ARCH__ >= 800
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;

    const int m_tile = blockIdx.x * TILE_M;
    const int n_tile = blockIdx.y * TILE_N;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m = wid >> 1;
    const int warp_n = wid & 1;
    const int wm_off = warp_m * 32;
    const int wn_off = warp_n * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // A: cp.async fp16 (8 KB per buf double-buffered)
    __shared__ __half A_smem[2][TILE_M * TILE_K];
    // B uint8 raw (1 KB per buf double-buffered)
    __shared__ uint8_t B_u8_smem[2][TILE_K * TILE_N];
    // B dequant fp16 (4 KB per buf double-buffered)
    __shared__ __half B_smem[2][TILE_K * TILE_N];
    // Per-K-tile scale + offset (broadcast across N) — 32 + 32 bytes per buf
    __shared__ __half k_scale[2][TILE_K];
    __shared__ __half k_offset[2][TILE_K];

    auto issue_A = [&](int buf, int k_off) {
        const int row = tid >> 1;
        const int col0 = (tid & 1) * 8;
        const int gm = m_tile + row;
        if (gm < M && (k_off + col0 + 7) < K_DIM) {
            cp_async_16B_u8(&A_smem[buf][row * TILE_K + col0],
                             &A[gm * K_DIM + (k_off + col0)]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) A_smem[buf][row * TILE_K + col0 + j] = __float2half(0.0f);
        }
    };
    // B uint8 load: 1024 bytes per K-tile / 128 threads = 8 bytes/thread.
    // Layout: thread tid loads B_u8_smem[buf][row*TILE_N + col0..col0+7]
    //   row = tid/8 (0..15), col0 = (tid&7)*8 (0,8,...,56). Each load is
    //   8 uint8 = uint64 word (no cp.async needed; 1 vectorized load).
    auto issue_B_u8 = [&](int buf, int k_off) {
        const int row = tid >> 3;
        const int col0 = (tid & 7) * 8;
        const int gn = n_tile + col0;
        if ((k_off + row) < K_DIM && gn < N) {
            const uint64_t* src = reinterpret_cast<const uint64_t*>(
                &W_u8[(k_off + row) * N + gn]);
            uint64_t* dst = reinterpret_cast<uint64_t*>(
                &B_u8_smem[buf][row * TILE_N + col0]);
            *dst = *src;
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) B_u8_smem[buf][row * TILE_N + col0 + j] = 0;
        }
    };
    // Per-K-tile scale + offset: 16 elements each. Use first 16 threads.
    auto issue_scale_offset = [&](int buf, int k_off) {
        if (tid < TILE_K) {
            const int gk = k_off + tid;
            if (gk < K_DIM) {
                k_scale[buf][tid]  = scale[gk];
                k_offset[buf][tid] = offset[gk];
            } else {
                k_scale[buf][tid]  = __float2half(0.0f);
                k_offset[buf][tid] = __float2half(0.0f);
            }
        }
    };

    // Prologue
    issue_A(0, 0);
    cp_async_commit_u8();
    issue_B_u8(0, 0);
    issue_scale_offset(0, 0);
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += TILE_K) {
        const int curr = (k / TILE_K) & 1;
        const int next = curr ^ 1;
        const int k_next = k + TILE_K;

        // Issue prefetch for next tile (if any)
        if (k_next < K_DIM) {
            issue_A(next, k_next);
            cp_async_commit_u8();
            cp_async_wait_n_u8<1>();
            issue_B_u8(next, k_next);
            issue_scale_offset(next, k_next);
        } else {
            cp_async_wait_n_u8<0>();
        }
        __syncthreads();

        // Dequant uint8 → fp16 in B_smem[curr].
        // 1024 elements per K-tile / 128 threads = 8 elements/thread.
        // Layout matches issue_B_u8: row = tid/8, col0 = (tid&7)*8.
        {
            const int row = tid >> 3;
            const int col0 = (tid & 7) * 8;
            const float s = __half2float(k_scale[curr][row]);
            const float o = __half2float(k_offset[curr][row]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint8_t u = B_u8_smem[curr][row * TILE_N + col0 + j];
                const float v = float(u) * s + o;
                B_smem[curr][row * TILE_N + col0 + j] = __float2half(v);
            }
        }
        __syncthreads();

        // WMMA on dequanted B
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

    __shared__ float Cs[TILE_M * TILE_N];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* dst = &Cs[(wm_off + i*16) * TILE_N + (wn_off + j*16)];
            wmma::store_matrix_sync(dst, acc[i][j], TILE_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

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
#endif
}

extern "C" void launch_det_matmul_tc_uint8(
    const void* A,         // [M, K] fp16
    const void* W_u8,      // [K, N] uint8
    const void* scale,     // [K] fp16
    const void* offset,    // [K] fp16
    void* C, int write_fp32,
    int M, int K, int N, cudaStream_t stream)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int M_blocks = (M + TILE_M - 1) / TILE_M;
    const int N_blocks = (N + TILE_N - 1) / TILE_N;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_tc_uint8_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const uint8_t*>(W_u8),
            reinterpret_cast<const __half*>(scale),
            reinterpret_cast<const __half*>(offset),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_uint8_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const uint8_t*>(W_u8),
            reinterpret_cast<const __half*>(scale),
            reinterpret_cast<const __half*>(offset),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
}
