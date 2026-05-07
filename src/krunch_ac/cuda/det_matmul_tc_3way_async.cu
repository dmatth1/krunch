// det_matmul_tc_3way_async.cu — 3-way fused matmul with cp.async (sm_80+).
//
// Computes y0=A0@W0, y1=A1@W1, y2=A2@W2 in ONE kernel launch where A0/A1/A2
// share shape [M, K] and W0/W1/W2 share shape [K, N]. Used by RWKV-4
// layer step's KVR matmul (kx@Kw, vx@Vw, rx@Rw — same shapes).
//
// Combines:
//   - 3-way fusion (saves 2 grid syncs vs 3 separate matmul launches;
//     saves redundant K-axis traversal of inputs since K is shared across
//     all 3 matmuls).
//   - cp.async double-buffered K-loop (sm_80+ — A10G/A100/H100/L40S).
//     Hides global-load latency behind WMMA mma_sync.
//   - WMMA 16×16 fragments (fp16 input, fp32 accumulator).
//
// Layout: 64×64 output tile per block, 4 warps, 2×2 frags per warp per
// matmul (12 frags per warp total). Bit-stable across M by construction.
//
// Bit-pattern DIFFERS from det_matmul_tc_3way (old single-warp variant)
// because the WMMA reduction order differs. Encoder + decoder must use
// the same kernel for AC roundtrip.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#if __CUDA_ARCH__ >= 800

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

template<int N>
__device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif

template<int K_DIM>
__global__ void det_matmul_tc_3way_async_kernel(
    const __half* __restrict__ A0, const __half* __restrict__ A1, const __half* __restrict__ A2,
    const __half* __restrict__ B0, const __half* __restrict__ B1, const __half* __restrict__ B2,
    void* __restrict__ Y0_raw, void* __restrict__ Y1_raw, void* __restrict__ Y2_raw,
    int wf0, int wf1, int wf2,  // write-as-fp32 flags per output
    int M, int N)
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

    // 3 sets of accumulators: 2×2 frags per warp per matmul.
    // 3 × 4 frags × 8 fp32 elements/thread = 96 fp32 per thread = 384 bytes
    // = 96 registers. Comfortable.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc2[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc0[i][j], 0.0f);
            wmma::fill_fragment(acc1[i][j], 0.0f);
            wmma::fill_fragment(acc2[i][j], 0.0f);
        }
    }

    // Double-buffered shared mem.
    // 3 A buffers × 2 (double-buffered) × TILE_M × TILE_K × 2 bytes
    //   = 3 × 2 × 64 × 16 × 2 = 12 KB
    // 3 B buffers × 2 × TILE_K × TILE_N × 2 = 12 KB
    // Total: 24 KB — fits T4 64 KB / A10G 100 KB shared budget.
    __shared__ __half A0s[2][TILE_M * TILE_K];
    __shared__ __half A1s[2][TILE_M * TILE_K];
    __shared__ __half A2s[2][TILE_M * TILE_K];
    __shared__ __half B0s[2][TILE_K * TILE_N];
    __shared__ __half B1s[2][TILE_K * TILE_N];
    __shared__ __half B2s[2][TILE_K * TILE_N];

    // Each thread loads 16 bytes (8 halves) per A buffer per K iter.
    // 64×16 elements / 128 threads = 8 halves/thread = 1 cp.async.16B.
    // Layout: thread tid loads A_smem[buf][row*TILE_K + col0..col0+7]
    //   row = tid/2 (0..63), col0 = (tid&1)*8 (0 or 8).
    auto issue_A_all = [&](int buf, int k_off) {
        const int row = tid >> 1;
        const int col0 = (tid & 1) * 8;
        const int gm = m_tile + row;
        if (gm < M) {
            cp_async_16B(&A0s[buf][row * TILE_K + col0], &A0[gm * K_DIM + (k_off + col0)]);
            cp_async_16B(&A1s[buf][row * TILE_K + col0], &A1[gm * K_DIM + (k_off + col0)]);
            cp_async_16B(&A2s[buf][row * TILE_K + col0], &A2[gm * K_DIM + (k_off + col0)]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                A0s[buf][row * TILE_K + col0 + j] = __float2half(0.0f);
                A1s[buf][row * TILE_K + col0 + j] = __float2half(0.0f);
                A2s[buf][row * TILE_K + col0 + j] = __float2half(0.0f);
            }
        }
    };
    auto issue_B_all = [&](int buf, int k_off) {
        const int row = tid >> 3;        // 0..15
        const int col0 = (tid & 7) * 8;  // 0,8,...,56
        const int gn = n_tile + col0;
        if ((k_off + row) < K_DIM && gn < N) {
            cp_async_16B(&B0s[buf][row * TILE_N + col0], &B0[(k_off + row) * N + gn]);
            cp_async_16B(&B1s[buf][row * TILE_N + col0], &B1[(k_off + row) * N + gn]);
            cp_async_16B(&B2s[buf][row * TILE_N + col0], &B2[(k_off + row) * N + gn]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                B0s[buf][row * TILE_N + col0 + j] = __float2half(0.0f);
                B1s[buf][row * TILE_N + col0 + j] = __float2half(0.0f);
                B2s[buf][row * TILE_N + col0 + j] = __float2half(0.0f);
            }
        }
    };

    // Prologue: prefetch K-tile 0
    issue_A_all(0, 0);
    issue_B_all(0, 0);
    cp_async_commit();

    // Main loop
    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += TILE_K) {
        const int curr = (k / TILE_K) & 1;
        const int next = curr ^ 1;
        const int k_next = k + TILE_K;

        if (k_next < K_DIM) {
            issue_A_all(next, k_next);
            issue_B_all(next, k_next);
            cp_async_commit();
            cp_async_wait_n<1>();
        } else {
            cp_async_wait_n<0>();
        }
        __syncthreads();

        // Compute on curr buffers — 3 matmuls × 4 frags each per warp.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0_frag, a1_frag, a2_frag;
            wmma::load_matrix_sync(a0_frag, &A0s[curr][(wm_off + i*16) * TILE_K], TILE_K);
            wmma::load_matrix_sync(a1_frag, &A1s[curr][(wm_off + i*16) * TILE_K], TILE_K);
            wmma::load_matrix_sync(a2_frag, &A2s[curr][(wm_off + i*16) * TILE_K], TILE_K);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0_frag, b1_frag, b2_frag;
                wmma::load_matrix_sync(b0_frag, &B0s[curr][wn_off + j*16], TILE_N);
                wmma::load_matrix_sync(b1_frag, &B1s[curr][wn_off + j*16], TILE_N);
                wmma::load_matrix_sync(b2_frag, &B2s[curr][wn_off + j*16], TILE_N);
                wmma::mma_sync(acc0[i][j], a0_frag, b0_frag, acc0[i][j]);
                wmma::mma_sync(acc1[i][j], a1_frag, b1_frag, acc1[i][j]);
                wmma::mma_sync(acc2[i][j], a2_frag, b2_frag, acc2[i][j]);
            }
        }
        __syncthreads();
    }

    // Output staging — fp32 buffer in shared, then per-output dtype write.
    __shared__ float Cs[TILE_M * TILE_N];

    auto store_one = [&](void* Y_raw, int wf, wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2]) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                float* dst = &Cs[(wm_off + i*16) * TILE_N + (wn_off + j*16)];
                wmma::store_matrix_sync(dst, acc[i][j], TILE_N, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // 64*64 cells / 128 threads = 32 cells/thread.
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            const int idx = i * 128 + tid;
            const int row = idx / TILE_N;
            const int col = idx % TILE_N;
            const int gm = m_tile + row;
            const int gn = n_tile + col;
            if (gm < M && gn < N) {
                const float v = Cs[row * TILE_N + col];
                if (wf) {
                    reinterpret_cast<float*>(Y_raw)[gm * N + gn] = v;
                } else {
                    reinterpret_cast<__half*>(Y_raw)[gm * N + gn] = __float2half(v);
                }
            }
        }
        __syncthreads();
    };
    store_one(Y0_raw, wf0, acc0);
    store_one(Y1_raw, wf1, acc1);
    store_one(Y2_raw, wf2, acc2);
#endif  // __CUDA_ARCH__ >= 800
}

extern "C" void launch_det_matmul_tc_3way_async(
    const void* A0, const void* A1, const void* A2,
    const void* B0, const void* B1, const void* B2,
    void* Y0, void* Y1, void* Y2,
    int wf0, int wf1, int wf2,
    int M, int K, int N, cudaStream_t stream)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int M_blocks = (M + TILE_M - 1) / TILE_M;
    const int N_blocks = (N + TILE_N - 1) / TILE_N;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_tc_3way_async_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A0), reinterpret_cast<const __half*>(A1),
            reinterpret_cast<const __half*>(A2),
            reinterpret_cast<const __half*>(B0), reinterpret_cast<const __half*>(B1),
            reinterpret_cast<const __half*>(B2),
            Y0, Y1, Y2, wf0, wf1, wf2, M, N);
    } else if (K == 3072) {
        det_matmul_tc_3way_async_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A0), reinterpret_cast<const __half*>(A1),
            reinterpret_cast<const __half*>(A2),
            reinterpret_cast<const __half*>(B0), reinterpret_cast<const __half*>(B1),
            reinterpret_cast<const __half*>(B2),
            Y0, Y1, Y2, wf0, wf1, wf2, M, N);
    }
}
