// det_matmul_tc_xl.cu — extra-large-tile WMMA matmul.
//
// 128×128 output tile per block, 8 warps (256 threads), 2×4 = 8 WMMA
// fragments per warp. Closes part of the cuBLAS HGEMM gap measured
// 2026-05-01 on T4: at M=1024 K=N=768, det_matmul_tc_mw 0.42 ms vs
// cuBLAS 0.05 ms = 8.4× gap. Bigger tiles → more compute per K-tile
// memory load → better latency hiding even without cp.async.
//
// Gates:
//   - M, N must be ≥ 16 (caller pads or chooses smaller tile kernel).
//   - K must be a multiple of 16 (true for K∈{768, 3072} which is all
//     RWKV-4 layer + head shapes).
//
// Bit-stable across M by construction: tile schedule
// (m_tile = blockIdx.x*128, n_tile = blockIdx.y*128) is M-independent.
// Each output element (m, n) is owned by exactly one warp; reduction
// order over K is fixed (k=0..K-16 step 16).
//
// On sm_80+ this should use cp.async for double-buffered K-loop. T4
// (sm_75) doesn't support cp.async hardware; fallback is sync loads.
// TODO(T3.6 follow-up): #if __CUDA_ARCH__ >= 800 path.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 16;       // one WMMA frag deep
constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_M = 4;        // 4 warps along M
constexpr int WARP_N = 2;        // 2 warps along N
constexpr int WARP_TILE_M = TILE_M / WARP_M;  // 32 (2 frags tall per warp)
constexpr int WARP_TILE_N = TILE_N / WARP_N;  // 64 (4 frags wide per warp)
constexpr int FRAG_M = WARP_TILE_M / 16;      // 2
constexpr int FRAG_N = WARP_TILE_N / 16;      // 4

template<int K_DIM>
__global__ void det_matmul_tc_xl_kernel(
    const __half* __restrict__ A,    // [M, K_DIM] row-major
    const __half* __restrict__ B,    // [K_DIM, N] row-major
    __half*       __restrict__ C16,  // [M, N], used if write_fp32 == 0
    float*        __restrict__ C32,  // [M, N], used if write_fp32 == 1
    int M, int N, int write_fp32)
{
    const int m_tile = blockIdx.x * TILE_M;
    const int n_tile = blockIdx.y * TILE_N;

    const int tid = threadIdx.x;          // 0..255
    const int wid = tid >> 5;             // 0..7
    const int lane = tid & 31;            // 0..31
    const int warp_m = wid / WARP_N;      // 0..3
    const int warp_n = wid % WARP_N;      // 0..1
    const int wm_off = warp_m * WARP_TILE_M;  // 0, 32, 64, 96
    const int wn_off = warp_n * WARP_TILE_N;  // 0, 64

    // Accumulators: 2x4 = 8 frags per warp, fp32.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++)
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Shared-mem K-tile staging. 128×16 + 16×128 = 8 KB, easily fits.
    __shared__ __half As[TILE_M * TILE_K];   // 4 KB
    __shared__ __half Bs[TILE_K * TILE_N];   // 4 KB

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += TILE_K) {
        // Cooperative load A[m_tile..m_tile+128, k..k+16] into As.
        // 128×16 = 2048 elements / 256 threads = 8 elements per thread.
        #pragma unroll
        for (int idx = tid; idx < TILE_M * TILE_K; idx += 256) {
            const int row = idx / TILE_K;        // 0..127
            const int col = idx % TILE_K;        // 0..15
            const int gm = m_tile + row;
            __half v = (gm < M) ? A[gm * K_DIM + (k + col)] : __float2half(0.0f);
            As[row * TILE_K + col] = v;
        }
        // Cooperative load B[k..k+16, n_tile..n_tile+128] into Bs.
        #pragma unroll
        for (int idx = tid; idx < TILE_K * TILE_N; idx += 256) {
            const int row = idx / TILE_N;        // 0..15
            const int col = idx % TILE_N;        // 0..127
            const int gn = n_tile + col;
            __half v = (gn < N) ? B[(k + row) * N + gn] : __float2half(0.0f);
            Bs[row * TILE_N + col] = v;
        }
        __syncthreads();

        // Each warp computes its 2x4 frags: WMMA mma_sync over the staged tile.
        #pragma unroll
        for (int i = 0; i < FRAG_M; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &As[(wm_off + i*16) * TILE_K], TILE_K);
            #pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &Bs[wn_off + j*16], TILE_N);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Store fragments directly to global. wmma::store_matrix_sync handles
    // intra-warp coalescing. With 16 frags per warp, ~1 KB written per
    // store call; T4 HBM coalescer absorbs the strided pattern fine.
    //
    // PER-WARP shared scratch (not shared across warps — race fix).
    // 8 warps × 256 floats × 4 bytes = 8 KB.
    __shared__ float warp_scratch_all[WARPS_PER_BLOCK * 16 * 16];
    float* warp_scratch = &warp_scratch_all[wid * 256];

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            const int gm0 = m_tile + wm_off + i*16;
            const int gn0 = n_tile + wn_off + j*16;
            const bool full_tile = (gm0 + 16 <= M) && (gn0 + 16 <= N);

            if (full_tile) {
                // Direct global store. C is row-major with stride N.
                if (write_fp32) {
                    wmma::store_matrix_sync(&C32[gm0 * N + gn0], acc[i][j], N,
                                             wmma::mem_row_major);
                } else {
                    // No direct fp16 store from fp32 accumulator; stage in
                    // shared mem then convert + write. Per-warp scratch is
                    // safe because each warp owns disjoint (i, j) loops.
                    __syncwarp();
                    wmma::store_matrix_sync(warp_scratch, acc[i][j], 16,
                                             wmma::mem_row_major);
                    __syncwarp();
                    for (int idx = lane; idx < 256; idx += 32) {
                        const int dm = idx >> 4, dn = idx & 15;
                        C16[(gm0 + dm) * N + (gn0 + dn)]
                            = __float2half(warp_scratch[dm * 16 + dn]);
                    }
                }
            } else {
                // Partial tile — stage fp32 then mask the global write.
                __syncwarp();
                wmma::store_matrix_sync(warp_scratch, acc[i][j], 16,
                                         wmma::mem_row_major);
                __syncwarp();
                for (int idx = lane; idx < 256; idx += 32) {
                    const int dm = idx >> 4, dn = idx & 15;
                    const int gm = gm0 + dm;
                    const int gn = gn0 + dn;
                    if (gm < M && gn < N) {
                        const float v = warp_scratch[dm * 16 + dn];
                        if (write_fp32) {
                            C32[gm * N + gn] = v;
                        } else {
                            C16[gm * N + gn] = __float2half(v);
                        }
                    }
                }
            }
        }
    }
}

extern "C" void launch_det_matmul_tc_xl(
    const void* A, const void* B, void* C,
    int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + TILE_M - 1) / TILE_M;
    const int N_blocks = (N + TILE_N - 1) / TILE_N;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(WARPS_PER_BLOCK * 32);  // 256 threads

    if (K == 768) {
        det_matmul_tc_xl_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_xl_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__half*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
}
