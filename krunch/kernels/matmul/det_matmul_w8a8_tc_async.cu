// W8A8 int8-Tensor-Core matmul + cp.async double-buffered K-loop (sm_80+).
// Same per-row act scale + per-col weight scale as det_matmul_w8a8_tc.cu,
// but with cp.async overlapping the next-tile load with current WMMA work.
//
// Pattern mirrors det_matmul_tc_async.cu but with int8 fragments (16-byte
// cp.async loads 16 int8 elements per thread vs 8 fp16). 128 threads /
// 1024 bytes per int8 tile = exactly 8 bytes/thread per tile, so we
// alternate: tid 0..127 each load 8 bytes to A_smem (or B_smem on next
// load) — ONE 16-byte cp.async per thread covers ONE A-row + half a
// B-row. Simpler: each thread does TWO 8-byte loads (one A, one B) per
// tile-pair via 2× cp.async.cg.
//
// Why this matters: A10G fp16 baseline uses cp.async fp16; the
// non-cp.async W8A8 (det_matmul_w8a8_tc.cu) loses to it because cp.async
// hides global-mem latency behind WMMA mma_sync. To beat fp16-cp.async
// on A10G, our W8A8 ALSO needs cp.async to hide latency. The int8 TC's
// 2× compute throughput then materializes.
//
// Bit-stable across M by construction: tile schedule fixed, K-loop
// order fixed, cp.async load order fixed → same output rows for any
// M ≥ row.
//
// Output: Y[m, n] = scale_x[m] * scale_w[n] * sum_k(X_q[m,k] * W_q[k,n])
//
// sm_75 (T4) has no cp.async. Compile-guarded; routing must NOT select
// this kernel on sm_75 (USE_SM80_PLUS gate handles it).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;


#if __CUDA_ARCH__ >= 800

// 16-byte cp.async (16 int8 elements per call).
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
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
__global__ void det_matmul_w8a8_tc_async_kernel(
    const int8_t* __restrict__ X_q,
    const __half* __restrict__ scale_x,
    const int8_t* __restrict__ W_q,
    const __half* __restrict__ scale_w,
    void* __restrict__ Y_raw,
    int wf,  // write_fp32 flag
    int M, int N)
{
#if __CUDA_ARCH__ >= 800
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;

    const int m_tile = blockIdx.x * TILE_M;
    const int n_tile = blockIdx.y * TILE_N;

    const int tid = threadIdx.x;          // 0..127
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = wid >> 1;
    const int warp_n = wid & 1;
    const int wm_off = warp_m * 32;
    const int wn_off = warp_n * 32;

    // 4 int32 accumulator frags per warp (2×2 grid).
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    // Double-buffered int8 shared mem.
    // A_smem[buf][64*16] int8 = 1024 B per buf, 2 KB total
    // B_smem[buf][16*64] int8 = 1024 B per buf, 2 KB total
    __shared__ int8_t A_smem[2][TILE_M * TILE_K];
    __shared__ int8_t B_smem[2][TILE_K * TILE_N];

    // Issue helpers. 128 threads × 8 bytes = 1024 bytes per A or B tile.
    // For 16-byte cp.async, we need each thread to do TWO loads (one A,
    // one B) of 16 bytes... wait, 16-byte load needs 16 int8 elements;
    // A tile is 1024 bytes → 64 threads × 16 bytes covers it. Use first
    // 64 threads for A, next 64 for B.
    auto issue_A = [&](int buf, int k_start) {
        // tid 0..63: each loads row tid (16 cols = 16 bytes) of A.
        if (tid < 64) {
            const int gm = m_tile + tid;
            // Use a safe pointer: if out of bounds, point to a known-zero
            // location. Simpler: bounds-check + zero-fill at warmup time.
            // For now, gate cp.async on gm < M; out-of-bounds rows are
            // handled by initializing the smem tile to 0 once at start.
            if (gm < M) {
                cp_async_16B(&A_smem[buf][tid * TILE_K],
                              &X_q[gm * K_DIM + k_start]);
            }
            // else: leave smem stale; row not used in epilogue
        }
    };
    auto issue_B = [&](int buf, int k_start) {
        // tid 64..127: each loads 16 cols of B.
        // B tile is 16 rows × 64 cols. 64 threads cover it as
        // 16 rows × 4 chunks/row of 16 cols each.
        if (tid >= 64) {
            const int t = tid - 64;
            const int row = t >> 2;        // 0..15
            const int chunk = t & 3;       // 0..3
            const int col0 = chunk * 16;
            const int gn = n_tile + col0;
            if (gn + 15 < N) {
                cp_async_16B(&B_smem[buf][row * TILE_N + col0],
                              &W_q[(k_start + row) * N + gn]);
            } else if (gn < N) {
                // Edge tile — fall back to scalar load (rare; only the
                // last N-block in each row partition).
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    int g = gn + c;
                    int8_t v = (g < N) ? W_q[(k_start + row) * N + g] : (int8_t)0;
                    B_smem[buf][row * TILE_N + col0 + c] = v;
                }
            }
        }
    };

    // Init smem to zero (handles partial M / N tiles cleanly).
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int idx = i * 128 + tid;
        if (idx < TILE_M * TILE_K) {
            A_smem[0][idx] = 0;
            A_smem[1][idx] = 0;
            B_smem[0][idx] = 0;
            B_smem[1][idx] = 0;
        }
    }
    __syncthreads();

    // Pre-issue first tile.
    issue_A(0, 0);
    issue_B(0, 0);
    cp_async_commit();

    int curr = 0;
    for (int k = 0; k < K_DIM; k += TILE_K) {
        const int next = curr ^ 1;
        const int k_next = k + TILE_K;

        // Issue next tile while current computes.
        if (k_next < K_DIM) {
            issue_A(next, k_next);
            issue_B(next, k_next);
            cp_async_commit();
            cp_async_wait_n<1>();
        } else {
            cp_async_wait_n<0>();
        }
        __syncthreads();

        // 4 mma_sync per warp on the current buffer.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
            wmma::load_matrix_sync(
                a_frag, &A_smem[curr][(wm_off + i * 16) * TILE_K], TILE_K);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag;
                wmma::load_matrix_sync(
                    b_frag, &B_smem[curr][(wn_off + j * 16)], TILE_N);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
        curr = next;
    }

    // Epilogue: store frags → shared int32, then coalesced write with scaling.
    __shared__ int32_t Cs[TILE_M * TILE_N];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int32_t* dst = &Cs[(wm_off + i * 16) * TILE_N + (wn_off + j * 16)];
            wmma::store_matrix_sync(dst, acc[i][j], TILE_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 64*64 = 4096 / 128 = 32 elem/thread.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int idx = i * 128 + tid;
        const int row = idx >> 6;
        const int col = idx & 63;
        const int gm = m_tile + row;
        const int gn = n_tile + col;
        if (gm < M && gn < N) {
            const float acc_f = (float)Cs[row * TILE_N + col];
            const float sx = __half2float(scale_x[gm]);
            const float sw = __half2float(scale_w[gn]);
            const float v = acc_f * sx * sw;
            if (wf) {
                reinterpret_cast<float*>(Y_raw)[gm * N + gn] = v;
            } else {
                reinterpret_cast<__half*>(Y_raw)[gm * N + gn] = __float2half(v);
            }
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}


extern "C" void launch_det_matmul_w8a8_tc_async(
    const void* X_q, const void* scale_x,
    const void* W_q, const void* scale_w,
    void* Y, int write_fp32, int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 63) / 64;
    const int N_blocks = (N + 63) / 64;
    dim3 grid(M_blocks, N_blocks);
    dim3 block(128);

    if (K == 768) {
        det_matmul_w8a8_tc_async_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(X_q),
            reinterpret_cast<const __half*>(scale_x),
            reinterpret_cast<const int8_t*>(W_q),
            reinterpret_cast<const __half*>(scale_w),
            Y, write_fp32, M, N);
    } else if (K == 3072) {
        det_matmul_w8a8_tc_async_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(X_q),
            reinterpret_cast<const __half*>(scale_x),
            reinterpret_cast<const int8_t*>(W_q),
            reinterpret_cast<const __half*>(scale_w),
            Y, write_fp32, M, N);
    } else {
        fprintf(stderr, "FATAL: launch_det_matmul_w8a8_tc_async requires "
                "K in {768, 3072}, got K=%d\n", K);
        abort();
    }
}
