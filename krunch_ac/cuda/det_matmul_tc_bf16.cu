// det_matmul_tc_bf16.cu — bf16 cp.async double-buffered WMMA matmul (sm_80+).
//
// Direct bf16 port of det_matmul_tc_async.cu. Same tile schedule + K
// reduction order — only the input dtype + fragment type differ.
//
// Why: peer-codec note (PEER_CODEC_NOTES §5.4 Q4) — Bellard uses bf16
// in ts_zip; we use fp16. bf16 has 8 exponent bits (vs fp16's 5),
// wider dynamic range, less precision. For RWKV's long-tail logits
// the dynamic range may matter more than the fractional precision.
// Predicted lift: ~1.05× speed, neutral-or-better ratio. Bytes will
// differ from fp16 codec → v2 model_id territory, not v1.
//
// Bit-stable across M by construction (tile schedule fixed, K reduction
// order fixed). Bytes DIFFER from fp16 path by design (different dtype).
//
// sm_75 (T4) doesn't support bf16 WMMA. Compile guarded.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_16B_bf(__nv_bfloat16* smem_dst, const __nv_bfloat16* gmem_src) {
    unsigned int smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int_ptr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_bf() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_n_bf() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif

template<int K_DIM>
__global__ void det_matmul_tc_async_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C16,
    float*               __restrict__ C32,
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

    __shared__ __nv_bfloat16 A_smem[2][TILE_M * TILE_K];
    __shared__ __nv_bfloat16 B_smem[2][TILE_K * TILE_N];

    auto issue_A = [&](int buf, int k_off) {
        const int row = tid >> 1;
        const int col0 = (tid & 1) * 8;
        const int gm = m_tile + row;
        if (gm < M && (k_off + col0 + 7) < K_DIM) {
            cp_async_16B_bf(&A_smem[buf][row * TILE_K + col0],
                            &A[gm * K_DIM + (k_off + col0)]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) A_smem[buf][row * TILE_K + col0 + j] = __float2bfloat16(0.0f);
        }
    };
    auto issue_B = [&](int buf, int k_off) {
        const int row = tid >> 3;
        const int col0 = (tid & 7) * 8;
        const int gn = n_tile + col0;
        if ((k_off + row) < K_DIM && gn < N) {
            cp_async_16B_bf(&B_smem[buf][row * TILE_N + col0],
                            &B[(k_off + row) * N + gn]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) B_smem[buf][row * TILE_N + col0 + j] = __float2bfloat16(0.0f);
        }
    };

    issue_A(0, 0);
    issue_B(0, 0);
    cp_async_commit_bf();

    #pragma unroll 1
    for (int k = 0; k < K_DIM; k += TILE_K) {
        const int curr = (k / TILE_K) & 1;
        const int next = curr ^ 1;
        const int k_next = k + TILE_K;

        if (k_next < K_DIM) {
            issue_A(next, k_next);
            issue_B(next, k_next);
            cp_async_commit_bf();
            cp_async_wait_n_bf<1>();
        } else {
            cp_async_wait_n_bf<0>();
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &A_smem[curr][(wm_off + i*16) * TILE_K], TILE_K);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
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
                C16[gm * N + gn] = __float2bfloat16(v);
            }
        }
    }
#endif
}

extern "C" void launch_det_matmul_tc_async_bf16(
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
        det_matmul_tc_async_bf16_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A),
            reinterpret_cast<const __nv_bfloat16*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__nv_bfloat16*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    } else if (K == 3072) {
        det_matmul_tc_async_bf16_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A),
            reinterpret_cast<const __nv_bfloat16*>(B),
            write_fp32 ? nullptr : reinterpret_cast<__nv_bfloat16*>(C),
            write_fp32 ? reinterpret_cast<float*>(C) : nullptr,
            M, N, write_fp32);
    }
}
