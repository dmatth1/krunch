// 3-way batched Tensor-Core matmul: computes
//   y0 = x0 @ W0
//   y1 = x1 @ W1
//   y2 = x2 @ W2
// in a single kernel launch. All three GEMMs share the same M, K, N
// shape — used in the RWKV-4 attention block to fuse the kx@Kw,
// vx@Vw, rx@Rw sequence (3 launches → 1).
//
// Per-output bit-identical to running det_matmul_tc 3× separately
// (same WMMA tile schedule, fixed K-loop reduction order, same
// shared-mem staging).
//
// Block grid: (M_blocks, N_blocks, 3). blockIdx.z selects which of
// the three GEMMs.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template<int K_DIM>
__global__ void det_matmul_tc_3way_kernel(
    const __half* __restrict__ x0, const __half* __restrict__ x1, const __half* __restrict__ x2,
    const __half* __restrict__ W0, const __half* __restrict__ W1, const __half* __restrict__ W2,
    __half*       __restrict__ y0_h, __half*       __restrict__ y1_h, __half*       __restrict__ y2_h,
    float*        __restrict__ y0_f, float*        __restrict__ y1_f, float*        __restrict__ y2_f,
    int M, int N, int write_fp32_mask)  // bit 0/1/2 → write y0_f/y1_f/y2_f
{
    const int which = blockIdx.z;
    const __half* x; const __half* W; __half* y_h; float* y_f;
    int write_fp32;
    switch (which) {
        default:
        case 0: x = x0; W = W0; y_h = y0_h; y_f = y0_f; write_fp32 = (write_fp32_mask & 1); break;
        case 1: x = x1; W = W1; y_h = y1_h; y_f = y1_f; write_fp32 = (write_fp32_mask & 2) >> 1; break;
        case 2: x = x2; W = W2; y_h = y2_h; y_f = y2_f; write_fp32 = (write_fp32_mask & 4) >> 2; break;
    }

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
        // Stage A[m_tile..m_tile+16, k..k+16] with bounds check.
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = e * 32 + tid;
            const int dm = idx >> 4;
            const int dk = idx & 15;
            const int gm = m_tile + dm;
            const int gk = k + dk;
            __half v = __float2half(0.0f);
            if (gm < M) v = x[gm * K_DIM + gk];
            a_stage[idx] = v;
        }
        // Stage B[k..k+16, n_tile..n_tile+16] with bounds check.
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = e * 32 + tid;
            const int dk = idx >> 4;
            const int dn = idx & 15;
            const int gn = n_tile + dn;
            __half v = __float2half(0.0f);
            if (gn < N) v = W[(k + dk) * N + gn];
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
                y_f[gm * N + gn] = tile_f32[dm * 16 + dn];
            } else {
                y_h[gm * N + gn] = __float2half(tile_f32[dm * 16 + dn]);
            }
        }
    }
}

extern "C" void launch_det_matmul_tc_3way(
    const void* x0, const void* x1, const void* x2,
    const void* W0, const void* W1, const void* W2,
    void* y0, void* y1, void* y2,
    int wf0, int wf1, int wf2,           // 1 if y_i is fp32 output
    int M, int K, int N, cudaStream_t stream)
{
    const int M_blocks = (M + 15) / 16;
    const int N_blocks = (N + 15) / 16;
    dim3 grid(M_blocks, N_blocks, 3);
    dim3 block(32);
    const int wf_mask = (wf0) | (wf1 << 1) | (wf2 << 2);

    auto h0 = wf0 ? nullptr : reinterpret_cast<__half*>(y0);
    auto h1 = wf1 ? nullptr : reinterpret_cast<__half*>(y1);
    auto h2 = wf2 ? nullptr : reinterpret_cast<__half*>(y2);
    auto f0 = wf0 ? reinterpret_cast<float*>(y0) : nullptr;
    auto f1 = wf1 ? reinterpret_cast<float*>(y1) : nullptr;
    auto f2 = wf2 ? reinterpret_cast<float*>(y2) : nullptr;

    if (K == 768) {
        det_matmul_tc_3way_kernel<768><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x0),
            reinterpret_cast<const __half*>(x1),
            reinterpret_cast<const __half*>(x2),
            reinterpret_cast<const __half*>(W0),
            reinterpret_cast<const __half*>(W1),
            reinterpret_cast<const __half*>(W2),
            h0, h1, h2, f0, f1, f2,
            M, N, wf_mask);
    } else if (K == 3072) {
        det_matmul_tc_3way_kernel<3072><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x0),
            reinterpret_cast<const __half*>(x1),
            reinterpret_cast<const __half*>(x2),
            reinterpret_cast<const __half*>(W0),
            reinterpret_cast<const __half*>(W1),
            reinterpret_cast<const __half*>(W2),
            h0, h1, h2, f0, f1, f2,
            M, N, wf_mask);
    } else {
        // Generic-K fallback: caller handles. (Not used in RWKV-4 paths.)
        // Skipping for v0.
    }
}
