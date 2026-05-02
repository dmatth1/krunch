// rwkv_step_v2.cu — multi-warp fused single-token RWKV-4 layer.
//
// Replaces v1's <<<1, 768>>> (1 SM) with cooperative <<<6, 128>>> grid:
//   6 blocks × 128 threads = 768 active threads = 1 thread per channel
//   each block has 4 warps → 4 warps per SM → memory latency hiding
//
// Why 6×128 instead of 24×32: per-SM warp count drives latency hiding.
// Earlier 24×32 layout placed 1 warp per SM on T4 — every memory load
// stalled the whole SM (no other warps to schedule). Switching to
// 4 warps per block keeps a single SM busy across stalls. Cost: only
// 6 SMs active out of 40 on T4 (15% utilization), but per-SM throughput
// rises ~4×; net gain on memory-latency-bound work.
//
// Per-call HBM scratch holds cross-block intermediates (LN partials,
// kx/vx/rx, r*y, ffn_kx/rx/r_ffn, ffn_k). Each GEMV phase cooperatively
// stages its 768- (or 3072-)element input vector into a 12 KB shared
// buffer per block before the per-thread inner-loop GEMV.
//
// Numerical contract: matches `scripts/rwkv4_step_ref.py::_layer_step`
// within fp16 noise. NOT bit-equal to v1 (different reduction order).
// Encoder + decoder must use the SAME kernel for AC roundtrip — packed
// variant in T3.4e.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int N_EMBD       = 768;
constexpr int N_ATT        = 768;
constexpr int N_FFN        = 3072;
constexpr int V2_BLOCK_DIM = 128;
constexpr int V2_GRID_DIM  = N_EMBD / V2_BLOCK_DIM;  // 6
constexpr int V2_WARPS_PER_BLOCK = V2_BLOCK_DIM / 32; // 4

struct V2Scratch {
    float ln1_partials[V2_GRID_DIM];     // [6]
    float ln1_partial_sq[V2_GRID_DIM];
    float ln2_partials[V2_GRID_DIM];
    float ln2_partial_sq[V2_GRID_DIM];
    float buf[N_EMBD * 4];               // 3072 floats — kx/vx/rx, ffn_kx/rx, r_ffn, r*y
    float ffn_k_buf[N_FFN];              // 3072 floats — k_ffn after relu²
};

__device__ __forceinline__ float warp_red_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum across 128 threads (4 warps). Result valid in all threads.
__device__ __forceinline__ float blk_red_sum(float v, float* warp_scratch) {
    v = warp_red_sum(v);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_scratch[wid] = v;
    __syncthreads();
    // First warp reduces the 4 partials.
    if (wid == 0) {
        float w = (lane < V2_WARPS_PER_BLOCK) ? warp_scratch[lane] : 0.0f;
        w = warp_red_sum(w);
        if (lane == 0) warp_scratch[0] = w;
    }
    __syncthreads();
    return warp_scratch[0];
}

// Cooperative load: 128 threads load N elements (N a multiple of 128).
__device__ __forceinline__ void coop_load_f32(
    float* __restrict__ sh, const float* __restrict__ src, int N, int tid)
{
    #pragma unroll
    for (int i = tid; i < N; i += V2_BLOCK_DIM) sh[i] = src[i];
}

extern "C" __global__ void rwkv4_layer_step_v2_kernel(
    const __half* __restrict__ x_in,    __half* __restrict__ x_out,
    __half* __restrict__ att_xx,
    float* __restrict__ aa, float* __restrict__ bb, float* __restrict__ pp,
    __half* __restrict__ ffn_xx,
    const __half* __restrict__ ln1_w,   const __half* __restrict__ ln1_b,
    const __half* __restrict__ tm_k,    const __half* __restrict__ tm_v,
    const __half* __restrict__ tm_r,
    const float*  __restrict__ time_decay, const float* __restrict__ time_first,
    const __half* __restrict__ Kw,      const __half* __restrict__ Vw,
    const __half* __restrict__ Rw,      const __half* __restrict__ Ow,
    const __half* __restrict__ ln2_w,   const __half* __restrict__ ln2_b,
    const __half* __restrict__ ffn_tm_k, const __half* __restrict__ ffn_tm_r,
    const __half* __restrict__ ffn_Kw,  const __half* __restrict__ ffn_Vw,
    const __half* __restrict__ ffn_Rw,
    V2Scratch* __restrict__ scratch)
{
    auto grid = cg::this_grid();

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int c   = bid * V2_BLOCK_DIM + tid;  // channel 0..767

    __shared__ float sh[N_FFN];                // 12 KB — reused across phases
    __shared__ float warp_scratch[V2_WARPS_PER_BLOCK];  // 16 bytes

    // ============ Phase 1: LN1 ============
    const float xv = __half2float(x_in[c]);
    const float blk_sum   = blk_red_sum(xv,      warp_scratch);
    const float blk_sumsq = blk_red_sum(xv * xv, warp_scratch);
    if (tid == 0) {
        scratch->ln1_partials[bid]   = blk_sum;
        scratch->ln1_partial_sq[bid] = blk_sumsq;
    }
    grid.sync();

    // Warp 0 reduces 6 partials → broadcast mean/inv_std via shared mem.
    // (warp_red_sum within wid==0 only — warps 1..3 read garbage if they
    //  do it themselves, since their lanes never hold the partials.)
    __shared__ float s_mean, s_inv_std;
    {
        const int wid = tid >> 5;
        const int lane = tid & 31;
        if (wid == 0) {
            const float p_sum   = (lane < V2_GRID_DIM) ? scratch->ln1_partials[lane]   : 0.0f;
            const float p_sumsq = (lane < V2_GRID_DIM) ? scratch->ln1_partial_sq[lane] : 0.0f;
            const float total_sum   = warp_red_sum(p_sum);
            const float total_sumsq = warp_red_sum(p_sumsq);
            if (lane == 0) {
                const float m = total_sum / float(N_EMBD);
                const float v = total_sumsq / float(N_EMBD) - m * m;
                s_mean    = m;
                s_inv_std = rsqrtf(v + 1e-5f);
            }
        }
        __syncthreads();
    }
    const float mean    = s_mean;
    const float inv_std = s_inv_std;

    const float xx_v = (xv - mean) * inv_std * __half2float(ln1_w[c])
                     + __half2float(ln1_b[c]);

    // ============ Phase 2: time-mix premix ============
    const float att_xx_v = __half2float(att_xx[c]);
    const float kx = xx_v * __half2float(tm_k[c]) + att_xx_v * (1.0f - __half2float(tm_k[c]));
    const float vx = xx_v * __half2float(tm_v[c]) + att_xx_v * (1.0f - __half2float(tm_v[c]));
    const float rx = xx_v * __half2float(tm_r[c]) + att_xx_v * (1.0f - __half2float(tm_r[c]));

    att_xx[c] = __float2half(xx_v);

    scratch->buf[N_EMBD     + c] = kx;
    scratch->buf[N_EMBD * 2 + c] = vx;
    scratch->buf[N_EMBD * 3 + c] = rx;

    grid.sync();

    // ============ Phase 3: K, V, R GEMVs ============
    coop_load_f32(&sh[0],          &scratch->buf[N_EMBD],     N_EMBD, tid);
    coop_load_f32(&sh[N_EMBD],     &scratch->buf[N_EMBD * 2], N_EMBD, tid);
    coop_load_f32(&sh[N_EMBD * 2], &scratch->buf[N_EMBD * 3], N_EMBD, tid);
    __syncthreads();

    float k_acc = 0.0f, v_acc = 0.0f, r_acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_EMBD; i++) {
        k_acc += sh[i]              * __half2float(Kw[i * N_ATT  + c]);
        v_acc += sh[N_EMBD + i]     * __half2float(Vw[i * N_ATT  + c]);
        r_acc += sh[N_EMBD * 2 + i] * __half2float(Rw[i * N_EMBD + c]);
    }

    const float r_val = 1.0f / (1.0f + __expf(-r_acc));

    // ============ Phase 4: WKV ============
    const float u  = time_first[c];
    const float w  = time_decay[c];
    const float aa_v = aa[c];
    const float bb_v = bb[c];
    const float pp_v = pp[c];

    const float ww = u + k_acc;
    const float p1 = fmaxf(pp_v, ww);
    const float e1 = __expf(pp_v - p1);
    const float e2 = __expf(ww   - p1);
    const float y_val = (e1 * aa_v + e2 * v_acc) / (e1 * bb_v + e2);

    const float ww2 = w + pp_v;
    const float p2  = fmaxf(ww2, k_acc);
    const float e1_2 = __expf(ww2   - p2);
    const float e2_2 = __expf(k_acc - p2);
    aa[c] = e1_2 * aa_v + e2_2 * v_acc;
    bb[c] = e1_2 * bb_v + e2_2;
    pp[c] = p2;

    scratch->buf[c] = r_val * y_val;
    grid.sync();

    // ============ Phase 5: Ow GEMV + residual ============
    coop_load_f32(&sh[0], &scratch->buf[0], N_ATT, tid);
    __syncthreads();

    float ow_acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_ATT; i++) {
        ow_acc += sh[i] * __half2float(Ow[i * N_EMBD + c]);
    }
    const float x_after_att = xv + ow_acc;

    // ============ Phase 6: LN2 ============
    const float blk_sum2   = blk_red_sum(x_after_att,                warp_scratch);
    const float blk_sumsq2 = blk_red_sum(x_after_att * x_after_att,  warp_scratch);
    if (tid == 0) {
        scratch->ln2_partials[bid]   = blk_sum2;
        scratch->ln2_partial_sq[bid] = blk_sumsq2;
    }
    grid.sync();

    __shared__ float s_mean2, s_inv_std2;
    {
        const int wid = tid >> 5;
        const int lane = tid & 31;
        if (wid == 0) {
            const float p2_sum   = (lane < V2_GRID_DIM) ? scratch->ln2_partials[lane]   : 0.0f;
            const float p2_sumsq = (lane < V2_GRID_DIM) ? scratch->ln2_partial_sq[lane] : 0.0f;
            const float total_sum2   = warp_red_sum(p2_sum);
            const float total_sumsq2 = warp_red_sum(p2_sumsq);
            if (lane == 0) {
                const float m = total_sum2 / float(N_EMBD);
                const float v = total_sumsq2 / float(N_EMBD) - m * m;
                s_mean2    = m;
                s_inv_std2 = rsqrtf(v + 1e-5f);
            }
        }
        __syncthreads();
    }
    const float mean2    = s_mean2;
    const float inv_std2 = s_inv_std2;

    const float xx2_v = (x_after_att - mean2) * inv_std2
                          * __half2float(ln2_w[c])
                        + __half2float(ln2_b[c]);

    // ============ Phase 7: FFN premix ============
    const float ffn_xx_v   = __half2float(ffn_xx[c]);
    const float ffn_kx = xx2_v * __half2float(ffn_tm_k[c]) + ffn_xx_v * (1.0f - __half2float(ffn_tm_k[c]));
    const float ffn_rx = xx2_v * __half2float(ffn_tm_r[c]) + ffn_xx_v * (1.0f - __half2float(ffn_tm_r[c]));

    ffn_xx[c] = __float2half(xx2_v);

    scratch->buf[N_EMBD     + c] = ffn_kx;
    scratch->buf[N_EMBD * 2 + c] = ffn_rx;
    grid.sync();

    // ============ Phase 8: ffn_R GEMV ============
    coop_load_f32(&sh[0], &scratch->buf[N_EMBD * 2], N_EMBD, tid);
    __syncthreads();

    float ffn_r_acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_EMBD; i++) {
        ffn_r_acc += sh[i] * __half2float(ffn_Rw[i * N_EMBD + c]);
    }
    const float r_ffn_val = 1.0f / (1.0f + __expf(-ffn_r_acc));
    scratch->buf[N_EMBD * 3 + c] = r_ffn_val;

    // ============ Phase 9: ffn_K GEMV + relu² ============
    __syncthreads();
    coop_load_f32(&sh[0], &scratch->buf[N_EMBD], N_EMBD, tid);
    __syncthreads();

    // 3072 outputs / 768 threads = 4 outputs per thread. With 128 threads
    // per block × 6 blocks, output index = c + j*N_EMBD (covers 0..3071).
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int oc = c + j * N_EMBD;
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < N_EMBD; i++) {
            acc += sh[i] * __half2float(ffn_Kw[i * N_FFN + oc]);
        }
        const float relu_v = (acc > 0.0f) ? acc : 0.0f;
        scratch->ffn_k_buf[oc] = relu_v * relu_v;
    }
    grid.sync();

    // ============ Phase 10: ffn_V GEMV + final residual ============
    coop_load_f32(&sh[0], &scratch->ffn_k_buf[0], N_FFN, tid);
    __syncthreads();

    float ffn_v_acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < N_FFN; i++) {
        ffn_v_acc += sh[i] * __half2float(ffn_Vw[i * N_EMBD + c]);
    }

    const float r_ffn   = scratch->buf[N_EMBD * 3 + c];
    const float x_final = x_after_att + r_ffn * ffn_v_acc;
    x_out[c] = __float2half(x_final);
}

extern "C" int v2_scratch_bytes() {
    return (int)sizeof(V2Scratch);
}

extern "C" int launch_rwkv4_layer_step_v2(
    const void* x_in, void* x_out,
    void* att_xx, float* aa, float* bb, float* pp, void* ffn_xx,
    const void* ln1_w, const void* ln1_b,
    const void* tm_k, const void* tm_v, const void* tm_r,
    const float* time_decay, const float* time_first,
    const void* Kw, const void* Vw, const void* Rw, const void* Ow,
    const void* ln2_w, const void* ln2_b,
    const void* ffn_tm_k, const void* ffn_tm_r,
    const void* ffn_Kw, const void* ffn_Vw, const void* ffn_Rw,
    void* scratch_v2,
    cudaStream_t stream)
{
    void* args[] = {
        (void*)&x_in, (void*)&x_out,
        (void*)&att_xx, (void*)&aa, (void*)&bb, (void*)&pp, (void*)&ffn_xx,
        (void*)&ln1_w, (void*)&ln1_b,
        (void*)&tm_k, (void*)&tm_v, (void*)&tm_r,
        (void*)&time_decay, (void*)&time_first,
        (void*)&Kw, (void*)&Vw, (void*)&Rw, (void*)&Ow,
        (void*)&ln2_w, (void*)&ln2_b,
        (void*)&ffn_tm_k, (void*)&ffn_tm_r,
        (void*)&ffn_Kw, (void*)&ffn_Vw, (void*)&ffn_Rw,
        (void*)&scratch_v2,
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)rwkv4_layer_step_v2_kernel,
        dim3(V2_GRID_DIM), dim3(V2_BLOCK_DIM),
        args, /*shmem=*/0, stream);
    return (int)err;
}
