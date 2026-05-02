// rwkv_step_v3.cu — B-batched persistent fused layer kernel (in progress).
//
// Multi-session work. Architecture per docs/V3_KERNEL_DESIGN.md.
// THIS FILE: M2 (Phase 1+2) + M3 (Phase 4 WKV) + M4 (Phase 5 Ow+residual,
// Phase 6 LN2, Phase 7 ffn-premix). Phase 3 (KVR matmul) + Phases 8-10
// (ffn R/K/V + output residual) still stubbed.
//
// Block layout: 6 blocks × 128 threads = 768 threads, one per channel.
// Each thread iterates over B for per-channel ops. Cooperative launch
// (cg::this_grid().sync()) coordinates phases.
//
// Per-call HBM scratch holds intermediates [B, C] / [B, n_ffn] etc.
// Caller allocates via krunch_ac_cuda.v3_scratch_bytes(B) and passes
// as a uint8 tensor.
//
// Bit-correctness contract (when complete): matches rwkv4_step_ref
// within fp16 noise. NOT bit-equal to v2 or cpp_path. Encoder + decoder
// must use SAME v3 kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int N_EMBD       = 768;
constexpr int N_ATT        = 768;
constexpr int N_FFN        = 3072;
constexpr int V3_BLOCK_DIM = 128;
constexpr int V3_GRID_DIM  = N_EMBD / V3_BLOCK_DIM;  // 6
constexpr int V3_WARPS_PER_BLOCK = V3_BLOCK_DIM / 32; // 4
constexpr int V3_B_MAX = 256;

// Per-call scratch. Contiguous layout for clean pointer arithmetic.
//   ln1_partials[B_MAX][V3_GRID_DIM]    fp32 — per-block per-batch partial sum
//   ln1_partial_sq[B_MAX][V3_GRID_DIM]  fp32
//   kx[B_MAX][N_EMBD]                    fp16 — Phase 2 time-mix premix output
//   vx[B_MAX][N_EMBD]                    fp16
//   rx[B_MAX][N_EMBD]                    fp16
//   k_acc[B_MAX][N_ATT]                  fp32 — Phase 3 KVR matmul output (K)
//   v_acc[B_MAX][N_ATT]                  fp32
//   r_pre[B_MAX][N_EMBD]                 fp16 — Phase 3 KVR (R, pre-sigmoid)
//   y_buf[B_MAX][N_ATT]                  fp32 — Phase 4 WKV output
//   x_attn[B_MAX][N_EMBD]                fp16 — Phase 5 Ow output + residual
//   ln2_partials[B_MAX][V3_GRID_DIM]     fp32 — Phase 6 LN2 partial sums
//   ln2_partial_sq[B_MAX][V3_GRID_DIM]   fp32
//   ffn_kx[B_MAX][N_EMBD]                fp16 — Phase 7 ffn-premix kx
//   ffn_rx[B_MAX][N_EMBD]                fp16 — Phase 7 ffn-premix rx
//   ffn_K_act[B_MAX][N_FFN]              fp16 — Phase 8 relu²(ffn_Kw @ ffn_kx)
//   ffn_V_out[B_MAX][N_EMBD]             fp16 — Phase 9 ffn_Vw @ ffn_K_act
//   ffn_R_act[B_MAX][N_EMBD]             fp16 — Phase 8b sigmoid(ffn_Rw @ ffn_rx)
//
// Layout chosen for stable strides regardless of caller's B (always
// V3_B_MAX). Wastes ~1 MB at small B but simplifies kernel addressing.

__device__ __forceinline__ float warp_red_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide reduction across 128 threads (4 warps). Result valid in lane 0
// of warp 0; broadcast via warp_scratch[0].
__device__ __forceinline__ float blk_red_sum(float v, float* warp_scratch) {
    v = warp_red_sum(v);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_scratch[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float w = (lane < V3_WARPS_PER_BLOCK) ? warp_scratch[lane] : 0.0f;
        w = warp_red_sum(w);
        if (lane == 0) warp_scratch[0] = w;
    }
    __syncthreads();
    return warp_scratch[0];
}

// ============================================================
// Kernel: phases 1-2 implemented, 3-10 stubbed.
// ============================================================

extern "C" __global__ void rwkv4_layer_step_v3_kernel(
    int B,
    const __half* __restrict__ x_in,        // [B, C] fp16
    __half* __restrict__ x_out,              // [B, C] fp16 (final output; UNTOUCHED in M2)
    __half* __restrict__ att_xx,             // [B, C] fp16 in-place (read prev, write new)
    float*  __restrict__ aa,                 // [B, n_att] fp32 in-place (UNTOUCHED in M2)
    float*  __restrict__ bb,                 // [B, n_att]
    float*  __restrict__ pp,                 // [B, n_att]
    __half* __restrict__ ffn_xx,             // [B, C] fp16 (UNTOUCHED in M2)
    const __half* __restrict__ ln1_w,        // [C]
    const __half* __restrict__ ln1_b,        // [C]
    const __half* __restrict__ tm_k,         // [C]
    const __half* __restrict__ tm_v,         // [C]
    const __half* __restrict__ tm_r,         // [C]
    const float*  __restrict__ time_decay,   // [n_att]   (unused in M2)
    const float*  __restrict__ time_first,   // [n_att]
    const __half* __restrict__ Kw,           // [C, n_att]   (unused in M2)
    const __half* __restrict__ Vw,
    const __half* __restrict__ Rw,
    const __half* __restrict__ Ow,
    const __half* __restrict__ ln2_w,
    const __half* __restrict__ ln2_b,
    const __half* __restrict__ ffn_tm_k,
    const __half* __restrict__ ffn_tm_r,
    const __half* __restrict__ ffn_Kw,
    const __half* __restrict__ ffn_Vw,
    const __half* __restrict__ ffn_Rw,
    void* __restrict__ scratch_raw)
{
    auto grid = cg::this_grid();

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int c = bid * V3_BLOCK_DIM + tid;  // 0..767

    // Scratch pointers (offsets in floats / halves as appropriate).
    float*  scratch_f      = reinterpret_cast<float*>(scratch_raw);
    float*  ln1_partials   = scratch_f + 0;
    float*  ln1_partial_sq = scratch_f + V3_B_MAX * V3_GRID_DIM;
    __half* kx_buf = reinterpret_cast<__half*>(
        scratch_f + 2 * V3_B_MAX * V3_GRID_DIM);
    __half* vx_buf = kx_buf + V3_B_MAX * N_EMBD;
    __half* rx_buf = vx_buf + V3_B_MAX * N_EMBD;
    // After kx/vx/rx (3 × B_MAX × N_EMBD halves):
    float*  k_acc_buf = reinterpret_cast<float*>(rx_buf + V3_B_MAX * N_EMBD);
    float*  v_acc_buf = k_acc_buf + V3_B_MAX * N_ATT;
    __half* r_pre_buf = reinterpret_cast<__half*>(v_acc_buf + V3_B_MAX * N_ATT);
    float*  y_buf     = reinterpret_cast<float*>(r_pre_buf + V3_B_MAX * N_EMBD);
    // M4 buffers: x_attn (Ow output + residual), ln2 partials, ffn_kx, ffn_rx
    __half* x_attn_buf = reinterpret_cast<__half*>(y_buf + V3_B_MAX * N_ATT);
    float*  ln2_partials   = reinterpret_cast<float*>(x_attn_buf + V3_B_MAX * N_EMBD);
    float*  ln2_partial_sq = ln2_partials + V3_B_MAX * V3_GRID_DIM;
    __half* ffn_kx_buf = reinterpret_cast<__half*>(ln2_partial_sq + V3_B_MAX * V3_GRID_DIM);
    __half* ffn_rx_buf = ffn_kx_buf + V3_B_MAX * N_EMBD;
    // M5 buffers: ffn_K_act (post-relu²), ffn_V_out, ffn_R_act (post-sigmoid)
    __half* ffn_K_act = ffn_rx_buf + V3_B_MAX * N_EMBD;
    __half* ffn_V_out = ffn_K_act + V3_B_MAX * N_FFN;
    __half* ffn_R_act = ffn_V_out + V3_B_MAX * N_EMBD;

    // Per-block warp scratch (block reductions)
    __shared__ float warp_scratch[V3_WARPS_PER_BLOCK];
    __shared__ float s_mean[V3_B_MAX];
    __shared__ float s_inv_std[V3_B_MAX];
    __shared__ float s_mean2[V3_B_MAX];     // Phase 6 LN2 mean
    __shared__ float s_inv_std2[V3_B_MAX];  // Phase 6 LN2 inv_std

    // ============ Phase 1: LN1 partial sums (per b, per block) ============
    //
    // For each b in [0, B): compute block partial sum + sumsq over the
    // block's 128 channels. tid==0 writes partial to scratch.
    //
    // Total syncs: 2 per b iteration (block-wide blk_red_sum has 2
    // __syncthreads internally). With B=128, that's 256 block syncs per
    // LN. Not free, but acceptable for correctness foundation; further
    // optimization (vectorize across b) is a follow-up.

    for (int b = 0; b < B; b++) {
        const float xv = __half2float(x_in[b * N_EMBD + c]);
        const float blk_sum = blk_red_sum(xv, warp_scratch);
        const float blk_sumsq = blk_red_sum(xv * xv, warp_scratch);
        if (tid == 0) {
            ln1_partials[b * V3_GRID_DIM + bid] = blk_sum;
            ln1_partial_sq[b * V3_GRID_DIM + bid] = blk_sumsq;
        }
    }
    grid.sync();

    // ============ Phase 1b: LN1 finalize (per b, all blocks) ============
    //
    // Each block reads all V3_GRID_DIM=6 partials per b, reduces to
    // scalar mean/inv_std, broadcasts via shared mem.

    // Warp 0 (lanes 0..31) handles up to 32 b values per pass; loop for B>32.
    {
        const int wid = tid >> 5;
        const int lane = tid & 31;
        if (wid == 0) {
            for (int b = lane; b < B; b += 32) {
                float p_sum = 0.0f;
                float p_sumsq = 0.0f;
                #pragma unroll
                for (int j = 0; j < V3_GRID_DIM; j++) {
                    p_sum   += ln1_partials[b * V3_GRID_DIM + j];
                    p_sumsq += ln1_partial_sq[b * V3_GRID_DIM + j];
                }
                const float m = p_sum / float(N_EMBD);
                const float v = p_sumsq / float(N_EMBD) - m * m;
                s_mean[b]    = m;
                s_inv_std[b] = rsqrtf(v + 1e-5f);
            }
        }
        __syncthreads();
    }

    // ============ Phase 2: LN1 apply + time-mix premix ============
    //
    // For each (b, c): xx = (x - mean[b]) * inv_std[b] * gamma[c] + beta[c]
    // Then premix: kx = xx*tm_k + att_xx_prev*(1-tm_k), etc.
    // att_xx is updated in place at end (with last token's xx).
    //
    // For T=1 stepped, "last token's xx" == this token's xx. So we just
    // overwrite att_xx with xx_v after the premix computation.

    const float ln1_w_c = __half2float(ln1_w[c]);
    const float ln1_b_c = __half2float(ln1_b[c]);
    const float tm_k_c = __half2float(tm_k[c]);
    const float tm_v_c = __half2float(tm_v[c]);
    const float tm_r_c = __half2float(tm_r[c]);

    for (int b = 0; b < B; b++) {
        const float xv = __half2float(x_in[b * N_EMBD + c]);
        const float xx_v = (xv - s_mean[b]) * s_inv_std[b] * ln1_w_c + ln1_b_c;

        const float att_xx_v = __half2float(att_xx[b * N_EMBD + c]);
        const float kx = xx_v * tm_k_c + att_xx_v * (1.0f - tm_k_c);
        const float vx = xx_v * tm_v_c + att_xx_v * (1.0f - tm_v_c);
        const float rx = xx_v * tm_r_c + att_xx_v * (1.0f - tm_r_c);

        kx_buf[b * N_EMBD + c] = __float2half(kx);
        vx_buf[b * N_EMBD + c] = __float2half(vx);
        rx_buf[b * N_EMBD + c] = __float2half(rx);

        // Update att_xx state (in place) with this token's xx
        att_xx[b * N_EMBD + c] = __float2half(xx_v);
    }

    grid.sync();

    // ============ Phase 3: KVR matmul (scalar scaffold) ============
    //
    // k_acc[b, c] = sum_j Kw[j, c] * kx[b, j]    (fp32 acc)
    // v_acc[b, c] = sum_j Vw[j, c] * vx[b, j]    (fp32 acc)
    // r_pre[b, c] = sum_j Rw[j, c] * rx[b, j]    (fp16 store, fp32 acc)
    //
    // Kw/Vw/Rw shape [N_EMBD, N_ATT] (= [N_EMBD, N_EMBD]) row-major.
    // 3-way fusion: amortize loads of kx/vx/rx across the K-loop.

    if (c < N_EMBD) {
        for (int b = 0; b < B; b++) {
            float k_acc_v = 0.0f, v_acc_v = 0.0f, r_pre_v = 0.0f;
            for (int j = 0; j < N_EMBD; j++) {
                const float kxv = __half2float(kx_buf[b * N_EMBD + j]);
                const float vxv = __half2float(vx_buf[b * N_EMBD + j]);
                const float rxv = __half2float(rx_buf[b * N_EMBD + j]);
                k_acc_v += __half2float(Kw[j * N_ATT + c]) * kxv;
                v_acc_v += __half2float(Vw[j * N_ATT + c]) * vxv;
                r_pre_v += __half2float(Rw[j * N_EMBD + c]) * rxv;
            }
            k_acc_buf[b * N_ATT + c] = k_acc_v;
            v_acc_buf[b * N_ATT + c] = v_acc_v;
            r_pre_buf[b * N_EMBD + c] = __float2half(r_pre_v);
        }
    }

    grid.sync();

    // ============ Phase 4: WKV recurrence (per-(b, channel)) ============
    //
    // Same math as wkv_kernel.cu / v2 inline WKV. Each thread handles
    // 1 channel × loops over B sequentially. State (aa, bb, pp) is
    // updated in place.
    //
    // Inputs: k_acc_buf[B][N_ATT], v_acc_buf[B][N_ATT] (fp32)
    // State:  aa[B][N_ATT], bb[B][N_ATT], pp[B][N_ATT] (fp32)
    // Output: y_buf[B][N_ATT] (fp32)

    // n_att == n_embd for RWKV-4-Pile-169M (both 768). Each thread's
    // channel `c` indexes into n_att (size N_ATT).
    if (c < N_ATT) {
        const float u = time_first[c];        // boost for current step
        const float w = time_decay[c];        // already -exp(raw); use directly

        for (int b = 0; b < B; b++) {
            const int idx = b * N_ATT + c;
            const float k_v = k_acc_buf[idx];
            const float v_v = v_acc_buf[idx];

            float aa_v = aa[idx];
            float bb_v = bb[idx];
            float pp_v = pp[idx];

            // y_t = (exp(pp - p1) * aa + exp(u + k - p1) * v) /
            //       (exp(pp - p1) * bb + exp(u + k - p1))
            const float ww = u + k_v;
            const float p1 = fmaxf(pp_v, ww);
            const float e1 = __expf(pp_v - p1);
            const float e2 = __expf(ww   - p1);
            y_buf[idx] = (e1 * aa_v + e2 * v_v) / (e1 * bb_v + e2);

            // State update: pp ← max(w + pp, k); aa, bb rebased.
            const float ww2 = w + pp_v;
            const float p2  = fmaxf(ww2, k_v);
            const float e1_2 = __expf(ww2 - p2);
            const float e2_2 = __expf(k_v - p2);
            aa[idx] = e1_2 * aa_v + e2_2 * v_v;
            bb[idx] = e1_2 * bb_v + e2_2;
            pp[idx] = p2;
        }
    }

    grid.sync();

    // ============ Phase 5: Ow matmul + residual ============
    //
    // For each (b, c): pre[k] = sigmoid(r_pre[b, k]) * y_buf[b, k]
    //                  acc    = sum_k pre[k] * Ow[k, c]
    //                  x_attn[b, c] = x_in[b, c] + acc
    //
    // Ow shape [n_att, C] row-major. Each thread handles 1 channel c,
    // loops over B and inner-product of length n_att. Scalar scaffold;
    // M5/M6 will swap to WMMA.

    if (c < N_EMBD) {
        for (int b = 0; b < B; b++) {
            float acc = 0.0f;
            for (int k = 0; k < N_ATT; k++) {
                const float r_k = __half2float(r_pre_buf[b * N_EMBD + k]);
                const float sig = 1.0f / (1.0f + __expf(-r_k));
                const float y_k = y_buf[b * N_ATT + k];
                const float w   = __half2float(Ow[k * N_EMBD + c]);
                acc += sig * y_k * w;
            }
            const float x_v = __half2float(x_in[b * N_EMBD + c]);
            x_attn_buf[b * N_EMBD + c] = __float2half(x_v + acc);
        }
    }

    grid.sync();

    // ============ Phase 6: LN2 partial + finalize ============
    //
    // Same pattern as Phase 1, reading from x_attn_buf.

    for (int b = 0; b < B; b++) {
        const float xv = __half2float(x_attn_buf[b * N_EMBD + c]);
        const float blk_sum = blk_red_sum(xv, warp_scratch);
        const float blk_sumsq = blk_red_sum(xv * xv, warp_scratch);
        if (tid == 0) {
            ln2_partials[b * V3_GRID_DIM + bid] = blk_sum;
            ln2_partial_sq[b * V3_GRID_DIM + bid] = blk_sumsq;
        }
    }
    grid.sync();

    {
        const int wid = tid >> 5;
        const int lane = tid & 31;
        if (wid == 0) {
            for (int b = lane; b < B; b += 32) {
                float p_sum = 0.0f;
                float p_sumsq = 0.0f;
                #pragma unroll
                for (int j = 0; j < V3_GRID_DIM; j++) {
                    p_sum   += ln2_partials[b * V3_GRID_DIM + j];
                    p_sumsq += ln2_partial_sq[b * V3_GRID_DIM + j];
                }
                const float m = p_sum / float(N_EMBD);
                const float v = p_sumsq / float(N_EMBD) - m * m;
                s_mean2[b]    = m;
                s_inv_std2[b] = rsqrtf(v + 1e-5f);
            }
        }
        __syncthreads();
    }

    // ============ Phase 7: LN2 apply + ffn-premix ============
    //
    // For each (b, c): xx2 = (x_attn - mean) * inv_std * ln2_w[c] + ln2_b[c]
    //                  ffn_kx = xx2 * ffn_tm_k + ffn_xx_prev * (1 - ffn_tm_k)
    //                  ffn_rx = xx2 * ffn_tm_r + ffn_xx_prev * (1 - ffn_tm_r)
    // Update ffn_xx in place with xx2 (T=1 last-token convention).

    const float ln2_w_c = __half2float(ln2_w[c]);
    const float ln2_b_c = __half2float(ln2_b[c]);
    const float ffn_tm_k_c = __half2float(ffn_tm_k[c]);
    const float ffn_tm_r_c = __half2float(ffn_tm_r[c]);

    for (int b = 0; b < B; b++) {
        const float xv = __half2float(x_attn_buf[b * N_EMBD + c]);
        const float xx2 = (xv - s_mean2[b]) * s_inv_std2[b] * ln2_w_c + ln2_b_c;
        const float ffn_xx_v = __half2float(ffn_xx[b * N_EMBD + c]);
        const float fkx = xx2 * ffn_tm_k_c + ffn_xx_v * (1.0f - ffn_tm_k_c);
        const float frx = xx2 * ffn_tm_r_c + ffn_xx_v * (1.0f - ffn_tm_r_c);
        ffn_kx_buf[b * N_EMBD + c] = __float2half(fkx);
        ffn_rx_buf[b * N_EMBD + c] = __float2half(frx);
        ffn_xx[b * N_EMBD + c] = __float2half(xx2);
    }

    grid.sync();

    // ============ Phase 8: ffn_K matmul + relu² ============
    //
    // ffn_K[b, k] = sum_j ffn_Kw[j, k] * ffn_kx[b, j]   (j ∈ [0, N_EMBD))
    // ffn_K_act[b, k] = max(0, ffn_K[b, k])²
    //
    // ffn_Kw shape [N_EMBD, N_FFN] row-major. N_FFN=3072 outputs >
    // V3_BLOCK_DIM*V3_GRID_DIM=768 threads, so each thread covers 4
    // output channels via stride 768.
    //
    // Iteration order: thread c_out covers c_out, c_out+768, c_out+1536, c_out+2304.

    if (c < N_EMBD) {
        #pragma unroll
        for (int k_out = c; k_out < N_FFN; k_out += N_EMBD) {
            for (int b = 0; b < B; b++) {
                float acc = 0.0f;
                for (int j = 0; j < N_EMBD; j++) {
                    const float w  = __half2float(ffn_Kw[j * N_FFN + k_out]);
                    const float xv = __half2float(ffn_kx_buf[b * N_EMBD + j]);
                    acc += w * xv;
                }
                const float relu = fmaxf(acc, 0.0f);
                ffn_K_act[b * N_FFN + k_out] = __float2half(relu * relu);
            }
        }
    }

    // ============ Phase 8b: ffn_R matmul + sigmoid ============
    //
    // ffn_R[b, c] = sum_j ffn_Rw[j, c] * ffn_rx[b, j]   (j ∈ [0, N_EMBD))
    // ffn_R_act[b, c] = sigmoid(ffn_R[b, c])
    //
    // ffn_Rw shape [N_EMBD, N_EMBD] row-major.
    // Same threading as Phase 5 (1 channel per thread).

    if (c < N_EMBD) {
        for (int b = 0; b < B; b++) {
            float acc = 0.0f;
            for (int j = 0; j < N_EMBD; j++) {
                const float w  = __half2float(ffn_Rw[j * N_EMBD + c]);
                const float xv = __half2float(ffn_rx_buf[b * N_EMBD + j]);
                acc += w * xv;
            }
            const float sig = 1.0f / (1.0f + __expf(-acc));
            ffn_R_act[b * N_EMBD + c] = __float2half(sig);
        }
    }

    grid.sync();

    // ============ Phase 9: ffn_V matmul ============
    //
    // ffn_V[b, c] = sum_k ffn_Vw[k, c] * ffn_K_act[b, k]   (k ∈ [0, N_FFN))
    //
    // ffn_Vw shape [N_FFN, N_EMBD] row-major. 1 channel per thread.

    if (c < N_EMBD) {
        for (int b = 0; b < B; b++) {
            float acc = 0.0f;
            for (int k = 0; k < N_FFN; k++) {
                const float w  = __half2float(ffn_Vw[k * N_EMBD + c]);
                const float kv = __half2float(ffn_K_act[b * N_FFN + k]);
                acc += w * kv;
            }
            ffn_V_out[b * N_EMBD + c] = __float2half(acc);
        }
    }

    grid.sync();

    // ============ Phase 10: final residual + x_out write ============
    //
    // x_out[b, c] = x_attn[b, c] + ffn_R_act[b, c] * ffn_V_out[b, c]

    if (c < N_EMBD) {
        for (int b = 0; b < B; b++) {
            const float xa = __half2float(x_attn_buf[b * N_EMBD + c]);
            const float vv = __half2float(ffn_V_out[b * N_EMBD + c]);
            const float rr = __half2float(ffn_R_act[b * N_EMBD + c]);
            x_out[b * N_EMBD + c] = __float2half(xa + rr * vv);
        }
    }
}

// ============================================================
// Launch wrapper + scratch-bytes helper
// ============================================================

extern "C" int v3_scratch_bytes(int B) {
    // Always allocate for V3_B_MAX (stable strides). Layout:
    //   2 × B_MAX × GRID  floats  (LN1 partials)
    //   3 × B_MAX × N_EMBD halves (kx, vx, rx)
    //   2 × B_MAX × N_ATT  floats (k_acc, v_acc)
    //   1 × B_MAX × N_EMBD halves (r_pre)
    //   1 × B_MAX × N_ATT  floats (y_buf)
    //   1 × B_MAX × N_EMBD halves (x_attn)         [M4]
    //   2 × B_MAX × GRID  floats  (LN2 partials)   [M4]
    //   2 × B_MAX × N_EMBD halves (ffn_kx, ffn_rx) [M4]
    //   1 × B_MAX × N_FFN  halves (ffn_K_act)      [M5]
    //   2 × B_MAX × N_EMBD halves (ffn_V_out, ffn_R_act) [M5]
    (void)B;
    const int bytes = (2 * V3_B_MAX * V3_GRID_DIM) * sizeof(float)
                    + (3 * V3_B_MAX * N_EMBD) * sizeof(__half)
                    + (2 * V3_B_MAX * N_ATT) * sizeof(float)
                    + (1 * V3_B_MAX * N_EMBD) * sizeof(__half)
                    + (1 * V3_B_MAX * N_ATT) * sizeof(float)
                    + (1 * V3_B_MAX * N_EMBD) * sizeof(__half)
                    + (2 * V3_B_MAX * V3_GRID_DIM) * sizeof(float)
                    + (2 * V3_B_MAX * N_EMBD) * sizeof(__half)
                    + (1 * V3_B_MAX * N_FFN) * sizeof(__half)
                    + (2 * V3_B_MAX * N_EMBD) * sizeof(__half);
    return bytes;
}

extern "C" int launch_rwkv4_layer_step_v3(
    int B,
    const void* x_in, void* x_out,
    void* att_xx, float* aa, float* bb, float* pp, void* ffn_xx,
    const void* ln1_w, const void* ln1_b,
    const void* tm_k, const void* tm_v, const void* tm_r,
    const float* time_decay, const float* time_first,
    const void* Kw, const void* Vw, const void* Rw, const void* Ow,
    const void* ln2_w, const void* ln2_b,
    const void* ffn_tm_k, const void* ffn_tm_r,
    const void* ffn_Kw, const void* ffn_Vw, const void* ffn_Rw,
    void* scratch_v3,
    cudaStream_t stream)
{
    void* args[] = {
        (void*)&B,
        (void*)&x_in, (void*)&x_out,
        (void*)&att_xx, (void*)&aa, (void*)&bb, (void*)&pp, (void*)&ffn_xx,
        (void*)&ln1_w, (void*)&ln1_b,
        (void*)&tm_k, (void*)&tm_v, (void*)&tm_r,
        (void*)&time_decay, (void*)&time_first,
        (void*)&Kw, (void*)&Vw, (void*)&Rw, (void*)&Ow,
        (void*)&ln2_w, (void*)&ln2_b,
        (void*)&ffn_tm_k, (void*)&ffn_tm_r,
        (void*)&ffn_Kw, (void*)&ffn_Vw, (void*)&ffn_Rw,
        (void*)&scratch_v3,
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)rwkv4_layer_step_v3_kernel,
        dim3(V3_GRID_DIM), dim3(V3_BLOCK_DIM),
        args, /*shmem=*/0, stream);
    return (int)err;
}
