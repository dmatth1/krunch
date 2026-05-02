// Graph-safe drop-in replacement for `rwkv::wkv_forward`.
//
// Why exists: the rwkv package's wkv_forward dispatches via
// c10::Dispatcher and either uses host-side state or stream-ordered
// memory that doesn't replay deterministically inside torch.cuda.graph.
// Captured graphs of `rwkv4_layer_step_cpp` diverge from a fresh call
// on step 0 (out_diff = 0.18, then explodes). Replacing the dispatcher
// call with this kernel — pure CUDA, no workspaces, no dispatcher —
// makes the whole layer step graph-safe.
//
// Numerical contract: bit-identical regardless of T. One thread per
// (batch, channel) pair walks T timesteps sequentially in fp32, so
// reduction order for any (b, c) is t=0, 1, 2, … T-1 — independent of
// how the caller batches. Encoder packed (T=N) and decoder stepped
// (T=1) produce the same per-(b,c) state evolution, preserving AC
// roundtrip.
//
// Drift vs `rwkv::wkv_forward` is fp16-noise-level on the boundary
// between fp32 inner-loop and fp16-cast at the layer output (verified
// by scripts/test_own_wkv.py); both paths use the same RWKV-4 math.
// AC encoder + decoder both must run with the same WKV implementation
// or roundtrip breaks — toggle via KRUNCH_OWN_WKV=1 on both sides.

#include <cuda_runtime.h>

extern "C" __global__ void krunch_wkv_forward_kernel(
    const float* __restrict__ time_decay,   // [C]
    const float* __restrict__ time_first,   // [C]
    const float* __restrict__ k,            // [B, T, C]
    const float* __restrict__ v,            // [B, T, C]
    float*       __restrict__ y,            // [B, T, C]
    float*       __restrict__ aa,           // [B, C] mutated
    float*       __restrict__ bb,           // [B, C] mutated
    float*       __restrict__ pp,           // [B, C] mutated
    int B, int T, int C)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= C || b >= B) return;

    // u = time_first (boost for current step). w = time_decay AS PASSED IN
    // (already -exp(raw) by rwkv's model loader; we use it directly, no exp).
    // This matches BlinkDL/RWKV-LM `kernel_wkv_forward`'s math.
    const float w = time_decay[c];
    const float u = time_first[c];

    const int state_off = b * C + c;
    float aa_v = aa[state_off];
    float bb_v = bb[state_off];
    float pp_v = pp[state_off];

    for (int t = 0; t < T; t++) {
        const int idx = (b * T + t) * C + c;
        const float k_v = k[idx];
        const float v_v = v[idx];

        // Output: y_t = (exp(pp - p) * aa + exp(u + k - p) * v) /
        //              (exp(pp - p) * bb + exp(u + k - p))
        const float ww = u + k_v;
        const float p1 = fmaxf(pp_v, ww);
        const float e1 = __expf(pp_v - p1);
        const float e2 = __expf(ww - p1);
        y[idx] = (e1 * aa_v + e2 * v_v) / (e1 * bb_v + e2);

        // State update: pp ← max(w + pp, k); aa, bb rebased to new pp.
        const float ww2 = w + pp_v;
        const float p2 = fmaxf(ww2, k_v);
        const float e1_2 = __expf(ww2 - p2);
        const float e2_2 = __expf(k_v - p2);
        aa_v = e1_2 * aa_v + e2_2 * v_v;
        bb_v = e1_2 * bb_v + e2_2;
        pp_v = p2;
    }

    aa[state_off] = aa_v;
    bb[state_off] = bb_v;
    pp[state_off] = pp_v;
}

extern "C" void launch_krunch_wkv_forward(
    int B, int T, int C,
    const float* time_decay, const float* time_first,
    const float* k, const float* v, float* y,
    float* aa, float* bb, float* pp,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks_x = (C + threads - 1) / threads;
    dim3 grid(blocks_x, B);
    krunch_wkv_forward_kernel<<<grid, threads, 0, stream>>>(
        time_decay, time_first, k, v, y, aa, bb, pp, B, T, C);
}
