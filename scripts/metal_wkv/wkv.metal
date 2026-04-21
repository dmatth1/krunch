// Fused Metal Shading Language WKV kernel for RWKV-v4.
//
// Ported 1:1 from vendor/L3TC/models/RWKV_V4/cuda/wkv_cuda.cu (MIT, BlinkDL).
// One GPU thread per (batch, channel) pair — the whole T-loop runs inside
// that thread. Forward is pure feed-forward; backward uses Tmax scratch
// arrays (y, z, zexp) stored in thread-local/threadgroup memory.
//
// Tensor layout: k, v, y are (B, T, C) row-major. w, u are (C,). Same as
// the reference CUDA. All in fp32 (we convert bf16 tensors to fp32 before
// launching; the recurrence accumulator has to stay fp32 for stability).

#include <metal_stdlib>
using namespace metal;

// MIN_VALUE: must be low enough that exp(MIN_VALUE - finite) underflows
// to 0, but finite in fp32 so max() works normally. -1e38 was triggering
// Metal fast-math to produce NaN/inf at step 0. -30 is still below every
// practical k + u we'll see (k, u are tiny floats) and exp(-30) ≈ 1e-13
// which is below representable precision for the running sum but above
// fp32 denormal range, so no underflow misbehavior.
#define MIN_VALUE (-1e30f)

// Compile-time T_MAX — match the reference CUDA kernel (Tmax=2048).
// Scratch arrays y, z, zexp each take 4 * T_MAX bytes per thread.
#define T_MAX 2048

kernel void wkv_forward(
    device const float* w        [[ buffer(0) ]],  // (C,)
    device const float* u        [[ buffer(1) ]],  // (C,)
    device const float* k        [[ buffer(2) ]],  // (B, T, C)
    device const float* v        [[ buffer(3) ]],  // (B, T, C)
    device       float* y        [[ buffer(4) ]],  // (B, T, C)
    constant int& B              [[ buffer(5) ]],
    constant int& T              [[ buffer(6) ]],
    constant int& C              [[ buffer(7) ]],
    uint tid                      [[ thread_position_in_grid ]]
) {
    // tid indexes (b * C + c); one thread per (batch, channel).
    if (tid >= uint(B * C)) return;
    int b = int(tid) / C;
    int c = int(tid) - b * C;
    int offset = b * T * C + c;

    float uu = u[c];
    float ww = w[c];

    float p = 0.0f, q = 0.0f, o = MIN_VALUE;

    for (int i = 0; i < T; ++i) {
        int ii = i * C;
        float ki = k[offset + ii];
        float vi = v[offset + ii];

        // Current-step output with time_first bonus (u + k).
        float no = max(o, uu + ki);
        float A = exp(o - no);
        float Bv = exp(uu + ki - no);
        y[offset + ii] = (A * p + Bv * vi) / (A * q + Bv);

        // Update running state with time_decay step (w + o).
        no = max(ww + o, ki);
        A = exp(ww + o - no);
        Bv = exp(ki - no);
        p = A * p + Bv * vi;
        q = A * q + Bv;
        o = no;
    }
}

// Backward kernel — per (b, c) thread. Uses T_MAX thread-local scratch.
// Writes gk, gv elementwise; accumulates gw, gu into per-(b, c) outputs
// that the host code then reduces across batch (sum over B) to match
// the CUDA behavior in wkv_cuda.cu:108-110.
kernel void wkv_backward(
    device const float* w        [[ buffer(0) ]],
    device const float* u        [[ buffer(1) ]],
    device const float* k        [[ buffer(2) ]],
    device const float* v        [[ buffer(3) ]],
    device const float* gy       [[ buffer(4) ]],
    device       float* gw       [[ buffer(5) ]],  // (B, C)  — per-batch accum
    device       float* gu       [[ buffer(6) ]],  // (B, C)
    device       float* gk       [[ buffer(7) ]],  // (B, T, C)
    device       float* gv       [[ buffer(8) ]],  // (B, T, C)
    constant int& B              [[ buffer(9) ]],
    constant int& T              [[ buffer(10) ]],
    constant int& C              [[ buffer(11) ]],
    uint tid                      [[ thread_position_in_grid ]]
) {
    if (tid >= uint(B * C)) return;
    int b = int(tid) / C;
    int c = int(tid) - b * C;
    int offset = b * T * C + c;

    // NOTE: this is per-thread stack memory. With T_MAX=2048 and 3
    // float scratch arrays we burn 24 KB per thread. Metal allows up
    // to ~32 KB of thread-local memory per thread on Apple7+; tight
    // but within budget. If we ever need longer T we'd need to move
    // these into threadgroup memory with explicit partitioning.
    float y[T_MAX];
    float z[T_MAX];
    float zexp[T_MAX];

    float uu = u[c];
    float ww = w[c];
    float gw_acc = 0.0f, gu_acc = 0.0f;
    float p = 0.0f, q = 0.0f;
    float dpdw = 0.0f, dqdw = 0.0f;
    float o = MIN_VALUE;

    // Forward sweep — same loop as forward kernel, but additionally
    // records y[i], z[i], zexp[i] for the backward sweep AND accumulates
    // gw, gu contributions via the chain rule (dy/dw, dy/du at step i).
    for (int i = 0; i < T; ++i) {
        int ii = i * C;
        float ki = k[offset + ii];
        float vi = v[offset + ii];
        float gyi = gy[offset + ii];

        float no = max(o, ki + uu);
        float A = exp(o - no);
        float Bv = exp(ki + uu - no);

        float num = A * p + Bv * vi;
        float iden = 1.0f / (A * q + Bv);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = ki + uu - no;

        gw_acc += gyi * (dpdw - dqdw * y[i]) * iden * A;
        gu_acc += gyi * (vi - y[i]) * Bv * iden;

        // Advance running state.
        no = max(ww + o, ki);
        A = exp(ww + o - no);
        Bv = exp(ki - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + Bv * vi;
        q = A * q + Bv;
        o = no;
    }

    // Backward sweep — produces gk, gv. Mirrors wkv_cuda.cu:90-105.
    float gp = 0.0f, gq = 0.0f;
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; --i) {
        int ii = i * C;
        float ki = k[offset + ii];
        float vi = v[offset + ii];
        float gyi = gy[offset + ii];

        float A = gyi * z[i] * exp(zexp[i]);
        float Bv = exp(ki + o);
        gk[offset + ii] = A * (vi - y[i]) + Bv * (gp * vi + gq);
        gv[offset + ii] = A + Bv * gp;

        float no = max(ww + o, zexp[i] - ki - uu);
        A = exp(ww + o - no);
        Bv = gyi * z[i] * exp(zexp[i] - ki - uu - no);
        gp = A * gp + Bv;
        gq = A * gq - Bv * y[i];
        o = no;
    }

    // Finalize: multiply gw by w because the original tensor was -exp(w),
    // so chain rule contributes * d(-exp(w))/dw = -exp(w) = w_neg itself
    // (since we pre-exponentiated in the wrapper, _w passed in is w_neg).
    // See wkv_cuda.cu line 107-109 for the same trick.
    int offsetBC = b * C + c;
    gw[offsetBC] += gw_acc * w[c];
    gu[offsetBC] += gu_acc;
}
