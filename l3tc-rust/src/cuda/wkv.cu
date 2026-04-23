// WKV forward kernel — fused RWKV-v4 time-mix recurrence.
//
// Port of BlinkDL/RWKV-LM/RWKV-v4/cuda/wkv_cuda.cu (Apache-2.0).
// Attribution: Peng Bo (BlinkDL). Original repo:
// https://github.com/BlinkDL/RWKV-LM
//
// Changes from the original:
// - Backward pass removed (we only need forward for compression).
// - `kernel_forward` renamed to `wkv_forward` and exported as
//   `extern "C"` so cudarc can resolve the symbol by name.
// - Templated F specialized to float — we run in mixed precision
//   (fp16 activations, fp32 accumulation); the kernel takes fp32
//   inputs and the Rust wrapper does the fp16→fp32 promotion.
//
// Compiled at build time by `build.rs` via nvcc to `wkv.ptx`, which
// is `include_bytes!`'d into the Rust binary and loaded at runtime
// by cudarc.

#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

extern "C" __global__ void wkv_forward(
        const int B, const int T, const int C,
        const float *__restrict__ const _w,
        const float *__restrict__ const _u,
        const float *__restrict__ const _k,
        const float *__restrict__ const _v,
        float *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = _u[_c];
    float w = _w[_c];
    const float *__restrict__ const k = _k + _offset;
    const float *__restrict__ const v = _v + _offset;
    float *__restrict__ const y = _y + _offset;

    // p, q: running numerator/denominator of the softmax-weighted
    // rolling sum, divided by exp(o) to keep magnitudes in float range.
    // o: current log-max for numerical stability.
    float p = 0, q = 0, o = MIN_VALUE;

    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        // Emit y[i] = softmax-weighted average of past v's, with
        // current token k/v getting weight exp(u) relative to past.
        float no = fmaxf(o, u + k[ii]);
        float A = expf(o - no);
        float B = expf(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        // Update running state: discount past by exp(w), add current.
        no = fmaxf(w + o, k[ii]);
        A = expf(w + o - no);
        B = expf(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}
