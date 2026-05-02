// C++ orchestration of one RWKV-4 layer's forward at B=1, T=1.
// Replaces the Python `_att_seq_batched + _ffn_seq_batched` chain
// (~7 ATen op calls × ~75 us Python overhead each = ~525 us/layer).
// One C++ call: same ATen ops dispatch to cuBLAS/cuDNN, but no Python
// round-trips between them.
//
// Specialized for T=1 (decompress single-step). The Python forward
// path keeps the T>1 packed form for compress; this path is used
// only on the decode side.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <string>

// Fused elementwise kernels — see premix_kernels.cu.
extern "C" {
void launch_premix_3(const void*, const void*, const void*, const void*, const void*,
                      void*, void*, void*, int, int, int, cudaStream_t);
void launch_premix_2(const void*, const void*, const void*, const void*,
                      void*, void*, int, int, int, cudaStream_t);
void launch_relu_sq(void*, int, cudaStream_t);
// Deterministic matmul — see det_matmul.cu.
void launch_det_matmul(const void* x, const void* W, void* y,
                        int write_fp32, int M, int K, int N, cudaStream_t stream);
// Tensor-Core-accelerated deterministic matmul — see det_matmul_tc.cu.
// Requires M to be a multiple of 16 (caller must pad).
void launch_det_matmul_tc(const void* A, const void* B, void* C,
                           int write_fp32, int M, int K, int N,
                           cudaStream_t stream);
// 3-way batched TC matmul — see det_matmul_tc_3way.cu.
void launch_det_matmul_tc_3way(
    const void* x0, const void* x1, const void* x2,
    const void* W0, const void* W1, const void* W2,
    void* y0, void* y1, void* y2,
    int wf0, int wf1, int wf2,
    int M, int K, int N, cudaStream_t stream);
// Pinned-algo cuBLAS GEMM — see det_matmul_cublas.cu.
void launch_det_matmul_cublas(const void* x, const void* W, void* y,
                               int write_fp32, int M, int K, int N,
                               cudaStream_t stream);
void set_det_matmul_cublas_algo(int algo_id);
// 32×32-tile WMMA matmul — see det_matmul_tc_v2.cu.
void launch_det_matmul_tc_v2(const void* A, const void* B, void* C,
                              int write_fp32, int M, int K, int N,
                              cudaStream_t stream);
// Multi-warp 64x64-tile WMMA matmul — see det_matmul_tc_mw.cu.
// Only K=768 and K=3072 are supported; intended for the head matmul shape.
void launch_det_matmul_tc_mw(const void* A, const void* B, void* C,
                              int write_fp32, int M, int K, int N,
                              cudaStream_t stream);
// 128x128-tile / 8-warp WMMA matmul — see det_matmul_tc_xl.cu.
void launch_det_matmul_tc_xl(const void* A, const void* B, void* C,
                              int write_fp32, int M, int K, int N,
                              cudaStream_t stream);
// cp.async double-buffered K-loop (sm_80+ only) — see det_matmul_tc_async.cu.
// Closes the cuBLAS gap on A10G/L40S/A100/H100. K∈{768,3072}, M+N must be
// multiples of 64 (or partial-tile masking handles bounds).
void launch_det_matmul_tc_async(const void* A, const void* B, void* C,
                                 int write_fp32, int M, int K, int N,
                                 cudaStream_t stream);
// bf16 variant of det_matmul_tc_async — see det_matmul_tc_bf16.cu.
void launch_det_matmul_tc_async_bf16(
    const void* A, const void* B, void* C,
    int write_fp32, int M, int K, int N, cudaStream_t stream);
// uint8 weights + inline dequant + fp16 WMMA — see det_matmul_tc_uint8.cu.
void launch_det_matmul_tc_uint8(
    const void* A, const void* W_u8, const void* scale, const void* offset,
    void* C, int write_fp32, int M, int K, int N, cudaStream_t stream);
// 3-way fused cp.async + WMMA matmul — see det_matmul_tc_3way_async.cu.
// Computes (Y0=A0@B0, Y1=A1@B1, Y2=A2@B2) in one kernel launch.
// All inputs/outputs share shape; saves 2 launches + amortizes K-axis
// memory load latency. Used by KVR matmul (kx, vx, rx all M=B, K=n_embd,
// N=n_att). sm_80+ only.
void launch_det_matmul_tc_3way_async(
    const void* A0, const void* A1, const void* A2,
    const void* B0, const void* B1, const void* B2,
    void* Y0, void* Y1, void* Y2,
    int wf0, int wf1, int wf2,
    int M, int K, int N, cudaStream_t stream);
// Custom row-wise LayerNorm — see layer_norm.cu.
void launch_layer_norm(const void* x, const void* gamma, const void* beta,
                        void* y, int N, int C, float eps,
                        cudaStream_t stream);
// Deterministic softmax + CDF — see det_softmax_cdf.cu.
void launch_det_softmax_cdf(const void* logits, void* cdf,
                             int T, int V, int cdf_T_value,
                             cudaStream_t stream);
// Graph-safe WKV — see wkv_kernel.cu.
void launch_krunch_wkv_forward(
    int B, int T, int C,
    const float* time_decay, const float* time_first,
    const float* k, const float* v, float* y,
    float* aa, float* bb, float* pp,
    cudaStream_t stream);
}

// Toggle our own graph-safe WKV via KRUNCH_OWN_WKV=1. Default OFF — only
// flip when we need CUDA-graph capture of the layer step (KRUNCH_CPP_GRAPH=1).
// MUST match across encoder + decoder or AC roundtrip breaks.
static bool use_own_wkv() {
    static const bool USE = []{
        const char* e = std::getenv("KRUNCH_OWN_WKV");
        return e != nullptr && std::string(e) == "1";
    }();
    return USE;
}

// =============================================================================
// Phase profiler (T3.7) — gated by KRUNCH_PHASE_PROFILE=1.
//
// Records cudaEvents between phases of rwkv4_layer_step_cpp; sums elapsed
// times across all calls. Read via `read_phase_times()`. Adds per-call
// stream sync overhead (~50 µs) — only enable for diagnostic runs.
// =============================================================================

constexpr int N_PHASES = 11;
static const char* PHASE_NAMES[N_PHASES] = {
    "ln1", "premix3", "kvr_matmul", "sigmoid_r",
    "wkv", "ow_matmul_residual", "ln2", "premix2",
    "ffn_R_matmul_sigmoid", "ffn_K_matmul_relu", "ffn_V_matmul_residual"
};

static cudaEvent_t g_phase_events[N_PHASES + 1];
static double g_phase_sum_ms[N_PHASES] = {0};
static int g_phase_call_count = 0;
static bool g_phase_events_init = false;

static bool phase_profile_on() {
    static const bool E = []{
        const char* e = std::getenv("KRUNCH_PHASE_PROFILE");
        return e != nullptr && std::string(e) == "1";
    }();
    return E;
}

static inline void rec_phase(int idx, cudaStream_t s) {
    if (!phase_profile_on()) return;
    if (!g_phase_events_init) {
        for (int i = 0; i <= N_PHASES; i++) cudaEventCreate(&g_phase_events[i]);
        g_phase_events_init = true;
    }
    cudaEventRecord(g_phase_events[idx], s);
}

static inline void finalize_phase_call() {
    if (!phase_profile_on()) return;
    cudaEventSynchronize(g_phase_events[N_PHASES]);
    for (int i = 0; i < N_PHASES; i++) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, g_phase_events[i], g_phase_events[i+1]);
        g_phase_sum_ms[i] += ms;
    }
    g_phase_call_count++;
}

// Custom row-wise LayerNorm wrapper. Replaces at::layer_norm in
// the layer-step hot path: same per-row math (mean → var → normalize
// → affine), but as one custom kernel call (one block per row,
// fp32 reductions) instead of going through the PyTorch dispatcher.
// Bit-stable across batch dim by construction.
//
// Bit-pattern differs from at::layer_norm (different reduction order
// + we accumulate in fp32 throughout) — both compress and decompress
// must use the SAME code path or AC roundtrip breaks. Default OFF
// (KRUNCH_LAYERNORM_CUSTOM=1 to enable).
static at::Tensor layer_norm_cust(at::Tensor x, at::Tensor gamma,
                                   at::Tensor beta, double eps) {
    auto x_c = x.contiguous();
    const auto sizes = x_c.sizes();
    const int64_t C = sizes[sizes.size() - 1];
    const int64_t N = x_c.numel() / C;
    auto y = at::empty_like(x_c);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_layer_norm(x_c.data_ptr(), gamma.contiguous().data_ptr(),
                       beta.contiguous().data_ptr(), y.data_ptr(),
                       (int)N, (int)C, (float)eps, stream);
    return y;
}

static bool use_custom_layer_norm() {
    static const bool USE_CUSTOM_LN = []{
        const char* e = std::getenv("KRUNCH_LAYERNORM_CUSTOM");
        return e != nullptr && std::string(e) == "1";
    }();
    return USE_CUSTOM_LN;
}

static at::Tensor maybe_layer_norm(at::Tensor x, at::IntArrayRef shape,
                                    at::Tensor gamma, at::Tensor beta) {
    if (use_custom_layer_norm()) {
        return layer_norm_cust(x, gamma, beta, 1e-5);
    }
    return at::layer_norm(x, shape, gamma, beta);
}

// Match BlinkDL's matmul exactly: call rwkv::gemm_fp16_cublas via the
// dispatcher (different cuBLAS algorithm than gemm_fp16). Without this
// match, encode/decode logits drift by 2-4 abs and AC breaks.
// Decides per-call whether to use cuBLAS (fast) or the deterministic
// kernel (bit-identical across shapes, slower). Toggle via env
// `KRUNCH_DETERMINISTIC_MATMUL=1` — set on both compress and decompress
// paths to ensure encoder/decoder produce identical CDFs.
static at::Tensor gemm_fp16(at::Tensor x, at::Tensor w,
                             c10::optional<at::ScalarType> out_dtype = c10::nullopt) {
    auto x_c = x.contiguous();
    auto w_c = w.contiguous();
    const auto dtype = out_dtype.has_value() ? out_dtype.value() : x_c.scalar_type();
    const int M = (int)x_c.size(0);
    const int K = (int)x_c.size(1);
    const int N = (int)w_c.size(-1);
    auto out = at::empty({M, N}, x_c.options().dtype(dtype));

    static const bool USE_DET = []{
        const char* e = std::getenv("KRUNCH_DETERMINISTIC_MATMUL");
        return e != nullptr && std::string(e) == "1";
    }();

    if (USE_DET) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        // Routing precedence (T3.6/T3.7 fix 2026-05-01):
        //   1. cp.async kernel (sm_80+) — KRUNCH_HEAD_ASYNC default ON,
        //      runs at K∈{768,3072}, N % 8 == 0, M ≥ 64. Layer matmul
        //      shapes (N=768/3072) qualify; closes 1.7-2× of cuBLAS gap.
        //   2. KRUNCH_TC_V2 / KRUNCH_CUBLAS_PINNED — opt-in benches.
        //   3. Default 16×16 WMMA single-warp.
        // PRIOR BUG: layer-matmul gemm_fp16 fell through to the default
        // det_matmul_tc, bypassing async/MW/XL. Phase profile 2026-05-01
        // identified this — async kernel was correct + bit-stable but
        // never ran in production for layer matmuls.
        static const bool USE_ASYNC_GEMM = []{
            const char* e = std::getenv("KRUNCH_HEAD_ASYNC");
            return e == nullptr || std::string(e) != "0";
        }();
        const bool gemm_async_aligned = ((K == 768 || K == 3072)
                                          && (N % 8 == 0) && M >= 64);
        static const bool USE_CUBLAS_PIN = []{
            const char* e = std::getenv("KRUNCH_CUBLAS_PINNED");
            return e != nullptr && std::string(e) == "1";
        }();
        static const bool USE_TC_V2 = []{
            const char* e = std::getenv("KRUNCH_TC_V2");
            return e != nullptr && std::string(e) == "1";
        }();
        if (USE_ASYNC_GEMM && gemm_async_aligned) {
            launch_det_matmul_tc_async(x_c.data_ptr(), w_c.data_ptr(),
                                        out.data_ptr(), write_fp32,
                                        M, K, N, stream);
        } else if (USE_CUBLAS_PIN) {
            launch_det_matmul_cublas(x_c.data_ptr(), w_c.data_ptr(),
                                      out.data_ptr(), write_fp32,
                                      M, K, N, stream);
        } else if (USE_TC_V2 && (K == 768 || K == 3072)) {
            launch_det_matmul_tc_v2(x_c.data_ptr(), w_c.data_ptr(),
                                     out.data_ptr(), write_fp32,
                                     M, K, N, stream);
        } else {
            launch_det_matmul_tc(x_c.data_ptr(), w_c.data_ptr(),
                                  out.data_ptr(), write_fp32, M, K, N, stream);
        }
    } else {
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("rwkv::gemm_fp16_cublas", "")
            .typed<void(at::Tensor, at::Tensor, at::Tensor)>();
        op.call(x_c, w_c, out);
    }
    return out;
}

// State updates are in place via copy_(). Returns new x. The wkv state
// tensors (aa, bb, pp) and att_xx, ffn_xx all mutate via copy_.
at::Tensor rwkv4_layer_step_cpp_t1(
    at::Tensor x,                      // [1, 1, C] fp16
    at::Tensor att_xx,                 // [1, C]    fp16  (mutated)
    at::Tensor aa,                     // [1, n_att] fp32 (mutated)
    at::Tensor bb,                     // [1, n_att] fp32 (mutated)
    at::Tensor pp,                     // [1, n_att] fp32 (mutated)
    at::Tensor ffn_xx,                 // [1, C]    fp16  (mutated)
    at::Tensor ln1_w, at::Tensor ln1_b,
    at::Tensor tm_k, at::Tensor tm_v, at::Tensor tm_r,
    at::Tensor time_decay, at::Tensor time_first,
    at::Tensor Kw, at::Tensor Vw, at::Tensor Rw, at::Tensor Ow,
    at::Tensor ln2_w, at::Tensor ln2_b,
    at::Tensor ffn_tm_k, at::Tensor ffn_tm_r,
    at::Tensor ffn_Kw, at::Tensor ffn_Vw, at::Tensor ffn_Rw)
{
    const int64_t C = x.size(2);

    // LayerNorm 1
    auto xx = maybe_layer_norm(x, {C}, ln1_w, ln1_b);  // [1, 1, C]
    auto xx_2d = xx.contiguous().view({1, C});        // [1, C]

    // Time-mix premix via the SAME fused kernel used by the packed path.
    // This guarantees bit-identical kx/vx/rx between encode (T=N packed)
    // and decode (T=1 stepped) — required for AC roundtrip.
    auto kx = at::empty({1, C}, xx_2d.options());
    auto vx = at::empty({1, C}, xx_2d.options());
    auto rx = at::empty({1, C}, xx_2d.options());
    cudaStream_t stream_t1 = at::cuda::getCurrentCUDAStream();
    launch_premix_3(
        xx_2d.data_ptr(), att_xx.contiguous().data_ptr(),
        tm_k.contiguous().data_ptr(),
        tm_v.contiguous().data_ptr(),
        tm_r.contiguous().data_ptr(),
        kx.data_ptr(), vx.data_ptr(), rx.data_ptr(),
        1, 1, (int)C, stream_t1);

    // K, V, R linears — fused into one TC kernel launch (3 GEMMs in 1).
    // Outputs:
    //   k = kx @ Kw  → fp32 [1, n_att]
    //   v = vx @ Vw  → fp32 [1, n_att]
    //   r_pre = rx @ Rw → fp16 [1, C]   (sigmoid applied below)
    const int n_att = (int)Kw.size(-1);
    auto k = at::empty({1, n_att}, kx.options().dtype(at::kFloat));
    auto v = at::empty({1, n_att}, vx.options().dtype(at::kFloat));
    auto r_pre = at::empty({1, C}, rx.options());
    auto Kw_c = Kw.contiguous();
    auto Vw_c = Vw.contiguous();
    auto Rw_c = Rw.contiguous();
    // T=1 path: M=1 doesn't qualify for 3way_async (WMMA needs M≥16).
    // 3way_async routing only happens in the packed rwkv4_layer_step_cpp
    // function below.
    static const bool DISABLE_3WAY = []{
        const char* e = std::getenv("KRUNCH_NO_3WAY");
        return (e == nullptr) ? false : std::string(e) == "1";
    }();
    static const bool USE_DET_3WAY = []{
        const char* e = std::getenv("KRUNCH_DETERMINISTIC_MATMUL");
        return e != nullptr && std::string(e) == "1";
    }();
    if (USE_DET_3WAY && !DISABLE_3WAY) {
        // n_att == C for RWKV-4 (768 == 768), so M=N=K=768; K dim of input
        // must match. Kw is [C, n_att], Vw is [C, n_att], Rw is [C, C].
        // All have K=C=768 and N=n_att=C=768 → same shape, batchable.
        TORCH_CHECK(n_att == (int)C, "3-way fusion needs n_att == C");
        launch_det_matmul_tc_3way(
            kx.data_ptr(), vx.data_ptr(), rx.data_ptr(),
            Kw_c.data_ptr(), Vw_c.data_ptr(), Rw_c.data_ptr(),
            k.data_ptr(), v.data_ptr(), r_pre.data_ptr(),
            /*wf0=*/1, /*wf1=*/1, /*wf2=*/0,
            /*M=*/1, /*K=*/(int)C, /*N=*/n_att, stream_t1);
    } else {
        // Fallback: 3 separate matmuls via the dispatcher.
        k.copy_(gemm_fp16(kx, Kw, at::kFloat));
        v.copy_(gemm_fp16(vx, Vw, at::kFloat));
        r_pre.copy_(gemm_fp16(rx, Rw));
    }
    auto r = at::sigmoid(r_pre);                                   // [1, C]

    // WKV via rwkv's CUDA op (handles B*T*C = 1*1*C = C elements).
    auto k_flat = k.contiguous();
    auto v_flat = v.contiguous();
    auto y_flat = at::empty_like(k_flat);
    auto aa_c = aa.contiguous();
    auto bb_c = bb.contiguous();
    auto pp_c = pp.contiguous();
    auto td_c = time_decay.contiguous();
    auto tf_c = time_first.contiguous();

    if (use_own_wkv()) {
        launch_krunch_wkv_forward(
            /*B=*/1, /*T=*/1, (int)C,
            td_c.data_ptr<float>(), tf_c.data_ptr<float>(),
            k_flat.data_ptr<float>(), v_flat.data_ptr<float>(),
            y_flat.data_ptr<float>(),
            aa_c.data_ptr<float>(), bb_c.data_ptr<float>(), pp_c.data_ptr<float>(),
            stream_t1);
    } else {
        // Lookup the wkv_forward op once and call it.
        static auto wkv_op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("rwkv::wkv_forward", "")
            .typed<void(int64_t, int64_t, int64_t,
                         at::Tensor&, at::Tensor&,
                         at::Tensor&, at::Tensor&, at::Tensor&,
                         at::Tensor&, at::Tensor&, at::Tensor&)>();
        wkv_op.call(1, 1, C,
                    td_c, tf_c, k_flat, v_flat, y_flat,
                    aa_c, bb_c, pp_c);
    }

    // State update via copy_ — caller's tensors see the new values.
    aa.copy_(aa_c);
    bb.copy_(bb_c);
    pp.copy_(pp_c);
    att_xx.copy_(xx_2d);

    auto y = y_flat.to(x.scalar_type());                 // [1, n_att]
    auto out_att = gemm_fp16(r * y, Ow);                // [1, C]
    auto x_after_att = x.view({1, C}) + out_att;          // [1, C]

    // LayerNorm 2
    auto xx2 = maybe_layer_norm(x_after_att, {C}, ln2_w, ln2_b);  // [1, C]
    auto xx2_c = xx2.contiguous();

    // FFN premix via fused kernel (matches packed path bit-for-bit).
    auto ffn_kx = at::empty({1, C}, xx2_c.options());
    auto ffn_rx = at::empty({1, C}, xx2_c.options());
    launch_premix_2(
        xx2_c.data_ptr(), ffn_xx.contiguous().data_ptr(),
        ffn_tm_k.contiguous().data_ptr(),
        ffn_tm_r.contiguous().data_ptr(),
        ffn_kx.data_ptr(), ffn_rx.data_ptr(),
        1, 1, (int)C, stream_t1);

    // FFN linears (relu_sq fused kernel matches packed path)
    auto r_ffn = at::sigmoid(gemm_fp16(ffn_rx, ffn_Rw));     // [1, C]
    auto k_ffn = gemm_fp16(ffn_kx, ffn_Kw);                   // [1, n_ffn]
    launch_relu_sq(k_ffn.data_ptr(), (int)k_ffn.numel(), stream_t1);
    auto v_ffn = gemm_fp16(k_ffn, ffn_Vw);                    // [1, C]

    auto x_final = x_after_att + r_ffn * v_ffn;                // [1, C]

    // FFN state update
    ffn_xx.copy_(xx2);

    return x_final.view({1, 1, C});
}

// =============================================================================
// Generalized: handles any T (compress packed forward), B=1.
// Same code path as rwkv4_layer_step_cpp_t1 → numerical match by
// construction. Replaces _att_seq_batched + _ffn_seq_batched.
//
// Inputs:
//   x:           [1, T, C]  fp16
//   att_xx:      [1, C]     fp16  (mutated to xx[:, T-1, :])
//   aa, bb, pp:  [1, n_att] fp32  (mutated to post-T state)
//   ffn_xx:      [1, C]     fp16  (mutated to xx2[:, T-1, :])
// Returns x_out: [1, T, C] fp16
at::Tensor rwkv4_layer_step_cpp(
    at::Tensor x,
    at::Tensor att_xx,
    at::Tensor aa, at::Tensor bb, at::Tensor pp,
    at::Tensor ffn_xx,
    at::Tensor ln1_w, at::Tensor ln1_b,
    at::Tensor tm_k, at::Tensor tm_v, at::Tensor tm_r,
    at::Tensor time_decay, at::Tensor time_first,
    at::Tensor Kw, at::Tensor Vw, at::Tensor Rw, at::Tensor Ow,
    at::Tensor ln2_w, at::Tensor ln2_b,
    at::Tensor ffn_tm_k, at::Tensor ffn_tm_r,
    at::Tensor ffn_Kw, at::Tensor ffn_Vw, at::Tensor ffn_Rw)
{
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    // B>1 supported as of 2026-04-30 for cross-chunk batched decode.
    // Premix kernel + WKV op + matmul tile schedules all generalize
    // over B; state tensors must be [B, C] / [B, n_att].

    cudaStream_t prof_stream = at::cuda::getCurrentCUDAStream();
    rec_phase(0, prof_stream);

    // LayerNorm 1
    auto xx = maybe_layer_norm(x, {C}, ln1_w, ln1_b);                // [1, T, C]
    auto xx_flat = xx.contiguous().view({B * T, C});
    rec_phase(1, prof_stream);

    // Fused premix: kx, vx, rx = mix(xx, sx_full, tm_*).
    // sx_full[t=0] = att_xx, sx_full[t>0] = xx[t-1] — kernel handles both
    // without materializing sx_full (no at::cat).
    auto kx_flat = at::empty({B * T, C}, xx_flat.options());
    auto vx_flat = at::empty({B * T, C}, xx_flat.options());
    auto rx_flat = at::empty({B * T, C}, xx_flat.options());
    cudaStream_t stream0 = at::cuda::getCurrentCUDAStream();
    launch_premix_3(
        xx_flat.data_ptr(), att_xx.contiguous().data_ptr(),
        tm_k.contiguous().data_ptr(),
        tm_v.contiguous().data_ptr(),
        tm_r.contiguous().data_ptr(),
        kx_flat.data_ptr(), vx_flat.data_ptr(), rx_flat.data_ptr(),
        (int)B, (int)T, (int)C, stream0);
    rec_phase(2, prof_stream);

    // K, V, R linears — fused into one TC kernel launch (3 GEMMs in 1).
    const int n_att_p = (int)Kw.size(-1);
    auto k = at::empty({B * T, n_att_p}, kx_flat.options().dtype(at::kFloat));
    auto v = at::empty({B * T, n_att_p}, vx_flat.options().dtype(at::kFloat));
    auto r_pre_flat = at::empty({B * T, C}, rx_flat.options());
    auto Kw_c_p = Kw.contiguous();
    auto Vw_c_p = Vw.contiguous();
    auto Rw_c_p = Rw.contiguous();
    static const bool USE_DET_3WAY_P = []{
        const char* e = std::getenv("KRUNCH_DETERMINISTIC_MATMUL");
        return e != nullptr && std::string(e) == "1";
    }();
    // 3-way KVR matmul routing precedence (T3.S2a, 2026-05-02):
    //   1. 3way_async (cp.async + WMMA, sm_80+, M≥16) — fastest
    //   2. Old 3way (single-warp WMMA) — fallback
    //   3. KRUNCH_NO_3WAY=1 → fall through to 3 separate gemm_fp16
    //      (each routed to async via gemm_fp16's own routing)
    static const bool USE_3WAY_ASYNC = []{
        const char* e = std::getenv("KRUNCH_3WAY_ASYNC");
        return e == nullptr || std::string(e) != "0";
    }();
    static const bool DISABLE_3WAY_P = []{
        const char* e = std::getenv("KRUNCH_NO_3WAY");
        return (e == nullptr) ? false : std::string(e) == "1";
    }();
    // M ≥ 256 threshold: at M=128 (decompress B=128) the 64×64 tile only
    // produces 24 blocks (2 M-tiles × 12 N-tiles) vs old 3way's 16×16
    // tiles producing 384 blocks. A10G has 80 SMs; 24 blocks underfills
    // (regression measured: 47 → 43.5 KB/s decompress when 3way_async
    // was used). Compress packed M=1024 produces 192 blocks → fully
    // saturates SMs. Tune via KRUNCH_3WAY_ASYNC_M_MIN.
    static const int THREEWAY_ASYNC_M_MIN_P = []{
        const char* e = std::getenv("KRUNCH_3WAY_ASYNC_M_MIN");
        return (e != nullptr) ? atoi(e) : 256;
    }();
    const bool can_3way_async_p = USE_3WAY_ASYNC
                                   && (B * T) >= THREEWAY_ASYNC_M_MIN_P
                                   && (n_att_p == 768 || n_att_p == 3072);
    if (USE_DET_3WAY_P && !DISABLE_3WAY_P && can_3way_async_p) {
        TORCH_CHECK(n_att_p == (int)C, "3-way fusion needs n_att == C");
        launch_det_matmul_tc_3way_async(
            kx_flat.data_ptr(), vx_flat.data_ptr(), rx_flat.data_ptr(),
            Kw_c_p.data_ptr(), Vw_c_p.data_ptr(), Rw_c_p.data_ptr(),
            k.data_ptr(), v.data_ptr(), r_pre_flat.data_ptr(),
            /*wf0=*/1, /*wf1=*/1, /*wf2=*/0,
            /*M=*/(int)(B * T), /*K=*/(int)C, /*N=*/n_att_p, stream0);
    } else if (USE_DET_3WAY_P && !DISABLE_3WAY_P) {
        TORCH_CHECK(n_att_p == (int)C, "3-way fusion needs n_att == C");
        launch_det_matmul_tc_3way(
            kx_flat.data_ptr(), vx_flat.data_ptr(), rx_flat.data_ptr(),
            Kw_c_p.data_ptr(), Vw_c_p.data_ptr(), Rw_c_p.data_ptr(),
            k.data_ptr(), v.data_ptr(), r_pre_flat.data_ptr(),
            /*wf0=*/1, /*wf1=*/1, /*wf2=*/0,
            /*M=*/(int)(B * T), /*K=*/(int)C, /*N=*/n_att_p, stream0);
    } else {
        k.copy_(gemm_fp16(kx_flat, Kw, at::kFloat));
        v.copy_(gemm_fp16(vx_flat, Vw, at::kFloat));
        r_pre_flat.copy_(gemm_fp16(rx_flat, Rw));
    }
    rec_phase(3, prof_stream);
    auto r = at::sigmoid(r_pre_flat).view({B, T, C});
    rec_phase(4, prof_stream);

    // WKV: kernel expects [B*T, C] in/out.
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto y_flat = at::empty_like(k_c);
    auto aa_c = aa.contiguous();
    auto bb_c = bb.contiguous();
    auto pp_c = pp.contiguous();
    auto td_c = time_decay.contiguous();
    auto tf_c = time_first.contiguous();

    if (use_own_wkv()) {
        cudaStream_t stream_pk = at::cuda::getCurrentCUDAStream();
        launch_krunch_wkv_forward(
            (int)B, (int)T, (int)C,
            td_c.data_ptr<float>(), tf_c.data_ptr<float>(),
            k_c.data_ptr<float>(), v_c.data_ptr<float>(), y_flat.data_ptr<float>(),
            aa_c.data_ptr<float>(), bb_c.data_ptr<float>(), pp_c.data_ptr<float>(),
            stream_pk);
    } else {
        static auto wkv_op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("rwkv::wkv_forward", "")
            .typed<void(int64_t, int64_t, int64_t,
                         at::Tensor&, at::Tensor&,
                         at::Tensor&, at::Tensor&, at::Tensor&,
                         at::Tensor&, at::Tensor&, at::Tensor&)>();
        wkv_op.call(B, T, C, td_c, tf_c, k_c, v_c, y_flat, aa_c, bb_c, pp_c);
    }

    aa.copy_(aa_c);
    bb.copy_(bb_c);
    pp.copy_(pp_c);
    // att_xx update: last position's xx
    att_xx.copy_(xx.select(1, T - 1));
    rec_phase(5, prof_stream);

    auto y = y_flat.view({B, T, C}).to(x.scalar_type());            // [1, T, C]
    auto ry_flat = (r * y).contiguous().view({B * T, C});
    auto att_out = gemm_fp16(ry_flat, Ow).view({B, T, C});
    auto x_after_att = x + att_out;                                  // [1, T, C]
    rec_phase(6, prof_stream);

    // LayerNorm 2 + FFN
    auto xx2 = maybe_layer_norm(x_after_att, {C}, ln2_w, ln2_b);
    auto xx2_flat = xx2.contiguous().view({B * T, C});
    rec_phase(7, prof_stream);

    // Fused FFN premix: ffn_kx, ffn_rx = mix(xx2, ffn_xx, ffn_tm_*).
    auto ffn_kx_flat = at::empty({B * T, C}, xx2_flat.options());
    auto ffn_rx_flat = at::empty({B * T, C}, xx2_flat.options());
    launch_premix_2(
        xx2_flat.data_ptr(), ffn_xx.contiguous().data_ptr(),
        ffn_tm_k.contiguous().data_ptr(),
        ffn_tm_r.contiguous().data_ptr(),
        ffn_kx_flat.data_ptr(), ffn_rx_flat.data_ptr(),
        (int)B, (int)T, (int)C, stream0);
    rec_phase(8, prof_stream);

    auto r_ffn = at::sigmoid(gemm_fp16(ffn_rx_flat, ffn_Rw)).view({B, T, C});
    rec_phase(9, prof_stream);
    // FFN K + fused relu² (eliminates 2 ATen op launches: at::relu + at::pow).
    auto k_ffn = gemm_fp16(ffn_kx_flat, ffn_Kw);                    // [B*T, n_ffn]
    launch_relu_sq(k_ffn.data_ptr(), (int)k_ffn.numel(), stream0);
    rec_phase(10, prof_stream);
    auto v_ffn = gemm_fp16(k_ffn, ffn_Vw).view({B, T, C});           // [B, T, C]

    auto x_final = x_after_att + r_ffn * v_ffn;
    ffn_xx.copy_(xx2.select(1, T - 1));
    rec_phase(11, prof_stream);
    finalize_phase_call();

    return x_final;
}

// =============================================================================
// CUDA Graph wrapper for the packed forward.
// Captures one layer's forward at a fixed T into a graph the first time
// it's called with that T; subsequent calls replay the graph (one
// launch instead of ~13 ATen op launches → saves ~3-4 ms/call on T4).
//
// Caveat: graphs are bound to specific tensor data pointers. We require
// the caller to pass STATIC pre-allocated input/output buffers + state
// tensors so they don't change across calls.
// =============================================================================

#include <ATen/cuda/CUDAGraph.h>
#include <unordered_map>
#include <memory>

struct LayerGraphCache {
    std::unordered_map<int64_t, std::shared_ptr<at::cuda::CUDAGraph>> graphs;
};

// Per-layer-index cache (12 layers in the model). Indexed by layer_id × 100000 + T.
static std::unordered_map<int64_t, std::shared_ptr<at::cuda::CUDAGraph>> g_graphs;

at::Tensor rwkv4_layer_step_cpp_graphed(
    int64_t layer_id,
    at::Tensor x_buf,                  // [1, T, C] fp16 — static buffer
    at::Tensor x_out_buf,               // [1, T, C] fp16 — static output
    at::Tensor att_xx, at::Tensor aa, at::Tensor bb, at::Tensor pp,
    at::Tensor ffn_xx,
    at::Tensor ln1_w, at::Tensor ln1_b,
    at::Tensor tm_k, at::Tensor tm_v, at::Tensor tm_r,
    at::Tensor time_decay, at::Tensor time_first,
    at::Tensor Kw, at::Tensor Vw, at::Tensor Rw, at::Tensor Ow,
    at::Tensor ln2_w, at::Tensor ln2_b,
    at::Tensor ffn_tm_k, at::Tensor ffn_tm_r,
    at::Tensor ffn_Kw, at::Tensor ffn_Vw, at::Tensor ffn_Rw)
{
    const int64_t T = x_buf.size(1);
    const int64_t key = layer_id * 100000 + T;

    auto it = g_graphs.find(key);
    if (it == g_graphs.end()) {
        // Warm up on default stream first (cuBLAS workspaces, alloc).
        for (int i = 0; i < 2; i++) {
            auto out = rwkv4_layer_step_cpp(
                x_buf, att_xx, aa, bb, pp, ffn_xx,
                ln1_w, ln1_b, tm_k, tm_v, tm_r, time_decay, time_first,
                Kw, Vw, Rw, Ow,
                ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
                ffn_Kw, ffn_Vw, ffn_Rw);
            x_out_buf.copy_(out);
        }
        at::cuda::getCurrentCUDAStream().synchronize();

        // Capture on a side stream — required by CUDA Graph API.
        // Replay later happens on whatever stream is current.
        auto capture_stream = at::cuda::getStreamFromPool();
        auto saved_stream = at::cuda::getCurrentCUDAStream();
        at::cuda::setCurrentCUDAStream(capture_stream);

        auto graph = std::make_shared<at::cuda::CUDAGraph>();
        graph->capture_begin();
        auto out = rwkv4_layer_step_cpp(
            x_buf, att_xx, aa, bb, pp, ffn_xx,
            ln1_w, ln1_b, tm_k, tm_v, tm_r, time_decay, time_first,
            Kw, Vw, Rw, Ow,
            ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
            ffn_Kw, ffn_Vw, ffn_Rw);
        x_out_buf.copy_(out);
        graph->capture_end();

        at::cuda::setCurrentCUDAStream(saved_stream);
        g_graphs[key] = graph;
    } else {
        it->second->replay();
    }

    return x_out_buf;
}

// Reset the per-process graph cache (call when weights change or
// between unrelated runs).
void clear_layer_graph_cache() {
    g_graphs.clear();
}

// Direct binding for launch_det_matmul, used by tests to verify
// shape-invariance of the kernel without going through the full layer.
// Computes y = x @ W. x: [M,K] fp16, W: [K,N] fp16, returns [M,N] in
// the requested dtype (fp16 or fp32).
static at::Tensor det_matmul_py(at::Tensor x, at::Tensor W,
                                 c10::optional<at::ScalarType> out_dtype) {
    auto xc = x.contiguous();
    auto Wc = W.contiguous();
    const int M = (int)xc.size(0);
    const int K = (int)xc.size(1);
    const int N = (int)Wc.size(-1);
    const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
    auto out = at::empty({M, N}, xc.options().dtype(dtype));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
    // Routing:
    //   - Head matmul w/ LARGE M (K∈{768,3072}, N≥16384, M≥256): multi-warp
    //     64×64-tile kernel (det_matmul_tc_mw). Compress packed forward
    //     hits this (M=SEQ_BATCH=1024). 1.78× speedup on T4 microbench.
    //     Excludes decompress stepped (M=B≤256) where 64-row tiles waste
    //     loads (regression measured: T4 B=64 decompress 13.6→12.1 KB/s
    //     when MW was used). M-threshold default 256 — a 64×64 kernel on
    //     M=128 leaves half its rows padded and runs slower than 1-warp
    //     16×16 tiles which fully use M=128. Tune via KRUNCH_HEAD_MW_M_MIN.
    //   - KRUNCH_TC_V2=1: opt-in to 32×32-tile single-warp v2.
    //   - Default: original 16×16-tile single-warp det_matmul_tc.
    // KRUNCH_HEAD_MW=0 disables MW routing entirely (safety valve).
    static const bool USE_TC_V2 = []{
        const char* e = std::getenv("KRUNCH_TC_V2");
        return e != nullptr && std::string(e) == "1";
    }();
    static const bool DISABLE_HEAD_MW = []{
        const char* e = std::getenv("KRUNCH_HEAD_MW");
        return e != nullptr && std::string(e) == "0";
    }();
    static const int HEAD_MW_M_MIN = []{
        const char* e = std::getenv("KRUNCH_HEAD_MW_M_MIN");
        return (e != nullptr) ? atoi(e) : 256;
    }();
    // XL kernel landed but on T4 doesn't beat MW (no cp.async on sm_75).
    // Default OFF; flip on A10G/L40S/H100 (sm_80+) where cp.async-style
    // optimizations actually pay off. Currently off pending sm_80+ path
    // in det_matmul_tc_xl.cu.
    static const bool USE_XL = []{
        const char* e = std::getenv("KRUNCH_HEAD_XL");
        return e != nullptr && std::string(e) == "1";
    }();
    // cp.async kernel (sm_80+). Default ON; set KRUNCH_HEAD_ASYNC=0 to disable.
    // Kernel itself is empty on sm_75 — routing should detect and fall back.
    static const bool USE_ASYNC = []{
        const char* e = std::getenv("KRUNCH_HEAD_ASYNC");
        return e == nullptr || std::string(e) != "0";
    }();
    static const int HEAD_XL_M_MIN = []{
        const char* e = std::getenv("KRUNCH_HEAD_XL_M_MIN");
        return (e != nullptr) ? atoi(e) : 256;
    }();
    const bool head_shape = (K == 768 || K == 3072) && N >= 16384
                             && M >= HEAD_MW_M_MIN;
    // Async kernel needs 16-byte aligned global loads. For B, row stride
    // is N halves = 2*N bytes; aligned iff N % 8 == 0. Layer matmul shapes
    // (N=768, N=3072) qualify; head matmul N=50277 doesn't.
    const bool async_aligned = ((K == 768 || K == 3072) && (N % 8 == 0)
                                 && M >= 64);
    // Async + N-padding (T3.7 follow-up): if head_shape (large N) and N not
    // 8-aligned (e.g., 50277), pad W internally to N_pad = (N+7)&~7, run
    // async, slice output. Saves vs MW path; small per-call alloc overhead.
    const bool head_shape_unaligned = (K == 768 || K == 3072)
                                       && N >= 16384 && M >= 64
                                       && (N % 8 != 0);
    if (USE_ASYNC && async_aligned) {
        launch_det_matmul_tc_async(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                     write_fp32, M, K, N, stream);
    } else if (USE_ASYNC && head_shape_unaligned) {
        const int N_pad = (N + 7) & ~7;
        auto W_pad = at::zeros({K, N_pad}, Wc.options());
        // Copy W's N cols into first N cols of W_pad (rest stays zero).
        W_pad.narrow(1, 0, N).copy_(Wc);
        auto out_pad = at::empty({M, N_pad}, out.options());
        launch_det_matmul_tc_async(xc.data_ptr(), W_pad.data_ptr(),
                                     out_pad.data_ptr(),
                                     write_fp32, M, K, N_pad, stream);
        out.copy_(out_pad.narrow(1, 0, N));
    } else if (head_shape && USE_XL && M >= HEAD_XL_M_MIN) {
        // 128×128 tile / 8-warp kernel — better M=1024 perf than MW.
        launch_det_matmul_tc_xl(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                 write_fp32, M, K, N, stream);
    } else if (head_shape && !DISABLE_HEAD_MW) {
        launch_det_matmul_tc_mw(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                 write_fp32, M, K, N, stream);
    } else if (USE_TC_V2 && (K == 768 || K == 3072)) {
        launch_det_matmul_tc_v2(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                 write_fp32, M, K, N, stream);
    } else {
        launch_det_matmul_tc(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                              write_fp32, M, K, N, stream);
    }
    return out;
}

void register_layer_cpp(pybind11::module& m) {
    m.def("read_phase_times", []() {
        pybind11::dict d;
        for (int i = 0; i < N_PHASES; i++) {
            d[PHASE_NAMES[i]] = g_phase_sum_ms[i];
        }
        d["call_count"] = g_phase_call_count;
        return d;
    }, "Read accumulated phase timings (ms). Set KRUNCH_PHASE_PROFILE=1 to enable.");
    m.def("reset_phase_times", []() {
        for (int i = 0; i < N_PHASES; i++) g_phase_sum_ms[i] = 0;
        g_phase_call_count = 0;
    }, "Reset phase-time accumulators to zero.");

    m.def("rwkv4_layer_step_cpp_t1", &rwkv4_layer_step_cpp_t1,
          "C++ orchestration of one RWKV-4 layer at B=1, T=1.");
    m.def("rwkv4_layer_step_cpp", &rwkv4_layer_step_cpp,
          "C++ orchestration of one RWKV-4 layer at B=1, any T.");
    m.def("rwkv4_layer_step_cpp_graphed", &rwkv4_layer_step_cpp_graphed,
          "Graph-captured version of rwkv4_layer_step_cpp; replays after first call.");
    m.def("clear_layer_graph_cache", &clear_layer_graph_cache,
          "Drop all cached CUDA graphs (call when weights change).");
    m.def("det_matmul", &det_matmul_py,
          "Deterministic shape-invariant matmul: y = x @ W.",
          pybind11::arg("x"), pybind11::arg("W"),
          pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_bf16", [](at::Tensor x, at::Tensor W,
                                 c10::optional<at::ScalarType> out_dtype) {
        // bf16 cp.async + WMMA matmul (sm_80+). Used by the bf16 microbench
        // to compare against det_matmul (fp16) for speed + numerical drift.
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_async_bf16(
            xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
            write_fp32, M, K, N, stream);
        return out;
    }, "bf16 deterministic matmul (sm_80+ only). y = x @ W with both "
       "inputs bf16 and fp32 accumulator.",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_uint8", [](at::Tensor x, at::Tensor W_u8,
                                   at::Tensor scale, at::Tensor offset,
                                   c10::optional<at::ScalarType> out_dtype) {
        // uint8 weights + inline dequant + fp16 WMMA. Used by uint8 microbench.
        // x: [M, K] fp16, W_u8: [K, N] uint8, scale/offset: [K] fp16.
        auto xc = x.contiguous();
        auto Wc = W_u8.contiguous();
        auto sc = scale.contiguous();
        auto oc = offset.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_uint8(
            xc.data_ptr(), Wc.data_ptr(), sc.data_ptr(), oc.data_ptr(),
            out.data_ptr(), write_fp32, M, K, N, stream);
        return out;
    }, "uint8 weight matmul with per-input-channel scale + offset. "
       "x fp16, W_u8 uint8, scale + offset fp16.",
       pybind11::arg("x"), pybind11::arg("W_u8"),
       pybind11::arg("scale"), pybind11::arg("offset"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_cublas_pinned", [](at::Tensor x, at::Tensor W,
                                          c10::optional<at::ScalarType> out_dtype) {
        // Test-only binding: forces the cuBLAS pinned-algo path so we
        // can verify shape-stability of THE actual code path used by
        // gemm_fp16 when KRUNCH_CUBLAS_PINNED=1. The plain det_matmul
        // pybind goes through launch_det_matmul_tc (WMMA), so testing
        // that doesn't tell us anything about cuBLAS pinned.
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_cublas(xc.data_ptr(), Wc.data_ptr(),
                                  out.data_ptr(), write_fp32,
                                  M, K, N, stream);
        return out;
    }, "Run cuBLAS pinned-algo matmul (the path gemm_fp16 takes when "
       "KRUNCH_CUBLAS_PINNED=1). Use to verify shape-stability before "
       "trusting the path in production.",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("set_cublas_pinned_algo", [](int algo_id) {
        set_det_matmul_cublas_algo(algo_id);
    }, "Override the cuBLAS algo enum used by det_matmul_cublas_pinned + "
       "the in-engine cuBLAS path. Useful for sweeping algos.",
       pybind11::arg("algo_id"));
    m.def("det_matmul_tc_3way_async", [](
            at::Tensor A0, at::Tensor A1, at::Tensor A2,
            at::Tensor B0, at::Tensor B1, at::Tensor B2,
            c10::optional<at::ScalarType> out_dtype) {
        // 3-way fused cp.async + WMMA matmul for testing.
        // All inputs share [M, K], all weights share [K, N].
        auto a0 = A0.contiguous();
        auto a1 = A1.contiguous();
        auto a2 = A2.contiguous();
        auto b0 = B0.contiguous();
        auto b1 = B1.contiguous();
        auto b2 = B2.contiguous();
        const int M = (int)a0.size(0);
        const int K = (int)a0.size(1);
        const int N = (int)b0.size(-1);
        TORCH_CHECK(K == 768 || K == 3072, "3way_async needs K∈{768,3072}");
        TORCH_CHECK(a1.size(0) == M && a2.size(0) == M, "M must match");
        TORCH_CHECK(b1.size(-1) == N && b2.size(-1) == N, "N must match");
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : a0.scalar_type();
        auto y0 = at::empty({M, N}, a0.options().dtype(dtype));
        auto y1 = at::empty({M, N}, a0.options().dtype(dtype));
        auto y2 = at::empty({M, N}, a0.options().dtype(dtype));
        const int wf = (dtype == at::kFloat) ? 1 : 0;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        launch_det_matmul_tc_3way_async(
            a0.data_ptr(), a1.data_ptr(), a2.data_ptr(),
            b0.data_ptr(), b1.data_ptr(), b2.data_ptr(),
            y0.data_ptr(), y1.data_ptr(), y2.data_ptr(),
            wf, wf, wf, M, K, N, stream);
        return std::make_tuple(y0, y1, y2);
    }, "3-way fused cp.async + WMMA matmul (sm_80+).",
       pybind11::arg("A0"), pybind11::arg("A1"), pybind11::arg("A2"),
       pybind11::arg("B0"), pybind11::arg("B1"), pybind11::arg("B2"),
       pybind11::arg("out_dtype") = c10::nullopt);

    m.def("det_matmul_tc_async", [](at::Tensor x, at::Tensor W,
                                     c10::optional<at::ScalarType> out_dtype) {
        // Direct binding for cp.async double-buffered kernel (sm_80+).
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        TORCH_CHECK(K == 768 || K == 3072, "det_matmul_tc_async needs K∈{768,3072}");
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_async(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                    write_fp32, M, K, N, stream);
        return out;
    }, "cp.async double-buffered WMMA matmul (sm_80+).",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_tc_xl", [](at::Tensor x, at::Tensor W,
                                  c10::optional<at::ScalarType> out_dtype) {
        // Direct binding for testing det_matmul_tc_xl (8-warp, 128×128 tile).
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        TORCH_CHECK(K == 768 || K == 3072, "det_matmul_tc_xl needs K∈{768,3072}");
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_xl(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                 write_fp32, M, K, N, stream);
        return out;
    }, "8-warp 128x128-tile WMMA matmul. Bit-stable across M.",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_tc_mw", [](at::Tensor x, at::Tensor W,
                                  c10::optional<at::ScalarType> out_dtype) {
        // Direct binding for testing det_matmul_tc_mw (4-warps-per-block,
        // 64×64 tile). Requires K=768 or K=3072. Verifies bit-stability across
        // M without going through head_shape auto-routing.
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        TORCH_CHECK(K == 768 || K == 3072, "det_matmul_tc_mw only supports K∈{768,3072}");
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_mw(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                                 write_fp32, M, K, N, stream);
        return out;
    }, "Multi-warp 64x64-tile WMMA matmul (4 warps/block). Bit-stable across M.",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_matmul_tc_v2", [](at::Tensor x, at::Tensor W,
                                  c10::optional<at::ScalarType> out_dtype) {
        // 32×32-tile WMMA. Bit-stable across M; produces DIFFERENT bits
        // than det_matmul (16×16 tile). Used for testing the v2 kernel
        // shape-stability + speed.
        auto xc = x.contiguous();
        auto Wc = W.contiguous();
        const int M = (int)xc.size(0);
        const int K = (int)xc.size(1);
        const int N = (int)Wc.size(-1);
        const auto dtype = out_dtype.has_value() ? out_dtype.value() : xc.scalar_type();
        auto out = at::empty({M, N}, xc.options().dtype(dtype));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int write_fp32 = (dtype == at::kFloat) ? 1 : 0;
        launch_det_matmul_tc_v2(xc.data_ptr(), Wc.data_ptr(),
                                 out.data_ptr(), write_fp32,
                                 M, K, N, stream);
        return out;
    }, "32x32-tile WMMA matmul, bit-stable across M.",
       pybind11::arg("x"), pybind11::arg("W"),
       pybind11::arg("out_dtype") = c10::nullopt);
    m.def("det_softmax_cdf", [](at::Tensor logits_TxV, int cdf_T_value) {
        TORCH_CHECK(logits_TxV.dim() == 2 && logits_TxV.scalar_type() == at::kHalf);
        auto x = logits_TxV.contiguous();
        const int T = (int)x.size(0);
        const int V = (int)x.size(1);
        auto out = at::empty({T, V + 1}, x.options().dtype(at::kInt));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        launch_det_softmax_cdf(x.data_ptr(), out.data_ptr(),
                                T, V, cdf_T_value, stream);
        return out;
    }, "Deterministic batched softmax + CDF (per-row, shape-invariant).",
       pybind11::arg("logits"), pybind11::arg("cdf_T_value"));
}
