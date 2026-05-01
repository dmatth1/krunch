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
// Deterministic softmax + CDF — see det_softmax_cdf.cu.
void launch_det_softmax_cdf(const void* logits, void* cdf,
                             int T, int V, int cdf_T_value,
                             cudaStream_t stream);
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
        // Pinned-algo cuBLAS path is faster on A10G (uses tuned TC
        // tile schedules) AND shape-stable (algo enum is fixed across
        // M values). Default ON; disable via KRUNCH_CUBLAS_PINNED=0
        // for benchmark comparison or if a future cuBLAS version
        // drops the chosen algo.
        static const bool USE_CUBLAS_PIN = []{
            const char* e = std::getenv("KRUNCH_CUBLAS_PINNED");
            return e == nullptr || std::string(e) != "0";
        }();
        if (USE_CUBLAS_PIN) {
            launch_det_matmul_cublas(x_c.data_ptr(), w_c.data_ptr(),
                                      out.data_ptr(), write_fp32,
                                      M, K, N, stream);
        } else {
            // Fallback WMMA kernel — bit-identical across M, slower.
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
    auto xx = at::layer_norm(x, {C}, ln1_w, ln1_b);  // [1, 1, C]
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
    static const bool USE_DET_3WAY = []{
        const char* e = std::getenv("KRUNCH_DETERMINISTIC_MATMUL");
        return e != nullptr && std::string(e) == "1";
    }();
    if (USE_DET_3WAY) {
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

    // State update via copy_ — caller's tensors see the new values.
    aa.copy_(aa_c);
    bb.copy_(bb_c);
    pp.copy_(pp_c);
    att_xx.copy_(xx_2d);

    auto y = y_flat.to(x.scalar_type());                 // [1, n_att]
    auto out_att = gemm_fp16(r * y, Ow);                // [1, C]
    auto x_after_att = x.view({1, C}) + out_att;          // [1, C]

    // LayerNorm 2
    auto xx2 = at::layer_norm(x_after_att, {C}, ln2_w, ln2_b);  // [1, C]
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

    // LayerNorm 1
    auto xx = at::layer_norm(x, {C}, ln1_w, ln1_b);                // [1, T, C]
    auto xx_flat = xx.contiguous().view({B * T, C});

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
    if (USE_DET_3WAY_P) {
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
    auto r = at::sigmoid(r_pre_flat).view({B, T, C});

    // WKV: kernel expects [B*T, C] in/out.
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto y_flat = at::empty_like(k_c);
    auto aa_c = aa.contiguous();
    auto bb_c = bb.contiguous();
    auto pp_c = pp.contiguous();
    auto td_c = time_decay.contiguous();
    auto tf_c = time_first.contiguous();

    static auto wkv_op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("rwkv::wkv_forward", "")
        .typed<void(int64_t, int64_t, int64_t,
                     at::Tensor&, at::Tensor&,
                     at::Tensor&, at::Tensor&, at::Tensor&,
                     at::Tensor&, at::Tensor&, at::Tensor&)>();
    wkv_op.call(B, T, C, td_c, tf_c, k_c, v_c, y_flat, aa_c, bb_c, pp_c);

    aa.copy_(aa_c);
    bb.copy_(bb_c);
    pp.copy_(pp_c);
    // att_xx update: last position's xx
    att_xx.copy_(xx.select(1, T - 1));

    auto y = y_flat.view({B, T, C}).to(x.scalar_type());            // [1, T, C]
    auto ry_flat = (r * y).contiguous().view({B * T, C});
    auto att_out = gemm_fp16(ry_flat, Ow).view({B, T, C});
    auto x_after_att = x + att_out;                                  // [1, T, C]

    // LayerNorm 2 + FFN
    auto xx2 = at::layer_norm(x_after_att, {C}, ln2_w, ln2_b);
    auto xx2_flat = xx2.contiguous().view({B * T, C});

    // Fused FFN premix: ffn_kx, ffn_rx = mix(xx2, ffn_xx, ffn_tm_*).
    auto ffn_kx_flat = at::empty({B * T, C}, xx2_flat.options());
    auto ffn_rx_flat = at::empty({B * T, C}, xx2_flat.options());
    launch_premix_2(
        xx2_flat.data_ptr(), ffn_xx.contiguous().data_ptr(),
        ffn_tm_k.contiguous().data_ptr(),
        ffn_tm_r.contiguous().data_ptr(),
        ffn_kx_flat.data_ptr(), ffn_rx_flat.data_ptr(),
        (int)B, (int)T, (int)C, stream0);

    auto r_ffn = at::sigmoid(gemm_fp16(ffn_rx_flat, ffn_Rw)).view({B, T, C});
    // FFN K + fused relu² (eliminates 2 ATen op launches: at::relu + at::pow).
    auto k_ffn = gemm_fp16(ffn_kx_flat, ffn_Kw);                    // [B*T, n_ffn]
    launch_relu_sq(k_ffn.data_ptr(), (int)k_ffn.numel(), stream0);
    auto v_ffn = gemm_fp16(k_ffn, ffn_Vw).view({B, T, C});           // [B, T, C]

    auto x_final = x_after_att + r_ffn * v_ffn;
    ffn_xx.copy_(xx2.select(1, T - 1));

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
    // Use TC kernel (handles arbitrary M/N via shared-mem staging).
    launch_det_matmul_tc(xc.data_ptr(), Wc.data_ptr(), out.data_ptr(),
                          write_fp32, M, K, N, stream);
    return out;
}

void register_layer_cpp(pybind11::module& m) {
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
