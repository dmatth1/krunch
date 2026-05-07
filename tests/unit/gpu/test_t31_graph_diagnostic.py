"""T3.1 diagnostic: does CUDA-graph capture of `rwkv4_layer_step_cpp`
replay deterministically?

Two regimes tested in one run:
  (A) default WKV via rwkv::wkv_forward dispatcher
  (B) graph-safe WKV via launch_krunch_wkv_forward (KRUNCH_OWN_WKV=1)

For each regime, capture one layer step, replay 3×, compare to a
ground-truth direct-call sequence from the same starting state.

KRUNCH_DETERMINISTIC_MATMUL is forced ON so the gemm path is via our
own bit-stable WMMA kernel (NOT rwkv::gemm_fp16_cublas, which was
believed to be the other graph-unsafe op).

Outputs: per-regime, per-step `out_diff` max-abs vs reference. Pass
criterion: out_diff < 1e-3 across all 3 replay steps.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import sys
import torch
import krunch_ac_cuda
import rwkv.model  # noqa — registers rwkv::* ops


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def run_regime(label, use_own_wkv: bool):
    """Capture one layer step graph, replay 3×, compare each replayed
    output to a ground-truth direct call from clean state."""
    # Toggle our WKV. NOTE: must be set before the C++ static-const reads
    # it; this works on first invocation per process.
    if use_own_wkv:
        os.environ["KRUNCH_OWN_WKV"] = "1"
    else:
        os.environ.pop("KRUNCH_OWN_WKV", None)

    device = "cuda"
    torch.manual_seed(0)
    C = 768; n_att = 768; n_ffn = 3072

    def w(*shape, scale=0.05):
        return torch.randn(*shape, dtype=torch.float16, device=device) * scale

    ln1_w = torch.ones(C, dtype=torch.float16, device=device)
    ln1_b = torch.zeros(C, dtype=torch.float16, device=device)
    ln2_w = torch.ones(C, dtype=torch.float16, device=device)
    ln2_b = torch.zeros(C, dtype=torch.float16, device=device)
    tm_k = w(C); tm_v = w(C); tm_r = w(C)
    time_decay = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1 - 1.0
    time_first = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1
    Kw = w(C, n_att); Vw = w(C, n_att); Rw = w(C, C); Ow = w(n_att, C)
    ffn_tm_k = w(C); ffn_tm_r = w(C)
    ffn_Kw = w(C, n_ffn); ffn_Vw = w(n_ffn, C); ffn_Rw = w(C, C)
    layer = [ln1_w, ln1_b, tm_k, tm_v, tm_r, time_decay, time_first,
             Kw, Vw, Rw, Ow, ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
             ffn_Kw, ffn_Vw, ffn_Rw]

    def fresh():
        return (
            torch.zeros(1, C, dtype=torch.float16, device=device),
            torch.zeros(1, n_att, dtype=torch.float32, device=device),
            torch.zeros(1, n_att, dtype=torch.float32, device=device),
            torch.full((1, n_att), -1e30, dtype=torch.float32, device=device),
            torch.zeros(1, C, dtype=torch.float16, device=device),
        )

    # Ground truth: 3 direct calls from clean state.
    inputs = [torch.randn(1, 1, C, dtype=torch.float16, device=device) * 0.1
              for _ in range(3)]
    sa = list(fresh())
    refs = []
    for x in inputs:
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(x.contiguous(), *sa, *layer)
        refs.append(out.clone())

    # Graph capture w/ snapshot/restore.
    sb = list(fresh())
    x_buf = torch.empty(1, 1, C, dtype=torch.float16, device=device)
    out_buf = torch.empty(1, 1, C, dtype=torch.float16, device=device)

    snap = [t.clone() for t in sb]
    x_buf.copy_(inputs[0])
    for _ in range(2):  # warmup mutates sb
        o = krunch_ac_cuda.rwkv4_layer_step_cpp(x_buf, *sb, *layer)
        out_buf.copy_(o)
    torch.cuda.synchronize()
    for k in range(5):
        sb[k].copy_(snap[k])

    g = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream()
    cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        with torch.cuda.graph(g):
            o = krunch_ac_cuda.rwkv4_layer_step_cpp(x_buf, *sb, *layer)
            out_buf.copy_(o)
    torch.cuda.current_stream().wait_stream(cap)
    for k in range(5):
        sb[k].copy_(snap[k])

    print(f"\n[{label}] use_own_wkv={use_own_wkv}")
    diffs = []
    for t in range(3):
        x_buf.copy_(inputs[t])
        g.replay()
        torch.cuda.synchronize()
        d = maxabs(out_buf, refs[t])
        diffs.append(d)
        print(f"  step {t}: out_diff = {d:.6e}")
    return diffs


def main():
    print("T3.1 graph-replay diagnostic")
    print("KRUNCH_DETERMINISTIC_MATMUL =",
          os.environ.get("KRUNCH_DETERMINISTIC_MATMUL"))

    # NOTE: the static-const read of KRUNCH_OWN_WKV inside the C++ extension
    # is process-lifetime, so we can only test ONE regime per process.
    # Pick which based on argv: --regime A (default) or B.
    regime = "A"
    for arg in sys.argv[1:]:
        if arg.startswith("--regime="):
            regime = arg.split("=", 1)[1].upper()

    if regime == "A":
        d = run_regime("A: rwkv::wkv_forward (current default)", use_own_wkv=False)
    elif regime == "B":
        d = run_regime("B: launch_krunch_wkv_forward (graph-safe)", use_own_wkv=True)
    else:
        print(f"unknown regime {regime}")
        sys.exit(2)

    THRESHOLD = 1e-3
    pass_ = all(x < THRESHOLD for x in d)
    print(f"\n  verdict: {'PASS' if pass_ else 'FAIL'} "
          f"(max diff {max(d):.6e} vs threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()
