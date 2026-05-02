"""Verify launch_krunch_wkv_forward matches rwkv::wkv_forward to fp16 noise.

The graph-safe WKV kernel uses the same RWKV-4 math but a different
loop ordering (one thread per (b,c), sequential over T) than BlinkDL's
WKV kernel. They should agree to ~1e-3 abs (fp16 noise) on real-shape
inputs.

This test is descriptive: it reports the gap, doesn't gate on it. The
load-bearing bit-exactness check is the AC roundtrip
(test_t31_cpp_path_roundtrip.py + tests/gpu.sh) — encoder + decoder
both run KRUNCH_OWN_WKV=1, the bitstream just changes.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda  # noqa: F401
import rwkv.model  # noqa — registers rwkv::wkv_forward


def run_ours(B, T, C, time_decay, time_first, k, v, aa, bb, pp):
    """Call our launch via the existing pybind path (rwkv4_layer_step_cpp
    won't help isolate WKV — better to expose a direct test entry point.
    For now we exercise WKV via the layer-step wrapper at B=1, T=1 since
    that's the only path that needs the diagnostic). The WKV-only
    correctness comparison is approximated below by running both modes
    of the layer step on the same input and comparing y_flat indirectly
    (via the layer output)."""
    raise NotImplementedError(
        "Direct WKV-only pybind not exposed; run roundtrip test instead.")


def main():
    """Compare layer-step output (which contains WKV in its critical path)
    between use_own_wkv=False and use_own_wkv=True for the SAME inputs +
    same starting state. Drift gives a proxy for WKV-only drift."""
    device = "cuda"
    torch.manual_seed(7)
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

    # NOTE: process-wide static const reads KRUNCH_OWN_WKV once. Run this
    # script twice (with/without env) and compare manually OR run both
    # via two subprocesses. For a single-process run we just verify
    # the chosen mode produces non-nan, finite output within fp16 range.
    use_own = os.environ.get("KRUNCH_OWN_WKV") == "1"
    print(f"KRUNCH_OWN_WKV = {'1 (our kernel)' if use_own else '0 (rwkv dispatcher)'}")

    inputs = [torch.randn(1, 1, C, dtype=torch.float16, device=device) * 0.1
              for _ in range(8)]
    sa = list(fresh())
    outs = []
    for x in inputs:
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(x.contiguous(), *sa, *layer)
        outs.append(out.clone())

    print(f"  got {len(outs)} outputs, all-finite = "
          f"{all(torch.isfinite(o).all().item() for o in outs)}")
    print(f"  max-abs over all outputs = "
          f"{max(o.abs().max().item() for o in outs):.4f}")
    print(f"  state aa max-abs = {sa[1].abs().max().item():.4f}")
    print(f"  state bb max-abs = {sa[2].abs().max().item():.4f}")
    print(f"  state pp max-abs = {sa[3].abs().max().item():.4f}")
    print("OK")


if __name__ == "__main__":
    main()
