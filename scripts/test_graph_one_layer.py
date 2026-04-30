"""Minimal repro: capture one layer step via torch.cuda.graph, replay,
compare output to a direct call from clean state.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda
import rwkv.model  # noqa


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def main():
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

    # GROUND TRUTH: 3 sequential T=1 calls from clean state.
    inputs = [torch.randn(1, 1, C, dtype=torch.float16, device=device) * 0.1
              for _ in range(3)]
    sa = list(fresh())
    refs = []
    for x in inputs:
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(x.contiguous(), *sa, *layer)
        refs.append(out.clone())

    # GRAPH: capture once, replay 3×, compare outputs.
    sb = list(fresh())
    x_buf = torch.empty(1, 1, C, dtype=torch.float16, device=device)
    out_buf = torch.empty(1, 1, C, dtype=torch.float16, device=device)

    # Snapshot before warmup
    snap = [t.clone() for t in sb]

    # Warmup ×2 to allocate workspaces (mutates sb)
    x_buf.copy_(inputs[0])
    for _ in range(2):
        o = krunch_ac_cuda.rwkv4_layer_step_cpp(x_buf, *sb, *layer)
        out_buf.copy_(o)
    torch.cuda.synchronize()

    # Restore state
    for k in range(5):
        sb[k].copy_(snap[k])

    # Capture
    g = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream()
    cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        with torch.cuda.graph(g):
            o = krunch_ac_cuda.rwkv4_layer_step_cpp(x_buf, *sb, *layer)
            out_buf.copy_(o)
    torch.cuda.current_stream().wait_stream(cap)

    # Restore state again (capture mutated it)
    for k in range(5):
        sb[k].copy_(snap[k])

    # Replay 3×, compare to refs
    print("Capture+replay vs ground truth:")
    for t in range(3):
        x_buf.copy_(inputs[t])
        g.replay()
        torch.cuda.synchronize()
        d = maxabs(out_buf, refs[t])
        print(f"  step {t}: out_diff={d:.6e}")

if __name__ == "__main__":
    main()
