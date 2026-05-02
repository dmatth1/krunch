"""Single-layer CUDA-graph determinism bisect.

forward_stepped_graphed_v2 is documented "KNOWN BROKEN — same failing
token sequence as v1. Most likely cause: cuBLAS workspaces or the
rwkv::wkv_forward dispatcher lookup don't replay deterministically."

This test isolates the suspect by:
1. Running rwkv4_layer_step_cpp 5x against fixed inputs (raw runs).
2. Capturing a CUDA graph after warmup, replaying 5x.
3. Comparing raw[i] vs graphed[i] step-by-step.

Run under different env-flag combinations to identify which third-party
component breaks graph capture:

  KRUNCH_DETERMINISTIC_MATMUL=0/1   (gemm_fp16_cublas vs our det_matmul)
  KRUNCH_OWN_WKV=0/1                 (rwkv::wkv_forward vs our krunch_wkv)

If (DET=1, OWN=1) is deterministic and others aren't, third-party
kernels are the culprit and the fix is: set both flags during capture.

Threshold: max-abs diff < 1e-3 step-by-step.
"""
import os, sys
import torch


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"  KRUNCH_DETERMINISTIC_MATMUL={os.environ.get('KRUNCH_DETERMINISTIC_MATMUL', '0')}")
    print(f"  KRUNCH_OWN_WKV={os.environ.get('KRUNCH_OWN_WKV', '0')}")
    print(f"  KRUNCH_HEAD_ASYNC={os.environ.get('KRUNCH_HEAD_ASYNC', '1')}")
    print()

    import krunch_ac_cuda

    device = "cuda"
    torch.manual_seed(42)
    C = 768
    n_att = 768
    n_ffn = 3072

    # Fixed inputs
    x = torch.randn(1, 1, C, dtype=torch.float16, device=device) * 0.1
    att_xx = torch.zeros(1, C, dtype=torch.float16, device=device)
    aa = torch.zeros(1, n_att, dtype=torch.float32, device=device)
    bb = torch.zeros(1, n_att, dtype=torch.float32, device=device)
    pp = torch.full((1, n_att), -1e30, dtype=torch.float32, device=device)
    ffn_xx = torch.zeros(1, C, dtype=torch.float16, device=device)

    # Layer weights — use realistic-shaped random weights
    def hf(*shape):
        return (torch.randn(*shape, device=device) * 0.05).to(torch.float16)
    def hf_one(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=device) * 0.5 + 1.0
    def f32(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device) * 0.3

    layer = [
        hf_one(C),                  # ln1.weight
        hf(C),                       # ln1.bias
        torch.rand(C, dtype=torch.float16, device=device),  # tm_k
        torch.rand(C, dtype=torch.float16, device=device),  # tm_v
        torch.rand(C, dtype=torch.float16, device=device),  # tm_r
        -torch.rand(n_att, dtype=torch.float32, device=device) - 0.1,  # time_decay
        f32(n_att),                  # time_first
        hf(C, n_att),                # Kw
        hf(C, n_att),                # Vw
        hf(C, C),                    # Rw
        hf(n_att, C),                # Ow
        hf_one(C),                   # ln2.weight
        hf(C),                       # ln2.bias
        torch.rand(C, dtype=torch.float16, device=device),  # ffn.tm_k
        torch.rand(C, dtype=torch.float16, device=device),  # ffn.tm_r
        hf(C, n_ffn),                # ffn_Kw
        hf(n_ffn, C),                # ffn_Vw
        hf(C, C),                    # ffn_Rw
    ]

    # Snapshot initial state
    def snap():
        return [att_xx.clone(), aa.clone(), bb.clone(), pp.clone(), ffn_xx.clone(), x.clone()]
    def restore(s):
        att_xx.copy_(s[0]); aa.copy_(s[1]); bb.copy_(s[2]); pp.copy_(s[3]); ffn_xx.copy_(s[4]); x.copy_(s[5])

    initial = snap()

    # === Run 1: raw runs, 5 steps ===
    raw_outs = []
    raw_states = []
    for step in range(5):
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(x, att_xx, aa, bb, pp, ffn_xx, *layer)
        raw_outs.append(out.clone())
        raw_states.append((att_xx.clone(), aa.clone(), bb.clone(), pp.clone(), ffn_xx.clone()))
        x.copy_(out)  # advance x

    # === Run 2: graph capture + 5 replays ===
    restore(initial)

    # Warmup (initialize lazy state — cuBLAS workspaces, dispatchers).
    out_buf = torch.empty(1, 1, C, dtype=torch.float16, device=device)
    for _ in range(3):
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(x, att_xx, aa, bb, pp, ffn_xx, *layer)
        out_buf.copy_(out)
    torch.cuda.synchronize()

    # Restore state for capture (the warmup advanced it 3 times).
    restore(initial)

    # Capture the graph.
    g = torch.cuda.CUDAGraph()
    cap_stream = torch.cuda.Stream()
    cap_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(g):
            out = krunch_ac_cuda.rwkv4_layer_step_cpp(
                x, att_xx, aa, bb, pp, ffn_xx, *layer)
            out_buf.copy_(out)
    torch.cuda.current_stream().wait_stream(cap_stream)
    torch.cuda.synchronize()

    # Restore state and replay 5 times.
    restore(initial)
    graph_outs = []
    for step in range(5):
        g.replay()
        graph_outs.append(out_buf.clone())
        x.copy_(out_buf)  # advance x same way

    # === Diff ===
    print("Step-by-step max-abs diff (raw vs graph):")
    THRESHOLD = 1e-3
    pass_ = True
    for i in range(5):
        d = maxabs(raw_outs[i], graph_outs[i])
        ok = d < THRESHOLD
        pass_ = pass_ and ok
        print(f"  step {i}: {d:.6e}  {'PASS' if ok else 'FAIL'}")

    print(f"\nverdict: {'GRAPH IS DETERMINISTIC' if pass_ else 'GRAPH BROKEN'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()
