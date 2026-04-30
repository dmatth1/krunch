"""Pure AC roundtrip with model-free CDFs.

Goal: isolate whether krunch_ac_cuda's encode_step / decode_step
kernels alone roundtrip correctly. Construct fake CDFs that mimic
model output (one tall peak per step + uniform background), encode N
known symbols, then decode and compare.

If this passes → AC kernels are fine and the e2e bug is in
model→CDF→AC handoff. If it fails → the kernels are the bug.
"""
import torch
import krunch_ac_cuda
from krunch_ac.gpu_encode import probs_to_cdf_gpu


def make_synthetic_probs(T, V, seed=42, peakiness=10.0):
    """N rows of (mostly-flat + sharp-peak) probabilities. Different
    peak channel per row, similar to LM output statistics."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.rand(T, V, generator=g, device="cuda") * 0.01
    peak_channels = torch.randint(0, V, (T,), generator=g, device="cuda")
    for t in range(T):
        base[t, peak_channels[t]] += peakiness
    probs = base / base.sum(dim=1, keepdim=True)
    return probs.float(), peak_channels


def main():
    T = 64
    V = 50277
    device = "cuda"

    probs, _ = make_synthetic_probs(T, V)
    cdfs = probs_to_cdf_gpu(probs).contiguous()  # [T, V+1]

    # Pick a target symbol per step. Mix peaks (high-prob, narrow
    # interval) and random low-prob symbols (wide intervals) to
    # stress the AC kernel across the full range.
    g = torch.Generator(device=device).manual_seed(7)
    targets = torch.randint(0, V, (T,), generator=g, device=device, dtype=torch.int32).contiguous()

    # ============== ENCODE (single batched call) ==============
    output_buf_b = torch.zeros(64 * T, dtype=torch.uint8, device=device)
    ac_state_b = torch.zeros(4, dtype=torch.uint32, device=device)
    ac_state_b[1] = 0xFFFFFFFF
    krunch_ac_cuda.encode_step(cdfs, targets, output_buf_b, ac_state_b)
    krunch_ac_cuda.encode_finalize(output_buf_b, ac_state_b)
    torch.cuda.synchronize()
    n_b = (int(ac_state_b[3].item()) + 7) // 8
    bs_b = bytes(output_buf_b[:n_b].cpu().numpy())

    # ============== ENCODE (row-by-row, mirror decode) ==============
    output_buf_r = torch.zeros(64 * T, dtype=torch.uint8, device=device)
    ac_state_r = torch.zeros(4, dtype=torch.uint32, device=device)
    ac_state_r[1] = 0xFFFFFFFF
    for t in range(T):
        krunch_ac_cuda.encode_step(cdfs[t:t+1].contiguous(),
                                    targets[t:t+1].contiguous(),
                                    output_buf_r, ac_state_r)
    krunch_ac_cuda.encode_finalize(output_buf_r, ac_state_r)
    torch.cuda.synchronize()
    n_r = (int(ac_state_r[3].item()) + 7) // 8
    bs_r = bytes(output_buf_r[:n_r].cpu().numpy())

    print(f"batched: {n_b} bytes ({n_b*8/T:.2f} bits/tok)")
    print(f"row-by-row: {n_r} bytes")
    print(f"bitstream batched == row-by-row: {bs_b == bs_r}")

    # ============== DECODE ==============
    bs_padded = bs_b + b"\x00" * 64
    in_buf = torch.frombuffer(bytearray(bs_padded), dtype=torch.uint8).to(device)
    ac_state_d = torch.zeros(4, dtype=torch.uint32, device=device)
    out_sym = torch.empty(1, dtype=torch.int32, device=device)
    krunch_ac_cuda.decode_init(in_buf, ac_state_d)

    decoded = []
    for t in range(T):
        cdf_row = cdfs[t].contiguous()
        krunch_ac_cuda.decode_step(cdf_row, in_buf, ac_state_d, out_sym)
        decoded.append(int(out_sym.item()))

    targets_l = targets.cpu().tolist()
    if decoded == targets_l:
        print("PASS — AC kernels roundtrip cleanly.")
    else:
        bad = next(i for i, (a, b) in enumerate(zip(decoded, targets_l)) if a != b)
        print(f"FAIL — first mismatch step {bad}: decoded={decoded[bad]} expected={targets_l[bad]}")
        # Look at the CDF interval around expected vs decoded
        c = cdfs[bad].cpu().tolist()
        e = targets_l[bad]
        d = decoded[bad]
        print(f"  expected sym {e}: cdf=[{c[e]}, {c[e+1]}) width={c[e+1]-c[e]}")
        print(f"  decoded  sym {d}: cdf=[{c[d]}, {c[d+1]}) width={c[d+1]-c[d]}")


if __name__ == "__main__":
    main()
