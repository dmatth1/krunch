"""Test if quantizing activations BETWEEN layers (not just at head) makes
encoder C++ packed and decoder C++ stepped produce identical CDFs.

Hypothesis: drift compounds across 12 layers. If we quantize x to a
coarse grid after each layer, both paths re-synchronize at every layer
boundary. Drift per layer << drift accumulated through all layers.

Sweep over per-layer quantization SCALEs to find the largest (least
ratio cost) that gives byte-exact AC roundtrip.
"""
import os, time, torch
import torch.nn.functional as F

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch_ac.gpu_encode import probs_to_cdf_gpu
import krunch_ac_cuda

N_LAYER = 12


def get_layer_weights(model, device):
    w = model.w; layers = []
    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt: t = t.to(dtype=dt).contiguous()
        return t
    for i in range(N_LAYER):
        bbb = f"blocks.{i}."; att = f"blocks.{i}.att."; ffn = f"blocks.{i}.ffn."
        layers.append([
            fix(w[bbb+'ln1.weight'], torch.float16), fix(w[bbb+'ln1.bias'], torch.float16),
            fix(w[att+'time_mix_k'].squeeze(), torch.float16),
            fix(w[att+'time_mix_v'].squeeze(), torch.float16),
            fix(w[att+'time_mix_r'].squeeze(), torch.float16),
            fix(w[att+'time_decay'], torch.float32),
            fix(w[att+'time_first'], torch.float32),
            fix(w[att+'key.weight'], torch.float16),
            fix(w[att+'value.weight'], torch.float16),
            fix(w[att+'receptance.weight'], torch.float16),
            fix(w[att+'output.weight'], torch.float16),
            fix(w[bbb+'ln2.weight'], torch.float16), fix(w[bbb+'ln2.bias'], torch.float16),
            fix(w[ffn+'time_mix_k'].squeeze(), torch.float16),
            fix(w[ffn+'time_mix_r'].squeeze(), torch.float16),
            fix(w[ffn+'key.weight'], torch.float16),
            fix(w[ffn+'value.weight'], torch.float16),
            fix(w[ffn+'receptance.weight'], torch.float16),
        ])
    return (layers,
            fix(w['emb.weight'], torch.float16),
            fix(w['ln_out.weight'], torch.float16),
            fix(w['ln_out.bias'], torch.float16),
            fix(w['head.weight'], torch.float16))


def fresh_state(device, n_embd, n_att):
    return ([torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)])


def quantize_activation(x, scale):
    """Round to fixed-point grid, both encoder/decoder apply same op.
    State tensors (aa, bb, pp) NOT quantized — they're fp32 and
    different scales/dynamics."""
    if scale is None:
        return x
    return (torch.round(x.float() * scale) / scale).to(x.dtype)


def encode_decode_test(model, layers, emb_w, ln_out_w, ln_out_b, head_w,
                       seq, scale, device, n_embd, n_att):
    T = len(seq)
    inputs = [BOS_TOKEN] + seq[:-1]
    targets = seq

    # ENCODE
    state_e = fresh_state(device, n_embd, n_att)
    idx = torch.tensor(inputs, dtype=torch.long, device=device)
    x = emb_w[idx].view(1, T, n_embd).contiguous()
    if scale is not None:
        x = quantize_activation(x, scale)
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_e[0][i], state_e[1][i], state_e[2][i], state_e[3][i], state_e[4][i],
            *layers[i],
        )
        if scale is not None:
            x = quantize_activation(x, scale)
    x_norm = F.layer_norm(x.view(T, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
    logits_enc = (x_norm @ head_w).float()
    if scale is not None:
        logits_enc = quantize_activation(logits_enc, scale)
    probs_enc = torch.softmax(logits_enc, dim=-1)
    cdfs_enc = probs_to_cdf_gpu(probs_enc).contiguous()

    output_buf = torch.zeros(64 * T, dtype=torch.uint8, device=device)
    ac_state = torch.zeros(4, dtype=torch.uint32, device=device); ac_state[1] = 0xFFFFFFFF
    targets_t = torch.tensor(targets, dtype=torch.int32, device=device).contiguous()
    krunch_ac_cuda.encode_step(cdfs_enc, targets_t, output_buf, ac_state)
    krunch_ac_cuda.encode_finalize(output_buf, ac_state)
    torch.cuda.synchronize()
    bit_offset = int(ac_state[3].item())
    n_bytes = (bit_offset + 7) // 8
    bitstream = bytes(output_buf[:n_bytes].cpu().numpy())

    # DECODE
    state_d = fresh_state(device, n_embd, n_att)
    bs_padded = bitstream + b"\x00" * 64
    input_buf = torch.frombuffer(bytearray(bs_padded), dtype=torch.uint8).to(device)
    ac_state_d = torch.zeros(4, dtype=torch.uint32, device=device)
    out_sym = torch.empty(1, dtype=torch.int32, device=device)
    krunch_ac_cuda.decode_init(input_buf, ac_state_d)

    decoded = []
    last = BOS_TOKEN
    for step in range(T):
        x = emb_w[last].view(1, 1, n_embd).contiguous()
        if scale is not None:
            x = quantize_activation(x, scale)
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                x.contiguous(),
                state_d[0][i], state_d[1][i], state_d[2][i], state_d[3][i], state_d[4][i],
                *layers[i],
            )
            if scale is not None:
                x = quantize_activation(x, scale)
        x_norm = F.layer_norm(x.view(1, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
        logits_d = (x_norm @ head_w).float().flatten()
        if scale is not None:
            logits_d = quantize_activation(logits_d, scale)
        probs_d = torch.softmax(logits_d.reshape(1, -1), dim=-1)
        cdf_row = probs_to_cdf_gpu(probs_d)[0].contiguous()
        krunch_ac_cuda.decode_step(cdf_row, input_buf, ac_state_d, out_sym)
        tok = int(out_sym.item())
        decoded.append(tok)
        last = tok

    matches = (decoded == targets)
    return n_bytes, matches, decoded


def main():
    RWKV = _load_rwkv()
    print("loading RWKV-4-Pile-169M ...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    print("loaded", flush=True)
    device = "cuda"
    n_embd = model.args.n_embd
    n_att = model.args.n_att
    layers, emb_w, ln_out_w, ln_out_b, head_w = get_layer_weights(model, device)

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file("/models/20B_tokenizer.json")
    text = ("The quick brown fox jumps over the lazy dog. "
            "This is a test of the krunch neural compression codec. "
            "We are running a byte-exact AC roundtrip with C++ orchestration "
            "on both compress and decompress sides. The forward pass uses "
            "fused CUDA kernels for elementwise premix and a captured graph "
            "for the whole 12-layer forward.")
    seq = [BOS_TOKEN] + list(tk.encode(text).ids)[:63]

    # Baseline (no quantization)
    n_baseline, ok_baseline, _ = encode_decode_test(
        model, layers, emb_w, ln_out_w, ln_out_b, head_w,
        seq, None, device, n_embd, n_att)
    print(f"BASELINE (no quant) {len(seq)} tokens → {n_baseline} bytes — roundtrip {'OK' if ok_baseline else 'FAIL'}", flush=True)

    # Mid-layer quantization sweep. Smaller SCALE = coarser grid = more
    # tolerance + more ratio cost. Per-layer drift << final-drift, so
    # we expect SCALE values much larger than the head-only sweep to work.
    for scale in [256, 128, 64, 32, 16, 8, 4]:
        n_bytes, ok, decoded = encode_decode_test(
            model, layers, emb_w, ln_out_w, ln_out_b, head_w,
            seq, scale, device, n_embd, n_att)
        cost = (n_bytes - n_baseline) / n_baseline * 100
        marker = "✓" if ok else "✗"
        print(f"SCALE={scale:>4} step={1.0/scale:.5f} → {n_bytes:>4} bytes "
              f"(ratio cost {cost:+.2f}%) — roundtrip {marker} {'OK' if ok else 'FAIL'}",
              flush=True)


if __name__ == "__main__":
    main()
