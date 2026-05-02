"""Sweep logit-quantization scales to find the smallest SCALE that gives
bit-exact AC roundtrip between C++ packed (encoder) and C++ stepped
(decoder). Then measure ratio cost vs un-quantized BlinkDL baseline.

Quantization rule: logits_q = round(logits.float() * SCALE) / SCALE.
Both encoder and decoder do same op on their (drifted) logits — if
SCALE is small enough, both round to same int and produce identical
CDFs.

Drift between C++ packed/stepped: 0.016-0.094 max abs (measured).
For drift δ: SCALE = 1/(2δ) ensures rounding can't flip a bin.
Trying SCALE ∈ {32, 16, 8, 4, 2}.
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


def quantize(logits_fp32, scale):
    """Quantize logits to fixed-point grid. Both encoder and decoder
    apply same op → identical quantized values when input drift < 1/(2*scale)."""
    if scale is None:
        return logits_fp32
    return torch.round(logits_fp32 * scale) / scale


def encode_decode_test(model, layers, emb_w, ln_out_w, ln_out_b, head_w,
                       seq, scale, device, n_embd, n_att):
    T = len(seq)
    inputs = [BOS_TOKEN] + seq[:-1]
    targets = seq

    # ENCODE
    state_e = fresh_state(device, n_embd, n_att)
    idx = torch.tensor(inputs, dtype=torch.long, device=device)
    x = emb_w[idx].view(1, T, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_e[0][i], state_e[1][i], state_e[2][i], state_e[3][i], state_e[4][i],
            *layers[i],
        )
    x_norm = F.layer_norm(x.view(T, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
    logits_enc = (x_norm @ head_w).float()
    logits_enc_q = quantize(logits_enc, scale)
    probs_enc = torch.softmax(logits_enc_q, dim=-1)
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
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                x.contiguous(),
                state_d[0][i], state_d[1][i], state_d[2][i], state_d[3][i], state_d[4][i],
                *layers[i],
            )
        x_norm = F.layer_norm(x.view(1, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
        logits_d = (x_norm @ head_w).float().flatten()
        logits_d_q = quantize(logits_d, scale)
        probs_d = torch.softmax(logits_d_q.reshape(1, -1), dim=-1)
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
    seq = [BOS_TOKEN] + list(tk.encode(text).ids)[:63]  # 64 tokens

    # FIRST: measure actual per-step drift between encoder and decoder
    # logits when both run the FULL EXPECTED PREFIX (so they're not
    # diverging due to wrong tokens — measures pure path drift).
    print("=== drift measurement: encoder packed vs decoder stepped ===", flush=True)
    inputs = [BOS_TOKEN] + seq[:-1]
    state_e = fresh_state(device, n_embd, n_att)
    idx = torch.tensor(inputs, dtype=torch.long, device=device)
    x = emb_w[idx].view(1, len(inputs), n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_e[0][i], state_e[1][i], state_e[2][i], state_e[3][i], state_e[4][i],
            *layers[i],
        )
    x_norm = F.layer_norm(x.view(len(inputs), n_embd), (n_embd,),
                          weight=ln_out_w, bias=ln_out_b)
    enc_logits_all = (x_norm @ head_w).float()  # [T, V]

    state_d = fresh_state(device, n_embd, n_att)
    last = BOS_TOKEN
    max_drift = 0.0
    drifts = []
    for step, expected in enumerate(seq):
        x_step = emb_w[last].view(1, 1, n_embd).contiguous()
        for i in range(N_LAYER):
            x_step = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                x_step.contiguous(),
                state_d[0][i], state_d[1][i], state_d[2][i], state_d[3][i], state_d[4][i],
                *layers[i],
            )
        x_n = F.layer_norm(x_step.view(1, n_embd), (n_embd,),
                           weight=ln_out_w, bias=ln_out_b)
        dec_logits = (x_n @ head_w).float().flatten()
        d = (enc_logits_all[step] - dec_logits).abs().max().item()
        drifts.append(d)
        max_drift = max(max_drift, d)
        last = expected  # force-feed the expected token (don't AC-decode)
    print(f"max drift across {len(seq)} steps: {max_drift:.4f}", flush=True)
    print(f"per-step drift sample (first 8): {[f'{d:.4f}' for d in drifts[:8]]}", flush=True)
    print(f"per-step drift sample (last 8):  {[f'{d:.4f}' for d in drifts[-8:]]}", flush=True)

    # First — measure baseline (no quantization) compressed size
    n_baseline, ok_baseline, _ = encode_decode_test(
        model, layers, emb_w, ln_out_w, ln_out_b, head_w,
        seq, None, device, n_embd, n_att)
    print(f"BASELINE (no quant) {len(seq)} tokens → {n_baseline} bytes — roundtrip {'OK' if ok_baseline else 'FAIL'}", flush=True)

    # Sweep scales: smaller SCALE = coarser quantization = more tolerance + more ratio cost
    # 1/(2*0.094) ≈ 5.3, so SCALE ≤ 5 should always tolerate the worst-case drift
    for scale in [64, 32, 16, 8, 4, 2, 1]:
        n_bytes, ok, _ = encode_decode_test(
            model, layers, emb_w, ln_out_w, ln_out_b, head_w,
            seq, scale, device, n_embd, n_att)
        cost = (n_bytes - n_baseline) / n_baseline * 100
        print(f"SCALE={scale:>3} step={1.0/scale:.4f} → {n_bytes:>4} bytes "
              f"(ratio cost {cost:+.2f}% vs baseline) — roundtrip {'OK' if ok else 'FAIL'}",
              flush=True)


if __name__ == "__main__":
    main()
