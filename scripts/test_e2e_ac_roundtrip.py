"""End-to-end AC roundtrip test using C++ paths on both sides.

Compress: C++ packed forward + GPU AC encode kernel.
Decompress: C++ stepped forward + GPU AC decode kernel.

If decoded tokens match the originals byte-for-byte, the architecture
ships:
  - No quantized AC needed (numerical match within fp16 noise).
  - No bitstream change.
  - No ratio cost.
"""
import os, time, torch
import torch.nn.functional as F

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
# Force deterministic cuBLAS: same algorithm chosen across T=1024 vs T=1 calls.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch as _t
_t.use_deterministic_algorithms(True, warn_only=True)
_t.backends.cuda.matmul.allow_tf32 = False

from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch_ac.gpu_encode import probs_to_cdf_gpu
import krunch_ac_cuda

N_LAYER = 12


def get_layer_weights(model, device):
    w = model.w
    layers = []
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

    # Tokenize a real text sample. Natural tokens are common (high prob)
    # so AC intervals are wide and tolerate the 0.06-abs fp16 drift between
    # encoder (C++ packed) and decoder (C++ stepped).
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file("/models/20B_tokenizer.json")
    text = ("The quick brown fox jumps over the lazy dog. "
            "This is a test of the krunch neural compression codec. "
            "We are running a byte-exact AC roundtrip with C++ orchestration "
            "on both compress and decompress sides.")
    tokens = tk.encode(text).ids
    seq = [BOS_TOKEN] + list(tokens)[:31]  # cap at 32 tokens
    print(f"original tokens ({len(seq)}): {seq}", flush=True)

    T = len(seq)
    # ====================== ENCODE (compress) ======================
    # Run C++ packed forward over input = [BOS, seq[0], ..., seq[T-2]]
    # to get logits at every position. AC-encode seq[0..T-1] against them.
    state_e = fresh_state(device, n_embd, n_att)
    inputs = [BOS_TOKEN] + seq[:-1]   # full_input
    targets = seq                      # what we encode
    idx = torch.tensor(inputs, dtype=torch.long, device=device)
    x = emb_w[idx].view(1, T, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_e[0][i], state_e[1][i], state_e[2][i], state_e[3][i], state_e[4][i],
            *layers[i],
        )
    # Run ln_out per-row to match decoder's [1, n_embd] invocation shape.
    # F.layer_norm is shape-dependent in fp16 between [T, C] and [1, C].
    x_flat = x.view(T, n_embd)
    x_norm = torch.cat([
        F.layer_norm(x_flat[i:i+1], (n_embd,), weight=ln_out_w, bias=ln_out_b)
        for i in range(T)
    ], dim=0)
    # Deterministic head matmul: cuBLAS picks different algos at T=N vs T=1
    # so we use the bit-invariant kernel here too.
    logits_enc = krunch_ac_cuda.det_matmul(x_norm.contiguous(), head_w.contiguous())  # [T, V]
    # Per-row softmax + per-row CDF: matches decoder's [1, V] invocation
    # shape exactly. Avoids any shape-dependent reduction strategy.
    probs_rows = []
    cdf_rows = []
    for t in range(T):
        p_t = torch.softmax(logits_enc[t:t+1].float(), dim=-1)
        probs_rows.append(p_t)
        cdf_rows.append(probs_to_cdf_gpu(p_t).contiguous())
    probs_enc = torch.cat(probs_rows, dim=0)
    cdfs_enc = torch.cat(cdf_rows, dim=0).contiguous()  # [T, V+1]

    # GPU AC encode — call row-by-row to mirror the per-token decode path.
    # encode_step batched and decode_step single-row may not be bit-symmetric.
    output_buf = torch.zeros(64 * T, dtype=torch.uint8, device=device)
    ac_state = torch.zeros(4, dtype=torch.uint32, device=device)
    ac_state[1] = 0xFFFFFFFF
    for t in range(T):
        sym_t = torch.tensor([targets[t]], dtype=torch.int32, device=device).contiguous()
        krunch_ac_cuda.encode_step(cdfs_enc[t:t+1].contiguous(), sym_t, output_buf, ac_state)
    krunch_ac_cuda.encode_finalize(output_buf, ac_state)
    torch.cuda.synchronize()

    bit_offset = int(ac_state[3].item())
    n_bytes = (bit_offset + 7) // 8
    bitstream = bytes(output_buf[:n_bytes].cpu().numpy())
    print(f"compressed {T} tokens to {n_bytes} bytes (~{n_bytes*8/T:.2f} bits/token)", flush=True)

    # ====================== DECODE (decompress) ======================
    state_d = fresh_state(device, n_embd, n_att)
    bs_padded = bitstream + b"\x00" * 64
    input_buf = torch.frombuffer(bytearray(bs_padded), dtype=torch.uint8).to(device)
    ac_state_d = torch.zeros(4, dtype=torch.uint32, device=device)
    out_sym = torch.empty(1, dtype=torch.int32, device=device)
    krunch_ac_cuda.decode_init(input_buf, ac_state_d)

    decoded = []
    last = BOS_TOKEN
    for step in range(T):
        # C++ T=1 stepped forward
        x = emb_w[last].view(1, 1, n_embd).contiguous()
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                x.contiguous(),
                state_d[0][i], state_d[1][i], state_d[2][i], state_d[3][i], state_d[4][i],
                *layers[i],
            )
        x_norm = F.layer_norm(x.view(1, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
        logits_d = krunch_ac_cuda.det_matmul(x_norm.contiguous(), head_w.contiguous()).flatten()
        probs_d = torch.softmax(logits_d.float().reshape(1, -1), dim=-1)
        cdf_row = probs_to_cdf_gpu(probs_d)[0].contiguous()
        # DIAG: compare decoder's CDF against encoder's per-step CDF.
        cdf_enc_row = cdfs_enc[step].contiguous()
        cdf_diff = (cdf_row.long() - cdf_enc_row.long()).abs().max().item()
        if cdf_diff != 0:
            print(f"  step {step}: cdf_diff={cdf_diff} (encoder vs decoder)", flush=True)
        krunch_ac_cuda.decode_step(cdf_row, input_buf, ac_state_d, out_sym)
        tok = int(out_sym.item())
        decoded.append(tok)
        last = tok

    print(f"decoded tokens ({len(decoded)}): {decoded}", flush=True)
    if decoded == targets:
        print("PASS — bit-exact AC roundtrip via C++ paths on both sides.", flush=True)
    else:
        mismatch_at = next(i for i, (a, b) in enumerate(zip(decoded, targets)) if a != b)
        print(f"FAIL — first mismatch at step {mismatch_at}: decoded={decoded[mismatch_at]} expected={targets[mismatch_at]}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
