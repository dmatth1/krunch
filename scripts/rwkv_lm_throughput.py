"""RWKV-LM throughput test — using BlinkDL's custom WKV CUDA kernel.

This bypasses HF transformers entirely. It instantiates RWKV_GPT from
BlinkDL/RWKV-LM/RWKV-v4/src/model_run.py, which compiles and uses the
fused WKV CUDA kernel (the one that makes ts_zip and L3TC fast).

Env requirements (set before this script runs):
  RWKV_RUN_DEVICE=cuda
  RWKV_FLOAT_MODE=fp16  (or bf16/fp32 — fp16 is recommended on A10G)

Args:
  --rwkv-lm-dir   path to RWKV-LM/RWKV-v4 (checkout of BlinkDL's repo)
  --model-path    path to .pth WITHOUT the extension
                  (e.g. /tmp/RWKV-4-Pile-169M-20220807-8023)
  --content       raw bytes to tokenize + feed through the model
  --seq-len       default 1024 (this is baked into the CUDA kernel via T_MAX)
  --batch-size    default 1
  --warmup-iters  default 3 (first iter compiles the kernel)
  --measure-iters default 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rwkv-lm-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True,
                   help="Path to .pth WITHOUT the extension")
    p.add_argument("--tokenizer-json", type=Path, required=True,
                   help="Path to 20B_tokenizer.json")
    p.add_argument("--content", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--vocab-size", type=int, default=50277)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--measure-iters", type=int, default=20)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--result-path", type=Path, default=Path("/tmp/rwkv_lm_throughput.json"))
    args = p.parse_args()

    # Env BEFORE import (module-level compile uses these)
    os.environ["RWKV_RUN_DEVICE"] = "cuda"
    os.environ["RWKV_FLOAT_MODE"] = args.dtype

    sys.path.insert(0, str(args.rwkv_lm_dir))
    sys.path.insert(0, str(args.rwkv_lm_dir / "src"))

    # Must chdir to rwkv_lm_dir because the cpp_extension.load() call
    # in model_run.py uses relative paths `cuda/wkv_op.cpp`.
    os.chdir(args.rwkv_lm_dir)

    import torch
    from tokenizers import Tokenizer

    assert torch.cuda.is_available(), "CUDA required"
    dev_name = torch.cuda.get_device_name(0)
    print(f"[rwkv-lm] device={dev_name}  dtype={args.dtype}")

    # Compiling the CUDA kernel happens at import time
    t_compile_start = time.perf_counter()
    from model_run import RWKV_GPT  # type: ignore
    t_compile_end = time.perf_counter()
    print(f"[rwkv-lm] CUDA kernel compile/load: {t_compile_end - t_compile_start:.1f}s")

    # Instantiate + load weights
    t0 = time.perf_counter()
    model = RWKV_GPT(
        MODEL_NAME=str(args.model_path),
        RUN_DEVICE="cuda",
        model_type="RWKV",
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        ctx_len=args.seq_len,
    ).cuda()
    # Dtype cast
    if args.dtype == "fp16":
        model = model.half()
    elif args.dtype == "bf16":
        model = model.bfloat16()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rwkv-lm] model loaded params={n_params/1e6:.1f}M  took={time.perf_counter()-t0:.1f}s")

    # Load tokenizer (GPT-NeoX BPE from 20B_tokenizer.json)
    tok = Tokenizer.from_file(str(args.tokenizer_json))

    # Read + tokenize
    raw = args.content.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    needed = (args.warmup_iters + args.measure_iters) * args.batch_size * (args.seq_len + 1)
    text = text[: needed * 5]  # rough 5 bytes/token upper bound
    enc = tok.encode(text).ids
    tokens = torch.tensor(enc, dtype=torch.long)
    bytes_per_token = len(raw[: len(text.encode())]) / max(1, tokens.numel())
    print(f"[rwkv-lm] tokens={tokens.numel()}  bytes/tok={bytes_per_token:.3f}")

    # Warmup
    print(f"[rwkv-lm] warmup…")
    t_warm = time.perf_counter()
    with torch.no_grad():
        for i in range(args.warmup_iters):
            t_i = time.perf_counter()
            batch = torch.stack([tokens[j * args.seq_len : (j + 1) * args.seq_len]
                                 for j in range(args.batch_size)]).cuda()
            logits = model(batch)
            torch.cuda.synchronize()
            print(f"  warmup iter {i}: {(time.perf_counter()-t_i)*1000:.0f} ms")

    # Measure
    per_step_ms = []
    total_nll_nats = 0.0
    total_positions = 0
    with torch.no_grad():
        for step in range(args.measure_iters):
            start = (args.warmup_iters + step) * args.batch_size * args.seq_len
            batch = torch.stack([tokens[start + j * args.seq_len : start + (j + 1) * args.seq_len]
                                 for j in range(args.batch_size)]).cuda()
            # target = shifted by 1 for entropy calc
            tgt = torch.stack([tokens[start + j * args.seq_len + 1 : start + (j + 1) * args.seq_len + 1]
                               for j in range(args.batch_size)]).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_step_ms.append((t1 - t0) * 1000.0)

            # Entropy sanity check (ratio consistency vs prior measurements)
            if tgt.numel() > 0:
                # Use all positions except the last (no target for it)
                valid_logits = logits[:, :-1, :].float()
                valid_tgt = tgt[:, :-1] if tgt.shape[1] == batch.shape[1] else tgt
                # Align: logits[:,t] predicts tokens[t+1] = tgt[:,t]
                log_probs = torch.log_softmax(valid_logits, dim=-1)
                n_predict = min(valid_logits.shape[1], valid_tgt.shape[1])
                lp = log_probs[:, :n_predict].gather(-1, valid_tgt[:, :n_predict].unsqueeze(-1)).squeeze(-1)
                total_nll_nats += -lp.sum().item()
                total_positions += n_predict * args.batch_size

    total_time_s = sum(per_step_ms) / 1000.0
    tokens_per_sec = (args.measure_iters * args.batch_size * args.seq_len) / max(1e-9, total_time_s)
    bytes_per_sec = tokens_per_sec * bytes_per_token
    kbps = bytes_per_sec / 1024.0

    bits_per_token = (total_nll_nats / math.log(2)) / max(1, total_positions)
    entropy_ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)

    sorted_ms = sorted(per_step_ms)
    result = {
        "impl": "BlinkDL/RWKV-LM (custom WKV CUDA kernel)",
        "model": str(args.model_path),
        "params_M": round(n_params / 1e6, 1),
        "device": dev_name,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "measure_iters": args.measure_iters,
        "total_tokens_forwarded": args.measure_iters * args.batch_size * args.seq_len,
        "total_time_s": round(total_time_s, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "bytes_per_sec": round(bytes_per_sec, 1),
        "KB_per_s": round(kbps, 2),
        "MB_per_s": round(kbps / 1024.0, 3),
        "per_step_ms": {
            "p50": round(sorted_ms[len(sorted_ms)//2], 2),
            "p95": round(sorted_ms[int(len(sorted_ms)*0.95)], 2),
            "max": round(sorted_ms[-1], 2),
        },
        "bytes_per_token": round(bytes_per_token, 3),
        "bits_per_token": round(bits_per_token, 4),
        "entropy_ratio": round(entropy_ratio, 5),
        "gate_300KBps_met": kbps >= 300.0,
        "gate_1MBps_stretch": kbps >= 1024.0,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
