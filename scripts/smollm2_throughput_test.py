"""Quick throughput test for SmolLM2 arithmetic-coding compression path.

Measures: forward-pass latency and throughput when running SmolLM2-*
over fixed-length windows of content. Reports KB/s of *input* bytes
processed — directly comparable to zstd KB/s and our 300 KB/s gate.

Arithmetic coding itself is fast (~100 MB/s) and parallelizable, so
the transformer forward pass is the bottleneck. This script times
JUST the forward pass + next-token distribution extraction, which is
what the arithmetic coder actually needs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--content", type=Path, required=True,
                   help="Raw text/content bytes to feed through the model")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--measure-iters", type=int, default=20,
                   help="Chunks to time")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Chunks processed in parallel (GPU batching lever)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--result-path", type=Path, default=Path("/tmp/smollm2_throughput.json"))
    args = p.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device
    device = torch.device(device_str)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dev_name = torch.cuda.get_device_name(0) if device_str == "cuda" else ("Apple MPS" if device_str == "mps" else "CPU")
    print(f"[sm2-tput] device={dev_name}  dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch_dtype).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sm2-tput] params={n_params/1e6:.1f}M")

    # Tokenize content
    raw = args.content.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    # Take enough text for warmup + measure * batch chunks
    needed_tokens_est = args.seq_len * (args.warmup_iters + args.measure_iters * args.batch_size) * 2
    needed_chars = needed_tokens_est * 4  # rough bytes-per-token upper bound
    text = text[:max(needed_chars, 100_000)]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor(ids, dtype=torch.long)
    bytes_per_token = len(raw[:len(text.encode())]) / max(1, tokens.numel())
    print(f"[sm2-tput] tokens_available={tokens.numel()}  bytes/tok={bytes_per_token:.3f}")

    def sync():
        if device_str == "cuda":
            torch.cuda.synchronize()
        elif device_str == "mps":
            torch.mps.synchronize()

    # Build batched inputs
    chunks_needed = args.warmup_iters + args.measure_iters * args.batch_size
    assert tokens.numel() >= chunks_needed * args.seq_len, \
        f"not enough tokens ({tokens.numel()}) for {chunks_needed} × {args.seq_len}"

    def get_batch(start_chunk: int, n: int) -> torch.Tensor:
        batch = []
        for i in range(n):
            t_start = (start_chunk + i) * args.seq_len
            batch.append(tokens[t_start : t_start + args.seq_len])
        return torch.stack(batch).to(device)

    # Warmup
    with torch.no_grad():
        for i in range(args.warmup_iters):
            inp = get_batch(i, args.batch_size)
            _ = model(inp)
    sync()

    # Measure
    per_step_ms = []
    total_tokens = 0
    with torch.no_grad():
        for step in range(args.measure_iters):
            inp = get_batch(args.warmup_iters + step * args.batch_size, args.batch_size)
            sync()
            t0 = time.perf_counter()
            out = model(inp)
            # Extract log-probs (what the arithmetic coder consumes).
            # Real AC does this per-token but for throughput the softmax
            # over the whole sequence dominates, so we include it.
            _ = torch.log_softmax(out.logits.float(), dim=-1)
            sync()
            t1 = time.perf_counter()
            per_step_ms.append((t1 - t0) * 1000.0)
            total_tokens += inp.numel()

    total_time_s = sum(per_step_ms) / 1000.0
    tokens_per_sec = total_tokens / max(1e-9, total_time_s)
    bytes_per_sec = tokens_per_sec * bytes_per_token
    kbps = bytes_per_sec / 1024.0

    sorted_ms = sorted(per_step_ms)
    result = {
        "model": args.base_model,
        "params_M": round(n_params / 1e6, 1),
        "device": dev_name,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "measure_iters": args.measure_iters,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time_s, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "bytes_per_sec": round(bytes_per_sec, 1),
        "KB_per_s": round(kbps, 2),
        "per_step_ms": {
            "p50": round(sorted_ms[len(sorted_ms)//2], 2),
            "p95": round(sorted_ms[int(len(sorted_ms)*0.95)], 2),
            "max": round(sorted_ms[-1], 2),
        },
        "gate_300KBps_met": kbps >= 300.0,
        "bytes_per_token": round(bytes_per_token, 3),
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
