"""RWKV-4 throughput test with the WKV CUDA kernel explicitly loaded.

The HF transformers RwkvForCausalLM auto-loads a custom CUDA kernel
(`wkv_cuda.cu` fused-recurrence) when:
  - dtype is fp32 or fp16 (NOT bf16 — the kernel has no bf16 impl)
  - CUDA is available
  - ninja build tool is installed
  - first forward runs with seq_len > 0

If any of those fail, transformers silently falls back to a
Python-level recurrence loop which is ~50× slower. This script:

  1. Forces fp16 and verifies torch.cuda availability
  2. Runs a short warm-up which triggers on-the-fly kernel compilation
     (~15-30s the first time — ninja builds wkv_cuda.cu)
  3. Prints whether the kernel actually got loaded
     (`transformers.models.rwkv.modeling_rwkv.rwkv_cuda_kernel` is
      None when fallback is used, a CUDA extension object when loaded)
  4. Then runs a proper timed measurement

Writes a JSON with throughput + per-step latency + diagnostic flags.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_kernel_loaded() -> dict:
    try:
        from transformers.models.rwkv import modeling_rwkv
        loaded = modeling_rwkv.rwkv_cuda_kernel is not None
        max_seq = getattr(modeling_rwkv.rwkv_cuda_kernel, "max_seq_length", None) if loaded else None
        return {"kernel_loaded": loaded, "kernel_max_seq": max_seq}
    except Exception as e:
        return {"kernel_loaded": False, "kernel_error": repr(e)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--content", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-iters", type=int, default=3,
                   help=">=2 recommended; first run triggers CUDA compile")
    p.add_argument("--measure-iters", type=int, default=20)
    p.add_argument("--result-path", type=Path, default=Path("/tmp/rwkv_kernel_throughput.json"))
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev_name = torch.cuda.get_device_name(0)
    print(f"[rwkv-tput] device={dev_name}  dtype=fp16 (kernel-compatible)")

    # Pre-import transformers.models.rwkv so we can observe kernel state
    from transformers.models.rwkv import modeling_rwkv
    print(f"[rwkv-tput] before load:  kernel_loaded={modeling_rwkv.rwkv_cuda_kernel is not None}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to("cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rwkv-tput] params={n_params/1e6:.1f}M  (fp16)")

    # Tokenize
    raw = args.content.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    needed_tokens = (args.warmup_iters + args.measure_iters) * args.batch_size * (args.seq_len + 1)
    text = text[: needed_tokens * 5]  # 5 bytes/token upper bound
    ids = tok(text, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor(ids, dtype=torch.long)
    bytes_per_token = len(raw[: len(text.encode())]) / max(1, tokens.numel())
    print(f"[rwkv-tput] tokens={tokens.numel()}  bytes/tok={bytes_per_token:.3f}")

    # Warmup: first iter triggers kernel compile (takes 15-60s the first time)
    print(f"[rwkv-tput] warmup (first iter compiles CUDA kernel, may take ~30s)…")
    t_warm0 = time.perf_counter()
    with torch.no_grad():
        for i in range(args.warmup_iters):
            t_i = time.perf_counter()
            batch = torch.stack([tokens[j*args.seq_len : (j+1)*args.seq_len] for j in range(args.batch_size)]).cuda()
            _ = model(input_ids=batch)
            torch.cuda.synchronize()
            print(f"  warmup iter {i}: {(time.perf_counter()-t_i)*1000:.0f} ms")
    print(f"[rwkv-tput] warmup done in {time.perf_counter()-t_warm0:.1f}s")

    # Check kernel state AFTER warmup (it gets loaded lazily on first forward)
    kernel_state = check_kernel_loaded()
    print(f"[rwkv-tput] after warmup: {kernel_state}")

    # Measure
    per_step_ms = []
    total_tokens_processed = 0
    with torch.no_grad():
        for step in range(args.measure_iters):
            start = (args.warmup_iters + step) * args.batch_size * args.seq_len
            batch = torch.stack([tokens[start + j*args.seq_len : start + (j+1)*args.seq_len]
                                 for j in range(args.batch_size)]).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=batch)
            _ = torch.log_softmax(out.logits.float(), dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_step_ms.append((t1 - t0) * 1000.0)
            total_tokens_processed += batch.numel()

    total_time_s = sum(per_step_ms) / 1000.0
    tokens_per_sec = total_tokens_processed / max(1e-9, total_time_s)
    bytes_per_sec = tokens_per_sec * bytes_per_token
    kbps = bytes_per_sec / 1024.0

    sorted_ms = sorted(per_step_ms)
    result = {
        "model": args.model_id,
        "params_M": round(n_params / 1e6, 1),
        "device": dev_name,
        "dtype": "fp16",
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "total_tokens": total_tokens_processed,
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
        **kernel_state,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
