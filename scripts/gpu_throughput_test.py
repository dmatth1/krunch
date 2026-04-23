"""GPU throughput test for Spike 5 v1.pth on g5.xlarge.

Measures PyTorch-on-CUDA tokens/sec and KB/s for the L3TC-12M model.
PyTorch-native (not ONNX) — gives a clean upper-bound number on the
model-itself throughput without ONNX export plumbing. If PyTorch hits
the gate, ORT will too; if PyTorch misses, we don't need to debug
ONNX export to know the architecture can't meet the gate.

Outputs a single-line JSON result for easy capture + a human-readable
summary to stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm


def load_model(checkpoint_path: Path, num_layers: int, vocab_size: int,
               hidden_size: int, intermediate_size: int, rwkv_rank: int,
               ctx_len: int, repo_root: Path | None = None):
    if repo_root is None:
        # Default to /app for the container case.
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/vendor/L3TC")
    else:
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, str(repo_root / "vendor" / "L3TC"))
    from scripts.train_l3tc_phase11 import build_model  # type: ignore

    device = torch.device("cpu")
    model = build_model(
        device,
        num_layers=num_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rwkv_rank=rwkv_rank,
        ctx_len=ctx_len,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    result = model.load_state_dict(state_dict, strict=False)
    model_params = sum(1 for _ in model.state_dict().keys())
    missing = len(result.missing_keys)
    if missing >= model_params * 0.9:
        print(f"FATAL: {missing}/{model_params} params missing", file=sys.stderr)
        sys.exit(3)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True,
                   help="Held-out text corpus to tokenize + feed through model")
    p.add_argument("--chunk-size-bytes", type=int, default=64 * 1024,
                   help="Dispatcher chunk size — tokens per chunk = chunk-size/bytes-per-token")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--intermediate-size", type=int, default=1024)
    p.add_argument("--rwkv-rank", type=int, default=4)
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--max-chunks", type=int, default=50,
                   help="Cap chunks measured (50 × 64KB = 3.2 MB sampled)")
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--repo-root", type=Path, default=None,
                   help="Path to repo root (for `scripts.train_l3tc_phase11` import). Default /app.")
    p.add_argument("--result-path", type=Path, default=Path("/tmp/gpu_result.json"))
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if device_str == "cuda":
        dev_name = torch.cuda.get_device_name(0)
    elif device_str == "mps":
        dev_name = "Apple MPS"
    else:
        dev_name = "CPU"
    print(f"[gpu] device={dev_name}  dtype={args.dtype}", file=sys.stderr)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    # Load + tokenize corpus
    raw = args.corpus.read_bytes()
    original_bytes = len(raw)
    text = raw.decode("utf-8", errors="replace")
    tokens = sp.EncodeAsIds(text)
    total_tokens = len(tokens)
    bytes_per_token = original_bytes / max(1, total_tokens)
    print(f"[gpu] corpus={original_bytes}B  tokens={total_tokens}  bytes/tok={bytes_per_token:.3f}",
          file=sys.stderr)

    # Chunking: each "chunk" is chunk_size_bytes of INPUT corpus.
    # Tokens per chunk ≈ chunk_size / bytes_per_token.
    tokens_per_chunk = int(args.chunk_size_bytes / bytes_per_token)
    tokens_per_chunk = min(tokens_per_chunk, args.ctx_len)
    n_chunks_possible = (total_tokens - 1) // tokens_per_chunk
    n_chunks = min(n_chunks_possible, args.max_chunks)
    print(f"[gpu] tokens_per_chunk={tokens_per_chunk}  n_chunks={n_chunks}", file=sys.stderr)

    # Load model
    print(f"[gpu] loading checkpoint…", file=sys.stderr)
    model = load_model(
        args.checkpoint, args.num_layers, args.vocab_size,
        hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
        rwkv_rank=args.rwkv_rank, ctx_len=args.ctx_len,
        repo_root=args.repo_root,
    )
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gpu] model loaded  params={n_params/1e6:.1f}M", file=sys.stderr)

    def sync():
        if device_str == "cuda":
            torch.cuda.synchronize()
        elif device_str == "mps":
            torch.mps.synchronize()

    # Warmup (kernel cache, autotune, allocator)
    with torch.no_grad():
        for i in range(args.warmup_iters):
            seg = tokens[:tokens_per_chunk]
            inp = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0)
            types = torch.zeros_like(inp)
            _ = model(inp, types, train=True)
    sync()

    # Measurement loop
    total_tokens_processed = 0
    total_bytes_processed = 0
    total_nll_nats = 0.0
    per_chunk_latency_ms = []

    with torch.no_grad():
        for i in range(n_chunks):
            start_tok = i * tokens_per_chunk
            seg = tokens[start_tok : start_tok + tokens_per_chunk + 1]
            if len(seg) < tokens_per_chunk + 1:
                break
            inp = torch.tensor(seg[:-1], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(seg[1:], dtype=torch.long, device=device).unsqueeze(0)
            types = torch.zeros_like(inp)

            sync()
            t0 = time.perf_counter()
            out = model(inp, types, train=True)
            if isinstance(out, (tuple, list)):
                out = out[0]
            sync()
            t1 = time.perf_counter()
            per_chunk_latency_ms.append((t1 - t0) * 1000.0)

            # Also accumulate entropy for a sanity-check vs held_out_ratio
            log_probs = F.log_softmax(out.float(), dim=-1)
            nll_nats = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll_nats += nll_nats
            total_tokens_processed += inp.numel()
            total_bytes_processed += int(inp.numel() * bytes_per_token)

    total_time_s = sum(per_chunk_latency_ms) / 1000.0
    tokens_per_sec = total_tokens_processed / max(1e-9, total_time_s)
    bytes_per_sec = total_bytes_processed / max(1e-9, total_time_s)
    kbps = bytes_per_sec / 1024.0

    bits_per_token = (total_nll_nats / math.log(2)) / max(1, total_tokens_processed)
    entropy_ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)

    per_chunk_ms = sorted(per_chunk_latency_ms)
    result = {
        "device": dev_name,
        "dtype": args.dtype,
        "params_M": round(n_params / 1e6, 2),
        "tokens_per_chunk": tokens_per_chunk,
        "chunks_measured": len(per_chunk_latency_ms),
        "total_tokens": total_tokens_processed,
        "total_bytes": total_bytes_processed,
        "total_time_s": round(total_time_s, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "bytes_per_sec": round(bytes_per_sec, 1),
        "KB_per_s": round(kbps, 2),
        "per_chunk_latency_ms": {
            "p50": round(per_chunk_ms[len(per_chunk_ms)//2], 3),
            "p95": round(per_chunk_ms[int(len(per_chunk_ms)*0.95)], 3),
            "max": round(per_chunk_ms[-1], 3),
        },
        "entropy_bits_per_token": round(bits_per_token, 4),
        "entropy_ratio": round(entropy_ratio, 4),
        "gate_300KBps_met": kbps >= 300.0,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[gpu] throughput={kbps:.1f} KB/s  (gate: 300 KB/s  → {'PASS' if kbps >= 300 else 'MISS'})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
