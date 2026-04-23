"""Zero-shot entropy-bound compression ratio for a pretrained RWKV model.

Loads any HuggingFace RWKV-4 or RWKV-5 checkpoint (e.g.
`RWKV/rwkv-4-169m-pile`), tokenizes the input content stream,
forward-passes through it, and reports bits/token + ratio vs the
original byte count. Same math as `scripts/measure_held_out_ratio.py`
but for a generic HF model instead of the custom L3TC build_model.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="RWKV/rwkv-4-169m-pile",
                   help="HF repo id for the pretrained RWKV checkpoint")
    p.add_argument("--content", type=Path, required=True,
                   help="Content stream (bytes). For chat, use the output of preprocess_chat.py.")
    p.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024,
                   help="Cap held-out measurement at this many input bytes (default 10 MB)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Cap chunks measured (useful for CPU dry runs)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--result-path", type=Path, default=Path("/tmp/rwkv_zeroshot.json"))
    args = p.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device
    device = torch.device(device_str)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dev_name = torch.cuda.get_device_name(0) if device_str == "cuda" else ("Apple MPS" if device_str == "mps" else "CPU")
    print(f"[rwkv0] device={dev_name}  dtype={args.dtype}")

    # Load
    print(f"[rwkv0] loading {args.model_id}…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch_dtype).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rwkv0] loaded  params={n_params/1e6:.1f}M  took={time.time()-t0:.1f}s")

    # Read content (first max_bytes only — takes held-out from TAIL so
    # a future fine-tune run using the same file trains on the head
    # and we evaluate on disjoint tail).
    full_bytes = args.content.read_bytes()
    corpus_bytes = full_bytes[-args.max_bytes:] if len(full_bytes) > args.max_bytes else full_bytes
    corpus_size = len(corpus_bytes)
    text = corpus_bytes.decode("utf-8", errors="replace")
    print(f"[rwkv0] held-out corpus={corpus_size/1e6:.1f} MB  tokenizing…")
    t0 = time.time()
    CHUNK_CHARS = 1 * 1024 * 1024
    all_ids: list[int] = []
    for i in range(0, len(text), CHUNK_CHARS):
        ids = tok(text[i : i + CHUNK_CHARS], add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    bytes_per_token = corpus_size / max(1, tokens.numel())
    print(f"[rwkv0] tokens={tokens.numel()}  bytes/tok={bytes_per_token:.3f}  took={time.time()-t0:.1f}s")

    # Measurement: fixed-length windows, no overlap
    tokens_per_chunk = args.seq_len
    n_possible = max(0, (tokens.numel() - 1) // tokens_per_chunk)
    n_chunks = n_possible if args.max_chunks is None else min(n_possible, args.max_chunks)
    print(f"[rwkv0] measuring {n_chunks} chunks × {tokens_per_chunk} tokens…")

    total_nll_nats = 0.0
    total_positions = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * tokens_per_chunk
            seg = tokens[start : start + tokens_per_chunk + 1]
            if seg.numel() < tokens_per_chunk + 1:
                break
            inp = seg[:-1].unsqueeze(0).to(device)
            tgt = seg[1:].unsqueeze(0).to(device)
            out = model(input_ids=inp)
            logits = out.logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll_nats += nll
            total_positions += inp.numel()
            if (i + 1) % 10 == 0:
                bpt = (total_nll_nats / math.log(2)) / total_positions
                print(f"  chunk {i+1}/{n_chunks}  running bits/tok={bpt:.3f}  elapsed={time.time()-t0:.0f}s")

    total_time = time.time() - t0
    bits_per_token = (total_nll_nats / math.log(2)) / max(1, total_positions)
    entropy_ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)
    tokens_per_sec = total_positions / max(1e-9, total_time)
    bytes_per_sec = tokens_per_sec * bytes_per_token

    result = {
        "model_id": args.model_id,
        "params_M": round(n_params / 1e6, 1),
        "device": dev_name,
        "dtype": args.dtype,
        "held_out_bytes": corpus_size,
        "held_out_tokens": tokens.numel(),
        "bytes_per_token": round(bytes_per_token, 4),
        "positions_evaluated": total_positions,
        "bits_per_token": round(bits_per_token, 4),
        "entropy_ratio": round(entropy_ratio, 5),
        "bits_per_byte": round(bits_per_token / bytes_per_token, 4),
        "measurement_time_s": round(total_time, 1),
        "throughput_KB_per_s": round(bytes_per_sec / 1024.0, 2),
        # Reference points for comparison:
        "reference_zstd_19_ratio": 0.16655,
        "reference_bzip3_wholefile_ratio": 0.14499,
        "reference_spike5_l3tc_12m_ratio": 0.2208,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
