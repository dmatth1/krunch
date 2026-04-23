"""Track 3: LoRA fine-tune SmolLM2-360M on chat content stream.

Objective: next-token LM loss on preprocessed WildChat-English content
(no JSON framing — that's stripped by preprocess_chat.py). Keep base
frozen, train rank-16 LoRA on attention + MLP linears.

Measures entropy-bound ratio at the end using the same bits/byte
formula as scripts/measure_held_out_ratio.py — directly comparable
to Spike 5's v1.metadata.held_out_ratio = 0.2208.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class ContentStreamDataset(Dataset):
    """Random contiguous windows over a content stream."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, n_samples: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Random window (deterministic per-idx for repro-in-epoch).
        g = torch.Generator().manual_seed(idx)
        start = torch.randint(0, self.tokens.numel() - self.seq_len - 1, (1,), generator=g).item()
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1].clone(),
            "labels": chunk[1:].clone(),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--content-stream", type=Path, required=True,
                   help="Raw content bytes (output of preprocess_chat.py)")
    p.add_argument("--train-val-split", type=float, default=0.95,
                   help="Fraction of content stream for training")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--epoch-length", type=int, default=5000,
                   help="Samples per epoch (approximates 1 epoch over the corpus)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--eval-samples", type=int, default=200,
                   help="Val chunks for held-out entropy bound")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-path", type=Path, default=Path("/tmp/smollm2_result.json"))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device
    device = torch.device(device_str)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev_name = torch.cuda.get_device_name(0) if device_str == "cuda" else ("Apple MPS" if device_str == "mps" else "CPU")
    print(f"[smollm2] device={dev_name}  dtype={args.dtype}")

    # Load tokenizer + base model
    print(f"[smollm2] loading {args.base_model}…")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype).to(device)
    n_base = sum(p.numel() for p in model.parameters())
    print(f"[smollm2] base params={n_base/1e6:.1f}M")

    # Attach LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize content stream
    raw = args.content_stream.read_bytes()
    total_bytes = len(raw)
    text = raw.decode("utf-8", errors="replace")
    print(f"[smollm2] corpus={total_bytes/1e6:.1f} MB  tokenizing…")
    t0 = time.time()
    # Tokenize in chunks to avoid max-length warnings / OOM on a giant string
    CHUNK = 2 * 1024 * 1024
    all_ids: list[int] = []
    for i in range(0, len(text), CHUNK):
        ids = tok(text[i : i + CHUNK], add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    bytes_per_token = total_bytes / max(1, tokens.numel())
    print(f"[smollm2] tokens={tokens.numel()}  bytes/tok={bytes_per_token:.3f}  took={time.time()-t0:.1f}s")

    # Split
    split_idx = int(args.train_val_split * tokens.numel())
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"[smollm2] train={train_tokens.numel()}  val={val_tokens.numel()}")

    train_ds = ContentStreamDataset(train_tokens, args.seq_len, args.epoch_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Optimizer — LoRA params only
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Train
    model.train()
    for epoch in range(args.epochs):
        t_ep = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 200 == 0:
                print(f"  epoch={epoch} batch={n_batches}/{len(train_loader)} loss={loss.item():.4f}")
        avg_loss = total_loss / max(1, n_batches)
        print(f"[smollm2] epoch {epoch}: avg_loss={avg_loss:.4f}  elapsed={time.time()-t_ep:.1f}s")

    # Save LoRA adapter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    print(f"[smollm2] saved LoRA to {args.output_dir}")

    # Held-out entropy bound — same formula as measure_held_out_ratio.py
    model.eval()
    total_nll_nats = 0.0
    total_positions = 0
    with torch.no_grad():
        for i in range(args.eval_samples):
            start = (i * args.seq_len) % max(1, val_tokens.numel() - args.seq_len - 1)
            chunk = val_tokens[start : start + args.seq_len + 1]
            if chunk.numel() < args.seq_len + 1:
                continue
            inp = chunk[:-1].unsqueeze(0).to(device)
            tgt = chunk[1:].unsqueeze(0).to(device)
            out = model(input_ids=inp)
            logits = out.logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll_nats += nll
            total_positions += inp.numel()

    bits_per_token = (total_nll_nats / math.log(2)) / max(1, total_positions)
    entropy_ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)

    result = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epoch_length": args.epoch_length,
        "lr": args.lr,
        "dtype": args.dtype,
        "corpus_bytes": total_bytes,
        "tokens": tokens.numel(),
        "bytes_per_token": round(bytes_per_token, 4),
        "val_tokens_evaluated": total_positions,
        "bits_per_token": round(bits_per_token, 4),
        "held_out_entropy_ratio": round(entropy_ratio, 5),
        # Reference points from prior spikes for easy comparison:
        "reference_zstd_19_ratio": 0.16655,
        "reference_bzip3_wholefile_ratio": 0.14499,
        "reference_spike5_held_out_ratio": 0.2208,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
