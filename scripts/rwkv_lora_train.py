"""Track 3c: LoRA fine-tune pretrained RWKV-4 on chat content stream.

Same contract as scripts/smollm2_lora_train.py but for RWKV
architecture. Targets the attention + FFN linears that RWKV exposes:
`key`, `value`, `receptance`, `output`. Skips the LM `head` (big
matrix, low marginal gain, keeps adapter small).

Outputs an entropy-bound ratio JSON directly comparable to
scripts/rwkv_zeroshot_ratio.py (Track 3b).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class ContentStreamDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int, n_samples: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx)
        start = torch.randint(0, self.tokens.numel() - self.seq_len - 1, (1,), generator=g).item()
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1].clone(), "labels": chunk[1:].clone()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--content-stream", type=Path, required=True)
    p.add_argument("--train-val-split", type=float, default=0.90,
                   help="First X of tokens = train. Remaining = held-out eval (disjoint).")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--epoch-length", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--eval-chunks", type=int, default=50,
                   help="Contiguous eval chunks from held-out tail for ratio measurement")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-path", type=Path, default=Path("/tmp/rwkv_lora_result.json"))
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
    print(f"[rwkv-lora] device={dev_name}  dtype={args.dtype}")

    # Load tokenizer + base
    print(f"[rwkv-lora] loading {args.base_model}…")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch_dtype).to(device)
    n_base = sum(p.numel() for p in model.parameters())
    print(f"[rwkv-lora] base params={n_base/1e6:.1f}M")

    # LoRA — RWKV target modules
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["key", "value", "receptance", "output"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize content stream
    raw = args.content_stream.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    print(f"[rwkv-lora] corpus={len(raw)/1e6:.1f} MB  tokenizing…")
    t0 = time.time()
    CHUNK = 1 * 1024 * 1024
    all_ids: list[int] = []
    for i in range(0, len(text), CHUNK):
        ids = tok(text[i : i + CHUNK], add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    bytes_per_token = len(raw) / max(1, tokens.numel())
    print(f"[rwkv-lora] tokens={tokens.numel()}  bytes/tok={bytes_per_token:.3f}  took={time.time()-t0:.1f}s")

    split_idx = int(args.train_val_split * tokens.numel())
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"[rwkv-lora] train={train_tokens.numel()}  val={val_tokens.numel()}")

    train_ds = ContentStreamDataset(train_tokens, args.seq_len, args.epoch_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Train
    model.train()
    epoch_losses = []
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
            if n_batches % 500 == 0:
                print(f"  epoch={epoch} batch={n_batches}/{len(train_loader)} loss={loss.item():.4f}")
        avg_loss = total_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        print(f"[rwkv-lora] epoch {epoch}: avg_loss={avg_loss:.4f}  elapsed={time.time()-t_ep:.1f}s")

    # Save adapter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    print(f"[rwkv-lora] saved adapter to {args.output_dir}")

    # Held-out entropy-bound ratio
    model.eval()
    total_nll_nats = 0.0
    total_positions = 0
    with torch.no_grad():
        for i in range(args.eval_chunks):
            start = i * args.seq_len
            chunk = val_tokens[start : start + args.seq_len + 1]
            if chunk.numel() < args.seq_len + 1:
                break
            inp = chunk[:-1].unsqueeze(0).to(device)
            tgt = chunk[1:].unsqueeze(0).to(device)
            out = model(input_ids=inp)
            log_probs = F.log_softmax(out.logits.float(), dim=-1)
            nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll_nats += nll
            total_positions += inp.numel()

    bits_per_token = (total_nll_nats / math.log(2)) / max(1, total_positions)
    entropy_ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)

    result = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["key", "value", "receptance", "output"],
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epoch_length": args.epoch_length,
        "lr": args.lr,
        "dtype": args.dtype,
        "device": dev_name,
        "corpus_bytes": len(raw),
        "tokens": tokens.numel(),
        "bytes_per_token": round(bytes_per_token, 4),
        "train_tokens": train_tokens.numel(),
        "val_tokens": val_tokens.numel(),
        "val_positions_evaluated": total_positions,
        "epoch_train_losses": [round(l, 4) for l in epoch_losses],
        "held_out_bits_per_token": round(bits_per_token, 4),
        "held_out_entropy_ratio": round(entropy_ratio, 5),
        "reference_zstd_19_ratio": 0.16655,
        "reference_bzip3_wholefile_ratio": 0.14499,
        "reference_spike5_l3tc_12m_ratio": 0.2208,
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
