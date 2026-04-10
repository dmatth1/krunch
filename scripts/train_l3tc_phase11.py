"""Phase 11.5 — minimal L3TC trainer.

Trains an L3TC-200K RWKV-v4 + HiRA model from scratch on a
broader corpus, importing only the model class from L3TC and
nothing else from L3TC's source tree. Replaces the
`vendor/L3TC/main.py` invocation we used through Phase 11
attempts 1-15, which dragged in yapf, termcolor, scipy,
deepspeed, ninja, tqdm, transformers, pkuseg, and a numpy 2.x
ABI break — none of which the actual training path needs.

What this script imports from L3TC:
  - `RWKV_TC_HIRA` from `models/RWKV_V4/rwkv_tc_hira_train.py`
    (the model class itself)
  - `L2Wrap` from `models/RWKV_V4/rwkv_v4_train.py` (the
    activation-norm regularizer L3TC uses for RWKV)

That's it. Everything else is our own code: the dataset, the
training loop, the optimizer setup, the LR schedule, the eval
function, the checkpointing.

Architecture and parameter count are pinned to match
`vendor/L3TC/config/l3tc/l3tc_200k.py` per Phase 11 hard
constraint #1 (architecture unchanged). Training recipe is
improved over L3TC: AdamW with weight_decay=0.01 (was Adam with
0 decay), cosine annealing with linear warmup (was a broken
double-stepping StepLR that collapsed LR to 0.3% by end of
training), bf16 mixed precision, and torch.compile.

Usage (local MPS validation):

    cd vendor/L3TC && source .venv/bin/activate
    python ../../scripts/train_l3tc_phase11.py \\
        --train-file data/train_data/train_enwik8_5mb.txt \\
        --val-file   data/train_data/train_enwik8_5mb.txt \\
        --output-dir ../../checkpoints-phase11-local \\
        --epochs 1 --epoch-length 200 --device mps

Usage (cloud, full Pass 1):

    python scripts/train_l3tc_phase11.py \\
        --train-file data/train_data/train_enwik9_bpe_16384_0.999.txt \\
        --val-file   data/val_data/val_enwik9_bpe_16384_0.999.txt \\
        --output-dir checkpoint \\
        --epochs 15 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Force fp32 for the WKV CUDA kernel selection. L3TC's
# rwkv_v4_train.py reads RWKV_FLOAT_MODE at module load time and
# the CUDA kernel only supports fp32. We use bf16 mixed precision
# only via torch.cuda.amp.autocast around the forward, not by
# changing the kernel float type.
os.environ.setdefault("RWKV_FLOAT_MODE", "fp32")
# When running on MPS / CPU we don't want the kernel; the WKV
# call is monkey-patched below to a pure-PyTorch implementation.
# When running on CUDA we DO want the kernel; leave it on.
import torch  # noqa: E402

if not torch.cuda.is_available():
    os.environ.pop("USE_WKV_CUDA_FOR_RWKV", None)
else:
    os.environ.setdefault("USE_WKV_CUDA_FOR_RWKV", "True")

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

# Make L3TC's models package importable
HERE = Path(__file__).resolve().parent
L3TC_DIR = HERE.parent / "vendor" / "L3TC"
sys.path.insert(0, str(L3TC_DIR))

# Pure-PyTorch WKV for MPS / CPU. We monkey-patch this BEFORE
# importing rwkv_tc_hira_train so that on non-CUDA devices the
# module's `RUN_CUDA` symbol points at our autograd-friendly
# implementation. Identical to the WKV monkey-patch in
# scripts/distill_l3tc.py.
def _wkv_cpu_forward(B, T, C, w, u, k, v):
    w_neg = -torch.exp(w.contiguous())
    device = k.device
    dtype = k.dtype
    aa = torch.zeros(B, C, device=device, dtype=dtype)
    bb = torch.zeros(B, C, device=device, dtype=dtype)
    pp = torch.full((B, C), -1e38, device=device, dtype=dtype)
    outs = []
    for t in range(T):
        kk = k[:, t]
        vv = v[:, t]
        ww = u + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        outs.append(a / b)
        ww = pp + w_neg
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p
    return torch.stack(outs, dim=1)


from models.RWKV_V4 import rwkv_tc_hira_train as _rwkv_train_mod  # noqa: E402

if not torch.cuda.is_available():
    _rwkv_train_mod.RUN_CUDA = lambda B, T, C, w, u, k, v: _wkv_cpu_forward(
        B, T, C, w, u, k, v
    )

from models.RWKV_V4.rwkv_tc_hira_train import RWKV_TC_HIRA  # noqa: E402
from models.RWKV_V4.rwkv_v4_train import L2Wrap  # noqa: E402


# === Hyperparameters pinned to vendor/L3TC/config/l3tc/l3tc_200k.py ===
# Per Phase 11 hard constraint #1 these do not change.
HIDDEN_SIZE = 96
NUM_HIDDEN_LAYERS = 2
INTERMEDIATE_SIZE = 96
RWKV_RANK = 4
VOCAB_SIZE = 16384
CTX_LEN = 2048
SENTENCE_LENGTH = CTX_LEN
CHUNK_SIZE = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8
CLIP_MAX_NORM = 5.0
# Cosine annealing with linear warmup. Replaces L3TC's broken
# double-stepping StepLR (which collapsed LR to 0.3% of initial by
# end of training). The warmup prevents instability on the first few
# steps from random init; the cosine tail gives a smooth convergence.
WARMUP_STEPS = 500
LR_MIN = 1e-6
# Weight decay for AdamW. L3TC used Adam with 0 decay; we use AdamW
# with mild decay (0.01) for better generalization.
WEIGHT_DECAY = 0.01
# L3TC's per-epoch sample budget. We follow it 1:1.
EPOCH_LENGTH = 1_000_000

# Token id constants — match L3TC's SPM training: pad=0, unk=1, bos=2.
PAD_TOKEN = 0
UNK_TOKEN = 1
BOS_TOKEN = 2


# ============================================================
# Dataset
# ============================================================
class L3TCTokenDataset(Dataset):
    """Reads a tokenized text file and yields fixed-length training
    samples in the shape L3TC's RWKV_TC_HIRA model expects.

    File format: auto-detected. Two formats supported:
      1. One integer per line (matches what
         scripts/spot-fleet-userdata.sh produces, and what we
         already have cached in S3).
      2. Single line of comma-separated integers (matches L3TC's
         own enwik_dataset.py).

    Each sample is `(input_token, output_token, input_types,
    output_types)` reshaped to (chunk_size=1, segment_length).
    Input is `[BOS] tok[start:start+seg-1]`, output is the
    one-token-shifted ground truth `tok[start:start+seg]`. Types
    are 0 for padding, 1 for real tokens (matches L3TC).

    The class is __len__-stable per epoch via `epoch_length` so
    we can resume / reproduce. Random sampling within an epoch is
    seeded by `np.random.RandomState`.

    Loading is done into a single np.int32 array, not a Python
    list, so a 1.3 GB token file uses ~1 GB of RAM instead of
    L3TC's ~17 GB Python-list-of-ints overhead.
    """

    def __init__(
        self,
        corpus_path: Path,
        epoch_length: int,
        segment_length: int = SENTENCE_LENGTH,
        chunk_size: int = CHUNK_SIZE,
        seed: int = 1204,
    ):
        self.segment_length = segment_length
        self.chunk_size = chunk_size
        self.epoch_length = epoch_length
        self.rng = np.random.RandomState(seed)

        print(f"loading token file: {corpus_path}")
        t0 = time.time()
        self.tokens = self._load_tokens(corpus_path)
        print(
            f"  loaded {len(self.tokens):,} tokens in {time.time() - t0:.1f}s "
            f"(~{self.tokens.nbytes / 1e6:.1f} MB int32)"
        )
        if len(self.tokens) < self.segment_length * self.chunk_size + 1:
            raise ValueError(
                f"corpus too short ({len(self.tokens)} tokens) for "
                f"segment {self.segment_length} x chunk {self.chunk_size}"
            )

    @staticmethod
    def _load_tokens(path: Path) -> np.ndarray:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4096)
        # Auto-detect format. If the head has commas and few newlines,
        # it's L3TC comma-separated. Otherwise, one int per line.
        n_commas = head.count(",")
        n_newlines = head.count("\n")
        if n_commas > n_newlines * 2:
            # Comma-separated single-line (L3TC native format).
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            return np.array(
                [int(x) for x in line.split(",") if x],
                dtype=np.int32,
            )
        else:
            # One int per line (our format from spot-fleet-userdata.sh).
            tokens = np.fromfile(path, sep="\n", dtype=np.int32)
            # np.fromfile leaves a trailing zero if the file ends with
            # a newline; trim if it doesn't look like a real token.
            if len(tokens) > 0 and tokens[-1] == 0 and not path.read_bytes()[-2:] == b"0\n":
                tokens = tokens[:-1]
            return tokens

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int):
        n = len(self.tokens)
        seg = self.segment_length
        chunk = self.chunk_size

        # Random start within the corpus.
        max_start = n - chunk * seg - 1
        start = int(self.rng.randint(0, max_start))

        input_chunks = []
        output_chunks = []
        input_types = []
        output_types = []

        for c in range(chunk):
            cs = start + c * seg
            ce = cs + seg
            target = self.tokens[cs:ce].astype(np.int64)
            if c == 0:
                # First segment: prepend BOS, drop the last target token
                # for the input. Matches L3TC's enwik_dataset.py logic.
                inp = np.concatenate(
                    [[BOS_TOKEN], self.tokens[cs : ce - 1].astype(np.int64)]
                )
            else:
                # Subsequent segments in a chunk: shifted by one token.
                inp = self.tokens[cs - 1 : ce - 1].astype(np.int64)

            in_type = np.where(inp == PAD_TOKEN, 0, 1).astype(np.int64)
            out_type = np.where(target == PAD_TOKEN, 0, 1).astype(np.int64)

            input_chunks.append(inp)
            output_chunks.append(target)
            input_types.append(in_type)
            output_types.append(out_type)

        return {
            "input_token": torch.from_numpy(np.stack(input_chunks)),
            "output_token": torch.from_numpy(np.stack(output_chunks)),
            "input_types": torch.from_numpy(np.stack(input_types)),
            "output_types": torch.from_numpy(np.stack(output_types)),
        }


# ============================================================
# Model + optimizer setup
# ============================================================
def build_model(device: torch.device) -> RWKV_TC_HIRA:
    print(
        f"building RWKV_TC_HIRA: layers={NUM_HIDDEN_LAYERS} hidden={HIDDEN_SIZE} "
        f"intermediate={INTERMEDIATE_SIZE} rank={RWKV_RANK} vocab={VOCAB_SIZE}"
    )
    model = RWKV_TC_HIRA(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        rwkv_rank=RWKV_RANK,
        ctx_len=CTX_LEN,
        dropout_prob=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_embed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "head" not in n and "emb" not in n
    )
    print(
        f"  total params: {n_params:,}, "
        f"non-embed/head params: {n_params_no_embed:,}"
    )
    model = model.to(device)

    return model


def maybe_compile(model, device: torch.device, no_compile: bool):
    """torch.compile for throughput on CUDA. Disabled by default because
    L2Wrap (custom autograd.Function) can't be traced by dynamo and
    causes fallback + extra memory overhead that leads to OOM."""
    if no_compile or device.type != "cuda":
        print("  torch.compile: disabled")
        return model
    try:
        model = torch.compile(model)
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: not available ({e})")
    return model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # AdamW with mild weight decay (0.01) for generalization. L3TC
    # used Adam with 0 decay; we've lifted the "match L3TC recipe
    # exactly" constraint. AdamW + decay is the modern standard.
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing with linear warmup. Replaces L3TC's broken
    double-stepping StepLR."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        # Cosine decay from 1.0 to LR_MIN/LEARNING_RATE
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = LR_MIN / LEARNING_RATE
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Training and eval
# ============================================================
def compute_masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    types_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Same masking pattern as L3TC's main.py train_one_epoch."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = types_mask.reshape(-1).to(flat_logits.dtype)
    per_token = criterion(flat_logits, flat_targets)
    return (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)


def train_one_epoch(
    model: RWKV_TC_HIRA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp_bf16: bool,
    log_every: int = 10,
    save_every_steps: int = 1000,
    save_fn=None,
    grad_accum: int = 1,
) -> dict:
    model.train()
    n_steps = 0
    sum_loss = 0.0
    t0 = time.time()
    last_log_t = t0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_token = batch["input_token"]
        input_types = batch["input_types"]
        output_token = batch["output_token"]
        output_types = batch["output_types"]

        if use_amp_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_token, input_types, train=True)
                loss = compute_masked_ce_loss(
                    logits, output_token, output_types, criterion
                )
                loss = L2Wrap.apply(loss, logits)
        else:
            logits = model(input_token, input_types, train=True)
            loss = compute_masked_ce_loss(
                logits, output_token, output_types, criterion
            )
            loss = L2Wrap.apply(loss, logits)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum

        loss_value = loss.item() * grad_accum  # un-scale for logging
        if not math.isfinite(loss_value):
            print(f"  WARNING: non-finite loss {loss_value} at step {step}")
            sys.exit(1)

        loss.backward()

        # Optimizer step every grad_accum micro-batches
        if (step + 1) % grad_accum == 0:
            if CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        sum_loss += loss_value
        n_steps += 1

        # Intra-epoch checkpoint so we don't lose multi-hour epochs
        # to spot reclaim or unexpected termination.
        if save_fn is not None and (step + 1) % save_every_steps == 0:
            save_fn(f"checkpoint_latest.pth")

        if (step + 1) % log_every == 0:
            now = time.time()
            avg_loss = sum_loss / n_steps
            steps_per_sec = log_every / (now - last_log_t)
            print(
                f"  epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={avg_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"{steps_per_sec:.2f} it/s"
            )
            last_log_t = now

    avg_loss = sum_loss / max(n_steps, 1)
    print(f"epoch {epoch} done: avg_loss={avg_loss:.4f} wall={time.time() - t0:.1f}s")
    return {"loss": avg_loss, "n_steps": n_steps}


@torch.no_grad()
def evaluate(
    model: RWKV_TC_HIRA,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    model.eval()
    total_ce = 0.0
    total_tokens = 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_token = batch["input_token"]
        input_types = batch["input_types"]
        output_token = batch["output_token"]
        output_types = batch["output_types"]

        logits = model(input_token, input_types, train=True)
        # Compute *unmasked* per-token CE for accumulation, then mask.
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = output_token.reshape(-1)
        flat_mask = output_types.reshape(-1).to(flat_logits.dtype)
        per_token = criterion(flat_logits, flat_targets)
        masked = (per_token * flat_mask).sum().item()
        n_tok = int(flat_mask.sum().item())
        total_ce += masked
        total_tokens += n_tok

    avg_ce = total_ce / max(total_tokens, 1)
    bpb = avg_ce / math.log(2) / 3.5  # rough — ~3.5 bytes per BPE token
    print(
        f"eval epoch {epoch}: tokens={total_tokens:,} avg_ce_nats={avg_ce:.4f} "
        f"approx_bpb={bpb:.4f}"
    )
    return {"avg_ce_nats": avg_ce, "approx_bpb": bpb, "n_tokens": total_tokens}


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(
    model: RWKV_TC_HIRA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    torch.save(state, path)
    print(f"  saved checkpoint: {path}")
    return path


def load_checkpoint(
    path: Path,
    model: RWKV_TC_HIRA,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw)
    cleaned = OrderedDict()
    for k, v in sd.items():
        cleaned[k[len("module.") :] if k.startswith("module.") else k] = v
    out = model.load_state_dict(cleaned, strict=False)
    print(f"loaded checkpoint {path}: missing={len(out.missing_keys)} "
          f"unexpected={len(out.unexpected_keys)}")
    start_epoch = 0
    if optimizer is not None and "optimizer" in raw:
        try:
            optimizer.load_state_dict(raw["optimizer"])
        except Exception as e:
            print(f"  could not restore optimizer state: {e}")
    if scheduler is not None and raw.get("lr_scheduler"):
        try:
            scheduler.load_state_dict(raw["lr_scheduler"])
        except Exception as e:
            print(f"  could not restore scheduler state: {e}")
    if "epoch" in raw:
        start_epoch = raw["epoch"] + 1
    return start_epoch


# ============================================================
# Main
# ============================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, required=True,
                   help="Held-out validation token file. For sanity-check "
                        "runs may point at the same file as --train-file.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--epoch-length", type=int, default=EPOCH_LENGTH,
                   help="Samples per training epoch (matches L3TC).")
    p.add_argument("--val-length", type=int, default=200,
                   help="Number of validation samples per eval pass.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1204)
    p.add_argument("--save-every-steps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bf16 mixed precision (use fp32). "
                        "Auto-disabled on non-CUDA devices.")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile. Required when using "
                        "custom autograd Functions like L2Wrap that "
                        "dynamo cannot trace.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch "
                        "= batch-size * grad-accum.")
    args = p.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    use_amp_bf16 = (
        not args.no_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    print(f"device: {device}, bf16 amp: {use_amp_bf16}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "phase11_args.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    # Datasets
    train_ds = L3TCTokenDataset(
        args.train_file,
        epoch_length=args.epoch_length,
        seed=args.seed,
    )
    val_ds = L3TCTokenDataset(
        args.val_file,
        epoch_length=args.val_length,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=max(1, args.num_workers // 2),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # Model + optimizer + scheduler
    model = build_model(device)
    model = maybe_compile(model, device, args.no_compile)
    optimizer = build_optimizer(model)
    steps_per_epoch = args.epoch_length // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, total_steps)
    print(f"schedule: {WARMUP_STEPS} warmup steps, {total_steps} total steps, "
          f"cosine {LEARNING_RATE} -> {LR_MIN}")
    # Per-token cross entropy. We mask + reduce ourselves to match L3TC.
    criterion = nn.CrossEntropyLoss(reduction="none")

    start_epoch = 0
    if args.resume is not None and args.resume.exists():
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        # Intra-epoch save callback for resilience against spot reclaim
        # or unexpected termination.
        def _save_fn(name: str):
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                args, args.output_dir, name,
            )

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            use_amp_bf16=use_amp_bf16,
            log_every=args.log_every,
            save_every_steps=args.save_every_steps,
            save_fn=_save_fn,
            grad_accum=args.grad_accum,
        )
        global_step += train_stats["n_steps"]

        # Per-epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, args,
            args.output_dir, f"checkpoint{epoch:04d}.pth",
        )
        # Mirror to "latest"
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, args,
            args.output_dir, "checkpoint_latest.pth",
        )

        # Eval
        eval_stats = evaluate(model, val_loader, criterion, device, epoch)

        with open(args.output_dir / "log.txt", "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train": train_stats,
                "eval": eval_stats,
                "global_step": global_step,
            }) + "\n")

    print("training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
