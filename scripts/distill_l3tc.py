"""Phase 4e: distill L3TC-3.2M into an L3TC-200K-shaped student.

Takes:
  - a teacher dump file produced by `l3tc dump-teacher`
    (top-K softmax distributions from the teacher model over a
    training corpus, tokenized at segment_bytes=2048)
  - the shipped L3TC-200K checkpoint as the student initializer
  - an input text corpus (typically enwik8)

Fine-tunes the student on the corpus with a blended loss:

    loss = (1 - alpha) * CE(student_logits, target) +
           alpha * KL(student_soft / T, teacher_soft / T) * T^2

where `alpha` is the distillation weight (default 0.7) and `T`
is the temperature (default 2.0). The standard knowledge
distillation formulation from Hinton et al. 2015.

Outputs a new checkpoint in the same format as the shipped
L3TC-200K .pth, ready to feed into
`l3tc-rust/scripts/convert_checkpoint.py`.

Compute reality check: 200K params × ~100 MB of text × a few
epochs is roughly 15-30 CPU-hours on Apple Silicon. Use a tiny
subset (--max-bytes 1000000) for a quick debug pass first. For
real training, use MPS if available (--device mps) or rent a
GPU (--device cuda).

Usage:

    # Debug run on first 1 MB of enwik8, ~5 min on CPU:
    cd vendor/L3TC && source .venv/bin/activate
    python ../../scripts/distill_l3tc.py \\
        --teacher-dump /tmp/enwik6_teacher.bin \\
        --init-checkpoint checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth \\
        --config config/l3tc/l3tc_200k.py \\
        --input ../../bench/corpora/enwik6 \\
        --output ../../l3tc-rust/checkpoints/l3tc_200k_distilled.pth \\
        --max-bytes 1000000 \\
        --segment-bytes 2048 \\
        --epochs 2 \\
        --batch-size 4 \\
        --lr 1e-5 \\
        --device cpu
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Force fp32 and disable the CUDA WKV kernel path BEFORE importing
# L3TC's training module — it reads `RWKV_FLOAT_MODE` at module
# load time and gates the `wkv_cuda` extension import on
# `USE_WKV_CUDA_FOR_RWKV`. We monkey-patch RUN_CUDA below to use
# a pure-PyTorch CPU/MPS implementation instead.
os.environ.setdefault("RWKV_FLOAT_MODE", "fp32")
os.environ.pop("USE_WKV_CUDA_FOR_RWKV", None)

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

HERE = Path(__file__).resolve().parent
L3TC_DIR = HERE.parent / "vendor" / "L3TC"
sys.path.insert(0, str(L3TC_DIR))

from models.RWKV_V4 import rwkv_tc_hira_train as _rwkv_train_mod  # noqa: E402
from models.RWKV_V4.rwkv_tc_hira_train import RWKV_TC_HIRA  # noqa: E402


def _wkv_cpu_forward(
    B: int,
    T: int,
    C: int,
    w: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch WKV kernel replacing L3TC's CUDA-only RUN_CUDA.

    Matches the recurrence in `rwkv_tc_hira_infer.py::RWKV_TimeMix.forward`,
    operating on a full (B, T, C) time batch instead of one token at
    a time. Autograd handles the backward pass automatically.

    Args:
        B, T, C: batch, time, channel
        w: (C,) time_decay (will be -exp'd internally, matching L3TC's
           CUDA WKV.forward which does `w = -torch.exp(w)` before use)
        u: (C,) time_first bonus
        k: (B, T, C) attention keys
        v: (B, T, C) attention values

    Returns: (B, T, C) — the WKV-mixed output, equivalent to what
    L3TC's `RUN_CUDA` produces. The sigmoid-gated receptance is
    applied outside this kernel by the caller.
    """
    # L3TC's WKV.forward applies `w = -torch.exp(w)` internally.
    # We do the same here so the recurrence uses the post-negation
    # decay directly.
    w_neg = -torch.exp(w.contiguous())  # (C,)

    device = k.device
    dtype = k.dtype
    aa = torch.zeros(B, C, device=device, dtype=dtype)
    bb = torch.zeros(B, C, device=device, dtype=dtype)
    pp = torch.full((B, C), -1e38, device=device, dtype=dtype)
    outs = []
    for t in range(T):
        kk = k[:, t]  # (B, C)
        vv = v[:, t]  # (B, C)

        # Output step: combine current state with (time_first + k).
        ww = u + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        outs.append(a / b)

        # State update: decay state then fold in current k, v.
        ww = pp + w_neg
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p

    return torch.stack(outs, dim=1)  # (B, T, C)


def _run_cuda_cpu(B, T, C, w, u, k, v):
    """Drop-in replacement for L3TC's RUN_CUDA that runs on CPU/MPS."""
    return _wkv_cpu_forward(B, T, C, w, u, k, v)


# Monkey-patch L3TC's training module to use the CPU WKV.
_rwkv_train_mod.RUN_CUDA = _run_cuda_cpu

# Default student shape constants (L3TC-200K from
# vendor/L3TC/config/l3tc/l3tc_200k.py). Override per run via CLI
# flags for Phase 4e3 smaller-student experiments.
DEFAULT_HIDDEN_SIZE = 96
DEFAULT_NUM_LAYERS = 2
DEFAULT_INTERMEDIATE_SIZE = 96
DEFAULT_RWKV_RANK = 4
VOCAB_SIZE = 16384
CTX_LEN = 2048
BOS_ID = 2


def load_teacher_dump(
    path: Path,
) -> tuple[int, int, list[list[tuple[int, int, list[tuple[int, float]]]]]]:
    """Load the `.bin` file written by `l3tc dump-teacher` (v2).

    Returns `(vocab_size, top_k, segments)` where `segments` is a
    list of segments, each of which is a list of per-step
    `(input_token_id, target_token_id, [(token_id, prob), ...])`
    tuples. All tokenization is taken from the dump; the Python
    caller never needs to re-tokenize the corpus, which avoids
    any drift between the Rust tokenizer's Phase 4b2 unk
    extraction and a simpler Python tokenizer.
    """
    data = path.read_bytes()
    pos = 0

    def read(n: int) -> bytes:
        nonlocal pos
        b = data[pos : pos + n]
        pos += n
        return b

    magic = read(4)
    if magic != b"L3TD":
        raise ValueError(f"bad magic: {magic!r}")
    (version,) = struct.unpack("<I", read(4))
    if version != 2:
        raise ValueError(
            f"unsupported teacher dump version {version}; expected 2. "
            "Re-run `l3tc dump-teacher` with the current binary."
        )
    (vocab,) = struct.unpack("<I", read(4))
    (top_k,) = struct.unpack("<I", read(4))
    (n_segs,) = struct.unpack("<I", read(4))
    (n_steps,) = struct.unpack("<Q", read(8))

    print(f"  teacher dump v2: vocab={vocab}, top_k={top_k}, segments={n_segs}, steps={n_steps}")

    # Per-segment step counts.
    seg_steps = []
    for _ in range(n_segs):
        (s,) = struct.unpack("<I", read(4))
        seg_steps.append(s)

    # Per-step records: input(4) + target(4) + K × (id(4) + prob(4))
    step_record_size = 4 + 4 + top_k * 8
    segments: list[list[tuple[int, int, list[tuple[int, float]]]]] = []
    for n_seg_steps in seg_steps:
        seg_records: list[tuple[int, int, list[tuple[int, float]]]] = []
        for _ in range(n_seg_steps):
            record = read(step_record_size)
            input_id = struct.unpack_from("<I", record, 0)[0]
            target = struct.unpack_from("<I", record, 4)[0]
            topk: list[tuple[int, float]] = []
            off = 8
            for _ in range(top_k):
                tid, prob = struct.unpack_from("<If", record, off)
                off += 8
                topk.append((tid, prob))
            seg_records.append((input_id, target, topk))
        segments.append(seg_records)
    return vocab, top_k, segments


def load_student(
    init_checkpoint: Path | None,
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
    rwkv_rank: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the student training model.

    If `init_checkpoint` is provided, load weights from it (for
    same-shape fine-tuning). Otherwise random-init — the Phase 4e3
    path, where the student layer count differs from any shipped
    checkpoint.
    """
    print(
        f"building student: layers={num_layers} hidden={hidden_size} "
        f"intermediate={intermediate_size} rank={rwkv_rank}"
    )
    model = RWKV_TC_HIRA(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        rwkv_rank=rwkv_rank,
        ctx_len=CTX_LEN,
        dropout_prob=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  student params: {n_params:,}")

    if init_checkpoint is not None:
        print(f"loading student init: {init_checkpoint}")
        raw = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        sd = raw.get("model", raw)

        # Strip DDP prefix if present
        cleaned = OrderedDict()
        for k, v in sd.items():
            cleaned[k[len("module.") :] if k.startswith("module.") else k] = v

        load_out = model.load_state_dict(cleaned, strict=False)
        print(
            f"  load result: missing={len(load_out.missing_keys)}, "
            f"unexpected={len(load_out.unexpected_keys)}"
        )
        if load_out.missing_keys:
            print(f"  missing sample: {load_out.missing_keys[:5]}")
        if load_out.unexpected_keys:
            print(f"  unexpected sample: {load_out.unexpected_keys[:5]}")
    else:
        print("  random init (no --init-checkpoint)")

    model = model.to(device)
    model.train()
    return model


def segments_to_tensors(
    segments: list[list[tuple[int, int, list[tuple[int, float]]]]],
    max_segments: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert loaded teacher-dump segments into per-segment
    (input, target, teacher_topk) tensors ready for training.

    Each input is the sequence of model-input token ids in that
    segment; each target is the ground-truth next token; teacher_topk
    is shape (T, K, 2) with (token_id, prob) pairs.
    """
    out = []
    total_steps = 0
    for seg_idx, seg in enumerate(segments):
        if max_segments is not None and seg_idx >= max_segments:
            break
        if not seg:
            continue
        n_steps = len(seg)
        inputs = torch.tensor([s[0] for s in seg], dtype=torch.long)
        targets = torch.tensor([s[1] for s in seg], dtype=torch.long)
        topk = torch.tensor(
            [[(tid, prob) for tid, prob in s[2]] for s in seg],
            dtype=torch.float32,
        )
        out.append((inputs, targets, topk))
        total_steps += n_steps
    print(f"  prepared {len(out)} segments, {total_steps} total prediction steps")
    return out


def distillation_loss(
    student_logits: torch.Tensor,
    target: torch.Tensor,
    teacher_topk: torch.Tensor,
    vocab_size: int,
    temperature: float,
    alpha: float,
    tail_floor: float = 1e-7,
) -> tuple[torch.Tensor, float, float]:
    """Blend CE and KL distillation losses.

    Args:
        student_logits: (T, V)   student model output (pre-softmax)
        target:         (T,)     ground-truth next tokens
        teacher_topk:   (T, K, 2) teacher top-K (token_id, prob) pairs
        vocab_size:     V
        temperature:    T in Hinton's formulation
        alpha:          distillation weight (0 = pure CE, 1 = pure KL)
        tail_floor:     probability to assign to non-top-K teacher tokens
                        so the KL divergence stays finite

    Returns (loss, ce_value, kl_value).
    """
    T = temperature

    # --- cross-entropy on ground truth ---
    ce = F.cross_entropy(student_logits, target)

    # --- KL(teacher || student) at temperature T ---
    # Reconstruct a sparse teacher distribution over the full vocab.
    # Top-K positions get their explicit probs; all others get
    # `tail_floor`. This is an approximation but preserves KL's
    # "penalize low student prob where teacher is high" signal.
    n_steps = student_logits.shape[0]
    teacher_ids = teacher_topk[:, :, 0].long()  # (T, K)
    teacher_probs = teacher_topk[:, :, 1]  # (T, K)

    # Renormalize the top-K over the available "probability mass"
    # after reserving some for the tail. Simple approach: sum the
    # top-K, compute the tail-mass, distribute uniformly among
    # (V - K) tail tokens.
    k = teacher_probs.shape[1]
    topk_sum = teacher_probs.sum(dim=-1, keepdim=True)  # (T, 1)
    tail_mass = torch.clamp(1.0 - topk_sum, min=0.0)  # (T, 1)
    tail_per_token = tail_mass / (vocab_size - k)  # (T, 1)

    # Build full teacher distribution
    teacher_full = tail_per_token.expand(-1, vocab_size).clone()
    # Scatter the top-K probs
    teacher_full.scatter_(1, teacher_ids, teacher_probs)

    # Apply temperature: teacher_soft = softmax(log(teacher_full) / T)
    # Using log and renormalizing is more stable than raising to a
    # power directly.
    teacher_log = torch.log(teacher_full.clamp(min=tail_floor))
    teacher_soft = F.softmax(teacher_log / T, dim=-1)

    student_log_soft = F.log_softmax(student_logits / T, dim=-1)

    # KL(teacher || student) = sum teacher * (log teacher - log student)
    # PyTorch's F.kl_div expects log(Q) as first arg, P as second.
    kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)

    loss = (1 - alpha) * ce + alpha * kl
    return loss, ce.item(), kl.item()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # --- data loading ---
    # We don't need the tokenizer on the Python side anymore:
    # the teacher dump (v2) includes input + target token ids
    # per step, so the Python training loop is fully
    # tokenization-free. This avoids any drift between the
    # Rust tokenizer's Phase 4b2 unk extraction and a simpler
    # Python tokenizer.

    print(f"loading teacher dump: {args.teacher_dump}")
    teacher_vocab, teacher_k, teacher_segments = load_teacher_dump(args.teacher_dump)
    if teacher_vocab != VOCAB_SIZE:
        raise ValueError(f"teacher vocab {teacher_vocab} != expected {VOCAB_SIZE}")

    print("preparing training tensors")
    aligned = segments_to_tensors(teacher_segments, max_segments=args.max_segments)

    # --- student init ---
    student = load_student(
        args.init_checkpoint,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        rwkv_rank=args.rwkv_rank,
        device=device,
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8
    )

    # --- training loop ---
    print(f"training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kl = 0.0
        n_seen = 0
        t0 = time.time()

        for seg_idx, (input_tokens, target, teacher_topk) in enumerate(aligned):
            input_tokens = input_tokens.unsqueeze(0).to(device)  # (1, T)
            target = target.to(device)
            teacher_topk = teacher_topk.to(device)

            optimizer.zero_grad()
            # RWKV_TC_HIRA.forward_train expects (B, 1, T) or (B, T)?
            # Looking at forward_train: input_token = input_token.squeeze(dim=1)
            # so the expected shape is (B, 1, T) before squeeze → (B, T) after.
            logits = student(
                input_tokens.unsqueeze(1),  # (1, 1, T)
                input_types=None,
                train=True,
            )
            # logits shape: (1, T, V)
            logits = logits.squeeze(0)  # (T, V)

            loss, ce_val, kl_val = distillation_loss(
                logits,
                target,
                teacher_topk,
                VOCAB_SIZE,
                args.temperature,
                args.alpha,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce += ce_val
            epoch_kl += kl_val
            n_seen += 1

            if (seg_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch} seg {seg_idx + 1}/{len(aligned)} "
                    f"loss={epoch_loss / n_seen:.4f} "
                    f"ce={epoch_ce / n_seen:.4f} "
                    f"kl={epoch_kl / n_seen:.4f} "
                    f"{elapsed:.1f}s"
                )

        print(
            f"epoch {epoch} done: loss={epoch_loss / n_seen:.4f}, "
            f"ce={epoch_ce / n_seen:.4f}, kl={epoch_kl / n_seen:.4f}, "
            f"wall={time.time() - t0:.1f}s"
        )

        # Save a per-epoch checkpoint so we can pick the best later.
        epoch_out = args.output.with_suffix(f".epoch{epoch}.pth")
        print(f"  saving epoch checkpoint: {epoch_out}")
        torch.save({"model": student.state_dict()}, epoch_out)

    # --- save final checkpoint ---
    print(f"saving final checkpoint: {args.output}")
    save_state = {"model": student.state_dict()}
    torch.save(save_state, args.output)
    print("done")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-dump", type=Path, required=True)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional same-shape .pth to warm-start from. Omit for "
        "random init (Phase 4e3 smaller-student path).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Cap the number of training segments (for debug runs).",
    )
    # Student architecture — default to L3TC-200K shape.
    p.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    p.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    p.add_argument(
        "--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE
    )
    p.add_argument("--rwkv-rank", type=int, default=DEFAULT_RWKV_RANK)
    p.add_argument("--segment-bytes", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    # Defaults tuned for Phase 4e3 from-scratch 1-layer distillation.
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
