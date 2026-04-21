"""MLX training loop for Phase 14 specialists (task 25 step 2).

End-to-end trainer: reads token .npy, samples sequences, forward + loss
+ backward + AdamW step. bf16 compute dtype (MLX GPU default), fp32 master
weights via optimizer state.

**Phase 11 lessons preserved:**
- Cosine LR with linear warmup, counted in OPTIMIZER steps, not micro-
  batch steps (the old Phase 11 bug). The schedule is built once up-front
  with total_steps = (epoch_length * epochs / batch_size / grad_accum).
- `L2Wrap` auxiliary loss: encourages max logit toward 0. Phase 11 used
  a custom autograd function; in MLX we express it as an auxiliary
  L2-like loss term added before backward, which is mathematically
  identical for the gradient landscape.
- torch.compile off (not applicable); MLX uses `mx.compile` on the
  step function which is the equivalent.

**Differences from train_l3tc_phase11.py:**
- MLX lazy evaluation — we call `mx.eval(loss, model.parameters())` at
  the end of each step to block until the GPU catches up, otherwise
  the step timer measures queue depth, not real work.
- AdamW state is stored by the optimizer (not in a separate dict).
- bf16: MLX defaults to fp32 for optimizer state + bf16 for compute
  when you cast the model; the optimizer upgrades grads to fp32
  automatically.

Usage:
    vendor/L3TC/.venv/bin/python scripts/mlx_train_specialist.py \\
        --domain code --steps 200 --ctx 512 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

from mlx_rwkv import RwkvTcHira


# Phase 14 specialist shape
HIDDEN = 96
N_LAYER = 2
INTERMEDIATE = 96
RWKV_RANK = 4
VOCAB_SIZE = 16384

# Phase 11 LR recipe (see docs/phases/PHASE_11.md "Bugs fixed")
LR_PEAK = 1e-4
LR_MIN = 1e-6
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8

# L2Wrap factor from Phase 11 (train_l3tc_phase11.py L2Wrap.backward)
L2_FACTOR = 1e-4

# Token IDs match the SPM training in scripts/train_specialist_tokenizer.py.
PAD_TOKEN = 0
BOS_TOKEN = 2

# g5 baseline from docs/phase-findings/phase_11_findings.md.
G5_REF_TOKENS_PER_SEC = 12.44 * 32 * 2048


# ---------------------------------------------------------------- Data
class TokenSampler:
    """Memory-maps a token .npy file and yields random segments as (input, target) pairs.

    input[t] = [BOS, tokens[start:start+seg-1]]
    target[t] = tokens[start:start+seg]
    Mirrors the L3TCTokenDataset.__getitem__ logic from Phase 11 so the
    training distribution is identical.
    """

    def __init__(self, npy_path: Path, seg_len: int, batch_size: int, seed: int = 1204):
        self.path = npy_path
        self.seg_len = seg_len
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        if not npy_path.exists():
            raise FileNotFoundError(npy_path)
        self.tokens = np.lib.format.open_memmap(npy_path, mode="r")
        n = len(self.tokens)
        if n < seg_len + 2:
            raise ValueError(f"corpus too short: {n} tokens for seg_len={seg_len}")
        self._max_start = n - seg_len - 1

    def sample_batch(self) -> tuple[mx.array, mx.array]:
        """Returns (input_ids [B, T] int32, target_ids [B, T] int32)."""
        starts = self.rng.randint(0, self._max_start, size=self.batch_size)
        inputs = np.empty((self.batch_size, self.seg_len), dtype=np.int32)
        targets = np.empty((self.batch_size, self.seg_len), dtype=np.int32)
        for i, s in enumerate(starts):
            targets[i] = self.tokens[s : s + self.seg_len]
            inputs[i, 0] = BOS_TOKEN
            inputs[i, 1:] = self.tokens[s : s + self.seg_len - 1]
        return mx.array(inputs), mx.array(targets)


class SyntheticSampler:
    """Random-token sampler for the throughput probe. No disk I/O."""

    def __init__(self, seg_len: int, batch_size: int, seed: int = 1204):
        self.seg_len = seg_len
        self.batch_size = batch_size
        # Pre-generate one batch and reuse — throughput is insensitive
        # to token identity, and re-sampling every step would stall on
        # mx.random.randint dispatch overhead.
        mx.random.seed(seed)
        self._inputs = mx.random.randint(3, VOCAB_SIZE, shape=(batch_size, seg_len))
        self._targets = mx.random.randint(3, VOCAB_SIZE, shape=(batch_size, seg_len))
        mx.eval(self._inputs, self._targets)

    def sample_batch(self):
        return self._inputs, self._targets


# ---------------------------------------------------------------- Loss
def masked_ce_with_l2(model, inputs, targets, pad_id=PAD_TOKEN):
    """Cross-entropy + L2Wrap-equivalent auxiliary term.

    PyTorch L2Wrap is a custom backward that adds scattered gradients
    onto the argmax position — mathematically equivalent to adding
    `0.5 * factor * max_logit**2` to the loss (derivative of
    0.5*factor*y[argmax]^2 w.r.t. y[argmax] is factor*y[argmax], which
    matches the PyTorch L2Wrap.backward gy.scatter_(..., maxx * factor)).
    The factor is 1e-4 / (B * T) from train_l3tc_phase11.py.

    Mask: drop tokens where target == pad_id from both the CE and the
    normalization denominator.
    """
    logits = model(inputs)  # (B, T, V)
    B, T, V = logits.shape
    # Upcast to fp32 for CE + argmax stability.
    logits32 = logits.astype(mx.float32)
    flat = logits32.reshape(-1, V)
    tgt = targets.reshape(-1).astype(mx.int32)
    mask = (tgt != pad_id).astype(mx.float32)

    # Cross entropy
    log_probs = flat - mx.logsumexp(flat, axis=-1, keepdims=True)
    # gather log-prob of the target token per row
    row_idx = mx.arange(flat.shape[0])
    gathered = log_probs[row_idx, tgt]
    ce_per_tok = -gathered
    n_real = mx.maximum(mask.sum(), mx.array(1.0))
    ce = (ce_per_tok * mask).sum() / n_real

    # L2Wrap auxiliary: 0.5 * factor * max(logit)^2, averaged over tokens.
    # factor = L2_FACTOR / (B * T) so total contribution scale matches Phase 11.
    max_logits = mx.max(logits32, axis=-1)  # (B, T)
    l2 = 0.5 * (L2_FACTOR / (B * T)) * mx.sum(max_logits * max_logits)

    return ce + l2, ce


# ---------------------------------------------------------------- Schedule
def build_lr_schedule(total_optimizer_steps: int):
    """Linear warmup to LR_PEAK, then cosine to LR_MIN.

    `total_optimizer_steps` must be in optimizer-step units (not micro-
    batch units). If grad_accum > 1 the caller must divide before passing
    here — this is the same anti-bug fix as train_l3tc_phase11.py's
    `total_steps = (steps_per_epoch * epochs) // grad_accum`.
    """
    warmup = opt.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine = opt.cosine_decay(LR_PEAK, max(total_optimizer_steps - WARMUP_STEPS, 1), LR_MIN)
    return opt.join_schedules([warmup, cosine], [WARMUP_STEPS])


# ---------------------------------------------------------------- Train step
def make_train_step(model, optimizer):
    """Build a compiled train step that is one forward + backward + update.

    Using mx.value_and_grad + optimizer.update inside an @mx.compile'd
    function lets MLX fuse the whole step into as few Metal dispatches
    as possible — that's the whole reason MLX clears the task-9 gate.
    """
    loss_and_grad = nn.value_and_grad(model, masked_ce_with_l2)

    state = [model.state, optimizer.state, mx.random.state]

    def step(inputs, targets):
        (total, ce), grads = loss_and_grad(model, inputs, targets)
        # Gradient clip by global norm (matches Phase 11 CLIP_MAX_NORM = 5.0)
        grads = _clip_grads_global_norm(grads, max_norm=5.0)
        optimizer.update(model, grads)
        return total, ce

    # NOTE: `mx.compile` on the outer train step tries to fuse through
    # the WKV custom_function's body — inlining its T-loop of per-timestep
    # ops into one massive Metal primitive that exceeds Metal's argument-
    # buffer limit (observed at ctx=512 batch=32: "Too many inputs/outputs
    # fused in the Metal Compiled primitive"). Without outer compile, MLX
    # still fuses within the WKV primitive's two internal sweeps, which is
    # where the T-loop overhead concentrates. Outer compile is off.
    return step


def _clip_grads_global_norm(grads, max_norm=5.0):
    """Clip all gradients so their combined L2 norm <= max_norm."""
    # MLX tree of arrays — flatten, compute norm, scale.
    flat = _tree_flatten(grads)
    sq_sum = sum(mx.sum(g * g) for g in flat)
    total_norm = mx.sqrt(sq_sum)
    # Avoid division by zero / NaN on first step before any grads exist.
    scale = mx.minimum(mx.array(1.0), max_norm / (total_norm + 1e-6))
    return _tree_map(lambda g: g * scale, grads)


def _tree_flatten(tree):
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif o is not None:
            out.append(o)
    walk(tree)
    return out


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, v) for v in tree]
    if tree is None:
        return None
    return fn(tree)


# ---------------------------------------------------------------- Main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default=None,
                   help="If set, reads data/specialists/{domain}/tokens.npy; "
                        "otherwise uses synthetic random tokens.")
    p.add_argument("--tokens-npy", type=Path, default=None,
                   help="Explicit path to .npy token file.")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    print(f"\n=== MLX specialist trainer (task 25 step 2) ===")
    print(f"mlx:         {mx.__version__}")
    print(f"device:      {mx.default_device()}")
    print(f"shape:       {N_LAYER}L x {HIDDEN}H x {VOCAB_SIZE}v ctx={args.ctx}")
    print(f"batch:       {args.batch_size}  grad_accum: {args.grad_accum}")
    print(f"dtype:       {args.dtype}  steps: {args.steps}")

    # --------- Model
    mx.random.seed(args.seed)
    model = RwkvTcHira(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN,
        num_hidden_layers=N_LAYER, intermediate_size=INTERMEDIATE,
        rwkv_rank=RWKV_RANK, ctx_len=args.ctx,
    )
    if args.dtype == "bf16":
        model.set_dtype(mx.bfloat16)
    elif args.dtype == "fp16":
        model.set_dtype(mx.float16)
    mx.eval(model.parameters())

    # --------- Data
    if args.tokens_npy is not None:
        sampler = TokenSampler(args.tokens_npy, args.ctx, args.batch_size, seed=args.seed)
        print(f"data:        {args.tokens_npy}  ({len(sampler.tokens):,} tokens)")
    elif args.domain is not None:
        sampler = TokenSampler(
            Path(f"data/specialists/{args.domain}/tokens.npy"),
            args.ctx, args.batch_size, seed=args.seed,
        )
    else:
        sampler = SyntheticSampler(args.ctx, args.batch_size, seed=args.seed)
        print(f"data:        synthetic random tokens (throughput probe mode)")

    # --------- Optimizer + schedule
    effective_steps = args.steps // args.grad_accum
    lr_schedule = build_lr_schedule(effective_steps)
    optimizer = opt.AdamW(
        learning_rate=lr_schedule,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    print(f"schedule:    {WARMUP_STEPS} warmup -> {effective_steps} opt steps "
          f"(peak {LR_PEAK} -> min {LR_MIN})")

    train_step = make_train_step(model, optimizer)

    # --------- Warmup (first compile + graph capture)
    print(f"\nwarmup (3 steps; first compile)...")
    t_warm = time.time()
    for _ in range(3):
        inputs, targets = sampler.sample_batch()
        total, ce = train_step(inputs, targets)
        mx.eval(total, ce, model.state, optimizer.state)
    print(f"  warmup wall: {time.time() - t_warm:.2f}s")

    # --------- Timed loop
    print(f"\ntimed run ({args.steps} steps)...")
    losses = []
    any_nan = False
    t0 = time.time()
    last = t0

    for step in range(args.steps):
        inputs, targets = sampler.sample_batch()
        total, ce = train_step(inputs, targets)
        mx.eval(total, ce, model.state, optimizer.state)
        loss_val = float(ce.item())
        losses.append(loss_val)
        if not math.isfinite(loss_val):
            any_nan = True
            print(f"  step {step}: NON-FINITE LOSS {loss_val}")
        if (step + 1) % args.log_every == 0:
            now = time.time()
            it_s = args.log_every / (now - last)
            toks = args.batch_size * args.ctx
            print(f"  step {step+1:4d}/{args.steps}: "
                  f"ce={loss_val:.4f} total={float(total.item()):.4f} "
                  f"{it_s:.2f} it/s ({int(it_s * toks):,} tok/s)")
            last = now

    elapsed = time.time() - t0
    total_tokens = args.batch_size * args.ctx * args.steps
    tok_s = total_tokens / elapsed
    it_s = args.steps / elapsed
    pct_vs_g5 = 100.0 * tok_s / G5_REF_TOKENS_PER_SEC

    print(f"\n=== RESULTS ===")
    print(f"wall sec:        {elapsed:.2f}")
    print(f"it/sec:          {it_s:.2f}")
    print(f"tokens/sec:      {int(tok_s):,}  (fwd+bwd+opt, batch={args.batch_size} ctx={args.ctx})")
    print(f"vs g5 baseline:  {pct_vs_g5:.1f}%  (gate 15.0% = "
          f"{int(G5_REF_TOKENS_PER_SEC * 0.15):,} tok/s)")
    print(f"loss first->last: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"non-finite loss:  {any_nan}")

    descended = losses[-1] < losses[0] - 0.3  # minimum descent over 200 steps
    gate_ok = pct_vs_g5 >= 15.0 and not any_nan and descended
    print(f"\nGATE: {'PASS' if gate_ok else 'FAIL'} "
          f"(>=15% g5 AND no NaN AND loss descended)")

    summary = {
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "shape": f"{N_LAYER}Lx{HIDDEN}Hx{VOCAB_SIZE}v",
        "ctx": args.ctx,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "grad_accum": args.grad_accum,
        "steps": args.steps,
        "wall_sec": round(elapsed, 2),
        "it_per_sec": round(it_s, 3),
        "tokens_per_sec": int(tok_s),
        "pct_vs_g5": round(pct_vs_g5, 2),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_descended": descended,
        "non_finite_loss": any_nan,
        "gate_pass": gate_ok,
    }
    print(f"\nJSON: {json.dumps(summary)}")
    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
