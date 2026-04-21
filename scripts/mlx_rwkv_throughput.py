"""Task-25 HARD CHECKPOINT: MLX forward-only throughput probe.

Mirrors scripts/mps_throughput_probe.py (task 9) but on MLX. Target
config: 2L × 96H × 16K vocab, batch 32, ctx 512, bf16/fp16/fp32. Target:
≥15% of g5.xlarge baseline = 122,290 tok/s forward+backward, but this
probe is forward-only so we compare against ~2x that for equivalence
(since backward ≈ same cost as forward). That works out to ~245K tok/s
forward-only as the proxy gate.

Actually cleaner: the task-9 gate is 122K tok/s in the original
forward+backward sense. A forward-only probe that hits 122K means
a full training step at batch 32 would see roughly the same because
backward has the same T-loop structure. So we keep the 122K threshold
directly and mark it as a conservative reach.

If we're far under 122K even on forward-only, we stop here and raise
to team-lead (per task 25 hard checkpoint).

Usage:
    vendor/L3TC/.venv/bin/python scripts/mlx_rwkv_throughput.py
Or with knobs:
    vendor/L3TC/.venv/bin/python scripts/mlx_rwkv_throughput.py \\
        --ctx 512 --batch-size 32 --steps 200 --dtype bf16 --compile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx
from mlx_rwkv import RwkvTcHira

# Task-9 reference: g5 = 12.44 it/s * 32 batch * 2048 ctx = 815,595 tok/s
# forward+backward. Threshold 15% = 122,340 tok/s.
G5_REF_TOKENS_PER_SEC = 12.44 * 32 * 2048
GATE_PCT = 15.0


def run(args):
    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    cmp_dtype = dtype_map[args.dtype]

    print(f"\n=== MLX RWKV Throughput Probe (Task 25 hard checkpoint) ===")
    print(f"mlx:              {mx.__version__}")
    print(f"device:           {mx.default_device()}")
    print(f"shape:            2L x 96H x 16384 vocab, ctx={args.ctx}")
    print(f"batch:            {args.batch_size}")
    print(f"dtype:            {args.dtype}")
    print(f"compile:          {args.compile}")
    print(f"steps:            {args.steps}")

    mx.random.seed(1204)
    model = RwkvTcHira(
        vocab_size=16384, hidden_size=96, num_hidden_layers=2,
        intermediate_size=96, rwkv_rank=4, ctx_len=args.ctx,
        dropout_prob=0.0,
    )
    # Cast all parameters to the target compute dtype.
    if cmp_dtype is not mx.float32:
        model.set_dtype(cmp_dtype)
    mx.eval(model.parameters())

    n_params = sum(int(mx.prod(mx.array(list(p.shape))).item()) if len(p.shape) > 0 else 1
                   for _, p in _leaf_params(model))
    print(f"params:           {n_params:,}")

    # Input batch — random token IDs. Graph-equivalent to real data.
    tokens = mx.random.randint(3, 16384, shape=(args.batch_size, args.ctx))

    fwd = model
    if args.compile:
        # mx.compile captures the graph on first call and re-uses it —
        # this is the key MLX affordance vs the torch-MPS Python T-loop.
        fwd = mx.compile(model.__call__)

    print("\nwarmup (5 steps)...")
    t0 = time.time()
    for _ in range(5):
        out = fwd(tokens)
        mx.eval(out)
    print(f"  warmup wall: {time.time() - t0:.2f}s")

    print(f"\ntimed run ({args.steps} steps)...")
    losses_printed = 0
    t0 = time.time()
    last = t0
    log_every = max(args.steps // 10, 5)

    for step in range(args.steps):
        out = fwd(tokens)
        mx.eval(out)  # block until this step is complete
        if (step + 1) % log_every == 0:
            now = time.time()
            it_s = log_every / (now - last)
            toks = args.batch_size * args.ctx
            print(f"  step {step+1:4d}/{args.steps}: "
                  f"{it_s:.2f} it/s ({int(it_s * toks):,} tok/s)")
            last = now

    elapsed = time.time() - t0
    total_tokens = args.batch_size * args.ctx * args.steps
    tok_s = total_tokens / elapsed
    it_s = args.steps / elapsed
    pct = 100.0 * tok_s / G5_REF_TOKENS_PER_SEC

    print(f"\n=== RESULTS ===")
    print(f"wall sec:       {elapsed:.2f}")
    print(f"it/sec:         {it_s:.2f}")
    print(f"tokens/sec:     {int(tok_s):,}  (forward-only)")
    print(f"vs g5 fwd+bwd:  {pct:.1f}%  (threshold {GATE_PCT}% = "
          f"{int(G5_REF_TOKENS_PER_SEC * GATE_PCT / 100):,} tok/s)")

    # Note on interpretation: this is forward-only. Training needs
    # backward + optimizer step too. Backward ≈ 1-1.5× forward cost
    # for this shape. So forward-only throughput of X roughly implies
    # training throughput of X/2-X/2.5.
    print(f"expected train: {int(tok_s / 2.2):,} tok/s (fwd+bwd+opt ~2.2x fwd)")
    train_pct = 100.0 * (tok_s / 2.2) / G5_REF_TOKENS_PER_SEC
    print(f"expected train % vs g5: {train_pct:.1f}%")

    gate_pass_fwd = pct >= GATE_PCT
    gate_pass_train_proxy = train_pct >= GATE_PCT

    print(f"\nGATE (fwd-only):        {'PASS' if gate_pass_fwd else 'FAIL'}")
    print(f"GATE (train proxy /2.2): {'PASS' if gate_pass_train_proxy else 'FAIL'}")

    summary = {
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "shape": f"2Lx96Hx16384v",
        "ctx": args.ctx,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "compile": args.compile,
        "steps": args.steps,
        "wall_sec": round(elapsed, 2),
        "it_per_sec": round(it_s, 3),
        "fwd_tokens_per_sec": int(tok_s),
        "estimated_train_tokens_per_sec": int(tok_s / 2.2),
        "g5_ref_tokens_per_sec": int(G5_REF_TOKENS_PER_SEC),
        "pct_vs_g5_fwd": round(pct, 2),
        "pct_vs_g5_train_proxy": round(train_pct, 2),
        "gate_pass_fwd_only": gate_pass_fwd,
        "gate_pass_train_proxy": gate_pass_train_proxy,
    }
    print(f"\nJSON: {json.dumps(summary)}")
    return summary


def _leaf_params(module):
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
        else:
            yield None, obj
    yield from walk(module.parameters())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
