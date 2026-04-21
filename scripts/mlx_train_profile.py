"""Profile which parts of the MLX training step dominate wall time.

Breaks down: forward-only, forward+loss, forward+loss+backward, full step.
Lets us see where the remaining gap to the gate comes from.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from mlx_rwkv import RwkvTcHira

VOCAB = 16384
HIDDEN = 96
LAYERS = 2
CTX = 512
BATCH = 32
STEPS = 30


def main():
    mx.random.seed(1204)
    model = RwkvTcHira(VOCAB, HIDDEN, LAYERS, HIDDEN, 4, CTX)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    inputs = mx.random.randint(3, VOCAB, shape=(BATCH, CTX))
    targets = mx.random.randint(3, VOCAB, shape=(BATCH, CTX))
    mx.eval(inputs, targets)

    # ---- forward only ----
    for _ in range(3):
        mx.eval(model(inputs))
    t0 = time.time()
    for _ in range(STEPS):
        mx.eval(model(inputs))
    dt = time.time() - t0
    print(f"forward-only:        {dt / STEPS * 1000:.1f} ms/step  "
          f"({int(STEPS * BATCH * CTX / dt):,} tok/s)")

    # ---- forward + CE loss ----
    def loss_fn(model, x, t):
        logits = model(x)
        B, T, V = logits.shape
        flat = logits.reshape(-1, V).astype(mx.float32)
        tgt = t.reshape(-1).astype(mx.int32)
        lp = flat - mx.logsumexp(flat, axis=-1, keepdims=True)
        return -lp[mx.arange(flat.shape[0]), tgt].mean()

    for _ in range(3):
        mx.eval(loss_fn(model, inputs, targets))
    t0 = time.time()
    for _ in range(STEPS):
        mx.eval(loss_fn(model, inputs, targets))
    dt = time.time() - t0
    print(f"fwd + CE loss:       {dt / STEPS * 1000:.1f} ms/step  "
          f"({int(STEPS * BATCH * CTX / dt):,} tok/s)")

    # ---- forward + loss + backward ----
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(3):
        l, g = loss_and_grad(model, inputs, targets)
        mx.eval(l, g)
    t0 = time.time()
    for _ in range(STEPS):
        l, g = loss_and_grad(model, inputs, targets)
        mx.eval(l, g)
    dt = time.time() - t0
    print(f"fwd + loss + bwd:    {dt / STEPS * 1000:.1f} ms/step  "
          f"({int(STEPS * BATCH * CTX / dt):,} tok/s)")

    # ---- full step with AdamW ----
    optimizer = opt.AdamW(learning_rate=1e-4)

    def step(x, t):
        l, g = loss_and_grad(model, x, t)
        optimizer.update(model, g)
        return l

    for _ in range(3):
        mx.eval(step(inputs, targets), model.state, optimizer.state)
    t0 = time.time()
    for _ in range(STEPS):
        l = step(inputs, targets)
        mx.eval(l, model.state, optimizer.state)
    dt = time.time() - t0
    print(f"fwd + bwd + AdamW:   {dt / STEPS * 1000:.1f} ms/step  "
          f"({int(STEPS * BATCH * CTX / dt):,} tok/s)")

    # g5 reference
    g5 = 12.44 * 32 * 2048
    print(f"\nvs g5 baseline ({int(g5):,} tok/s): "
          f"full-step at {100.0 * int(STEPS * BATCH * CTX / dt) / g5:.1f}% (gate 15%)")


if __name__ == "__main__":
    main()
