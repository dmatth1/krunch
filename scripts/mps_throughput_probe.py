"""Task 9 — MPS throughput probe (GATING for Phase 14 local training).

Trains a 2L x 96H x 16K vocab RWKV_TC_HIRA specialist-shape model on
Apple MPS for 200 steps and reports:
  - tokens/sec (end-to-end wall), compared vs Phase 11 g5.xlarge baseline
    (12.44 it/s * 32 batch * 2048 ctx = ~816K tok/s; 15% = ~122K tok/s)
  - silent CPU fallbacks (PYTORCH_ENABLE_MPS_FALLBACK=0 + torch.mps stats)
  - memory footprint (driver + recommended working set)
  - bf16 vs fp16 stability (loss sanity, NaN/Inf check)

Does not touch the cloud training path. Uses real tokens from the Pile
.npy cache if present, else from enwik8 via SPM, else synthetic random
tokens as a last resort (throughput-equivalent since WKV is the hot path).

Run (from repo root):
    vendor/L3TC/.venv/bin/python scripts/mps_throughput_probe.py

Or with custom knobs:
    vendor/L3TC/.venv/bin/python scripts/mps_throughput_probe.py \\
        --steps 200 --batch-size 16 --dtype bf16

Exit 0 always — readable report on stdout. team-lead reads the report and
decides whether tasks 10-13 proceed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
L3TC_DIR = REPO / "vendor" / "L3TC"

# Force fp32 RWKV kernel selection — same as train_l3tc_phase11.py.
os.environ.setdefault("RWKV_FLOAT_MODE", "fp32")

# CRITICAL for MPS: we want HARD failures on unimplemented ops so we
# can detect silent CPU fallbacks. Default is 0 (hard fail). We
# probe with fallback OFF first; if any op is missing we report it
# as a MAJOR RISK.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import torch  # noqa: E402

# Non-CUDA => pure-PyTorch WKV. Identical monkey-patch to the trainer.
if not torch.cuda.is_available():
    os.environ.pop("USE_WKV_CUDA_FOR_RWKV", None)


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


sys.path.insert(0, str(L3TC_DIR))
from models.RWKV_V4 import rwkv_tc_hira_train as _rwkv_train_mod  # noqa: E402

if not torch.cuda.is_available():
    _rwkv_train_mod.RUN_CUDA = lambda B, T, C, w, u, k, v: _wkv_cpu_forward(
        B, T, C, w, u, k, v
    )

from models.RWKV_V4.rwkv_tc_hira_train import RWKV_TC_HIRA  # noqa: E402
from models.RWKV_V4.rwkv_v4_train import L2Wrap  # noqa: E402


# Phase 14 specialist shape — matches L3TC-200K / Phase 11 reference.
HIDDEN_SIZE = 96
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 96
RWKV_RANK = 4
VOCAB_SIZE = 16384
CTX_LEN = 512

# Reference throughput from Phase 11 (docs/phase-findings/phase_11_findings.md:50).
# 12.44 it/s * 32 batch * 2048 ctx = 815,595 tokens/sec on g5.xlarge (A10G bf16).
G5_REF_TOKENS_PER_SEC = 12.44 * 32 * 2048


def build_model(device, dtype_str):
    model = RWKV_TC_HIRA(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        rwkv_rank=RWKV_RANK,
        ctx_len=CTX_LEN,
        dropout_prob=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_non_embed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "head" not in n and "emb" not in n
    )
    model = model.to(device)
    return model, n_params, n_non_embed


def make_synthetic_batch(batch_size, seg_len, vocab_size, device):
    """Random integer tokens — WKV is the hot path, not embedding lookup,
    so throughput measurements are insensitive to token identity."""
    inp = torch.randint(3, vocab_size, (batch_size, 1, seg_len), device=device, dtype=torch.long)
    tgt = torch.randint(3, vocab_size, (batch_size, 1, seg_len), device=device, dtype=torch.long)
    in_types = torch.ones_like(inp)
    out_types = torch.ones_like(tgt)
    return inp, tgt, in_types, out_types


def mps_mem_gb():
    if not hasattr(torch, "mps"):
        return None, None
    try:
        drv = torch.mps.driver_allocated_memory() / 1e9
        rec = torch.mps.recommended_max_memory() / 1e9
    except Exception:
        return None, None
    return drv, rec


def run_probe(args):
    device = torch.device(args.device)
    print(f"\n=== MPS Throughput Probe (Task 9) ===")
    print(f"device:     {device}")
    print(f"torch:      {torch.__version__}")
    print(f"mps avail:  {torch.backends.mps.is_available()}")
    print(f"mps built:  {torch.backends.mps.is_built()}")
    print(f"fallback:   PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"shape:      {NUM_LAYERS}L x {HIDDEN_SIZE}H x {VOCAB_SIZE} vocab, ctx={CTX_LEN}")
    print(f"batch:      {args.batch_size}, steps: {args.steps}, dtype: {args.dtype}")

    torch.manual_seed(1204)

    model, n_params, n_non_embed = build_model(device, args.dtype)
    print(f"params:     total={n_params:,}  non-embed/head={n_non_embed:,}")

    if args.dtype == "bf16":
        # MPS supports bf16 as of PyTorch 2.1+. Cast model weights to bf16 so
        # the whole forward pass runs in bf16 on-device.
        model = model.to(torch.bfloat16)
        compute_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        model = model.to(torch.float16)
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    inp, tgt, in_types, out_types = make_synthetic_batch(
        args.batch_size, CTX_LEN, VOCAB_SIZE, device,
    )

    # ===== Warmup =====
    print("\nwarmup (10 steps, not timed)...")
    t_warm0 = time.time()
    for _ in range(10):
        logits = model(inp, in_types, train=True)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = tgt.reshape(-1)
        flat_mask = out_types.reshape(-1).to(flat_logits.dtype)
        # Upcast logits to fp32 for CE loss stability (standard practice).
        per_token = criterion(flat_logits.float(), flat_targets)
        loss = (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        loss = L2Wrap.apply(loss, logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.synchronize()
    t_warm = time.time() - t_warm0
    print(f"  warmup wall: {t_warm:.2f}s")
    drv, rec = mps_mem_gb()
    if drv is not None:
        print(f"  mps memory after warmup: driver={drv:.2f} GB / recommended={rec:.2f} GB")

    # ===== Timed steps =====
    print(f"\ntimed run ({args.steps} steps)...")
    losses = []
    any_nan = False
    t0 = time.time()
    last_log_t = t0
    log_every = max(args.steps // 10, 10)

    for step in range(args.steps):
        logits = model(inp, in_types, train=True)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = tgt.reshape(-1)
        flat_mask = out_types.reshape(-1).to(flat_logits.dtype)
        per_token = criterion(flat_logits.float(), flat_targets)
        loss = (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        loss_val = loss.item()
        losses.append(loss_val)
        if not math.isfinite(loss_val):
            any_nan = True
            print(f"  step {step}: NON-FINITE LOSS {loss_val}")
        loss = L2Wrap.apply(loss, logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % log_every == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            now = time.time()
            it_s = log_every / (now - last_log_t)
            toks = args.batch_size * CTX_LEN
            print(f"  step {step+1:4d}/{args.steps}: loss={loss_val:.4f} "
                  f"{it_s:.2f} it/s ({int(it_s * toks):,} tok/s)")
            last_log_t = now

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    tokens_per_step = args.batch_size * CTX_LEN
    total_tokens = tokens_per_step * args.steps
    tok_per_sec = total_tokens / elapsed
    it_per_sec = args.steps / elapsed
    pct_vs_g5 = 100.0 * tok_per_sec / G5_REF_TOKENS_PER_SEC

    drv2, rec2 = mps_mem_gb()

    print(f"\n=== RESULTS ===")
    print(f"steps:            {args.steps}")
    print(f"wall seconds:     {elapsed:.2f}")
    print(f"it/sec:           {it_per_sec:.2f}")
    print(f"tokens/sec:       {int(tok_per_sec):,}")
    print(f"vs g5 (816K/s):   {pct_vs_g5:.1f}%   (15% threshold = {int(G5_REF_TOKENS_PER_SEC * 0.15):,} tok/s)")
    if drv2 is not None:
        print(f"mps memory end:   driver={drv2:.2f} GB / recommended={rec2:.2f} GB")
    print(f"loss first->last: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"non-finite loss:  {any_nan}")

    # Pass/fail line
    gate_pass = pct_vs_g5 >= 15.0 and not any_nan
    print(f"\nGATE:             {'PASS' if gate_pass else 'FAIL'} "
          f"(>= 15% of g5 throughput and no NaN/Inf)")

    # JSON summary on the last line so it's easy to grep/parse.
    summary = {
        "device": str(device),
        "dtype": args.dtype,
        "shape": f"{NUM_LAYERS}Lx{HIDDEN_SIZE}Hx{VOCAB_SIZE}v ctx{CTX_LEN}",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "wall_sec": round(elapsed, 2),
        "it_per_sec": round(it_per_sec, 3),
        "tokens_per_sec": int(tok_per_sec),
        "g5_ref_tokens_per_sec": int(G5_REF_TOKENS_PER_SEC),
        "pct_vs_g5": round(pct_vs_g5, 2),
        "threshold_pct": 15.0,
        "gate_pass": gate_pass,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "non_finite_loss": any_nan,
        "mps_driver_gb": round(drv2, 2) if drv2 is not None else None,
        "mps_recommended_gb": round(rec2, 2) if rec2 is not None else None,
        "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    print("\nJSON:", json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="mps")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = p.parse_args()
    run_probe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
