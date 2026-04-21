"""Numerical parity check: MLX RwkvTcHira vs PyTorch RWKV_TC_HIRA.

Builds both models, copies PyTorch weights into the MLX model, then
forwards the same input through both and reports max-abs-diff on logits.
Task-25 step 1 gate: parity must be close (< ~1e-3 abs diff on logits)
before we invest in the training loop port.

Usage:
    vendor/L3TC/.venv/bin/python scripts/mlx_rwkv_parity.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor" / "L3TC"))
sys.path.insert(0, str(HERE))

# Force MPS path in the Phase 11 trainer (we're comparing CPU/MPS paths,
# not CUDA), matching the monkey-patch in train_l3tc_phase11.py.
os.environ.setdefault("RWKV_FLOAT_MODE", "fp32")

import numpy as np
import torch
from models.RWKV_V4 import rwkv_tc_hira_train as _rwkv_mod


def _wkv_cpu_forward(B, T, C, w, u, k, v):
    """Same pure-PyTorch WKV as train_l3tc_phase11.py. Needed so the
    PyTorch reference is invocable on CPU (no CUDA kernel available)."""
    w_neg = -torch.exp(w.contiguous())
    device = k.device
    dtype = k.dtype
    aa = torch.zeros(B, C, device=device, dtype=dtype)
    bb = torch.zeros(B, C, device=device, dtype=dtype)
    pp = torch.full((B, C), -1e38, device=device, dtype=dtype)
    outs = []
    for t in range(T):
        kk = k[:, t]; vv = v[:, t]
        ww = u + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p); e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        outs.append(a / b)
        ww = pp + w_neg
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p); e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p
    return torch.stack(outs, dim=1)


_rwkv_mod.RUN_CUDA = lambda B, T, C, w, u, k, v: _wkv_cpu_forward(B, T, C, w, u, k, v)

from models.RWKV_V4.rwkv_tc_hira_train import RWKV_TC_HIRA

import mlx.core as mx
from mlx_rwkv import RwkvTcHira


def port_weights(pt_model: RWKV_TC_HIRA, mlx_model: RwkvTcHira):
    """Copy weights from the PyTorch model into the MLX model by name.

    PyTorch Linear.weight: (out_features, in_features). MLX nn.Linear
    also stores (out_features, in_features). So Linear weights copy
    1:1 without transpose.

    The tricky bits:
    - time_mix_* params are (1, 1, C) in PyTorch. MLX mirrors that.
    - time_decay, time_first are (C,) in PyTorch. Same in MLX.
    - embedding.weight: both (V, H).
    - LayerNorm weight/bias: both (C,).
    - time_shift is a module in PyTorch (no params); no-op here.

    We flatten both state_dicts to a dict of {name: tensor} then
    rename where the module-tree paths diverge (mostly: `blocks.N.` is
    the same on both sides since we mirror the structure).
    """
    sd = pt_model.state_dict()

    # Build a dict of MLX parameters keyed by their dotted name.
    # mlx.nn.Module.parameters() returns nested dict; we flatten it.
    flat_mlx = dict(_flatten_mlx_params(mlx_model))

    missing = []
    mismatched = []
    for k, v in sd.items():
        if k not in flat_mlx:
            missing.append(k)
            continue
        mx_arr = _pt_to_mx(v)
        # Shape check
        target_shape = tuple(flat_mlx[k].shape)
        if tuple(mx_arr.shape) != target_shape:
            mismatched.append((k, tuple(mx_arr.shape), target_shape))
            continue
        flat_mlx[k] = mx_arr

    if missing:
        print(f"WARN: {len(missing)} PyTorch params not in MLX model:")
        for k in missing[:10]:
            print(f"  {k}  shape={tuple(sd[k].shape)}")
    if mismatched:
        print(f"WARN: {len(mismatched)} shape mismatches:")
        for k, a, b in mismatched[:10]:
            print(f"  {k}  pt={a}  mx={b}")

    # Apply. MLX requires the nested-dict form.
    nested = _unflatten(flat_mlx)
    mlx_model.update(nested)
    mx.eval(mlx_model.parameters())
    return mlx_model


def _pt_to_mx(t: torch.Tensor) -> mx.array:
    """Torch tensor -> MLX array via numpy bridge."""
    return mx.array(t.detach().cpu().to(torch.float32).numpy())


def _flatten_mlx_params(module):
    """Yield (dotted_name, param) for every leaf param in an MLX Module."""
    params = module.parameters()

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}.{k}" if prefix else k
                yield from walk(v, name)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                name = f"{prefix}.{i}" if prefix else str(i)
                yield from walk(v, name)
        else:
            yield prefix, obj

    yield from walk(params)


def _unflatten(flat: dict) -> dict:
    """Flat {'a.b.0.c': arr} -> nested {'a': {'b': [{'c': arr}]}}."""
    root: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        cur = root
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            next_p = parts[i + 1] if not last else None
            if last:
                if p.isdigit():
                    idx = int(p)
                    # cur should be a list
                    while len(cur) <= idx:
                        cur.append(None)
                    cur[idx] = v
                else:
                    cur[p] = v
            else:
                # need to create container — list if next is numeric, dict otherwise
                if p.isdigit():
                    idx = int(p)
                    while len(cur) <= idx:
                        cur.append(None)
                    if cur[idx] is None:
                        cur[idx] = [] if next_p and next_p.isdigit() else {}
                    cur = cur[idx]
                else:
                    if p not in cur:
                        cur[p] = [] if next_p and next_p.isdigit() else {}
                    cur = cur[p]
    return root


def main():
    vocab_size = 256
    hidden = 96
    n_layer = 2
    inter = 96
    rank = 4
    ctx = 64  # small T for parity — throughput bench is a separate script
    B = 2

    print("=== MLX vs PyTorch parity ===")
    print(f"config: vocab={vocab_size} hidden={hidden} layers={n_layer} ctx={ctx} batch={B}")

    torch.manual_seed(1204)
    mx.random.seed(1204)

    pt = RWKV_TC_HIRA(
        vocab_size=vocab_size, hidden_size=hidden,
        num_hidden_layers=n_layer, intermediate_size=inter,
        rwkv_rank=rank, ctx_len=ctx, dropout_prob=0.0,
    ).to("cpu").eval()

    mlx_model = RwkvTcHira(
        vocab_size=vocab_size, hidden_size=hidden,
        num_hidden_layers=n_layer, intermediate_size=inter,
        rwkv_rank=rank, ctx_len=ctx, dropout_prob=0.0,
    )
    mx.eval(mlx_model.parameters())

    n_pt = sum(p.numel() for p in pt.parameters())
    n_mx = sum(int(np.prod(a.shape)) for _, a in _flatten_mlx_params(mlx_model))
    print(f"params: pt={n_pt:,}  mx={n_mx:,}")

    print("\nPorting PyTorch weights -> MLX...")
    port_weights(pt, mlx_model)

    # Sanity: re-count
    n_mx2 = sum(int(np.prod(a.shape)) for _, a in _flatten_mlx_params(mlx_model))
    print(f"params after port: mx={n_mx2:,}")
    assert n_mx == n_mx2, "param count drifted after port"

    # Run both forward
    tok_np = np.random.randint(0, vocab_size, size=(B, ctx), dtype=np.int64)
    tok_pt = torch.from_numpy(tok_np).unsqueeze(1)  # (B, 1, T) as Phase 11 expects
    types_pt = torch.ones_like(tok_pt)

    with torch.no_grad():
        logits_pt = pt(tok_pt, types_pt, train=True)
    logits_pt_np = logits_pt.detach().cpu().numpy()

    tok_mx = mx.array(tok_np)
    logits_mx = mlx_model(tok_mx)
    mx.eval(logits_mx)
    logits_mx_np = np.asarray(logits_mx)

    print(f"\npt  logits shape: {logits_pt_np.shape}")
    print(f"mx  logits shape: {logits_mx_np.shape}")

    if logits_pt_np.shape != logits_mx_np.shape:
        print("SHAPE MISMATCH — parity cannot proceed")
        return 1

    abs_diff = np.abs(logits_pt_np - logits_mx_np)
    rel_diff = abs_diff / (np.abs(logits_pt_np) + 1e-6)
    print(f"max abs diff: {abs_diff.max():.6e}")
    print(f"mean abs diff: {abs_diff.mean():.6e}")
    print(f"max rel diff: {rel_diff.max():.4f}")
    print(f"mean rel diff: {rel_diff.mean():.4f}")

    # Thresholds: we're comparing fp32 PyTorch CPU vs fp32 MLX GPU. Differences
    # come from accumulation order in matmuls + slightly different exp/log.
    # 1e-3 on logits is a comfortable parity bar.
    abs_ok = abs_diff.max() < 1e-2
    mean_ok = abs_diff.mean() < 1e-3
    gate = abs_ok and mean_ok

    print(f"\nPARITY: {'PASS' if gate else 'FAIL'} "
          f"(max_abs < 1e-2 AND mean_abs < 1e-3)")
    return 0 if gate else 2


if __name__ == "__main__":
    sys.exit(main())
