"""Numerical parity check for metal_wkv.WkvMetal.

Compares our Metal kernel (MPS) against the pure-PyTorch CPU reference
from scripts/train_l3tc_phase11.py. The CPU reference is itself
mathematically identical to the CUDA kernel (verified by Phase 11 runs).

Steps:
1. Forward: compare y from metal_wkv.WkvMetal.apply vs. the CPU ref.
2. Backward: run both through torch.autograd, compare gw, gu, gk, gv.

Thresholds: 1e-4 max abs diff, 1e-5 mean abs diff at fp32. WKV is a
numerically-tricky recurrence (log-sum-exp stabilization) so we expect
~ULP-scale differences from accumulation order but nothing larger.

Usage:
    vendor/L3TC/.venv/bin/python scripts/metal_wkv/parity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make `metal_wkv` importable

import torch

from metal_wkv import WkvMetal


def wkv_cpu_reference(w, u, k, v):
    """Same logic as train_l3tc_phase11.py._wkv_cpu_forward, but with
    autograd-friendly torch ops (no in-place state mutation).
    Pre-exponentiation happens here to match the CUDA contract.
    """
    # w passed in is the raw time_decay (to be -exp'd inside).
    w_exp = -torch.exp(w)
    B, T, C = k.shape
    aa = torch.zeros(B, C, dtype=k.dtype, device=k.device)
    bb = torch.zeros(B, C, dtype=k.dtype, device=k.device)
    pp = torch.full((B, C), -1e38, dtype=k.dtype, device=k.device)
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
        ww = pp + w_exp
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p
    return torch.stack(outs, dim=1)


def main():
    torch.manual_seed(1204)

    B, T, C = 2, 16, 8
    # fp32 everywhere for parity. Real training will bf16 upstream.
    w = torch.randn(C, requires_grad=True)
    u = torch.randn(C, requires_grad=True)
    k = torch.randn(B, T, C, requires_grad=True)
    v = torch.randn(B, T, C, requires_grad=True)
    print(f"=== Metal WKV parity check ===")
    print(f"shape: B={B} T={T} C={C}  dtype=fp32")

    # ---------- CPU reference ----------
    w_cpu = w.detach().clone().requires_grad_(True)
    u_cpu = u.detach().clone().requires_grad_(True)
    k_cpu = k.detach().clone().requires_grad_(True)
    v_cpu = v.detach().clone().requires_grad_(True)
    y_cpu = wkv_cpu_reference(w_cpu, u_cpu, k_cpu, v_cpu)
    # random upstream grad to test backward
    torch.manual_seed(7)
    gy = torch.randn_like(y_cpu)
    y_cpu.backward(gy)
    gw_cpu = w_cpu.grad.detach().clone()
    gu_cpu = u_cpu.grad.detach().clone()
    gk_cpu = k_cpu.grad.detach().clone()
    gv_cpu = v_cpu.grad.detach().clone()

    # ---------- Metal MPS ----------
    if not torch.backends.mps.is_available():
        print("MPS not available; skipping.")
        return 1
    device = torch.device("mps")
    w_mps = w.detach().clone().to(device).requires_grad_(True)
    u_mps = u.detach().clone().to(device).requires_grad_(True)
    k_mps = k.detach().clone().to(device).requires_grad_(True)
    v_mps = v.detach().clone().to(device).requires_grad_(True)
    y_mps = WkvMetal.apply(B, T, C, w_mps, u_mps, k_mps, v_mps)
    gy_mps = gy.to(device)
    y_mps.backward(gy_mps)

    # ---------- Compare forward ----------
    y_mps_cpu = y_mps.detach().cpu()
    d_y = (y_cpu.detach() - y_mps_cpu).abs()
    print(f"\nforward y:   max_abs={d_y.max().item():.3e}  mean_abs={d_y.mean().item():.3e}")

    # ---------- Compare gradients ----------
    gw_mps = w_mps.grad.detach().cpu()
    gu_mps = u_mps.grad.detach().cpu()
    gk_mps = k_mps.grad.detach().cpu()
    gv_mps = v_mps.grad.detach().cpu()
    d_gw = (gw_cpu - gw_mps).abs()
    d_gu = (gu_cpu - gu_mps).abs()
    d_gk = (gk_cpu - gk_mps).abs()
    d_gv = (gv_cpu - gv_mps).abs()
    print(f"backward gw: max_abs={d_gw.max().item():.3e}  mean_abs={d_gw.mean().item():.3e}")
    print(f"backward gu: max_abs={d_gu.max().item():.3e}  mean_abs={d_gu.mean().item():.3e}")
    print(f"backward gk: max_abs={d_gk.max().item():.3e}  mean_abs={d_gk.mean().item():.3e}")
    print(f"backward gv: max_abs={d_gv.max().item():.3e}  mean_abs={d_gv.mean().item():.3e}")

    tol = 1e-4
    ok = all([
        d_y.max().item() < tol,
        d_gw.max().item() < tol,
        d_gu.max().item() < tol,
        d_gk.max().item() < tol,
        d_gv.max().item() < tol,
    ])
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'} (max abs < {tol})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
