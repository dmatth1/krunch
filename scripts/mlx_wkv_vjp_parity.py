"""Check custom-VJP gradients match MLX auto-diff gradients of the
auto-diff WKV on small inputs.

Runs both implementations through `mx.value_and_grad` and compares
d_time_decay, d_time_first, dk, dv.

Thresholds: max abs diff 1e-4 on fp32 (accumulation noise expected).
Small shapes only (T up to 32) since the auto-diff reference gets
slow past that point.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx
import numpy as np

from mlx_wkv import wkv as wkv_custom

# Auto-diff reference: inline WKV forward sweep (no @mx.custom_function
# wrapper), so mx.value_and_grad uses MLX's automatic backward and we
# can compare to the kernel-computed gradients.
def wkv_autodiff(time_decay, time_first, k, v):
    w_exp = -mx.exp(time_decay)
    B, T, C = k.shape
    aa = mx.zeros((B, C), dtype=k.dtype)
    bb = mx.zeros((B, C), dtype=k.dtype)
    pp = mx.full((B, C), -1e30, dtype=k.dtype)
    outs = []
    for t in range(T):
        kk = k[:, t]; vv = v[:, t]
        ww = time_first + kk
        p = mx.maximum(pp, ww)
        e1 = mx.exp(pp - p); e2 = mx.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        outs.append(a / b)
        ww2 = pp + w_exp
        p2 = mx.maximum(ww2, kk)
        e1b = mx.exp(ww2 - p2); e2b = mx.exp(kk - p2)
        aa = e1b * aa + e2b * vv
        bb = e1b * bb + e2b
        pp = p2
    return mx.stack(outs, axis=1)


def l1_of_all(arrays):
    return sum(mx.sum(a) for a in arrays)


def main():
    B, T, C = 2, 16, 8
    mx.random.seed(1204)
    w = mx.random.normal((C,))
    u = mx.random.normal((C,))
    k = mx.random.normal((B, T, C))
    v = mx.random.normal((B, T, C))

    mx.eval(w, u, k, v)

    def loss_custom(w, u, k, v):
        y = wkv_custom(w, u, k, v)
        return mx.sum(y * mx.cos(y))  # non-trivial scalar loss

    def loss_auto(w, u, k, v):
        y = wkv_autodiff(w, u, k, v)
        return mx.sum(y * mx.cos(y))

    grad_custom = mx.value_and_grad(loss_custom, argnums=(0, 1, 2, 3))
    grad_auto = mx.value_and_grad(loss_auto, argnums=(0, 1, 2, 3))

    print("=== MLX custom-VJP vs auto-diff parity ===")
    print(f"shape: B={B} T={T} C={C}\n")

    (lc, (dw_c, du_c, dk_c, dv_c)) = grad_custom(w, u, k, v)
    (la, (dw_a, du_a, dk_a, dv_a)) = grad_auto(w, u, k, v)
    mx.eval(lc, la, dw_c, du_c, dk_c, dv_c, dw_a, du_a, dk_a, dv_a)

    print(f"loss custom={float(lc):.6f}  auto={float(la):.6f}  "
          f"diff={abs(float(lc) - float(la)):.3e}\n")

    def fmt(name, a, b):
        d = mx.abs(a - b)
        max_abs = float(mx.max(d))
        mean_abs = float(mx.mean(d))
        print(f"  {name:18s}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
              f"shape={tuple(a.shape)}")
        return max_abs

    print("gradient comparison (custom vs auto):")
    mx_dw = fmt("d_time_decay", dw_c, dw_a)
    mx_du = fmt("d_time_first", du_c, du_a)
    mx_dk = fmt("dk",           dk_c, dk_a)
    mx_dv = fmt("dv",           dv_c, dv_a)

    tol = 1e-4
    ok = all(m < tol for m in (mx_dw, mx_du, mx_dk, mx_dv))
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'}  (max abs < {tol})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
