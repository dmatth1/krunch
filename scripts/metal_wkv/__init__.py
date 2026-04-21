"""Fused Metal WKV kernel for RWKV-v4 on torch MPS.

Task 25. Compiles `wkv.metal` + `wkv_mps.mm` into a torch cpp_extension
at import time. Exposes a `WkvMetal` autograd Function that exactly
matches the CUDA WKV semantics: `y = WkvMetal.apply(B, T, C, w, u, k, v)`.

Math equivalence to vendor/L3TC/models/RWKV_V4/cuda/wkv_cuda.cu is the
release bar — see scripts/metal_wkv/parity_check.py.

Pre-exponentiation note: the CUDA WKV wrapper does `w = -exp(w)` before
calling the kernel (see rwkv_v4_train.py:60-69). The Metal side matches
— we do the same transform in the Python wrapper so the kernel itself
receives the post-exp w. This is critical for the gw multiply at the
end of the backward kernel, which expects w to be the post-exp value.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_METAL_PATH = HERE / "wkv.metal"
_MM_PATH = HERE / "wkv_mps.mm"
_BUILD_DIR = HERE / ".torch_build"


def _load_ext():
    """Compile + load the ObjC++ Metal extension directly via clang++.

    We bypass torch.utils.cpp_extension.load() because the repo path
    contains a space ("Claude Projects"), which torch's ninja-backed
    build system fails to quote correctly — it fails at link time when
    searching `-L<path-with-space>`. clang++ invoked directly from
    Python properly receives the path as a single argv element.
    """
    if not _METAL_PATH.exists():
        raise FileNotFoundError(_METAL_PATH)
    if not _MM_PATH.exists():
        raise FileNotFoundError(_MM_PATH)

    os.environ["L3TC_WKV_METAL_PATH"] = str(_METAL_PATH)
    _BUILD_DIR.mkdir(exist_ok=True)

    # Use the Python version-unique filename so rebuilds don't clash
    # across venvs.
    so_path = _BUILD_DIR / f"l3tc_wkv_mps.{sys.implementation.cache_tag}.so"
    obj_path = _BUILD_DIR / "wkv_mps.o"
    verbose = bool(os.environ.get("L3TC_WKV_VERBOSE", ""))

    # Rebuild if source newer than .so (or if .so doesn't exist).
    needs_build = (
        not so_path.exists()
        or so_path.stat().st_mtime < _MM_PATH.stat().st_mtime
        or so_path.stat().st_mtime < _METAL_PATH.stat().st_mtime
    )
    if needs_build:
        _build_so(so_path, obj_path, verbose)

    # Load the built module.
    spec = importlib.util.spec_from_file_location("l3tc_wkv_mps", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_so(so_path: Path, obj_path: Path, verbose: bool):
    """Two-step: compile .mm -> .o, then link .o -> .so."""
    torch_inc = Path(torch.__file__).parent / "include"
    torch_api_inc = torch_inc / "torch" / "csrc" / "api" / "include"
    torch_lib = Path(torch.__file__).parent / "lib"
    # Use sysconfig directly — venv 'include' often points at a non-
    # existent path; the real Python headers live under the system
    # Python.framework. sysconfig.get_paths() resolves this correctly.
    import sysconfig
    py_inc = Path(sysconfig.get_paths()["include"])

    compile_cmd = [
        "clang++",
        "-c", str(_MM_PATH),
        "-o", str(obj_path),
        "-std=c++17",
        "-ObjC++",
        "-fno-objc-arc",
        "-fPIC",
        "-O2",
        f"-I{torch_inc}",
        f"-I{torch_api_inc}",
        f"-I{py_inc}",
        "-DTORCH_EXTENSION_NAME=l3tc_wkv_mps",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-D_GLIBCXX_USE_CXX11_ABI=0",  # PyTorch's default ABI flag on macOS
    ]
    link_cmd = [
        "clang++",
        "-shared",
        str(obj_path),
        "-o", str(so_path),
        f"-L{torch_lib}",
        "-lc10", "-ltorch_cpu", "-ltorch", "-ltorch_python",
        "-framework", "Foundation",
        "-framework", "Metal",
        "-undefined", "dynamic_lookup",
        f"-Wl,-rpath,{torch_lib}",
    ]

    for label, cmd in [("compile", compile_cmd), ("link", link_cmd)]:
        if verbose:
            print(f"[{label}] {' '.join(repr(a) if ' ' in a else a for a in cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"{label} failed (exit {res.returncode}):\n"
                f"stderr:\n{res.stderr}\n\nstdout:\n{res.stdout}"
            )
        elif verbose and res.stderr:
            print(f"[{label} stderr]\n{res.stderr}")


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_ext()
    return _ext


class WkvMetal(torch.autograd.Function):
    """Drop-in replacement for `WKV.apply` from rwkv_v4_train.py on MPS.

    Signature matches the CUDA version: `WkvMetal.apply(B, T, C, w, u, k, v)`.
    All of w, u, k, v must be on MPS and fp32 (the kernel casts/converts
    bf16/fp16 upstream; see scripts/train_l3tc_phase11_mps.py wrapper).

    Mirrors the pre-exponentiation contract from rwkv_v4_train.py: the
    raw trainable `time_decay` goes in, we exponentiate to `-exp(w)`,
    and the kernel + backward gw multiply see this post-exp value.
    """

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        assert w.device.type == "mps", "WkvMetal is MPS-only"
        ctx.B, ctx.T, ctx.C = int(B), int(T), int(C)

        # Pre-exp w exactly like rwkv_v4_train.py:60-69.
        w_exp = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w_exp, u, k, v)

        ext = _get_ext()
        y = ext.wkv_forward(w_exp, u, k, v)
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w_exp, u, k, v = ctx.saved_tensors
        ext = _get_ext()
        gw, gu, gk, gv = ext.wkv_backward(
            w_exp, u, k, v, gy.float().contiguous(),
        )
        # gw, gu are (B, C) in the kernel — reduce across batch to match
        # the shapes of w, u (C,). Matches CUDA wrapper behavior
        # (rwkv_v4_train.py:96-97).
        gw = gw.sum(dim=0)
        gu = gu.sum(dim=0)
        # (B, T, C) are passed through as-is. Non-tensor inputs get None.
        return (None, None, None, gw, gu, gk, gv)


def run_wkv_metal(B, T, C, w, u, k, v):
    """Convenience wrapper mirroring `RUN_CUDA` in rwkv_v4_train.py."""
    return WkvMetal.apply(B, T, C, w, u, k, v)
