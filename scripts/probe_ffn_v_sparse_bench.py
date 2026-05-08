"""
A1.2 architecture-selection microbench: dense vs torch.sparse.mm vs
indices-based gather, at production FFN-V shapes with the same ~90%
sparsity pattern as real relu² output.

Three candidates:

  (A) Dense baseline — torch matmul fp16. The reference.
  (B) torch.sparse.mm via CSR. Cheap to prototype; cuSPARSE-backed.
  (C) torch.gather + torch.matmul on the per-row dense submatrix
      (only meaningful at very small M — at B=161 the union of
      non-zero columns across rows is ~100% so this collapses to
      dense). Run at M=1, M=8, M=32 to measure where it stops
      being viable.

Output drives the A1 implementation choice: if (B) clearly beats
(A) at our shapes, ship via torch.sparse.mm; if (B) doesn't beat
(A), commit to a custom CUDA kernel.

Per docs/TIER_3_OPTIMIZATION.md A1.2.
"""

import time

import torch


def _bench(fn, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iters


def _make_sparse_input(M: int, K: int, density: float, seed: int) -> torch.Tensor:
    """Build [M, K] fp16 with `density` fraction of non-zero entries
    randomly distributed per-row. Mirrors the relu² output shape."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mask = (torch.rand(M, K, generator=g, device="cuda") < density).to(torch.float16)
    vals = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float16) * 0.1
    return mask * vals


def _csr_mm(x_fp16, W_fp16):
    """SparseCSR ⊗ dense fp16 matmul via cuSPARSE."""
    csr = x_fp16.to_sparse_csr()
    return torch.sparse.mm(csr, W_fp16)


def _gather_dense(x_fp16, W_fp16):
    """Per-row indices→dense gather, then dense matmul of the
    [M, max_nnz] sub-input × W. Only beats dense when nnz << K
    AND row indices unify well across M."""
    nnz_per_row = (x_fp16 != 0).sum(dim=1)
    max_nnz = int(nnz_per_row.max().item())
    if max_nnz == 0:
        return torch.zeros(x_fp16.shape[0], W_fp16.shape[1],
                            dtype=torch.float16, device="cuda")
    # Per-row top-max_nnz nonzeros; pad with index 0 for short rows
    # (multiplied by 0 since x_padded[m, j] = 0 for j ≥ nnz[m]).
    M, K = x_fp16.shape
    sorted_vals, sorted_idx = torch.sort(x_fp16.abs(), dim=1, descending=True)
    idx_top = sorted_idx[:, :max_nnz]      # [M, max_nnz]
    vals_top = torch.gather(x_fp16, 1, idx_top)  # [M, max_nnz]
    W_gathered = W_fp16[idx_top]            # [M, max_nnz, N]
    # Per-row matmul: out[m] = vals_top[m] @ W_gathered[m]
    out = torch.einsum("mk,mkn->mn", vals_top.to(torch.float32),
                                       W_gathered.to(torch.float32))
    return out.to(torch.float16)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    p = torch.cuda.get_device_properties(0)
    print(f"[GPU] {p.name} sm_{p.major}{p.minor}, "
          f"{p.multi_processor_count} SMs")

    # FFN-V shape: M = encoder packed (1024) or decoder batched
    # (B=161 in production, also test small B). K = n_ffn = 3072.
    # N = n_embd = 768.
    K, N = 3072, 768
    density_zeros = 0.10  # 10% non-zero, matches measured 89.64% zeros
    Ms = [1, 8, 32, 161, 1024]

    # Build dense W once.
    W = (torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.02)
    print(f"\n[bench] FFN-V at K={K}, N={N}, ~{(1-density_zeros)*100:.0f}% sparse")
    print(f"{'M':>5} {'(A) dense ms':>14} {'(B) csr mm ms':>15} "
          f"{'(C) gather ms':>15} {'best vs A':>12}")
    for M in Ms:
        x = _make_sparse_input(M, K, density_zeros, seed=M)
        out_dense = torch.matmul(x.to(torch.float32), W.to(torch.float32)).to(torch.float16)

        # (A) dense fp16 matmul
        try:
            t_dense = _bench(lambda: torch.matmul(x, W))
        except Exception as e:
            t_dense = float("nan")

        # (B) csr mm
        try:
            out_csr = _csr_mm(x, W)
            err_csr = float((out_csr.float() - out_dense.float()).abs().max())
            t_csr = _bench(lambda: _csr_mm(x, W))
        except Exception as e:
            t_csr = float("nan")
            err_csr = float("nan")

        # (C) gather + dense matmul (only viable at small M)
        try:
            if M <= 32:
                out_gather = _gather_dense(x, W)
                err_g = float((out_gather.float() - out_dense.float()).abs().max())
                t_gather = _bench(lambda: _gather_dense(x, W))
            else:
                t_gather = float("nan")
                err_g = float("nan")
        except Exception as e:
            t_gather = float("nan")
            err_g = float("nan")

        speedups = []
        if t_csr == t_csr:
            speedups.append(("(B)", t_dense / t_csr))
        if t_gather == t_gather:
            speedups.append(("(C)", t_dense / t_gather))
        if speedups:
            best = max(speedups, key=lambda x: x[1])
            best_str = f"{best[0]} {best[1]:.2f}x"
        else:
            best_str = "—"

        print(f"{M:>5} {t_dense*1000:>14.3f} {t_csr*1000:>15.3f} "
              f"{t_gather*1000:>15.3f} {best_str:>12}")

    print("\n[bench] interpretation:")
    print("  - If (B) ≥1.5× at M≥161 → torch.sparse.mm is the implementation choice")
    print("  - If (B) ~ 1× at M≥161 → cuSPARSE generic doesn't win; need custom CUDA")
    print("  - (C) is only meaningful at small M; documents the per-row union")
    print("    approach's collapse as M grows")


if __name__ == "__main__":
    main()
