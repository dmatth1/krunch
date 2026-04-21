"""Phase 14 task 8: per-domain parallel re-tokenization to streaming .npy.

For each of the 7 specialist domains:
  1. Enumerate input text files under data/specialists/{domain}/.
  2. Tokenize with the domain-specific 16K SPM at
     tokenizers/specialists/{domain}/spm.model (trained by task 11).
  3. Write tokens to data/specialists/{domain}/tokens.npy as a
     pre-allocated memmap, streaming in chunks.
  4. Record the final length in a sidecar `tokens.npy.len`.

Phase 11 lessons applied:
  - Pre-allocated memmap streaming (not accumulated chunks). Phase 11
    built 55 GB cache in 26 min with this pattern.
  - Per-worker byte position tracked locally (never f.tell() in hot
    loop). Phase 11 observed 3.7× speedup from this.
  - Chunk size chosen so encode_as_ids() isn't called per-line (that
    blew up at ~5 MB/s per worker). 64 KB text chunks hit >20 MB/s.

Usage:
    python scripts/phase14_retokenize_to_npy.py \\
        --root . --workers 10 --domains prose,code,structured
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

CHUNK_BYTES = 64 * 1024  # per-worker text buffer size


def _estimate_tokens(input_size_bytes: int) -> int:
    """Upper bound on token count for a given text size. SPM unigram at
    16K vocab is typically 0.25-0.35 tokens/byte on prose, higher on
    short tokens (syslog) or structured data. Bound at 0.6 tokens/byte
    so we never under-estimate — retokenize adds a BOS per ~64 KB chunk
    which pushes this higher than pure subword density."""
    return int(input_size_bytes * 0.6) + 2_000_000


def _list_domain_inputs(domain_dir: Path) -> list[Path]:
    """All .txt files under domain/, excluding the tokens artifacts."""
    files = []
    for p in sorted(domain_dir.iterdir()):
        if p.is_symlink():
            # fallback symlink points at tokenized Phase 11 corpus (IDs);
            # that's not text — handled separately in main().
            continue
        if p.is_file() and p.suffix == ".txt":
            files.append(p)
    return files


def _tokenize_worker(args):
    """Tokenize [start, end) bytes of a source file, return token array."""
    src_path, spm_path, start, end, worker_id = args
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    tokens: list[int] = []
    bytes_read = start
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start)
        if start > 0:
            skip = f.readline()
            bytes_read += len(skip.encode("utf-8"))
        buf: list[str] = []
        buf_bytes = 0
        for line in f:
            bytes_read += len(line.encode("utf-8"))
            buf.append(line)
            buf_bytes += len(line)
            if buf_bytes >= CHUNK_BYTES:
                text = "".join(buf)
                ids = sp.encode_as_ids(text)
                tokens.extend(ids)
                tokens.append(sp.eos_id() if sp.eos_id() > 0 else 2)
                buf, buf_bytes = [], 0
            if bytes_read >= end:
                break
        if buf:
            text = "".join(buf)
            ids = sp.encode_as_ids(text)
            tokens.extend(ids)
            tokens.append(sp.eos_id() if sp.eos_id() > 0 else 2)
    return worker_id, np.asarray(tokens, dtype=np.int32)


def tokenize_domain(domain: str, domain_dir: Path, spm_path: Path,
                    workers: int, dtype) -> tuple[int, float]:
    """Tokenize all files in a domain, streaming to a .npy memmap.

    Uses `np.lib.format.open_memmap` so the output has a real .npy header
    (required by `mlx_train_specialist.py`, which loads via the same API).
    Pre-allocates with an over-estimate, then rewrites the header with the
    exact length at the end. This is the Phase 11 streaming pattern.
    """
    inputs = _list_domain_inputs(domain_dir)
    if not inputs:
        print(f"  {domain}: no input files")
        return 0, 0.0
    total_bytes = sum(p.stat().st_size for p in inputs)
    upper = _estimate_tokens(total_bytes)
    out_path = domain_dir / "tokens.npy"
    tmp_path = domain_dir / "tokens.npy.tmp"
    # Pre-allocate as a real .npy file. open_memmap writes the header
    # up front so the file is a valid .npy from the start.
    print(f"  {domain}: preallocating {upper:,} tokens "
          f"({upper * np.dtype(dtype).itemsize / 1e9:.2f} GB .npy)")
    mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype,
                                   shape=(upper,))
    offset = 0
    t0 = time.time()

    for src in inputs:
        size = src.stat().st_size
        chunk = max(1, size // workers)
        args_list = []
        for i in range(workers):
            s = i * chunk
            e = size if i == workers - 1 else (i + 1) * chunk
            args_list.append((str(src), str(spm_path), s, e, i))
        with mp.Pool(workers) as pool:
            for worker_id, arr in pool.imap_unordered(_tokenize_worker, args_list):
                if arr.size == 0:
                    continue
                end = offset + arr.size
                if end > upper:
                    # Grow: flush + resize .npy file and reopen at new shape.
                    new_upper = int(end * 1.3)
                    print(f"    {domain}: growing .npy {upper:,} -> {new_upper:,}")
                    mm.flush()
                    del mm
                    # Rewrite header for new shape by reallocating the file
                    # via a fresh open_memmap, copying old data in. Too
                    # expensive to copy; simpler: make the initial upper
                    # estimate safer next time. For now raise to prompt
                    # a bigger pre-allocation.
                    raise RuntimeError(
                        f"{domain} under-estimated tokens "
                        f"({upper} < {end}); rerun with a higher "
                        f"_estimate_tokens multiplier")
                mm[offset:end] = arr
                offset = end

    mm.flush()
    del mm

    # Rewrite .npy header with exact length. open_memmap will produce a
    # new header describing shape (offset,) when we reopen w+ with the
    # exact size — but that would wipe data. Instead, use numpy's helper
    # to write a fresh header in-place by re-writing the file with the
    # truncated array. For large arrays this is cheap because we only
    # need to rewrite the ~128-byte header, not the data; we do this by
    # loading the existing data as memmap, saving header + data via
    # np.save.
    # Simpler: mem-copy the first `offset` tokens to out_path via np.save.
    src_mm = np.lib.format.open_memmap(tmp_path, mode="r", dtype=dtype,
                                       shape=(upper,))
    np.save(out_path, src_mm[:offset])
    del src_mm
    tmp_path.unlink()
    (domain_dir / "tokens.npy.len").write_text(f"{offset}\n")
    elapsed = time.time() - t0
    print(f"  {domain}: {offset:,} tokens in {elapsed:.0f}s "
          f"({offset / max(total_bytes, 1):.3f} tokens/byte, "
          f"{total_bytes / 1e6 / elapsed:.1f} MB/s input)")
    return offset, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/Users/dmatt/Claude Projects/l3tc-prod"))
    p.add_argument("--specialists-dir", type=Path,
                   default=Path("data/specialists"))
    p.add_argument("--tokenizers-dir", type=Path,
                   default=Path("tokenizers/specialists"),
                   help="Where training-engineer puts the 7 domain SPMs "
                        "(task 11). Each should be named {domain}/spm.model.")
    p.add_argument("--domains", type=str,
                   default="prose,code,structured,logs,tabular,markup,fallback")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--dtype", choices=("int32", "uint16"), default="int32",
                   help="int32 is safe for 16K vocab; uint16 is half-size.")
    args = p.parse_args()

    root = args.root
    spec_root = args.specialists_dir if args.specialists_dir.is_absolute() else root / args.specialists_dir
    tok_root = args.tokenizers_dir if args.tokenizers_dir.is_absolute() else root / args.tokenizers_dir
    dtype = np.int32 if args.dtype == "int32" else np.uint16

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    print(f"=== Phase 14 re-tokenize → .npy ===")
    print(f"specialists: {spec_root}")
    print(f"tokenizers:  {tok_root}")
    print(f"workers:     {args.workers}")
    print(f"domains:     {domains}")

    summary: dict = {}
    t_start = time.time()
    for d in domains:
        domain_dir = spec_root / d
        # Check both layouts: tokenizers/specialists/{d}/spm.model and
        # data/specialists/{d}/spm.model (training-engineer dropped the
        # latter in task 11).
        candidates = [
            tok_root / d / "spm.model",
            domain_dir / "spm.model",
        ]
        spm_path = next((c for c in candidates if c.exists()), None)
        if spm_path is None:
            print(f"  {d}: SKIP (SPM missing; looked at "
                  f"{[str(c) for c in candidates]})")
            continue
        if not domain_dir.exists():
            print(f"  {d}: SKIP (no domain dir at {domain_dir})")
            continue
        n, elapsed = tokenize_domain(d, domain_dir, spm_path,
                                     args.workers, dtype)
        summary[d] = {"tokens": n, "elapsed_s": elapsed}

    print(f"\n=== Summary (wall {time.time()-t_start:.0f}s) ===")
    for d, s in summary.items():
        print(f"  {d}: {s['tokens']:,} tokens in {s['elapsed_s']:.0f}s")

    # Write manifest alongside
    (spec_root / "_retokenize_manifest.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
