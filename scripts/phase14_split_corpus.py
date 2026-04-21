"""Phase 14 task 6: split the Phase 11 51 GB corpus into 7 domain piles.

Strategy:
  - Port detect() from l3tc-rust/src/bin/l3tc/specialist.rs (reuses the
    module we built in phase14_audit_corpus.py).
  - For each raw-text source that went into the 51 GB balanced corpus,
    stream documents (chunked by blank-line or size cap), run detect()
    on the first 4 KB, append the full document to
    data/specialists/{domain}/from_{source}.txt.
  - Pile is tokenized on disk; we decode shards in parallel, reconstitute
    per-document text (BOS-separated token streams), detect, and write
    to the appropriate domain pile.
  - Fallback pile is a copy of the Phase 11 tokenized balanced corpus.

Per-source parallelism via multiprocessing (one worker per shard for
Pile; per-file workers for raw text).

Phase 11 lesson applied:
  - byte position tracked locally (never f.tell() in hot loop)
  - streaming write (no big in-memory accumulation)

Output layout:
    data/specialists/prose/from_*.txt
    data/specialists/code/from_*.txt
    data/specialists/structured/from_*.txt
    data/specialists/logs/from_*.txt
    data/specialists/tabular/from_*.txt
    data/specialists/markup/from_*.txt
    data/specialists/fallback/*  (see --fallback-mode)
    data/specialists/_manifest.json

Usage:
    python scripts/phase14_split_corpus.py --root . [--limit-gb N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

# Reuse the detect() port
sys.path.insert(0, str(Path(__file__).parent))
from phase14_audit_corpus import detect, PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK  # noqa: E402


DOMAINS = [PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK]


# --- Document splitting for raw text files ---
#
# Documents in the Phase 11 raw text files are separated by blank lines
# (build_real_corpus.py appends "\n" after each document, resulting in
# "content\n\n" between contents because content itself usually ends in
# "\n"). We use blank-line splitting with a byte cap so no single
# "document" exceeds 256 KB — detect() only reads 4 KB anyway and we
# want bounded memory.

CHUNK_MAX_BYTES = 256 * 1024  # 256 KB cap per document chunk
CHUNK_MIN_BYTES = 4 * 1024    # don't flush chunks shorter than 4 KB (detect
                              # needs 256 bytes; 4 KB gives stable signals
                              # and avoids the fallback black-hole)


def _process_text_file(args):
    in_path, out_dir, source_label, start_offset, end_offset, worker_id = args
    # Open per-domain writers lazily; keep them open for the worker's span.
    out_files = {}
    try:
        buf: list[str] = []
        buf_bytes = 0
        bytes_read = start_offset
        counts = {d: 0 for d in DOMAINS}
        bytes_written = {d: 0 for d in DOMAINS}
        docs = 0
        t0 = time.time()

        def _flush():
            nonlocal buf, buf_bytes, docs
            if not buf:
                return
            doc = "".join(buf)
            buf = []
            buf_bytes = 0
            # Detect on first 4 KB of doc as bytes
            sample = doc.encode("utf-8", errors="replace")[:4096]
            label, _conf = detect(sample)
            dom = label if label in DOMAINS else FALLBACK
            fh = out_files.get(dom)
            if fh is None:
                target = out_dir / dom / f"from_{source_label}.part{worker_id:02d}.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = open(target, "w", encoding="utf-8")
                out_files[dom] = fh
            fh.write(doc)
            if not doc.endswith("\n"):
                fh.write("\n")
            fh.write("\n")  # doc separator
            counts[dom] += 1
            bytes_written[dom] += len(doc.encode("utf-8", errors="replace"))
            docs += 1

        with open(in_path, "r", encoding="utf-8", errors="replace") as fin:
            fin.seek(start_offset)
            if start_offset > 0:
                skip = fin.readline()
                bytes_read += len(skip.encode("utf-8"))
            for line in fin:
                bytes_read += len(line.encode("utf-8"))
                # Blank line = document boundary, but only flush if we've
                # accumulated enough bytes for detect() to have a real
                # signal. Otherwise keep merging with the next doc — this
                # prevents the "hundreds of thousands of 200-byte fallback
                # fragments" pathology we saw in the smoke run.
                if line.strip() == "":
                    if buf_bytes >= CHUNK_MIN_BYTES:
                        _flush()
                    else:
                        # keep accumulating; add blank as separator so the
                        # detect sample still sees document structure
                        buf.append("\n")
                        buf_bytes += 1
                else:
                    buf.append(line)
                    buf_bytes += len(line)
                    if buf_bytes >= CHUNK_MAX_BYTES:
                        _flush()
                if bytes_read >= end_offset:
                    break
            _flush()
        return worker_id, docs, counts, bytes_written, time.time() - t0
    finally:
        for fh in out_files.values():
            fh.close()


def split_text_parallel(in_path: Path, out_dir: Path, source_label: str,
                        n_workers: int, limit_bytes: int | None = None):
    size = in_path.stat().st_size
    if limit_bytes is not None:
        size = min(size, limit_bytes)
    chunk = size // n_workers
    args_list = []
    for i in range(n_workers):
        start = i * chunk
        end = size if i == n_workers - 1 else (i + 1) * chunk
        args_list.append((str(in_path), out_dir, source_label, start, end, i))
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_text_file, args_list)
    total_docs = sum(r[1] for r in results)
    total_counts = {d: 0 for d in DOMAINS}
    total_bytes = {d: 0 for d in DOMAINS}
    for _, _, c, b, _ in results:
        for d in DOMAINS:
            total_counts[d] += c[d]
            total_bytes[d] += b[d]
    print(f"  {source_label}: {total_docs:,} docs in {time.time()-t0:.0f}s")
    for d in DOMAINS:
        if total_counts[d]:
            print(f"    -> {d:10s} {total_counts[d]:>8,} docs, {total_bytes[d]/1e9:>5.2f} GB")
    return total_counts, total_bytes


# --- Pile shard processing (tokenized) ---

def _process_pile_shard(args):
    in_path, out_dir, spm_path, worker_id = args
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)

    out_files = {}
    counts = {d: 0 for d in DOMAINS}
    bytes_written = {d: 0 for d in DOMAINS}
    docs = 0
    t0 = time.time()

    def _flush(ids):
        nonlocal docs
        if not ids:
            return
        try:
            text = sp.decode_ids(ids)
        except Exception:
            return
        if not text:
            return
        sample = text.encode("utf-8", errors="replace")[:4096]
        label, _ = detect(sample)
        dom = label if label in DOMAINS else FALLBACK
        fh = out_files.get(dom)
        if fh is None:
            target = out_dir / dom / f"from_pile.part{worker_id:02d}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            fh = open(target, "w", encoding="utf-8")
            out_files[dom] = fh
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n")
        counts[dom] += 1
        bytes_written[dom] += len(sample) if len(text.encode("utf-8", errors="replace")) < 4096 else len(text.encode("utf-8", errors="replace"))
        docs += 1

    try:
        with open(in_path, "r") as fin:
            current: list[int] = []
            for line in fin:
                try:
                    tid = int(line.strip())
                except ValueError:
                    continue
                if tid == 2:  # BOS
                    _flush(current)
                    current = []
                else:
                    current.append(tid)
            _flush(current)
    finally:
        for fh in out_files.values():
            fh.close()
    return worker_id, docs, counts, bytes_written, time.time() - t0


def split_pile_parallel(pile_parts: list[Path], out_dir: Path, spm_path: Path,
                        n_workers: int):
    args_list = [(str(p), out_dir, str(spm_path), i)
                 for i, p in enumerate(pile_parts)]
    n_workers = min(n_workers, len(pile_parts))
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_pile_shard, args_list)
    total_docs = sum(r[1] for r in results)
    total_counts = {d: 0 for d in DOMAINS}
    total_bytes = {d: 0 for d in DOMAINS}
    for _, _, c, b, _ in results:
        for d in DOMAINS:
            total_counts[d] += c[d]
            total_bytes[d] += b[d]
    print(f"  pile: {total_docs:,} docs in {time.time()-t0:.0f}s")
    for d in DOMAINS:
        if total_counts[d]:
            print(f"    -> {d:10s} {total_counts[d]:>8,} docs, {total_bytes[d]/1e9:>5.2f} GB")
    return total_counts, total_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/Users/dmatt/Claude Projects/l3tc-prod"))
    p.add_argument("--out-dir", type=Path, default=Path("data/specialists"),
                   help="Output root (will be joined to --root if relative).")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit-gb", type=float, default=None,
                   help="For smoke-test: cap bytes read from each source.")
    p.add_argument("--fallback-mode", choices=("symlink", "copy"),
                   default="symlink")
    p.add_argument("--skip-pile", action="store_true",
                   help="Skip the 40 GB pile decode (saves ~15 min for smoke tests).")
    args = p.parse_args()

    root = args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    limit_bytes = int(args.limit_gb * 1e9) if args.limit_gb else None

    raw_sources = [
        ("nick007x_code", root / "corpus_build/structured/code_diverse_real.txt"),
        ("lumees_struct_1", root / "corpus_build/structured/code_real.txt"),
        ("lumees_struct_2", root / "corpus_build/structured/code_real_extra.txt"),
        ("zenodo_loghub", root / "corpus_build/structured/logs_real.txt"),
        ("public_csv", root / "corpus_build/structured/csv_real.txt"),
        ("pile_raw_1gb", root / "corpus_build/pile_raw_1gb.txt"),
    ]

    manifest: dict = {
        "per_source": {},
        "per_domain_bytes": {d: 0 for d in DOMAINS},
        "per_domain_docs": {d: 0 for d in DOMAINS},
    }

    print(f"=== Phase 14 corpus split ===")
    print(f"output: {out_dir}")
    print(f"workers: {args.workers}\n")

    for label, path in raw_sources:
        if not path.exists():
            print(f"  SKIP {label} ({path} missing)")
            continue
        print(f"\n-- {label} ({path.stat().st_size/1e9:.2f} GB) --")
        counts, bytesw = split_text_parallel(path, out_dir, label,
                                             args.workers, limit_bytes)
        manifest["per_source"][label] = {
            "path": str(path.relative_to(root)),
            "docs": counts,
            "bytes": bytesw,
        }
        for d in DOMAINS:
            manifest["per_domain_bytes"][d] += bytesw[d]
            manifest["per_domain_docs"][d] += counts[d]

    if not args.skip_pile:
        pile_parts = sorted((root / "corpus_build/retokenized").glob("pile.part*.txt"))
        balanced_spm = root / "tokenizer_balanced_32k/spm_balanced_unigram_32768.model"
        if pile_parts and balanced_spm.exists():
            print(f"\n-- pile (decode from {len(pile_parts)} shards) --")
            counts, bytesw = split_pile_parallel(
                pile_parts, out_dir, balanced_spm, args.workers)
            manifest["per_source"]["pile_decoded"] = {
                "path": "corpus_build/retokenized/pile.part*.txt",
                "docs": counts,
                "bytes": bytesw,
            }
            for d in DOMAINS:
                manifest["per_domain_bytes"][d] += bytesw[d]
                manifest["per_domain_docs"][d] += counts[d]

    # Fallback pile: mirror the full Phase 11 balanced tokenized corpus.
    # Training on the generalist-style mix is the point of the fallback;
    # no re-tokenization needed if the training script reads .txt ids
    # the same way Phase 11 did. We leave a pointer file for task 8.
    fallback_src = root / "corpus_build/train_2l_corpus_balanced_32k.txt"
    fallback_target = out_dir / FALLBACK / "phase11_balanced_32k_tokens.txt"
    if fallback_src.exists():
        if fallback_target.exists() or fallback_target.is_symlink():
            fallback_target.unlink()
        if args.fallback_mode == "symlink":
            os.symlink(fallback_src.resolve(), fallback_target)
            print(f"\n-- fallback symlink: {fallback_target} -> {fallback_src}")
        else:
            shutil.copy2(fallback_src, fallback_target)
            print(f"\n-- fallback copied: {fallback_target}")
        try:
            manifest["fallback_pointer"] = str(fallback_target.relative_to(root))
        except ValueError:
            manifest["fallback_pointer"] = str(fallback_target)

    # Write manifest
    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {manifest_path}")
    print(f"\n=== Per-domain totals ===")
    for d in DOMAINS:
        print(f"  {d:10s} {manifest['per_domain_bytes'][d]/1e9:>6.2f} GB "
              f"({manifest['per_domain_docs'][d]:,} docs)")


if __name__ == "__main__":
    sys.exit(main())
