"""Train a 16K SentencePiece unigram tokenizer for one Phase 14 specialist.

Phase 14 task 11. One invocation per domain — corpus-engineer's task 6
produces `data/specialists/{domain}/corpus.txt`; this script reads it,
samples ~200 MB (Phase 11 finding: saturated B/T at this size — see
docs/phases/PHASE_11.md "Tokenizer: balanced 32K unigram" section),
trains an SPM unigram, writes the model + vocab + a small B/T report.

Mirrors the Phase 11 balanced-unigram recipe (see
`corpus_build/tokenizer_unigram_200mb.log`):
  - model_type=unigram (trains 5-6x faster than BPE on M1, slightly
    better B/T on most domains)
  - vocab_size=16384 (Phase 14 specialist target — matches
    L3TC-200K's original tokenizer width, which is the speed enabler)
  - character_coverage=0.9995 (one 9 higher than Phase 11's 0.999;
    domain corpora are narrower so we can afford more coverage)
  - pad_id=0, unk_id=1, bos_id=2 (matches L3TC SPM + rust decoder)
  - add_dummy_prefix=False, remove_extra_whitespaces=False
    (preserves literal whitespace for code/logs/structured text)
  - user_defined_symbols=["\\n","\\t"] (guaranteed tokens — prevents
    the unigram trainer from splitting whitespace weirdly on code)

Usage:

    vendor/L3TC/.venv/bin/python scripts/train_specialist_tokenizer.py \\
        --domain code \\
        --corpus data/specialists/code/corpus.txt \\
        --output-dir data/specialists/code \\
        --sample-mb 200

Output: data/specialists/code/spm.model + spm.vocab + spm.bt_report.json

B/T report is used by task 12 (specialist training) to set realistic
val target ratios (Phase 14 per-specialist target table).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _collect_sources(corpus_arg: Path) -> list[Path]:
    """Accept either a single file or a directory of `from_*.txt` parts.

    Task 6 writes per-source parts under `data/specialists/{domain}/` —
    this gathers all `from_*.txt` files so SPM sees the whole domain.
    Single-file invocation still works.
    """
    if corpus_arg.is_file():
        return [corpus_arg]
    if corpus_arg.is_dir():
        parts = sorted(corpus_arg.glob("from_*.txt"))
        if not parts:
            raise ValueError(f"no from_*.txt parts under {corpus_arg}")
        return parts
    raise FileNotFoundError(corpus_arg)


def sample_bytes_to_file(sources: list[Path], dst: Path, target_mb: float, seed: int = 1204):
    """Reservoir-sample `target_mb` of lines from one or more source files.

    Sources are sampled proportional to their byte size so the SPM
    training sample reflects the domain's actual content distribution.
    Random 64 KB chunks at random offsets — avoids a full pass over
    multi-GB corpora.
    """
    target_bytes = int(target_mb * 1e6)
    sizes = [s.stat().st_size for s in sources]
    total = sum(sizes)

    if total <= target_bytes:
        # Everything fits — concatenate all sources into dst.
        print(f"  corpus {total / 1e6:.1f} MB fits in budget; concatenating {len(sources)} file(s)")
        with open(dst, "wb") as f_out:
            for src in sources:
                f_out.write(src.read_bytes())
        return dst

    rng = random.Random(seed)
    chunk = 64 * 1024
    # Budget per source proportional to size.
    per_source_budget = [int(target_bytes * s / total) for s in sizes]
    print(f"  random-chunk sampling {target_mb} MB from {total / 1e9:.2f} GB across "
          f"{len(sources)} files")

    with open(dst, "wb") as f_out:
        for src, size, budget in zip(sources, sizes, per_source_budget):
            if budget < chunk:
                continue
            written = 0
            with open(src, "rb") as f_in:
                while written < budget:
                    offset = rng.randint(0, max(size - chunk - 1, 0))
                    f_in.seek(offset)
                    _ = f_in.readline()  # align to newline
                    buf = f_in.read(chunk)
                    last_nl = buf.rfind(b"\n")
                    if last_nl > 0:
                        buf = buf[: last_nl + 1]
                    try:
                        buf.decode("utf-8")
                    except UnicodeDecodeError:
                        buf = buf.decode("utf-8", errors="replace").encode("utf-8")
                    f_out.write(buf)
                    written += len(buf)
    print(f"  sampled -> {dst.stat().st_size / 1e6:.1f} MB")
    return dst


def train_spm(sample_path: Path, output_dir: Path, domain: str, vocab_size: int):
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "spm"

    print(f"training SPM unigram: domain={domain} vocab={vocab_size}")
    t0 = time.time()

    # Lossless round-trip is non-negotiable for a compression tokenizer:
    # - user_defined_symbols=["\n","\t"]: force literal control chars into
    #   the vocab as protected pieces. MUST be real \n/\t (not "\\n"/"\\t")
    #   — that was the bug in the first run that collapsed newlines to
    #   spaces on decode. Matches vendor/L3TC/train_tokenizer.py.
    # - normalization_rule_name="identity": disable SPM's default NFKC
    #   normalizer so `[INFO]`, `'alice'`, brackets, quotes, and non-ASCII
    #   all survive encode/decode untouched. Default NFKC was corrupting
    #   log syntax to `⁇`.
    # - add_dummy_prefix=False + remove_extra_whitespaces=False: don't
    #   munge leading/repeated whitespace (code + YAML indentation matter).
    # - split_by_unicode_script=False, split_digits=False: let the trainer
    #   merge domain-specific patterns (e.g. IPs, timestamps in logs;
    #   "0x3f" in code) instead of pre-splitting them.
    # - byte_fallback=True + character_coverage=1.0: guarantee every
    #   UTF-8 byte is representable even if it wasn't in the training
    #   sample. Without this, narrow-character corpora (e.g. the zenodo
    #   loghub sample has no `[`,`'`,`{` chars) produced SPMs that
    #   <unk>-replaced common syntax at inference time on other logs.
    #   byte_fallback renders unseen chars as <0xNN> pieces so round-trip
    #   is always lossless.
    spm.SentencePieceTrainer.Train(
        input=str(sample_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,
        character_coverage=1.0,
        byte_fallback=True,
        max_sentence_length=16384,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        split_by_unicode_script=False,
        split_digits=False,
        user_defined_symbols=["\n", "\t"],
        num_threads=os.cpu_count() or 4,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )

    elapsed = time.time() - t0
    model_path = Path(str(prefix) + ".model")
    vocab_path = Path(str(prefix) + ".vocab")
    print(f"  trained in {elapsed:.0f}s -> {model_path}")
    return model_path, vocab_path


def bytes_per_token_report(model_path: Path, sample_path: Path, domain: str):
    """Compute B/T on the sample to confirm convergence — Phase 11
    numbers are in PHASE_11.md table; we log ours for comparison."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    # Process in ~1 MB chunks; unigram encode is fast but we only
    # need a representative slice.
    text_bytes = 0
    n_tokens = 0
    with open(sample_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 50_000:
                break
            ids = sp.encode_as_ids(line)
            text_bytes += len(line.encode("utf-8"))
            n_tokens += len(ids)

    bpt = text_bytes / max(n_tokens, 1)
    print(f"  bytes/token on sample: {bpt:.3f}  ({n_tokens:,} tokens, {text_bytes / 1e6:.2f} MB)")

    report = {
        "domain": domain,
        "vocab_size": sp.vocab_size(),
        "bos_id": sp.bos_id(),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "sample_bytes": text_bytes,
        "sample_tokens": n_tokens,
        "bytes_per_token": round(bpt, 4),
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True,
                   choices=["prose", "code", "structured", "logs",
                            "tabular", "markup", "fallback"])
    p.add_argument("--corpus", type=Path, required=True,
                   help="Domain corpus produced by task 6 (plain text).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-mb", type=float, default=200.0,
                   help="Sample size for SPM training. Phase 11 found "
                        "200 MB saturates B/T; 1 GB runs 30x slower "
                        "with no gain.")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "spm_training_sample.txt"

    print(f"=== SPM trainer: {args.domain} ===")
    sources = _collect_sources(args.corpus)
    print(f"  {len(sources)} source file(s)")
    sample_bytes_to_file(sources, sample_path, args.sample_mb, seed=args.seed)

    model_path, vocab_path = train_spm(
        sample_path, args.output_dir, args.domain, args.vocab_size,
    )
    report = bytes_per_token_report(model_path, sample_path, args.domain)

    report_path = args.output_dir / "spm.bt_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote report: {report_path}")

    # Clean up the sample to save disk (can always regenerate).
    sample_path.unlink()
    print(f"\nDone. SPM: {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
