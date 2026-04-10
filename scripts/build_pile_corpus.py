"""Build the Pile dedup corpus for Phase 11 Pass 2.

Downloads a subset of EleutherAI's deduplicated Pile from HuggingFace,
tokenizes it with the existing enwik8-trained SPM (vocab 16384), and
writes it in the one-int-per-line format that train_l3tc_phase11.py
expects. Uploads the result to S3.

Run this on a machine with:
  - ~60 GB free disk (for download + tokenized output)
  - Good network (downloads ~50 GB from HuggingFace)
  - The L3TC venv with sentencepiece installed
  - AWS credentials for S3 upload

Intended to run on the baked AMI instance or a cheap CPU instance,
NOT on the laptop (50 GB download is too much for most home
connections).

Usage:
    cd vendor/L3TC && source .venv/bin/activate
    python ../../scripts/build_pile_corpus.py \\
        --target-gb 10 \\
        --spm-model dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model \\
        --output data/train_data/train_pile_dedup.txt \\
        --s3-upload s3://dmatth1-bnn-checkpoints/l3tc/corpora/train_pile_dedup.txt

    # Start small (1 GB) to verify the pipeline, then scale up:
    python ../../scripts/build_pile_corpus.py --target-gb 1 ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def download_and_tokenize(
    target_gb: float,
    spm_model_path: Path,
    output_path: Path,
    seed: int = 1204,
):
    """Stream Pile dedup from HuggingFace, tokenize, write to disk.

    Uses the `datasets` library to stream (no full download needed in
    memory), but writes the tokenized output to disk incrementally.
    Total disk usage: ~target_gb × 1.5 (raw text → tokenized).
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model_path))
    print(f"SPM loaded: vocab={sp.vocab_size()}, bos={sp.bos_id()}")

    target_bytes = int(target_gb * 1e9)
    print(f"target: {target_gb} GB ({target_bytes:,} bytes) of raw text")

    # Stream the Pile dedup from HuggingFace. The dataset is sharded;
    # streaming avoids downloading the full ~800 GB.
    print("streaming from EleutherAI/the_pile_deduplicated (split=train)...")
    ds = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )
    # Shuffle with a buffer for domain diversity (the Pile is sorted
    # by source). Buffer size is in number of examples, not bytes.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written_raw = 0
    tokens_written = 0
    docs_written = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for example in ds:
            if bytes_written_raw >= target_bytes:
                break

            text = example.get("text", "")
            if not text or len(text) < 100:
                continue  # skip very short documents

            # Tokenize
            ids = sp.encode_as_ids(text)
            if not ids:
                continue

            # Write BOS + token ids, one per line
            f_out.write("2\n")  # BOS
            for tid in ids:
                f_out.write(f"{tid}\n")

            bytes_written_raw += len(text.encode("utf-8"))
            tokens_written += len(ids)
            docs_written += 1

            if docs_written % 10_000 == 0:
                elapsed = time.time() - t0
                gb_done = bytes_written_raw / 1e9
                print(
                    f"  {docs_written:,} docs, {gb_done:.2f}/{target_gb} GB, "
                    f"{tokens_written:,} tokens, {elapsed:.0f}s"
                )

    elapsed = time.time() - t0
    file_size = output_path.stat().st_size
    print(
        f"\ndone: {docs_written:,} docs, {bytes_written_raw / 1e9:.2f} GB raw text, "
        f"{tokens_written:,} tokens, {file_size / 1e9:.2f} GB tokenized file, "
        f"{elapsed:.0f}s wall"
    )
    return output_path


def upload_to_s3(local_path: Path, s3_path: str):
    print(f"uploading {local_path} ({local_path.stat().st_size / 1e9:.2f} GB) to {s3_path}")
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_path), s3_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: S3 upload failed:\n{result.stderr}")
        sys.exit(1)
    print("upload complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target-gb",
        type=float,
        default=10.0,
        help="Target size of raw text to download in GB. Start with 1 "
        "for a quick test, 10-50 for the real run.",
    )
    p.add_argument(
        "--spm-model",
        type=Path,
        required=True,
        help="Path to the SPM .model file (enwik8 BPE 16384).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the tokenized file (one int per line).",
    )
    p.add_argument(
        "--s3-upload",
        type=str,
        default=None,
        help="Optional S3 path to upload the tokenized file to.",
    )
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    if not args.spm_model.exists():
        print(f"ERROR: SPM model not found: {args.spm_model}")
        sys.exit(1)

    out = download_and_tokenize(
        target_gb=args.target_gb,
        spm_model_path=args.spm_model,
        output_path=args.output,
        seed=args.seed,
    )

    if args.s3_upload:
        upload_to_s3(out, args.s3_upload)

    print("\nNext: update TRAIN_FILE in the userdata/launcher to point at")
    print(f"  {args.s3_upload or args.output}")


if __name__ == "__main__":
    main()
