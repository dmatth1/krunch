"""Train a SentencePiece BPE tokenizer on a sample of the Pile.

Downloads raw text from HuggingFace, saves it locally (and
optionally uploads to S3), then trains an SPM BPE tokenizer
with the same settings as L3TC's enwik8 tokenizer but on broad
data.

Usage (local, ~15 min download + ~10-30 min training):

    pip install datasets sentencepiece
    python scripts/train_tokenizer_pile.py \
        --target-gb 1 \
        --output-dir tokenizer_pile \
        --s3-raw-upload s3://dmatth1-bnn-checkpoints/l3tc/corpora/pile_raw_1gb.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def download_raw_pile(target_gb: float, output_path: Path, seed: int = 1204):
    """Stream raw Pile text from HuggingFace and save to disk."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    target_bytes = int(target_gb * 1e9)
    print(f"streaming {target_gb} GB of raw Pile text...")

    ds = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    docs = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            if bytes_written >= target_bytes:
                break
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue
            f.write(text)
            f.write("\n")
            bytes_written += len(text.encode("utf-8")) + 1
            docs += 1
            if docs % 10_000 == 0:
                elapsed = time.time() - t0
                gb = bytes_written / 1e9
                print(f"  {docs:,} docs, {gb:.2f}/{target_gb} GB, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"done: {docs:,} docs, {bytes_written / 1e9:.2f} GB, {elapsed:.0f}s")
    return output_path


def train_spm(raw_text_path: Path, output_dir: Path, vocab_size: int = 16384):
    """Train SentencePiece BPE on the raw text file."""
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / f"spm_pile_bpe_{vocab_size}")

    print(f"training SPM BPE: vocab={vocab_size}, input={raw_text_path}")
    t0 = time.time()

    spm.SentencePieceTrainer.Train(
        input=str(raw_text_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,
        character_coverage=0.999,
        max_sentence_length=10000,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        user_defined_symbols=["\\n", "\\t"],
        num_threads=os.cpu_count() or 4,
    )

    elapsed = time.time() - t0
    model_path = Path(f"{model_prefix}.model")
    vocab_path = Path(f"{model_prefix}.vocab")
    print(f"done: {model_path} + {vocab_path} in {elapsed:.0f}s")

    # Quick validation
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    test = "The quick brown fox jumps over the lazy dog."
    ids = sp.encode_as_ids(test)
    print(f"validation: '{test}' -> {len(ids)} tokens: {ids[:10]}...")
    print(f"vocab_size={sp.vocab_size()}, bos_id={sp.bos_id()}, pad_id={sp.pad_id()}")

    return model_path, vocab_path


def upload_to_s3(local_path: Path, s3_path: str):
    print(f"uploading {local_path} to {s3_path}")
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_path), s3_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: S3 upload failed: {result.stderr}")
    else:
        print("uploaded")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-gb", type=float, default=1.0,
                   help="GB of raw Pile text to download for SPM training.")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--output-dir", type=Path, default=Path("tokenizer_pile"))
    p.add_argument("--raw-text-path", type=Path, default=None,
                   help="Skip download; use this existing raw text file.")
    p.add_argument("--s3-raw-upload", type=str, default=None,
                   help="Upload raw text to this S3 path.")
    p.add_argument("--s3-model-upload", type=str, default=None,
                   help="Upload .model + .vocab to this S3 prefix.")
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    # Step 1: get raw text
    if args.raw_text_path and args.raw_text_path.exists():
        raw_path = args.raw_text_path
        print(f"using existing raw text: {raw_path}")
    else:
        raw_path = args.output_dir / f"pile_raw_{args.target_gb}gb.txt"
        download_raw_pile(args.target_gb, raw_path, args.seed)

    # Upload raw text to S3
    if args.s3_raw_upload:
        upload_to_s3(raw_path, args.s3_raw_upload)

    # Step 2: train SPM
    model_path, vocab_path = train_spm(raw_path, args.output_dir, args.vocab_size)

    # Upload model to S3
    if args.s3_model_upload:
        upload_to_s3(model_path, f"{args.s3_model_upload}/spm_pile_bpe_{args.vocab_size}.model")
        upload_to_s3(vocab_path, f"{args.s3_model_upload}/spm_pile_bpe_{args.vocab_size}.vocab")

    print(f"\nDone. Model: {model_path}")
    print(f"Next: re-tokenize the Pile corpus with this model.")


if __name__ == "__main__":
    main()
