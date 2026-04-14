# Phase 1 — Rust rewrite of the inference runtime  ✅ COMPLETE

Replaced L3TC's Python + PyTorch inference with a pure Rust implementation. Achieved 85 KB/s compress on enwik6 (6.43x faster than Python L3TC-200K's 13.24 KB/s) with byte-identical round trip on the full 1 MB corpus. See commit `4234d07` ("Phase 1 COMPLETE").

## Key results

| Metric | Target | Achieved |
|---|---|---|
| Speed (enwik6 compress) | >= 55 KB/s (5x Python) | 85 KB/s (6.43x) |
| Ratio (enwik6) | within 0.5 pp of Python | 0.2094 |
| Round trip | byte-identical | confirmed on enwik6, enwik8, Persian/Arabic |

## Architecture shipped

- Decoder-only RWKV-v4 forward pass with HiRA pre-merged at checkpoint conversion time
- SentencePiece tokenizer via `sentencepiece` Rust crate
- Pure-integer arithmetic coder (deterministic, zero floating-point)
- Two-stage checkpoint loading: Python converter (`scripts/convert_checkpoint.py`) produces a simple binary format, Rust reads it with zero dependencies
- CLI with zstd-ish ergonomics (`l3tc compress`, `l3tc decompress`, stdin/stdout, `-k`, `-f`)
- Self-consistent wire format (not Python-compatible by design)

## Dependencies

clap, anyhow, thiserror, sentencepiece, byteorder, memmap2. No ML frameworks.
