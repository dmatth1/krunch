# Archive: l3tc-prod history

This codebase was forked from `l3tc-prod` on 2026-04-21. l3tc-prod was
a CLI compressor project (learned lossless text compression) that was
archived when the market analysis didn't hold: competing with zstd at
1000× slower speed for ~15% ratio win on heterogeneous text isn't a
viable product shape.

What carries over (and is load-bearing for the service):

- RWKV-v4 + HiRA Rust inference runtime (`l3tc-rust/src/`)
- NEON INT8 matvec for CPU, Metal kernels for GPU
- RWKV-v4 training pipeline (`scripts/train_l3tc_phase11.py`)
- `.pth → .bin` checkpoint converter
- Phase 11 corpus-build + spot-fleet training infrastructure
- v4 compressed-file format (will evolve into per-blob service format)
- SentencePiece tokenizer wrapper + training scripts

The CLI product direction is done. No new CLI features. Runtime code
is kept because the service needs a compression engine and this one
is debugged, Metal-accelerated, and parity-tested.

## Key findings from l3tc-prod that inform the service

1. **Compression at 150 KB/s CPU is fine for amortized workloads.**
   Day-1 benchmark showed ~158 KB/s parallelized on enwik6.
   Per-customer batch compression is fine at this speed — customer
   PUTs complete in S3-speed; compression runs async.
2. **Ratio on Wikipedia text: 0.17 on enwik6 with a 200K-param
   RWKV-v4.** Beats zstd-22 (~0.25) by ~30% on pure English.
3. **Ratio on mixed prose is weaker.** ~0.33 aggregate on our
   v010_benchmark prose mix (Pile extracts + Dickens + Canterbury).
   Dickens file hit the `raw_store` fallback due to Latin-1 bytes.
   Takeaway: **homogeneity matters enormously.** The service wins
   by training on per-customer homogeneous data, not general text.
4. **Per-domain SPM tokenizers give meaningful B/T wins.** Tabular
   specialist SPM measured +1.34 B/T vs enwik8 on CSV. Prose domain
   SPMs were *worse* than enwik8 on prose (because our prose corpus
   was contaminated). For the service, each customer dataset gets
   its own SPM trained on just their data.
5. **TokenMonster as a tokenizer candidate**: 25-73% better B/T
   than SentencePiece on clean UTF-8, but drops non-UTF-8 bytes
   silently. Lossless variant requires training our own vocab with
   `-include-256-bytes` flag. Deferred to post-MVP service work.
6. **MLX forward pass at 426K tok/s (batch 32) on M1 Pro** proves
   the architecture is viable on Apple Silicon if we ever want a
   local-edge variant of the service for on-prem customers.

## Provenance

Git history in this fork preserves all l3tc-prod commits up through
2026-04-21. Commits after that date are learned-archive service work.
Pre-fork phase docs (Phase 0-14) lived in `docs/phases/` and
`docs/phase-findings/` and have been deleted from this fork — retrieve
from `l3tc-prod` git history if needed. Original
`docs/STORAGE_SERVICE_VISION.md` is superseded by the top-level
`STORAGE_SERVICE.md`.

Parallel fork: `rwkv-metal/` (also at `~/Claude Projects/`), which
took the same l3tc-prod codebase and attempted a Rust/Metal RWKV
inference runtime direction. That was also deprioritized when the
`web-rwkv` benchmark showed the niche was already occupied. The Rust
inference code lives in both forks; any future work consolidates
back into this one.
