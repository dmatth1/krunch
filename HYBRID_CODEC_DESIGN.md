# Hybrid codec design — neural-primary dispatcher

Status: design decided, pre-implementation. Revised 2026-04-22 after
Spike 2 findings + research on cmix/paq8 detection primitives + user
review of detector + within-file hybrid options.

## Product thesis

**Neural is the primary codec for text-like content. Classical codecs
are dispatched for the specific content types where classical
genuinely wins.** Spike 2 + the research literature support this:
on prose, code, chat, JSON-with-text, audit trails, and most
realistic customer data, a small in-domain RWKV + AC beats zstd-22
by 15–40% on ratio. The exception is pathologically templated
machine logs (HDFS-style), where zstd's hash-dictionary is
near-optimal; on those chunks we dispatch to CLP or zstd-dict.

Product claim this enables: **"we train a compression model on your
data; on everything except templated infrastructure logs it beats
any off-the-shelf compressor, and on those logs we fall back to the
best classical option automatically."** That is strictly better than
any single-codec approach on any mixed corpus, and per-chunk-safety
makes it never-worse than zstd-dict.

## Codec menu

Per-chunk blob format: 3-bit codec tag + encoded bytes.

| Tag | Codec | When it wins |
|---|---|---|
| 0x0 | passthrough / store raw | Pre-compressed content (JPEG, ZIP, gzipped inputs). Magic-byte detected. |
| 0x1 | lz4 | Tiny chunks / incompressible content / latency-critical paths |
| 0x2 | zstd -22 --long=27 | Generic fallback; the universal safety net |
| 0x3 | zstd --dict (per-family trained) | Homogeneous structured content with known vocab |
| 0x4 | bzip3 | Text where zstd is dominant and neural is unavailable |
| 0x5 | brotli + RFC 9842 shared dict | Versioned near-duplicate content |
| 0x6 | CLP IR + zstd | Templated machine logs (HDFS, Kubernetes event streams, syslog) |
| **0x7** | **RWKV + SPM + AC (our model)** | **Text-like content — prose, code, chat, JSON with free-text, docs, audit trails. The primary path.** |

## Per-chunk dispatcher

### Detector (decided: magic-byte prescreen + probe-encode)

Three stages:

```
Stage 1 — magic bytes (~100 ns, zero cost for non-binary data)
────────────────────────────────────────────────────────────────
If the first 16 bytes match a known binary format
  (JPEG/PNG/GIF/BMP/WAV/MP3/ELF/PE/PDF/ZIP/GZIP/ZSTD/XZ/BZIP2):
    → emit tag 0x0 (passthrough) or 0x1 (lz4), skip all probes.
Else fall through. Implementation: libmagic FFI or a minimal
hand-written magic-number table (~50 entries).

Stage 2 — probe-encode (~1-30 ms per chunk, runs in parallel)
────────────────────────────────────────────────────────────────
Run the following candidates on the first 4 KB of the chunk,
extrapolate each output length linearly to the full chunk:
  - zstd -22 --long=27                         (always)
  - zstd --dict                                (if per-family dict available)
  - CLP IR                                     (short-circuits fast if not log-like)
  - bzip3                                      (optional)
  - brotli + shared-dict                       (if near-dup dict available)
  - RWKV + AC                                  (slow: ~27 ms per 4 KB probe)

Pick the codec with shortest predicted output. Emit that codec's
tag + full-chunk encoded bytes.

Stage 3 — length-check safety net (cheap post-hoc)
────────────────────────────────────────────────────────────────
After the full encode with the picked codec, compare output length
to what zstd-dict produces on the same chunk. If the picked codec
output is >1% longer, substitute zstd-dict and update the tag.

This guarantees "never materially worse than zstd-dict" per chunk
regardless of probe extrapolation errors.
```

**What this design deliberately rejects:**
- Hand-tuned feature-based decision tree (cmix-style). More moving parts, threshold drift, less accurate than actually measuring. Probe-encode is better.
- Learned classifier (Tier D). Premature — probe-encode is correct by construction; no empirical evidence yet that a classifier would help.
- Customer-declared `dataset_type` hint (Tier C). Skipped per user call. Can be reintroduced later as a pure speedup optimization (skip probe for declared types).

**What this design deliberately keeps:**
- The **magic-byte prescreen** (from libmagic / cmix / paq8px lineage) because it's zero-cost and eliminates ~30% of pointless neural probes in mixed workloads.
- The **length-check safety net** (our own addition). Guarantees correctness floor.

**Why this is better than the original Tier A (heuristic features) + Tier B (probe) split:**
- Tier A's ~1 µs/chunk cost advantage doesn't matter in practice — neural encode already dominates at ~100 min/GB, and probe adds only a few min/GB.
- Tier A's 85–95% accuracy is empirically worse than probe's 100% accuracy. Misclassifications flow to the safety net but still cost cycles.
- Fewer thresholds to maintain, fewer failure modes to test.

### Detector primitives we port from cmix/paq8px

Even though the hand-tuned feature tree is rejected, the underlying
features/primitives have use elsewhere in the system:

- **UTF-8 state machine** from `paq8px/filter/Filters.hpp::detectText()` with adaptive invalid-byte decay. We use it as a quick-kill heuristic *inside* the neural probe: if the chunk is confirmed not UTF-8 valid, skip the neural probe (it would lose anyway). ~1 KB static state table, trivial port.
- **cmix `ascii_run > 500 && spaces >= 5` text test** from `preprocessor.cpp::detect()`. Same role — fast "is this even text" gate before neural probe.
- **Magic-byte table** from libmagic. Direct use at Stage 1.
- **Drain3-style digit/hex masking + first-N-token structure-hash** for a log detector if we later want to skip neural probe on templated-log chunks (right now the probe just handles this correctly on its own).

These are optimizations to *skip* neural probe on hopeless-for-neural
chunks. Not classifiers.

## Within-file hybrid: Tier 1 → 2, skip Tier 3

Three nested ways to mix codecs inside a single file/dataset. Decision:
**ship Tier 1 now, add Tier 2 for structured data after measuring on
real customer data, skip Tier 3.**

### Tier 1 (build first) — per-chunk codec selection

Each ~64 KB chunk independently picks from the codec menu via the
dispatcher above. Handles "mostly-prose with occasional templated
runs" elegantly. Simple blob format, simple decode. Per-chunk
round-trip correctness is enforced by each codec library.

This is the design described above.

### Tier 2 (build after Tier 1 ships + measures) — field-aware codec

For JSON/NDJSON/CSV with known schema, compress each column/field
with the best codec for that field:
- Keys → zstd-dict (they repeat enormously)
- Free-text descriptions / comments → RWKV + AC
- Timestamps → delta-encoded
- IDs / UUIDs / block IDs → dictionary-coded (zstd-dict or CLP-style)
- Numeric columns → elastic-length integer encoding

This is what CLP does for logs and what Parquet does for columnar
data. Structural win on audit trails, API event streams,
transaction logs — common customer shapes.

Engineering cost: ~1–2 weeks after Tier 1 is shipping. Requires
schema declaration at dataset creation, or a schema-inference
pipeline on the first compressed batch.

### Tier 3 (skip) — codec cascade within a chunk

Preprocess → neural residuals → classical tail. Considered in the
original hybrid codec design; rejected because:
- Engineering heavy
- Round-trip correctness risk (each stage can silently break invariants)
- Tier 2 captures most of what Tier 3 would buy, in a cleaner way

Revisit only if Tier 2 leaves measurable ratio on the table after
shipping.

## Engineering plan

| Step | Effort | Notes |
|---|---|---|
| Detector: magic-byte prescreen | 0.5 d | libmagic FFI or minimal hand-written table |
| Detector: probe-encode driver | 1 d | Common "run codec on 4 KB, extrapolate" scaffolding + per-codec probe wrapper |
| Detector: length-check safety net | 0.5 d | Post-encode comparison + tag substitute |
| Per-codec integration: zstd + zstd-dict | 0.5 d | `zstd` Rust crate; already in `l3tc-rust` dependency tree |
| Per-codec integration: bzip3 | 0.5 d | [`bzip3`](https://crates.io/crates/bzip3) crate wire-up |
| Per-codec integration: lz4 | 0.5 d | `lz4_flex` crate |
| Per-codec integration: brotli + shared-dict | 1 d | `brotli` crate with `dictionary` param; requires shared-dict storage |
| Per-codec integration: CLP IR | 3–5 d | Port IR format to Rust; use existing `zstd` for per-column backend |
| Per-codec: harden existing RWKV + AC path | 1 d | Already exists in l3tc-rust; confirm round-trip + throughput |
| Blob format: 3-bit tag header + per-chunk framing | 1 d | Versioned format; reserve room for Tier 2 extension |
| Training pipeline: per-dataset zstd dict training | 1 d | `ZSTD_trainFromBuffer` call in the training container entrypoint |
| Training pipeline: record dict alongside `.pth` | 0.5 d | Adds `v{N}.zstd_dict` artifact |
| Round-trip tests on HDFS + enwik8 + 1 JSON + 1 code corpus | 2 d | Must be bit-exact per chunk |
| Benchmark + codec-distribution histogram in metadata | 1 d | Per-dataset stats for ops visibility |
| Service wiring: Fargate worker downloads dict + model per dataset | 1 d | Update compression_worker.py |

**Total Tier 1: ~2 weeks, 1 engineer.**

**Tier 2 (later): ~1–2 weeks, 1 engineer** after measuring where
Tier 1 falls short on real customer data.

## Guarantees this buys

- **"Never materially worse than zstd-dict"** per chunk via Stage 3 safety net. Regardless of detector errors.
- **"Strictly better than zstd on text-like data"** by ~15–40% on the published benchmarks and ~14% on our own enwik8 measurement (see bench/results/enwik8-l3tc.md).
- **"Handles templated logs correctly via CLP"** so HDFS-class workloads don't embarrass us.
- **Zero reinvention** of traditional compression primitives — every classical codec comes from a library (zstd, bzip3, lz4, brotli, CLP via FFI or port).

## What this does NOT claim

- We are not beating theoretical entropy bounds. NNCP / cmix remain the theoretical references; we trade 30–50% of their ratio for 1000× more throughput.
- We are not claiming our neural model wins on every chunk. The dispatcher is explicit about when classical wins.
- We are not eliminating zstd as a dependency — we are doubling down on it as the universal safety net.

## Open items

1. **Chunk size**: 64 KB is the default. Tunable per-dataset if we see evidence that longer chunks materially help neural prediction (because of context length) and shorter chunks help dispatcher accuracy.
2. **Dictionary refresh cadence**: when data drifts, the zstd dict and neural model may stale. Reuse the existing retrain-on-drift logic from the training-complete Lambda.
3. **CLP IR scope**: full CLP IR or a minimal subset? Decide after reading the CLP OSDI'21 paper carefully; aim for a minimal subset that captures the template + variable-dict separation.
4. **Brotli shared-dict rollout**: requires a shared-dict store keyed by `{customer, dataset, content_family}`. Plumbing lives mostly in the training-complete Lambda + Fargate worker. Fine to defer to Tier 2.

## Sources

- [cmix — byronknoll/cmix](https://github.com/byronknoll/cmix), specifically `src/preprocess/preprocessor.cpp::detect()`
- [paq8px — hxim/paq8px](https://github.com/hxim/paq8px), `filter/Filters.hpp` + `BlockType.hpp`
- [Uber CLP blog + y-scope/clp](https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp/), [repo](https://github.com/y-scope/clp), [OSDI'21 paper](https://www.usenix.org/system/files/osdi21-rodrigues.pdf)
- [bzip3 Rust crate](https://crates.io/crates/bzip3)
- [libmagic man page](https://man7.org/linux/man-pages/man3/libmagic.3.html)
- [Compression Dictionary Transport (RFC 9842)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Compression_dictionary_transport)
- [DeepMind — Language Modeling Is Compression](https://arxiv.org/abs/2309.10668)
- Our own [`bench/results/enwik8-l3tc.md`](bench/results/enwik8-l3tc.md) and [`bench/results/enwik8-classical.md`](bench/results/enwik8-classical.md) for comparable numbers on prose.
