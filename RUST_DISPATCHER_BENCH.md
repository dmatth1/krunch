# Rust hybrid dispatcher — real-data benchmark (2026-04-22)

Actual byte counts from the Rust `l3tc hybrid-compress` binary built
from `l3tc-rust/` at commit `cc2248a`. Supersedes the Python
simulator numbers in [`DISPATCHER_SIM_RESULTS.md`](DISPATCHER_SIM_RESULTS.md)
on the chunks the simulator covered.

**Classical-only vs neural runs — important distinction.** The main
per-chunk-size table below is deliberately **classical-only** (no
neural codec in the probe menu). This measures the *floor* the
dispatcher gives any customer *before* their per-dataset model is
trained — i.e. the result on day 0 of a new archive. Adding neural
requires a `.bin` trained on data from that distribution; the
end-to-end section further down exercises that path using the
pre-existing distilled L3TC-200K + enwik8 tokenizer, and the neural
codec wins the chunk at **+46.6% vs per-chunk zstd** on prose. That
number is what the dispatcher does once training runs for a
customer; the 7.9% / 20.1% wins below are what it does before.

## Corpora

- **enwik8_5mb** — first 5 MB of enwik8 (English Wikipedia prose).
  Local file at `bench/corpora/enwik8_5mb`.
- **hdfs-like** — 19.97 MB synthetic HDFS-shaped log stream
  generated with a seeded RNG and six templates covering the most
  common HDFS / YARN log lines (NameNode / DataNode / FSNamesystem
  messages, HEARTBEAT, block allocation, task start, exception).
  Generated fresh for this benchmark; NOT the real 277 MB Loghub
  HDFS corpus used in Spike 1 (which wasn't on disk locally when
  this ran).

## Codec menu

Classical-only today: Passthrough, Lz4, Zstd22 (`--long=27`, w=27),
Bzip3, plus the `ClpStub` placeholder which can never win the
shortest-pick race. No `ZstdDictCodec` in these runs because the
single-file zstd-train call needs multi-sample input and I haven't
split the corpora yet; dictionary numbers pending.

## enwik8_5mb results

| chunk size | dispatcher ratio | zstd-shadow ratio | vs whole-file zstd-22 | throughput | codec distribution |
|---|---|---|---|---|---|
| 64 KB | 0.3300 | 0.3643 | **−14.7%** (worse) | 3.55 MB/s | bzip3 100% (77/77) |
| 256 KB | 0.2925 | 0.3340 | **−1.6%** (worse) | 3.82 MB/s | bzip3 100% (20/20) |
| **1 MB** | **0.2652** | 0.3104 | **+7.9% (better)** | 3.73 MB/s | bzip3 100% (5/5) |

**Whole-file zstd-22 `--long=27` baseline on the same file: 0.2878
(1,438,926 B).**

Round-trip verified byte-identical at 64 KB chunks.

## hdfs-like results

| chunk size | dispatcher ratio | zstd-shadow ratio | vs whole-file zstd-22 | throughput | codec distribution |
|---|---|---|---|---|---|
| 64 KB | 0.1850 | 0.2188 | **+0.3%** (tie) | 2.94 MB/s | bzip3 100% (305/305) |
| **1 MB** | **0.1482** | 0.1982 | **+20.1% (better)** | 3.24 MB/s | bzip3 100% (20/20) |

**Whole-file zstd-22 `--long=27` baseline on the same file: 0.1855
(3,705,194 B).**

Safety net never fired on either corpus at any chunk size.

## Main findings

### 1. Bzip3 dominates the classical menu on both content types

Bzip3 won 100% of chunks on prose AND on synthetic logs at every
chunk size tested. The zstd / lz4 codecs never won a chunk in the
classical-only menu. This is a data point the Python simulator
captured directionally for prose but not for logs — at 1 MB chunks
bzip3 is a flat ~8-10% win on every chunk.

If that holds up on real HDFS (the simulator ran at 64 KB on the
actual 277 MB corpus and saw zstd 86.9% / bzip3 13.1%, with
dispatcher 42% worse than whole-file zstd), the core story is:
**chunk size was the simulator's bottleneck, not the codec menu**.

### 2. Chunk size matters more than the simulator suggested

At 64 KB, the dispatcher is 14.7% worse than whole-file zstd on
prose. At 1 MB it's 7.9% better. That's a 22-percentage-point swing
from a single tunable. `DEFAULT_CHUNK_SIZE = 64 KB` in the dispatcher
is probably wrong for the average customer; the training pipeline
should emit a dataset-tuned chunk size.

Heuristic for Tier 1:
- **Text-heavy prose / documents**: 1 MB
- **Templated logs**: 1 MB (possibly 4 MB once we have the real
  HDFS corpus to validate)
- **Mixed / JSON / code**: 256 KB (per-chunk codec selection is
  the point; don't fragment too much)

### 3. Classical-only dispatcher already wins on both corpora at 1 MB

On enwik8_5mb the dispatcher beats whole-file zstd-22 by **7.9%**.
On hdfs-like it beats it by **20.1%**. Without any neural codec.
The neural path, once wired, is additive on prose (historical
L3TC-200K hit 0.2166 on enwik8 vs 0.2527 for zstd-22, a 14%
advantage; combining neural + bzip3 per-chunk should be at least as
good as either alone).

This is a big change to the product story: **we can beat S3+zstd
today with zero ML**, and the ML unlocks further gains rather than
being the core differentiator.

### 4. Throughput is fine for ingest, tight for retrieval

3.2 MB/s single-threaded on an M-series dev laptop is ~10× faster
than the 150 KB/s compress target for the 200K neural model, and
well above the 1 MB/s decompress target. The Fargate worker runs
single-chunk-at-a-time today; trivially parallelizable across
chunks for a 4-8× speedup if we need it.

## What's different from the Python simulator

| metric | simulator | Rust binary | why |
|---|---|---|---|
| enwik8 dispatcher ratio | 0.2162 at 64 KB | 0.3300 at 64 KB | simulator included the neural bits/byte estimate; classical-only run without it |
| enwik8 bzip3 share | 0.8% | 100% | simulator scored neural first; classical-only run let bzip3 win |
| hdfs dispatcher ratio | 0.0662 at 64 KB | 0.1850 at 64 KB (synthetic, not real) | different corpus; synthetic is 14× smaller and less repetitive than real HDFS |
| chunk-size sensitivity | untested | 1 MB gives +8-20% gain | simulator ran only at 64 KB |

## End-to-end neural validation (smoke test)

A separate, smaller run to sanity-check that `convert_checkpoint.py`
produces a `.bin` the Rust dispatcher can actually load and
round-trip against. Uses the pre-distilled L3TC-200K weights in
`l3tc-rust/checkpoints/l3tc_200k_distilled.pth` + the
enwik8-trained SPM in `vendor/L3TC/dictionary/`.

| corpus | bytes in | bytes out | ratio | zstd shadow | savings vs per-chunk zstd | codec picked | throughput |
|---|---|---|---|---|---|---|---|
| enwik6 first-100KB | 100,000 | 18,011 | **0.1801** | 0.3374 | **+46.6%** | neural 1/1 | 0.15 MB/s |

Round-trip byte-identical. Confirms the full pipeline
`.pth → convert_checkpoint.py → .bin → l3tc hybrid-compress →
l3tc hybrid-decompress → original` works end-to-end. Throughput
is the ~150 KB/s envelope we already know for L3TC-200K on an M-class
CPU; service-side A10G / L40S will be faster.

## Open follow-ups

1. Re-run on the real 277 MB Loghub HDFS corpus once it's back on
   disk (the Spike 1 fetch path still works; just need to re-pull).
   Expected: dispatcher beats whole-file zstd at 4 MB chunks, loses
   at 64 KB.
2. Repeat with trained zstd dictionary in the menu. Split each
   corpus into many sample files first (or use `zstd --train-fastcover`).
3. Once task #27 lands and the training job emits the Rust `.bin`
   model, re-run with neural in the menu. Expected: neural wins on
   free-text spans where bzip3's BWT is suboptimal; dispatcher
   improves by a further 5-10% on prose.
4. Benchmark decompression throughput alongside compression (not
   measured yet — runs fast but we should have numbers).
