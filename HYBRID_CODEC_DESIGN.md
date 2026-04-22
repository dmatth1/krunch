# Hybrid codec design — splitting skeleton from variable fields

Status: design spec, pre-implementation. Written 2026-04-22 off the
back of Spike 2 findings (see SPIKE_2_LOG.md).

## The insight from Spike 2

HDFS compression with a small RWKV model stalls at ~0.07 ratio —
50-60% worse than `zstd --long=27 -22`'s 0.047. We ran 7 experiments
varying vocab, model size, context, tokenizer config, and variable-
field normalization. Consistent pattern across all seven:

**The bytes of a log line split into two distinct populations**,
each compressible by very different mechanisms:

| Kind | Fraction of bytes | Example (HDFS) | What compresses it |
|---|---|---|---|
| **Skeleton / template** | ~80% | `INFO dfs.DataNode$DataXceiver: Receiving block` | Model prediction — a small RWKV can get to ~0.05 bits/byte on skeleton text because the template repeats endlessly. |
| **Variable fields** | ~20% | `081109 203518` (timestamp), `blk_-1608999687919862906` (block ID), `10.250.19.102:50010` (IP) | **Dictionary lookup**, not prediction. These values are near-random when you see them for the first time, but they *repeat* across the corpus — the same block ID appears when that block is written, allocated, replicated, and deleted; the same IPs appear in thousands of lines. zstd's 128 MB window captures them all and turns each reference into ~4 bytes. |

A learned model trying to beat zstd end-to-end has to do both jobs.
It turns out a 200 K-param RWKV is great at one (skeleton) and useless
at the other (variable fields are near-random from its short-context
view). Scaling the model to 25 M params helped only incrementally
because the bottleneck isn't capacity — it's access to a large
dictionary.

**zstd wins on HDFS by being a dictionary coder with a huge window;
loses on variable-content data where patterns need real prediction.
A hybrid that does both beats both.**

## The codec, at the level of an encode/decode flow

```
               ┌─────────────────────────────┐
plaintext ───▶ │ 1. NORMALIZE                │
               │   regex-scan for variable    │
               │   fields, emit placeholders  │
               │   into the skeleton stream   │
               │   and captured literals      │
               │   into per-field streams     │
               └──────────┬────────┬─────────┘
                          │        │
                          ▼        ▼
             skeleton stream   var streams
                          │    (TS, BLKID,
                          │     IP, TASKID, …)
                          ▼
               ┌─────────────────────────────┐
               │ 2. SKELETON ENCODE           │
               │   SPM (trained on normalized │
               │   corpus) → token ids        │
               │   RWKV → arithmetic-coded    │
               │   bitstream                  │
               │   ≈ 0.02 bits/byte on skel   │
               └──────────┬──────────────────┘
                          │
                          ▼
               ┌─────────────────────────────┐
               │ 3. VARIABLE ENCODE (per field)  │
               │   each field gets its own    │
               │   encoder keyed to the       │
               │   field's structure:         │
               │                              │
               │ · TS    → delta-encode from  │
               │           last value (times  │
               │           are monotonic)     │
               │ · BLKID → 63-bit int literal │
               │           + dictionary of    │
               │           previously seen    │
               │           block IDs          │
               │ · IP    → 4-byte literal +   │
               │           dictionary of seen │
               │           IPs (short keys)   │
               │ · TASKID → delta from last   │
               │           value in same job  │
               └──────────┬──────────────────┘
                          │
                          ▼
               ┌─────────────────────────────┐
               │ 4. MUX + HEADER              │
               │   tiny envelope: skeleton    │
               │   length, per-stream         │
               │   offsets, field-dictionary  │
               │   metadata                    │
               └──────────┬──────────────────┘
                          │
                          ▼
                   compressed blob

DECODE = reverse, stitching back in order.
```

## Why each stream compresses cheaply

### Skeleton (~80% of bytes)

- After normalization the skeleton has only ~2000 unique HDFS
  templates (measured in Spike 2 C4 — SPM complained it couldn't
  reach 16 K vocab because only 1119 distinct skeleton pieces
  existed).
- A 200 K-param RWKV easily models 2000 templates because the
  "next template given previous template" transition matrix is
  low-rank and learnable from ~100 K samples.
- Target: **≤ 0.05 bits/byte on skeleton**, i.e. 20× compression of
  the skeleton bytes.

### Variable fields (~20% of bytes)

Three sub-kinds; each gets a tailored encoder:

#### (a) Monotonic fields — timestamps

Delta-encoded against the previous value in the same field.
Deltas are small integers, Huffman or range-coded. Target: **1-2
bits per timestamp** instead of the 12 raw bytes.

#### (b) Dictionary fields — block IDs, IPs, task IDs

Maintain a per-field dictionary in encode order. When a value
repeats (which it does, heavily — HDFS block IDs repeat ~10-40
times each across allocation/replication/read/delete events), emit
a short code for its dictionary index. First occurrences emit the
raw value + a new-index marker.

Zstd does this already but at the byte level with a generic hash.
We do it at the **semantic-field level** so we can use smaller
indices and skip the byte-level redundancy inside a block ID (no
need to re-hash the "blk_-" prefix every line).

#### (c) Literal fields — first-occurrence variable values

Fall through to raw literal bytes. For genuinely-random fields
(rare UUIDs, unique error strings) this is the same cost as plain
storage. Small overall footprint.

## Throughput envelope

This design stays **well inside** the user's inference-speed
constraint (≥ 1 MB/s decompress single-stream on L4 GPU):

- **Skeleton decode** dominates total model work. 200 K-param RWKV
  at ctx 2048 on L4 does ~10 MB/s skeleton bytes per stream
  (from l3tc-rust benchmarks, extrapolating to skeleton
  throughput). Per corpus byte, that's `10 MB/s × 80% skeleton
  fraction = 8 MB/s` model-limited.
- **Variable decode** is just table lookup + arithmetic decoding
  of integer fields. Hundreds of MB/s per core.
- Net single-stream decompress: **~5-10 MB/s** on L4, well
  above the 1 MB/s floor.

## Engineering effort

| Chunk | Estimate | Notes |
|---|---|---|
| Finalize the normalizer for HDFS + 2 other corpora | 2 days | Already prototyped in `scripts/normalize_variable_fields.py`; needs generalization + round-trip tests. |
| Skeleton encode/decode wiring in `l3tc-rust` | 3-4 days | The Rust runtime already has SPM + RWKV AC paths; plumbing the normalized stream through is mechanical. |
| Per-field encoders (TS / BLKID / IP / TASKID) | 4-5 days | Each is 100-300 lines of Rust with round-trip property tests. |
| Mux + header format + versioning | 1-2 days | Small bit packer; important to get right once. |
| Integration test harness with a round-trip gate | 2-3 days | Must be **bit-exact**: decode(encode(x)) == x. No exceptions. |
| Benchmarking against zstd on HDFS + 2 corpora | 1-2 days | Confirms the design on the spike corpora. |
| Customer-agnostic discovery path | 1-2 weeks | How do we auto-learn the right regex patterns per customer? Two options: (a) customers declare a schema, (b) infer from a sample. Both have failure modes. Design + prototype only — deep learning-based field inference is a separate research bet. |

Fixed HDFS-specific path: **~2 weeks, 1 engineer**. Generalized
per-customer path: **~3-4 weeks**. The HDFS-specific path is enough
to prove the thesis and beat zstd on at least one corpus, which is
the Spike 2 hard constraint.

## What this buys us vs. what it costs

**Buys:**
- Beats zstd on HDFS (our hardest corpus).
- Stays inside the inference envelope (small model + cheap lookups).
- Gives us a compression story customers can reason about: "we
  model the structure, dictionary-encode the identifiers."
- Opens the door to field-aware tools for customers (e.g., "give
  me all logs where BLKID = X" is easier when BLKID is a named
  stream, not embedded raw bytes).

**Costs:**
- **Schema dependency.** The HDFS-specific encoder doesn't ship
  as-is for a customer with Stripe webhooks or nginx logs. We
  need per-customer regex configs or a field-discovery pipeline.
- **Round-trip correctness risk.** Every new field type is a
  chance to introduce an encode/decode mismatch. Needs aggressive
  property testing.
- **Rust runtime complexity.** Today `l3tc-rust` is a pure
  model + SPM pipeline. Adding field-stream mux + per-field
  encoders roughly doubles the code size.

## Open decision

Before building: **do we commit to HDFS-class corpora in the
product scope?** If the product really is "dump your logs here and
we give you good compression," HDFS-class templated data is a
realistic customer profile and we have to beat it. If the product
is "structured audit trails + free-text content," HDFS is an
adversarial outlier and we can skip this work, ship on realistic
data, and flag the limitation.

Either answer is viable. Don't start the 2-week build without the
answer.
