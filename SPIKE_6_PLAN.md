# Spike 6 — Schema preprocessor + GPU throughput + pretrained/LoRA pivot

Three-track spike. Tracks 1 & 2 run in parallel (independent). Track 3
starts after both complete and informs the Tier-2 architecture decision.

## Motivation

Spike 5 outcome: L3TC-12M from-scratch hits held-out **0.22 on
WildChat-English, 32% worse than zstd 0.167**. Deeper analysis (see
Spike 5 discussion thread) found:

- Context-mixing compressors (cmix, NNCP) are 200,000× too slow for
  our ≥300 KB/s throughput gate — off the table.
- L3TC's "train tiny model from scratch per-dataset" bet hits a
  ratio ceiling we can't engineer around at chat entropy.
- Architectural pivot is needed; Options A + C below were selected.

## Track 1 — Schema-aware preprocessor (A)

**Hypothesis**: chat NDJSON has 15–25% bytes of pure structural noise
(`conversation_id` UUIDs, `"role":"` field keys, ISO timestamps,
model names). Losslessly strip these before any codec runs. Free
ratio on top of whatever downstream codec we ship.

### Scope

Input corpus format (WildChat/OASST/ShareGPT-compatible):
```json
{"conversation_id": "<UUID>", "turn": <int>, "role": "<enum>",
 "content": "<text>", "timestamp": "<ISO8601>", "model": "<name>"}
```

### Design

Encoder output (3 streams, all losslessly reversible):
1. **Content stream** — `content` strings separated by a rare byte
   (`0x1E` record separator). Passes to dispatcher.
2. **Structural side-channel** — compact binary:
   - `conversation_id`: first-occurrence-full + subsequent-4-byte-refs
   - `turn`: delta-coded varint (usually 1)
   - `role`: 2-bit enum (`user`/`assistant`/`system`/`tool`)
   - `timestamp`: first-full-ISO + delta-ms-varint
   - `model`: per-conversation enum (1 byte/convo typical)
3. **Schema manifest** — JSON describing which fields, version, and
   a validation hash. Stored uncompressed (KB-scale).

Decoder: strict roundtrip. Property test: decode(encode(x)) == x for
10k real chat samples.

### Implementation

- Language: **Rust**, module in `l3tc-rust/src/preprocess/chat.rs`.
  Keeps the pipeline end-to-end fast and in one binary.
- CLI: `l3tc schema-encode --format wildchat input.ndjson → output.stream`
  and reverse.
- Dispatcher integration: new `hybrid-compress --preprocess=chat`
  flag wires encode → dispatcher → decoder.

### Measurement

Re-run WildChat-English 200 MB end-to-end on each codec stack:
| stack | ratio | vs baseline |
|---|---|---|
| zstd alone | 0.167 | reference |
| dispatcher alone (Spike 5 model) | 0.178 (measured) | — |
| preprocess + zstd | **TBD** | expected 5–15% lift |
| preprocess + dispatcher + Spike 5 model | **TBD** | ditto |

### Gate

Declare success if **preprocess + zstd** ratio ≤ 0.155 (≥7% better
than bare zstd). That proves the preprocessor has standalone value
independent of any neural model.

### Budget

~2 eng-days. No AWS compute beyond the 200 MB benchmark re-run.

## Track 2 — GPU throughput test

**Unfinished Spike 5 business.** Measure actual KB/s of the v1.bin
model on GPU. Independent of ratio — we need this number regardless
of the Track 3 pivot.

### Setup

- Instance: g5.xlarge on-demand (1× A10G, 24 GB), ~$1/hr
- Runtime: ONNX Runtime CUDA 12.x
- Corpus: WildChat-English 200 MB (same as Spike 5 eval)
- Model: existing `s3://archive-dev-archive/spike5/wildchat-en/models/v1.pth`

### Steps

1. Spin g5.xlarge with preinstalled CUDA+ORT AMI (DLAMI)
2. Pull v1.pth → export to ONNX (`torch.onnx.export` with opset 17+)
3. **Parity check**: run 1 MB holdout through both PyTorch and ORT;
   assert bits/token within 0.01
4. Run dispatcher sweep over 200 MB:
   - Log per-chunk: codec selected, encode time, decode time
   - Aggregate: neural-selection %, real dispatcher ratio, end-to-end KB/s
5. Compare to ≥300 KB/s gate
6. Tear down (hard stop at 2 hr)

### Deliverables

- Appendix to `SPIKE_5_LOG.md` with GPU numbers
- If throughput < 100 KB/s: architecture is fundamentally wrong for
  GPU path — this shapes Track 3 heavily.
- If throughput ≥ 300 KB/s: GPU path validated; Track 3 can assume it.

### Budget

~$2–3 compute, ~4 eng-hours.

## Track 3 — Pretrained base + per-tenant LoRA (C)

**Starts after Tracks 1 + 2 complete.** This is the real architectural
pivot. L3TC trains from scratch per-dataset; we flip to: one shared
pretrained base model + tiny per-tenant LoRA adapter.

### Why this works

- **Overfitting inverts into a feature**: pretrained prior handles
  common English/code. Adapter specializes on tenant corpus without
  memorizing it (rank-constrained).
- **Tenant overhead is small**: base model (~400 MB) amortized across
  all tenants. Per-tenant storage = LoRA weights (1–5 MB) + zstd-dict
  (128 KB) + compressed data.
- **Corpus-size mismatch disappears**: pretrained on trillions of
  tokens, fine-tuned on tenant's archive only for specialization.
  Spike 5's "200 MB is 1/5th the paper's regime" complaint goes away.

### Base model candidate evaluation

| candidate | params | tokenizer | license | pretrain budget |
|---|---|---|---|---|
| SmolLM2-135M | 135M | BPE 49K | Apache 2.0 | 2T tokens |
| SmolLM2-360M | 360M | BPE 49K | Apache 2.0 | 4T tokens |
| Qwen2.5-0.5B | 494M | BBPE 152K | Apache 2.0 | 18T tokens |
| TinyLlama-1.1B | 1.1B | BPE 32K | Apache 2.0 | 3T tokens |
| **(stretch) own 200M RWKV** | 200M | SPM 32K | — | ~$5k to pretrain |

**Default pick: SmolLM2-360M.** Strong English+code, Apache 2.0,
standard transformer with efficient decode, smaller vocab than Qwen
so ONNX path is lean.

Revisit own-pretrain if SmolLM2 misses throughput gate.

### Fine-tuning recipe (per tenant)

- Objective: next-token LM (not instruction-tuning — we don't need
  dialogue behavior, we need calibrated next-token probabilities)
- LoRA: rank 16, alpha 32, all attention+FFN linears
- LR: 1e-4, cosine schedule, no warmup
- Epochs: 2–3 over tenant corpus
- Data: raw corpus (post-schema-preprocess content stream from Track 1)
- Target train: val CE gap < 0.3 nats (keep adapter underfit)

### Spike 6 experiment

1. Download SmolLM2-360M base model
2. Apply Track 1 preprocessor to WildChat-English → content stream
3. LoRA fine-tune on content stream (2 epochs, ~200 MB corpus)
4. Export base + merged LoRA to ONNX
5. Run full dispatcher sweep on g5.xlarge:
   - Measure held-out ratio
   - Measure end-to-end throughput
6. Compare to:
   | target | ratio | throughput |
   |---|---|---|
   | zstd baseline | 0.167 | 300 MB/s |
   | Spike 5 L3TC-12M | 0.221 | (TBD Track 2) |
   | **Spike 6 commit** | ≤0.15 | ≥200 KB/s |
   | **Spike 6 stretch** | ≤0.12 | ≥400 KB/s |

### Decision gates

After Spike 6 measurement:

- **Pass commit**: Start Tier-2 product engineering. Build LoRA
  training pipeline, per-tenant model registry, GPU worker pool.
- **Pass stretch**: Above + start customer conversations with real
  numbers.
- **Miss both**: Re-examine model size. Try SmolLM2-135M (maybe speed
  wins) or Qwen-0.5B (maybe ratio wins). If neither hits commit,
  Option D (product reframe) becomes default.

### Budget

~$30–80 compute (one g5.xlarge-day for fine-tune + measurement),
~4 eng-days.

## Timeline

| Week | Track 1 | Track 2 | Track 3 |
|---|---|---|---|
| 1 | Rust preprocessor impl | GPU throughput test | (wait) |
| 2 | Roundtrip tests + 200 MB re-run | (done) | SmolLM2 fine-tune spike |
| 3 | Integration into dispatcher | — | Dispatcher + base-model benchmark |
| 4 | — | — | Decision gate write-up |

## Out of scope for Spike 6

- Multi-tenant LoRA serving (Tier-2 product work if Spike 6 passes)
- Base-model pretraining from scratch (fallback path only)
- Context-mixing ensemble (ruled out by throughput)
- Vertical-corpus collection (Harvey, Nabla) — parallel customer
  track, not a technical spike

## Why this ordering

Track 1 is the cheapest win and is **independent of architecture**;
its output (schema-stripped content stream) is an input to Track 3.
Track 2 is independent and has a one-shot compute cost, so run in
parallel. Track 3 is the load-bearing pivot and benefits from both
prior tracks' outputs.

Failure to hit Spike 6 gates still produces value: Track 1's
preprocessor ships regardless (lifts any codec); Track 2's GPU
numbers inform every future architectural decision.
