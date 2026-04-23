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

The preprocessor runs at **both training time and compression time**
— if training data is raw NDJSON but compression-time input is
stripped, the model sees a distribution shift and arithmetic coding
wastes bits. Tokenizer is also trained on the content stream, not
raw NDJSON, so merges optimize for real content rather than JSON
framing noise.

Consequence: Spike 5's `v1.bin` (trained on raw NDJSON) cannot be
re-used for preprocess-then-neural measurement. We'd need to retrain
an L3TC-12M on the preprocessed corpus to get an apples-to-apples
number — but that's Spike 5 compute all over again. So Track 1's
neural measurement is deferred; the primary gate is the
classical-codec-only stack:

| stack | ratio | vs baseline | status |
|---|---|---|---|
| zstd alone | 0.167 | reference | measured |
| dispatcher alone (Spike 5 model, raw NDJSON) | 0.178 | — | measured |
| **preprocess + zstd** | **TBD** | expected 5–15% lift | **Track 1 gate** |
| preprocess + bzip3 | **TBD** | similar | secondary |
| preprocess + dispatcher + *new* 12M model | TBD | deferred — needs retrain | out-of-scope |

### Gate

Declare success if **preprocess + zstd** ratio ≤ 0.155 (≥7% better
than bare zstd). That proves the preprocessor has standalone value
independent of any neural model — and importantly, that value
transfers to Track 3 (SmolLM2 + LoRA fine-tunes on the same
content stream, gets the same structural lift for free).

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

## Track 3 — Pretrained RWKV base + per-tenant LoRA (C, revised)

**Revision note (2026-04-23):** originally scoped as
SmolLM2-360M (transformer) + LoRA. After published-throughput
research, pivoted to **pretrained RWKV** as the base. Reasoning:

- L3TC paper (RWKV-4 + HiRA) reports **4.35 MB/s on A100** — 14× the
  300 KB/s gate.
- ts_zip (Bellard, RWKV-4 169M) reports **1 MB/s on RTX 4090** — 3×
  the gate.
- Transformer-based neural compressors (FineZip/Llama-3-8B,
  LLMZip/LLaMA-7B) are **0.67 KB/s and 0.012 KB/s respectively** —
  orders of magnitude below the gate.
- No published evidence of Mamba for AC text compression; the
  "Mamba is faster" claim doesn't translate directly to AC
  workloads.

RWKV is the demonstrated fast-GPU-compression substrate.
Transformer LMs would be a first-of-kind engineering bet.

**Starts after Tracks 1 + 2 complete.** L3TC trains from scratch
per-dataset; we flip to: one shared pretrained RWKV base + tiny
per-tenant LoRA adapter.

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

### Base model candidate evaluation (revised — RWKV only)

| candidate | params | tokenizer | license | pretrain |
|---|---|---|---|---|
| RWKV-4-169m-pile | 169M | GPT-NeoX 50K | Apache 2.0 | ~330B tokens (The Pile) |
| RWKV-4-430m-pile | 430M | GPT-NeoX 50K | Apache 2.0 | ~330B tokens |
| RWKV-5-world-1.5B | 1.5B | RWKV World 65K | Apache 2.0 | multilingual |
| RWKV-6-world-1.6B | 1.6B | RWKV World 65K | Apache 2.0 | multilingual |

**Default pick: RWKV-4-169m-pile.** Smallest, fastest decode, same
family as ts_zip (proven 1 MB/s at this exact config). HF hub:
`RWKV/rwkv-4-169m-pile`. English-Pile pretraining matches our
WildChat-English corpus. Upgrade to 430m if 169m ratio misses gate.

Skipping transformer candidates (SmolLM2, Qwen, Phi, TinyLlama) —
published throughput numbers rule them out for our 300 KB/s gate.

### Fine-tuning recipe (per tenant)

- Objective: next-token LM (not instruction-tuning — we don't need
  dialogue behavior, we need calibrated next-token probabilities)
- LoRA: rank 16, alpha 32, all attention+FFN linears
- LR: 1e-4, cosine schedule, no warmup
- Epochs: 2–3 over tenant corpus
- Data: raw corpus (post-schema-preprocess content stream from Track 1)
- Target train: val CE gap < 0.3 nats (keep adapter underfit)

### Spike 6 experiments (revised 2026-04-23 after RWKV pivot)

**Simpler scope.** After establishing that RWKV-4-169m-pile raw is
the likely product (zero-shot dry-run ratio 0.11 already beats
everything classical, and ts_zip proves the same model runs at 1
MB/s on RTX 4090), the spike narrows to two measurements:

**P1. RWKV-4-169m-pile zero-shot ratio on WildChat-English ≥10 MB.**
Previous sample was 2 MB. At 10+ MB the result is trustworthy for a
product decision. Apply Track 1 preprocessor → content stream, take
the tail 20 MB, forward-pass, compute entropy-bound ratio. Gate:
ratio ≤ 0.165 (25% better than zstd-19 at 0.167). Stretch: ≤ 0.115
(matches our dry-run).

**P2. RWKV-4-169m-pile decode throughput on A10G, bf16.** Use
`scripts/smollm2_throughput_test.py` (works on any HF CausalLM)
pointing at `RWKV/rwkv-4-169m-pile`. Expected: 500–700 KB/s based
on ts_zip's 1 MB/s on RTX 4090 scaled to A10G. Gate: ≥ 300 KB/s.
Stretch: ≥ 1 MB/s (matches ts_zip).

**Dropped from scope:** L3TC-12M throughput measurement and
RWKV + LoRA fine-tune. Rationale:

- L3TC-12M number is no longer load-bearing; the product candidate
  is RWKV-4-169m-pile (not L3TC from-scratch). Re-litigate later if
  needed.
- LoRA is premature optimization if raw zero-shot already hits the
  gate. Per-tenant fine-tuning adds training cost, adapter storage,
  versioning complexity, and onboarding latency for maybe 10–20%
  ratio improvement. Defer to v2.

### Deliverables

Results land in `s3://archive-dev-archive/spike6/gpu-throughput/`:
- `rwkv_zeroshot.json` (ratio + bits-per-token on ≥10 MB held-out)
- `rwkv_throughput.json` (KB/s on A10G bf16)

### Decision gates (post-measurement)

- **Both P1 and P2 pass gates** → Krunch v1 substrate is
  RWKV-4-169m-pile used raw. Next work: Rust adapter for HF RWKV,
  GPU decode path in `l3tc-rust`, dispatcher integration.
- **P1 passes, P2 misses** → ratio works, speed doesn't. Investigate
  the 2–5× speed lever stack (int8, torch.compile, CUDA graphs,
  custom CUDA kernel).
- **P1 misses** → larger base needed (RWKV-4-430m or 1.5B). Rerun
  with same harness; cost scales proportionally.
- **Both miss** → reconsider whether a general-chat benchmark is
  the right proving ground (pivot to vertical corpora: legal docs,
  medical notes, per-tenant chat logs).

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
