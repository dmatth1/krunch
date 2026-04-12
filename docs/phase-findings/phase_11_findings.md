# Phase 11 findings — broader corpus training

## Summary

Phase 11 trained the same 200K RWKV-v4 + HiRA architecture on
10 GB of the Pile dedup (Wikipedia + books + code + ArXiv +
StackExchange + web) to test whether a single broader-corpus
model could fix the OOD cliff while maintaining acceptable
in-distribution ratio.

**Result: the enwik6 hard floor (≤ 0.20) is broken.** The
broader model compresses enwik6 at 0.3156 — worse than bzip2.
The OOD improvement on webster is dramatic (1.26 → 0.49) but
the in-distribution regression is catastrophic. 200K non-embed
params cannot simultaneously cover Wikipedia and the broader
distribution at our ratio requirements.

## Ratio matrix

| corpus | Pile model (epoch 5) | Default 200K (enwik-only) | delta |
|---|---:|---:|---|
| **enwik6** | **0.3156** | **0.1699** | +86% worse (FLOOR BROKEN) |
| **webster** | **0.4886** | **1.2613** | -61% better (huge OOD win) |
| dickens | 1.0000 | 1.0000 | both raw-store (non-UTF-8) |

For context — classical compressor ratios on enwik6:
- bzip2-9: 0.2813
- xz-9e: 0.2907
- zstd-22: 0.3001

The broader model at 0.3156 is now **worse than all three
classical compressors** on enwik6. The entire value proposition
of the project (ratio better than classical) disappears for
Wikipedia-like text.

## Training details

- **Corpus:** 10 GB Pile dedup, 1.64M documents, 3.14B tokens
  (tokenized with enwik8-trained SPM vocab 16384)
- **Architecture:** RWKV-v4 + HiRA, 2L × 96H × 96I, rank 4
  (unchanged, per Phase 11 hard constraint #1)
- **Recipe:** AdamW (weight_decay=0.01), cosine warmup (500
  steps → peak 1e-4 → decay to 1e-6), bf16, batch 16,
  grad_accum 2 (effective batch 32)
- **Training:** 10 epochs × 500K samples planned, ~7 completed
  before auto-shutdown. 5 epoch checkpoints in S3.
- **Loss trajectory:** 9.86 → 4.87 (epoch 0) → 4.50 (epoch 2)
  → 4.39 (epoch 3) → 4.32 (epoch 5) → 4.30 (epoch 6).
  Converging fast, near floor by epoch 3.
- **Hardware:** g5.xlarge (A10G 24 GB), 12.44 it/s, ~42 min
  per epoch. Total wall: ~5 hours. Cost: ~$6 on-demand.
- **Memory:** 13 GB / 15 GB RAM (tight — the 14.7 GB tokenized
  file loaded as a 12.6 GB numpy int32 array). Worked but
  g5.2xlarge would be more comfortable for corpora > 5 GB.

## What the numbers mean

### The capacity finding

The broader model's loss (4.30 nats at epoch 6) is close to
the enwik8-only model's loss (4.26 nats at epoch 5 of the
recipe validation). **The model reached approximately the same
loss on both corpora.** But the enwik6 compression ratio is
drastically different:

- enwik8-only model at 4.26 loss → enwik6 ratio 0.2161 (5 ep)
- Pile model at 4.30 loss → enwik6 ratio 0.3156 (5 ep)

Same loss, very different ratios on the same test file. Why?
Because the Pile model's 4.30 loss is averaged over all domains
(Wikipedia, code, ArXiv, web). Its predictions on Wikipedia
text specifically are much worse than the enwik-only model's
predictions on the same text — the capacity that the enwik-only
model devoted to Wikipedia-specific patterns is now spread
across all Pile domains.

### Webster: the OOD win is real

Webster (41 MB, 1913 dictionary) went from 1.2613 (26% bigger
than raw) to 0.4886 (51% compression). The Pile includes
dictionary-like text and the model learned to predict definition
patterns. This confirms that broader training does improve OOD
performance — the issue is not "broader training doesn't help"
but "200K params can't do both."

### The 200K capacity ceiling

200K non-embed params (~4M total with embedding + head) is the
binding constraint. The model can learn ~one domain well (enwik8
→ 0.17 ratio) or spread itself thin across many domains (Pile
→ 0.49 on webster, 0.32 on enwik6). It cannot do both
simultaneously.

This is the outcome Phase 11's decision matrix predicted as a
possibility: "the architecture is too small; the problem is
capacity, not training."

## Decision per Phase 11 criteria

**Outcome #3: Floors break.** Phase 11 fails as a default-tier
replacement. The broader model cannot ship as the default
because it loses the ratio advantage over classical compressors
on in-distribution text.

**Escalation options:**

1. **Ship the broader model as an opt-in "generalist" tier**
   alongside the enwik-only default. Users who compress mixed
   input get 0.49 on dictionary text instead of 1.26. Users who
   compress Wikipedia-like text use the default and get 0.17.
   CLI: `--model checkpoints/l3tc_pile.bin`.

2. **Phase 5b: bigger model at the broader corpus.** Train a
   larger model (e.g., 800K or 3.2M non-embed params) on the
   Pile. More capacity should allow better per-domain
   predictions. Trade: slower compression (more FLOPs per
   token). The 3.2M model already runs at 26 KB/s vs 131 KB/s
   for the 200K — a broader-trained 3.2M might hit the sweet
   spot of "good ratio on everything at 26 KB/s."

3. **Phase 8: specialist dispatch.** Ship multiple small models
   (200K each, trained on different domains) and dispatch per
   file or per segment based on a lightweight classifier. Best
   ratio per domain but most engineering work.

4. **Hybrid: broader-trained 3.2M as default + enwik-specialized
   200K as fast tier.** The 3.2M has enough capacity for the
   broader distribution; the 200K stays as a speed-optimized
   tier for users who know their input is Wikipedia-like.

## Retained artifacts

- `s3://dmatth1-bnn-checkpoints/l3tc/phase11_pass2/` — 6 epoch
  checkpoints + train.log
- `s3://dmatth1-bnn-checkpoints/l3tc/corpora/train_pile_dedup.txt`
  — 14.7 GB tokenized Pile corpus (reusable for future runs)
- `s3://dmatth1-bnn-checkpoints/l3tc/enwik8_recipe_validation/`
  — enwik8 recipe validation checkpoints + log
- AMI `ami-07a4fc98c4ed4e19e` — pre-baked training environment
