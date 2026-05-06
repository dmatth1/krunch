# T4 1 MB WildChat — time breakdown

Phase-level profile of compress + decompress on g4dn.xlarge (T4, sm_75)
with the `KRUNCH_HEAD_ASYNC=0 KRUNCH_3WAY_ASYNC=0` fallback path
(cp.async kernels are no-ops on sm_75 — see commit 3ab6ef8).

Reproduce: `scripts/profile_1mb_breakdown.py` with
`KRUNCH_CPP_PROFILE=1 KRUNCH_PHASE_PROFILE=1` (adds ~50 µs per layer
call, so wall is slightly higher than the bench's; phase shares are
correct).

Date: 2026-05-06.

## Compress — 25.16 s wall, 40.7 KB/s

16 chunks × 15K-17K tokens each, packed T=large forward per chunk.

```
Total wall: 25.16 s
├── rwkv4_layer_step_cpp  (12 layers × packed forward)  16.40 s  (65 %)
│   ├── ffn_V matmul   (M=T, K=3072, N=768)              4.81 s   29.3 %
│   ├── ffn_K matmul   (M=T, K=768,  N=3072)             4.71 s   28.7 %
│   ├── KVR matmul     (M=T, K=768,  N=768 × 3)          3.40 s   20.8 %
│   ├── Ow matmul      (M=T, K=768,  N=768)              1.32 s    8.0 %
│   ├── ffn_R matmul   (M=T, K=768,  N=768)              1.20 s    7.3 %
│   ├── WKV recurrence                                   0.63 s    3.8 %
│   └── LN1+LN2+premix3+premix2+sigmoid                  0.34 s    2.0 %
└── Outside layer-step                                   8.76 s  (35 %)
    └── head matmul (M=T, K=768, N=50277) + softmax + CDF + AC encode
        + tokenize + per-chunk .cpu() / empty_cache
```

**Compress is matmul-bound, ~92 % of wall is in matmul kernels** (layer +
head). FFN_V + FFN_K together = 38 % of wall — the natural consequence of
their 4× larger N or K than the other layer matmuls. Almost no Python or
launch overhead because the packed forward amortizes it across thousands
of tokens.

### What would move the compress number

- cp.async WMMA on sm_75 — not possible (hardware doesn't support it).
- Better small-`det_matmul_tc` for K∈{768,3072}, N=768 — ~1.7× microbench
  vs current single-warp landed via `det_matmul_tc_mw` for head matmul,
  but layer matmul (N=768) doesn't route to mw (gated on N≥16384). Lifting
  that gate is a candidate.
- KRUNCH_CUBLAS_PINNED=1 — skipped because cuBLAS is non-deterministic
  across M, breaks AC roundtrip.
- Fundamentally, T4 is not the gate hardware — A10G+ already gets cp.async
  routing and lands much closer to 200 KB/s.

## Decompress — 94.95 s wall, 10.8 KB/s

17,309 timesteps × B=16 chunks, stepped T=1 forward in lockstep batch.

```
Total wall: 94.95 s = 5.49 ms / step at B=16
├── rwkv4_layer_step_cpp  (12 layers × T=1)              74.89 s  (79 %)
│   = 4.32 ms / step
│   ├── ffn_V matmul   (M=16, K=3072, N=768)            26.48 s   35.4 %
│   ├── Ow matmul      (M=16, K=768,  N=768)            10.62 s   14.2 %
│   ├── KVR matmul     (M=16, K=768,  N=768 × 3)        10.32 s   13.8 %
│   ├── ffn_K matmul   (M=16, K=768,  N=3072)            9.50 s   12.7 %
│   ├── ffn_R matmul   (M=16, K=768,  N=768)             7.59 s   10.1 %
│   ├── LN1                                              3.87 s    5.2 %
│   ├── premix3                                          2.32 s    3.1 %
│   └── WKV + LN2 + premix2 + sigmoid_r                  4.19 s    5.6 %
└── Outside layer-step                                  20.07 s  (21 %)
    = 1.16 ms / step
    └── softmax + CDF + decode_step + .item() sync + Python loop
```

### Per-call analysis at M=16

Take `ffn_V_matmul_residual` as the worst offender:
- 207,708 calls (12 layers × 17,309 steps) → **128 µs / call**
- Theoretical compute: 16 × 3072 × 768 × 2 = 75 MFLOPs ÷ 65 TFLOPS = **0.6 µs**
- → **~99.5 % of every call is kernel launch / scheduling overhead**, not arithmetic.

Same shape applies to all 5 layer-matmul phases. The phase profiler is
catching this under "matmul" because cudaEvents measure total kernel
in-flight time — but the time is launch latency, not work.

So V1_PLAN.md's claim ("decompress is ATen launch-overhead bound") still
holds — it's just buried inside the matmul phase numbers in this profile.

### What would move the decompress number

Per-step on T4 today (B=16):
```
5.49 ms / step
├── layer-step (12 layers × ~13 ATen launches each)  4.32 ms  (78 %)
└── outside (softmax + CDF + decode_step + .item() + Python)  1.16 ms  (22 %)
```

1. **CUDA Graph capture of the per-step batched forward** —
   `forward_stepped_batched_graphed` (already in V1_PLAN backlog).
   Collapses 12 × ~13 ATen launches per step → 12 graph replays.
   Layer-step drops from 4.32 ms → ~120-300 µs (mostly compute floor).
   Outside-layer-step (1.16 ms) **does not shrink** — `.item()` sync to
   feed next token is autoregressive-irreducible.
   Realistic per-step: ~1.3-1.5 ms → **~3-4× decompress speedup → ~50 KB/s on T4**.
   On A10G B=128-640 the relative win is bigger (launches dominate per-step
   wall more) → projected **150-200 KB/s, the tier-3 unlock**.
2. Fused softmax+CDF kernel — saves one launch + one cumsum pass per
   step. ~5-8 % decompress, would help all GPUs.
3. Multi-step / speculative decode — would amortize the `.item()` sync
   across N tokens. Significant architectural change; saves the
   irreducible 1.16 ms above. Distant lever.

## Bottom line

| Path | Bottleneck (T4) | Bottleneck (A10G+) | Lever (T4) | Lever (A10G+) |
|---|---|---|---|---|
| Compress | layer matmul kernel speed | softmax+CDF + remaining matmul | extend `det_matmul_tc_mw` to layer shapes (~1.5×) | softmax+CDF fusion (~5-10 %) |
| Decompress | launch overhead (hidden in matmul phases) | launch overhead | graph-capture stepped forward (~3-4×) | graph-capture stepped forward (~4×) |

**Highest-ROI lever overall: graph-capture decompress on A10G.** Closes
the 4× gap to the 200 KB/s gate on gate hardware. T4 gets a smaller
side-benefit (~3-4× → ~50 KB/s) but won't clear the gate regardless;
T4 is not the gate hardware. Captured as a follow-up in V1_PLAN.md.

### Why ~130 KB/s on T4 was wrong

An earlier draft of this doc claimed graph capture would hit ~130 KB/s
on T4. That assumed the outside-layer-step bucket (softmax + CDF +
`.item()` + Python loop) was small. The phase profile shows it's
**1.16 ms per step = 21 % of wall**, and `.item()` is irreducible
(autoregressive sync). A 12× speedup on the 4.3 ms layer-step bucket
still leaves ~1.4 ms/step floor → ~50 KB/s on T4, not 130.

### Graph capture: implemented, T4-neutral, A10G-TBD (2026-05-06)

`forward_stepped_batched_graphed_v2` is now wired into
`_decompress_chunks_batched_cpp` and defaults ON when `KRUNCH_OWN_WKV=1`
(the graph-safety prerequisite). State tensors are now cached across
calls (`fresh_state_batched_cached`) since CUDA graphs are pointer-bound.

**T4 measurement (g4dn.xlarge, 1 MB WildChat, B=16):**
- Eager: 13.6 KB/s, byte_exact ✓
- Graph: 13.4 KB/s, byte_exact ✓
- `eager_eq_graph_bytes: True` — graph replay is bit-exact

**Result: 0.99× speedup on T4 — no win.** Why:

The diagnosis I had ("decompress = ATen launch-overhead bound") was only
partly right. `rwkv4_layer_step_cpp` already fuses all 13 ATen ops per
layer into one C++ call → eager mode pays just 12 Python→C++ calls/step,
not 156 ATen launches. CPU-side dispatch overhead was already low.
The remaining 4.3 ms/step layer-step time is **GPU-side kernel launch
latency** — each WMMA matmul at M=16 has ~50–100 µs of in-flight
scheduling latency from the GPU's hardware scheduler. Graphs reduce
host-side launch dispatch, but the GPU-side launch latency on T4
(sm_75, older scheduler) is largely irreducible by graphs.

A10G (sm_86) has a newer scheduler that pipelines captured graphs more
aggressively + B=128–640 batches push GPU launch latency down to a
smaller fraction of per-step wall. The expected 3–4× win still holds
for A10G+; we'll confirm next time we spin one up.

The change shipped as opt-in-default-on under `KRUNCH_OWN_WKV=1`. T4
neutral (no regression), A10G-TBD positive. Disable explicitly via
`KRUNCH_DECOMPRESS_GRAPH=0`.
