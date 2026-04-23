# Compression speed + cost optimization

> Started 2026-04-22. Goal: speed up neural-codec compression on the
> service backend and drive cost-per-GB into a defensible range for
> the regulated-vertical AI-chat-archive pitch.

## Goals

1. **Speed**: match and then exceed M1 throughput (currently ~170 KB/s
   on 8-core M1) on service-side compression. Stretch: **≥ 1 MB/s**
   on a GPU-backed path for both compression and decompression.
2. **Cost**: **≤ $0.05 per GB compressed** steady-state.
3. **Hard constraint**: scale-to-zero when the compression queue is
   empty. No long-lived idle instances billing the project.

## What's measured (2026-04-22)

All on M1 Mac 8-core, 200K distilled enwik8 model, 16K vocab, 2L × 96h,
via `./l3tc-rust/target/release/l3tc`. Corpus: first 1 MB / 256 KB of
`bench/corpora/enwik8_5mb`.

### End-to-end hybrid-compress

| Config | Throughput | Wall-clock (1 MB) |
|---|---|---|
| Full hybrid (neural+classical), 8 threads | **170 KB/s** | 6.1 s |
| Full hybrid, 4 threads (Fargate proxy) | 80 KB/s | 11.9 s |
| Full hybrid, 6 threads | 120 KB/s | 8.2 s |
| Classical-only (no neural) | 3400 KB/s | 0.3 s |
| **Fargate 4 vCPU today (per user)** | **~30 KB/s** | |

Fargate Graviton per-core is ~2.7× slower than M1 per-core on this
workload (80 KB/s on M1@4-thread vs 30 KB/s on Fargate@4-vCPU). That
gap is architectural (Graviton's weaker NEON + L2 bandwidth + the
vCPU-is-a-thread-not-a-core penalty), not a code defect.

### M1 local sweep (2026-04-22, 200K distilled, enwik8 corpus)

Full neural path, 8 cores:

| File size | Compress | Decompress |
|---|---|---|
| 1 MB | 170 KB/s | 195 KB/s |
| 5 MB | 164 KB/s | 194 KB/s |
| 10 MB | 166 KB/s | 192 KB/s |

Flat across sizes → forward-pass compute is the bottleneck and
rayon is already saturating cores at 1 MB. Decompression is
marginally faster (AC decode < AC encode). This is the neural
ceiling on M1 8-core.

Projection for 16 vCPU Fargate ARM Spot (Graviton2, per-core ~0.35×
M1 per-core): ~130-150 KB/s compress, ~160-180 KB/s decompress.
**That's well below the 1 MB/s goal.** If confirmed empirically
after the neoverse-n1 image rebuild, Track 2 (GPU) is required.

### Per-phase breakdown (sequential, 256 KB, no rayon)

| Phase | Time | % | Per-step |
|---|---|---|---|
| **Forward pass** (RWKV) | 9.37 s | **87.4%** | 121 µs/token |
| cum_freqs (softmax → freq table) | 1.34 s | 12.5% | 17 µs/token |
| AC encode | 8 ms | 0.07% | 0.1 µs/token |
| Other bookkeeping | 3 ms | 0.03% | |

Forward pass dominates. Within it, the head matvec (16384 × 96 INT8,
already NEON+INT8 tuned) is the single biggest per-token cost — this
matmul is memory-bandwidth bound, which is exactly Graviton's weak
spot relative to Apple silicon.

### Segment-size sweep (ratio vs throughput on 256 KB, M1, 8 threads)

| segment_bytes | ratio | throughput |
|---|---|---|
| 2048 | 0.1723 | 181 KB/s |
| **4096 (default)** | **0.1692** | 168 KB/s |
| 8192 | 0.1677 | 153 KB/s |
| 16384 | 0.1669 | 112 KB/s |

Ratio improves slowly with bigger segments; throughput falls because
fewer segments means less segment-level parallelism. 4096 stays the
right default.

## Cost reality check

Fargate ARM 4 vCPU + 8 GB ≈ $0.198/hr. At today's 30 KB/s, 1 GB =
9.3 hr = **$1.84/GB**. The target of $0.05/GB requires **≥ 1.1 MB/s**
on Fargate — 8× current. Not reachable from CPU code optimization
alone; needs a combination of infra + eventually a GPU path.

| Throughput | Wall per GB | Fargate ARM $/GB |
|---|---|---|
| 30 KB/s (today) | 9.3 hr | $1.84 |
| 150 KB/s (M1 parity) | 1.9 hr | $0.37 |
| 500 KB/s (CPU max ceiling) | 33 min | $0.11 |
| 1 MB/s (GPU minimum goal) | 17 min | $0.056 |
| 5 MB/s (GPU realistic) | 3.3 min | $0.011 |

## Ruled-out levers (with reasons)

- **Skip classical probe when neural wins**: classical probes are
  only ~5% of wall time under the neural-codec path. Saves little.
- **Full-model INT8 quantization on CPU**: maybe 1.3-1.5×. Real gain
  but the GPU path blows past this.
- **Port Metal batched.rs → CPU**: 2-3× on M1. Nice locally but
  doesn't close the $/GB gap on Fargate. GPU does.
- **Inferentia2**: best long-term $/GB, but a 3-4 week Neuron SDK
  port. Wrong phase of the product to spend that. Revisit after the
  first paying customer.

## Plan

### Track 1 — infra scale-to-zero with the cheapest CPU compute (fast)

Objective: drop $/GB without code changes, keep scale-to-zero.

1. **Fargate Spot**: switch the compression service capacity provider
   to Fargate Spot (70% off Fargate). Keep scale-to-zero. Zero code
   change beyond CDK. Expected: ~$0.55/GB at today's 30 KB/s, which
   is a 3× cost reduction from one line of CDK.
2. **Lift Fargate vCPU to 16** on that Spot capacity: same container,
   more cores. Test whether rayon scales — M1 data suggests linear
   up to the core count. Expected 4× throughput, i.e. ~120 KB/s.
   Combined cost per GB: ~$0.15/GB.
3. (Optional, measured): **EC2 Graviton c7g.4xlarge spot capacity
   provider** as an alternative to Fargate Spot 16 vCPU for even
   better cost per core. ASG min=0, ECS Managed Scaling handles the
   drain. ~2-3 min cold-start latency acceptable given compression is
   async from ingest.

### Track 2 — GPU path on g5g (real speed)

Objective: cross 1 MB/s AND $0.05/GB simultaneously.

1. Port the existing Metal backend (`l3tc-rust/src/backend/mtl.rs`
   + `backend/batched.rs`) to CUDA + cuBLAS. Most of the design work
   (batched-session lane lockstep at batch=256, FP parity handling,
   tokenizer-side data flow) is already done and tested for Metal.
2. New ECS task type: **g5g.xlarge** (Graviton host + 1× T4g GPU,
   ~$0.42/hr on-demand, spot ~$0.15/hr). ASG min=0, scale-to-zero
   same pattern as Track 1.3.
3. Route compression via the GPU worker when a model is available;
   keep the CPU worker as fallback for zstd-only and small jobs.
4. Target: ≥ 5 MB/s on a 1 MB chunk at batch=256 segments. Expected
   $0.008-0.03/GB depending on occupancy.

### Track 3 — decompression (later)

Same shape as compression: segment-parallel, per-segment sequential
RWKV. The batched-GPU path reuses the same kernel; measure and ship
after Track 2 compression is validated.

### Non-goals for this work

- Inferentia2 port (defer to post-first-customer)
- Any compression ratio change — this is purely a throughput/$ effort
- Training-side optimization (different spike)

## Working log

### 2026-04-22 profile run results

Captured above. See `## What's measured`.

### 2026-04-22 Baseline on current Fargate (4 vCPU ARM64, on-demand)

End-to-end empirical test. 1 MB enwik8 slice PUT to S3, SQS message
submitted, service auto-scaled from 0→1 task, compressed, emitted EMF.

| metric | value |
|---|---|
| Input | 1,048,576 B |
| Output | 186,522 B (ratio 0.1779) |
| Compression throughput (EMF) | **0.0308 MB/s = 30.8 KB/s** |
| Ratio | 0.1779 (savings vs zstd-22 shadow: +40.4%) |
| Codec | neural (1/1 chunks) |
| Cold start (SQS in → task running) | ~170 s |
| Compress wall time | ~33 s |
| End-to-end (SQS in → S3 compressed out) | ~240 s |
| Task | `d10f65fae59349589600002745499f3b` |

**30.8 KB/s matches the user's prior estimate.** The cold-start latency
is 3-4× the compression time for a 1 MB job — that dominates $/GB for
small jobs and argues for batching many messages per scale-up cycle.

### 2026-04-22 Track 1.1 + 1.2 deployed together: Fargate Spot + 16 vCPU

Combined change because each requires a new task-definition revision
and we want one deploy round-trip:

- `enableFargateCapacityProviders: true` on cluster
- `capacityProviderStrategies: [FARGATE_SPOT weight=1 base=0]` on service
- cpu: 4096 → 16384; memory: 8192 → 32768 (Fargate's minimum at 16 vCPU)

Expected: ~4× throughput from cores (if rayon scales as on M1) + ~70%
cost drop from Spot. Combined: ~$0.14/GB at 120 KB/s estimated.

**First deploy measured result (MISLEADING — hit a SIGILL bug):**

| File | Throughput | Codec actually used |
|---|---|---|
| 1 MB | 0.82 MB/s | **zstd_fallback** (not neural!) |
| 5 MB | 1.08 MB/s | zstd_fallback |
| 10 MB | 2.43 MB/s | zstd_fallback |

Post-mortem: the worker emitted `rc=-4 (SIGILL); falling back to zstd`
for every message. Root cause: the image was built with
`RUSTFLAGS="-C target-cpu=neoverse-v1"` (Graviton3). Fargate Spot
lands on Graviton2 (neoverse-n1) for these task sizes, and the
LLVM-emitted instructions that require v1-only features crash the
process. The original 4 vCPU on-demand baseline happened to land on
Graviton3-class hardware, which is why it succeeded with neural —
pure placement luck.

The throughput numbers above are **real for zstd_fallback** (~3400
KB/s CPU ceiling, amortized across 2 parallel tasks + network),
and confirm that the Fargate configuration itself (Spot, 16 vCPU)
works. But we haven't yet measured the actual neural path on this
config.

### Fix + rebuild

- Changed Dockerfile `RUSTFLAGS` to `-C target-cpu=neoverse-n1`.
  Graviton3 is a superset of Graviton2; a neoverse-n1-compiled
  binary runs on both. Our hand-written NEON kernels in `tensor.rs`
  don't use any n1-vs-v1-differential instructions, so the 10×
  matvec speedup from Phase 2.5a is preserved.
- Triggered CodeBuild (`krunch-image-build`) from commit
  `29f8620` to produce a new image. Build in progress.
- After build completes: redeploy with new image tag, re-run
  neural-path measurements, update this log.

### Remaining status (pre-neural-remeasurement)

| Goal | Target | Status |
|---|---|---|
| Speed (stretch) | ≥ 1 MB/s | ⏳ awaiting neoverse-n1 rebuild |
| Cost | ≤ $0.05/GB | ⏳ awaiting above |
| Scale-to-zero | required | ✅ Spot capacity, min=0 |
| zstd_fallback throughput | n/a | ~1-2.4 MB/s confirmed on Spot 16 vCPU |

### 2026-04-22 Re-measurement on 16 vCPU Spot + neoverse-n1 image

Image `krunch-compress-29f862005c9e` deployed. Neural codec
confirmed working (Codec=neural in EMF, ratio matches expected
0.178). Results:

| File | Throughput | Codec | $/GB @ $0.19/hr |
|---|---|---|---|
| 1 MB | 80 KB/s | neural | $0.68 |
| 5 MB | 90 KB/s | neural | $0.61 |
| 10 MB | 90 KB/s | neural | $0.61 |
| 100 MB | (still running — same plateau expected) | | |

Plateau at ~90 KB/s across file sizes on 16 vCPU Fargate Spot.

**This is disappointing.** Only 2.6-3× the 4 vCPU baseline despite
4× the vCPU and Spot pricing. The scaling breakdown:

- 4 vCPU on-demand on Graviton-likely-3: 30 KB/s (baseline)
- 16 vCPU Spot on Graviton2 + neoverse-n1: 90 KB/s

So 4× cores ≠ 4× throughput on Graviton2. The 16 vCPU instance
probably has less per-core memory bandwidth than 4 individual
tasks would (NUMA / shared L3 / memory-controller contention), and
the neoverse-n1 codegen may lose a few more percent vs neoverse-v1.
The forward pass is memory-bandwidth bound on the head matvec
(16384 × 96 INT8 weights ≈ 1.5 MB streamed per token), which is
precisely where Graviton2's narrower memory subsystem loses ground.

At **90 KB/s / $0.61/GB**, we miss both top-level goals by an
order of magnitude. CPU Fargate is not a path to ≥1 MB/s at
<$0.05/GB.

### Track 1 verdict — CPU alone is not enough

Fargate Spot 16 vCPU on Graviton2 is a real improvement (3× the
throughput and ~3× the $/GB), but it is not a path to either
top-level goal. Keep the config as-is for day-to-day ops (it's
3× cheaper and faster than baseline) and move to Track 2.

### 2026-04-22 Track 2 (GPU) — starting

Approach: the repo **already has a Metal GPU backend** (`l3tc-rust/
src/backend/mtl.rs` + `batched.rs`, 5,000+ lines) that handles
batched-segment forward passes at `batch_size=256` with ULP-parity
bookkeeping (`FLAG_GPU_ENCODED`). Porting this to CUDA for a T4G-
class GPU is real work, but the design is proven.

**First step before writing any CUDA**: build the Metal path on
M1 and measure. M1 GPU is the same ballpark as T4G (similar
int8/fp16 TFLOPS, comparable memory bandwidth). If batch=256 on
M1 Metal hits ≥1 MB/s, we know the pattern works and the CUDA
port is pure engineering effort, not a research bet. If it
doesn't, we revisit the whole approach before spending days on a
port.

### M1 Metal measurement (2026-04-22)

Built `l3tc-rust` with `--features metal` locally and ran the
same 200K model over the same enwik8 slices:

| Config | 5 MB wall | Throughput |
|---|---|---|
| Metal batch=256 workers=1 (tuned default) | 39.2 s | **128 KB/s** |
| Metal batch=256 workers=4 | 47.5 s | 105 KB/s |
| Metal batch=512 workers=2 | 48.6 s | 103 KB/s |
| Metal batch=1024 workers=1 | 59.3 s | 84 KB/s |
| CPU NEON (reference) | 30.5 s | **164 KB/s** |

**M1 Metal is 25-50% SLOWER than tuned M1 CPU NEON** on the 200K
model at every tuning point. More workers and bigger batches made
it worse, not better. This is not a bug; it's the regime the model
is in:

- Per-token forward pass is ~300 µs of compute
- Per-token GPU dispatch + buffer prep is ~500 µs  
- Batching across 256 segments amortizes dispatch across lanes, but
  the per-lane compute is small enough that the GPU can't hide
  kernel-launch latency even at the amortization knee
- More workers introduces command-queue contention on one GPU

This pattern generalizes: a T4G GPU on g5g.xlarge would face the
same regime (same compute:dispatch ratio), so **porting Metal to
CUDA would not help**. I'm not doing the port.

### Honest verdict on both tracks

| Platform | Neural throughput | $/GB | Source |
|---|---|---|---|
| M1 8-core NEON (ceiling) | 170 KB/s | — (local) | measured |
| M1 GPU (Metal, batch=256) | 128 KB/s | — (local) | measured |
| Fargate 4 vCPU on-demand (baseline) | 31 KB/s | $1.84 | measured |
| **Fargate 16 vCPU Spot + neoverse-n1 (deployed)** | **90 KB/s** | **$0.61** | measured |
| Projected Fargate 32 vCPU (ceiling?) | ~150 KB/s | ~$0.45 | extrapolated |
| Projected T4G GPU on g5g.xlarge | likely ≤128 KB/s | ~$0.90 | inferred from M1 |

The 1 MB/s and $0.05/GB goals are **not reachable with the current
200K-model architecture**. The physics:

- Each byte compressed requires ~1-2 tokens through RWKV forward
- Each token takes ~300 µs minimum (memory-bound head matvec on 1.5 MB
  INT8 weights) even with perfect SIMD
- 1 MB/s ≈ 200-400 k tokens/s ≈ 2.5-5 µs/token across all cores
- Even with 64 cores that's 160-320 µs/core, just barely above the
  floor — and we'd need perfect scaling, which Graviton doesn't give

### Three honest options to present

1. **Accept the 3× improvement and ship current config.** 90 KB/s /
   $0.61/GB is real progress from 30 KB/s / $1.84/GB, and good
   enough for early design-partner customers archiving at moderate
   volumes. Revisit speed/cost after real customer usage data.

2. **Change the compression target: allow the dispatcher to drop
   neural on datasets where the ratio win is small.** We saw on
   Spike 3 that neural isn't always the right pick. For datasets
   where classical (bzip3) comes within a few pp of neural, skip
   neural and get 3400 KB/s + $0.02/GB. Tier the product:
   dataset-by-dataset, pick ratio-first or speed-first.

3. **Bigger model (1M-10M params) where GPU actually pays off.**
   At 10M params the GPU regime flips — per-token compute rises
   into the ms range, GPU dispatch becomes negligible. A 10M model
   on T4G could easily do 3-5 MB/s and deliver a better ratio on
   dialogue. But this is a Spike 5-scale commitment (weeks of
   training + eval), not a week of engineering.

The cheap-and-right answer for now is probably **(1) + (2)
together**: ship the 16 vCPU Spot config, and add a per-dataset
codec policy so we spend the neural budget where it materially
wins. Defer (3) until a customer commits.

### 2026-04-22 Decision: commit to (3) — bigger model + GPU

After discussion, decided to pursue the bigger-model + GPU path
(renamed Track 3 to distinguish from the dead Track 2). Before
committing to plan, a colleague review caught five errors in my
back-of-envelope math, which materially changed the expected
outcome. Fixing them so the plan isn't built on wrong numbers.

#### Corrections to my earlier math

| Claim (mine) | Reality (reviewer + our measurements) |
|---|---|
| CPU per-token at 200K = 120 µs | **30 µs measured** (170 KB/s ÷ 8 cores ÷ ~5 B/tok). CPU baseline is 4× cheaper than I wrote. |
| GPU dispatch overhead = 500 µs | 500 µs is naive launch; persistent command graphs (CUDA Graphs / MPSGraph, which our Metal path already uses) drop it to ~20 µs. |
| Batching amortizes dispatch 256× | RWKV's recurrent state + per-lane memory bandwidth mean **real speedups from batching are 20-50×**, not 256×. Our own Metal measurement saw ~4× from batch=256. |
| ~3 B/token SPM | **4-5 B/token** on our tokenizer. enwik8 ≈ 3.5, GH events 5.0, WildChat est 4-5. |
| "T4G GPU" | **T4G is Graviton2 CPU, not a GPU**. g4dn has T4, g5 has A10G, g6 has L40S. Name-confusion fix. |

#### Corrected crossover picture

Per-token latency at different model sizes (measured CPU + realistic
batched GPU with persistent command graph):

| model | CPU/token (measured-derived) | GPU batched/token (realistic) | winner |
|---|---|---|---|
| 200K | ~30 µs | ~20 µs (dispatch-dominated) | ~tie |
| 1M | ~150 µs | ~25 µs | GPU 6× |
| 10M | ~1.5 ms | ~50 µs | GPU 30× |
| 100M | ~15 ms | ~200 µs | GPU 75× |

Crossover is ~200-500K params (lower than my initial 1M claim).
GPU advantage at 10M is bigger than I first wrote. The dead Track
2 thesis was "200K on GPU beats 200K on CPU" — measured wrong.
**Track 3 thesis is "10M on GPU beats 200K on CPU"**, which the
math actually supports.

#### Recalibrated targets

Old optimistic extrapolation: 1.1 MB/s on 10M/T4G.
Reviewer's realistic: ~300 KB/s without serious tuning, 500 KB/s-
1 MB/s with a tuned persistent-command-graph backend + honest
batching.

- **Commit**: ≥ 500 KB/s compression at ≤ $0.17/GB on g6.xlarge spot
- **Stretch**: 1 MB/s at ~ $0.08/GB

The ratio improvement (est 20-30% from 200K → 10M on dialogue per
LMCompress scaling laws) is the stronger economic argument than raw
throughput. Ratio compounds across the archive's lifetime; speed is
mostly a per-job latency and one-time cost.

### What needs to be in place for Track 3

Five workstreams:

**1. Bigger model trained on chat data.** Pick a 10M-class RWKV-v4
config (likely 4-6 layers × 384h, vocab stays 16K). Existing
`scripts/train_l3tc_phase11.py` + `archive-dev-training` Batch on
g6.xlarge — only config bump needed. 10M × 10 epochs on 200 MB
WildChat should finish in 4-8 hours, ~$10-20. **Gate: ratio ≤ 0.15
on held-out WildChat; if ratio barely moves, abandon.**

**2. CUDA inference backend in `l3tc-rust`.** Today the neural
codec in `dispatcher.rs` hard-codes `crate::codec::compress` (CPU-
only). Options:
- `cudarc` + hand-port Metal kernels to CUDA (~1 week, full control)
- `candle` crate (has RWKV kernels already, but heavier deps, less
  control over batched-session pattern)
Leaning `cudarc`. **Persistent command graph is mandatory** — this
is what keeps dispatch under 50 µs and makes the Track 3 math
actually hold. Add `cuda` Cargo feature mirroring existing `metal`.

**3. ECS EC2-GPU compute with scale-to-zero.** Fargate doesn't
support GPU. Need EC2 launch type + capacity provider + ASG:
- Instance: `g6.xlarge` spot (L40S, 48 GB VRAM, ~$0.30/hr spot)
- ASG `min=0, max=N`, ECS Managed Scaling + Managed Termination
  Protection
- Spot + capacity-optimized allocation
- Image: NVIDIA CUDA runtime base + our Rust binary (~3 GB)
- Cold start estimate: 3-5 min (EC2 boot + image pull). Slower than
  Fargate's ~2 min.

**4. Routing / metadata.** Per-dataset metadata gets `backend`
field. Training pipeline routes >1M-param models to GPU queue.
Either a second SQS queue `archive-{env}-compression-gpu` or a
single queue with worker-side assertion. Leaning two queues so
CPU and GPU scaling are independent.

**5. CPU service stays.** Small zstd-only jobs and datasets
where neural doesn't win stay on the Fargate 16 vCPU Spot config.
Don't over-engineer: the CPU path is already deployed and doing
~90 KB/s / ~$0.61/GB; it's fine for small classical-only jobs.

### Proposed spike structure

| Day | Task | Gate |
|---|---|---|
| 1 | Model config + training on WildChat | Ratio ≤ 0.15 on held-out |
| 2-3 | `cudarc` BatchedSession port + persistent graph | ≥300 KB/s on g6.xlarge |
| 4 | Wire CUDA path into NeuralCodec + hybrid-compress | Round-trip tests pass |
| 5 | CDK: EC2-GPU capacity provider, ASG min=0, new queue | `cdk deploy` clean |
| 6 | Deploy to dev, end-to-end measurement | ≥500 KB/s empirically |
| 7 | Writeup, go/no-go | decision |

Budget ~$30-50 of AWS spend. Start with **Day 1 (train the bigger
model)** because that kills the whole idea cheapest if ratio
doesn't improve. If ratio wins, engineering is just execution.
If it's flat, save a week of CUDA porting.

### Session learnings (non-technical)

Worth keeping for future work on this kind of problem:

- **Measure before extrapolating.** My first Track 2 estimate
  claimed 1.1 MB/s on 10M GPU — off by 4× on the CPU baseline and
  5× on batching amortization. The 500 µs GPU overhead quote came
  from memory, not a measurement on our codebase. Using our own
  measurements would have caught these errors earlier.
- **A colleague review is worth more than extra self-polishing.**
  All five errors were caught by one review pass, not by me
  writing more carefully.
- **Instance-family name confusion is common.** g4dn=T4, g5=A10G,
  g6=L40S, g5g=Graviton2+T4G (where T4G is a GPU tile on Graviton2
  chassis — which I mistakenly called T4G and treated as on g5g,
  a separate mistake). Look up the actual chip before quoting.
- **Persistent command graphs are not a luxury for tight inference
  loops.** They're what makes the GPU regime work for autoregressive
  per-token inference at small model sizes. Writing "GPU" without
  checking whether we have CUDA Graphs / MPSGraph is the difference
  between 500 µs and 20 µs per step.
- **Framing commit-vs-stretch targets is better than single number.**
  "1 MB/s" invites over-promising. "Commit ≥500 KB/s, stretch 1 MB/s"
  gives room for honest measurement without reneging.
- **Ratio compounds; speed amortizes.** For a storage-as-a-service
  product, a 20% ratio win is worth more than a 20% speed win
  because ratio affects every byte forever and speed only affects
  the one-time compression cost. Lead product conversations with
  ratio when they're both available.

### Remaining open items

- **Decompression benchmark** (partially measured 2026-04-22):
  M1 8-core, local, neural 1 MB blob: **195 KB/s** (5.37 s wall,
  50.71 s user — similar 7× rayon scaling as compression). Slightly
  faster than compression on the same hardware because AC decode
  is a bit cheaper than AC encode. Fargate neural decompression
  throughput TBD (pending the neoverse-n1 rebuild for round-trip
  testing).
- **Cross-host decode compatibility issue**: a Fargate-encoded neural
  blob (`186,522 B`) from the baseline 4 vCPU on-demand run failed
  to decode on M1 with `tokenizer: decode: token stream has more
  <unk> tokens than unk payloads`. A locally-encoded blob of the
  same input round-trips fine. Suggests a tokenizer version or ULP
  drift between the Fargate image binary and the local release
  binary. Orthogonal to this work but should be investigated
  (independent of the Spot/16-vCPU change).
- **Larger-file behavior** (100 MB+): chunk count (= file bytes /
  1 MB) will exceed core count by 6×+. Expected to plateau at
  whatever single-chunk speed each core delivers × 16 / (1 + overhead).
  Empirically extrapolating from 10 MB → ~3 MB/s plateau at worst.
- **Chat-shape data (WildChat, OASST2) once models exist**: neural
  ratio + throughput may differ slightly from enwik8. Rerun the same
  test once Spike 4 lands trained models in S3.
- **Cold-start latency**: 3 min to spin up a Spot task is painful
  for small jobs. Mitigations: batch SQS messages, or keep min=1
  during business hours (costs ~$4/day). Defer unless customers
  complain.

### 2026-04-22 Track 2 scoping

(pending after Track 1 measurement)
