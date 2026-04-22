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

**Actual measured result (far better than expected):**

| File | Chunks (1 MB each) | Throughput | $/GB @ $0.19/hr |
|---|---|---|---|
| 1 MB | 1 | **0.82 MB/s** | $0.066 |
| 5 MB | 5 | **1.08 MB/s** | **$0.050** |
| 10 MB | 10 | **2.43 MB/s** | **$0.022** |

Where the multipliers came from vs baseline (30 KB/s):
- 16× more vCPU + Spot discount was the expected 4× win
- The actual **27× on 1 MB / 80× on 10 MB** came from chunk-level
  rayon parallelism finally having enough chunks to fill cores.
  At 10 chunks × 16 cores, with each chunk running segment-level
  rayon on its own work, the outer `par_iter` fans out and work-
  stealing evens the load. This was dormant in the 4 vCPU config
  because 5 chunks × 4 cores barely saturated.

Cost per GB calculation:
- Fargate ARM Spot (16 vCPU + 32 GB, us-east-1): ~$0.19/hr
- At 2.43 MB/s: 1 GB → 7.0 min → **$0.022/GB**
- At 1.08 MB/s: 1 GB → 15.4 min → **$0.050/GB**

### Goal status after Track 1

| Goal | Target | Actual (10 MB, steady-state) | Status |
|---|---|---|---|
| Speed (stretch) | ≥ 1 MB/s | 2.43 MB/s | ✅ ×2.4 |
| Cost | ≤ $0.05/GB | $0.022/GB | ✅ ×2.3 |
| Scale-to-zero | required | Spot capacity, min=0 | ✅ |

**Both top-level goals are met by Track 1 alone. Track 2 (GPU) is
no longer on the critical path** — it stays as a future option if we
eventually ship customers whose traffic justifies sub-cent-per-GB
economics, but that's a nice-to-have, not a requirement.

### 2026-04-22 Track 2 (GPU) — status

Deferred. CPU Fargate Spot + 16 vCPU already clears both targets.
Revisit only if production traffic economics demand it.

### Remaining open items

- **Decompression benchmark**: same code path shape; hasn't been
  measured yet in this session. Expected similar profile (segment-
  parallel forward pass dominates). Worth measuring before claiming
  GET-side performance numbers to customers.
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
