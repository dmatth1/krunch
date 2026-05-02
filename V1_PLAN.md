# Krunch v1 — open-source neural codec for text (library + image + format spec)

**Decided 2026-04-24**: ship as an open-source framework, not a hosted service.
Hosted is **v2** (`V2_PLAN.md`).

## What v1 is

An open-source neural codec for text. Ships as a Python library, a Docker
image, and a documented blob format. Customers run it on their own
infrastructure (any NVIDIA GPU + container runtime) and parallelize across
machines using whatever batch system they already use — `krunch plan` emits
the config for AWS Batch, GCP Batch, k8s, Modal, Ray, Slurm, or local.
Customer's data never leaves their network.

Two ways to run, same Docker image:

**Single-shot** — container starts, processes one input, exits.

```bash
# 1. Install (~5-10 min one-time — downloads CLI + pulls 3.5 GB image)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash

# 2. Compress (or decompress) — instant, image is cached
krunch compress   < data.jsonl  > data.krunch
krunch decompress < data.krunch > data.jsonl
```

**Single-machine parallel — automatic.** `krunch compress` autodetects
GPUs/cores and uses a multi-process worker pool by default.

**Distributed across machines** — `krunch plan` emits a runnable artifact
for the user's existing batch system. We don't run it; we generate the
config and they execute it with their own credentials:

```bash
krunch plan --target aws-batch --source s3://… --dest s3://… --workers 16 > job.json
krunch plan --target k8s       --source s3://… --dest s3://… --workers 16 > job.yaml
krunch plan --target modal     --source s3://… --dest s3://… --workers 16 > run.py
# user runs: aws batch submit-job ... | kubectl apply -f ... | modal run ...
```

Workers read their byte range directly from object storage, write a partial
blob back, and a finalize task stitches partials. No krunch control plane,
no krunch credentials.

That's the whole product surface for v1: a library + Docker image, a CLI
(`compress` / `decompress` / `plan` / `bench`), a documented blob format,
and a gallery of examples for popular batch frameworks.

## Repositioning (2026-04-30): codec, not framework

Earlier framing called krunch a "distributed neural compression framework."
We are not building a batch system — Modal/Ray/AWS Batch/k8s/GCP Batch/Slurm
already do that part well. What we ship is a neural codec; orchestration is
a templated artifact, not a maintained product.

What ships:
1. **A neural codec library** — `from krunch import Codec; codec.compress(bytes)`
2. **A Docker image** — GPU-baked packaging of the same library
3. **A documented blob format spec** — RFC-style, anyone can implement a decoder
4. **A container env-var contract** — the integration interface for any batch system
5. **`krunch plan`** — emits ready-to-run artifacts for AWS Batch, GCP Batch, k8s, Modal, Ray, Slurm, local
6. **`krunch bench`** — `--corpus path` reports ratio + compress/decompress KB/s on user's data
7. **An examples gallery** — short snippets per framework

What we **stop** owning: the AWS Batch deployer is one example, not the
headline. `krunch submit` is deprecated in favor of `krunch plan` +
user-side execution.

### Container env-var contract (the real interface)

```
KRUNCH_INPUT_URL    s3://…  |  gs://…  |  https://…  |  file://…
KRUNCH_OUTPUT_URL   destination prefix; worker writes to <prefix>.parts/<index>
KRUNCH_PART_INDEX   0-based, injected by the framework
KRUNCH_PART_COUNT   total worker count
KRUNCH_INPUT_LEN    total input bytes (read at split time)
KRUNCH_MODE         compress | decompress | finalize
```

On startup the worker: HEADs the source for size → computes its byte range
`[i * size / N, (i+1) * size / N)` → ranged GET → compress chunks → writes
to `parts/<index>`. Finalize task stitches partials and writes the final
header / footer.

### Splitting + stitching across plan targets

| target | worker indexing | finalize dependency |
|---|---|---|
| AWS Batch | `AWS_BATCH_JOB_ARRAY_INDEX` | `dependsOn: [{type: SEQUENTIAL}]` |
| GCP Batch | `BATCH_TASK_INDEX` | `taskGroups` ordering |
| K8s | `JOB_COMPLETION_INDEX` | second Job; init-container wait |
| Modal | function param `i` | `for_each(shards); assemble()` |
| Ray | `ray.remote()` task index | `ray.get([…]); assemble.remote(parts)` |
| Slurm | `SLURM_ARRAY_TASK_ID` | `sbatch --dependency=afterok:$JOBID` |
| local | bash `for i in $(seq …)` | sequential after worker loop |

Caveat: requires a seekable, ranged-readable source. Stdin can't be split —
distributed mode requires URL input.

## Multi-target, not AWS-only

Same Docker image runs anywhere NVIDIA GPUs + a container runtime exist.
Hardware contract: NVIDIA GPU with CUDA 12.x driver, ≥16 GB VRAM (A10G / L4 /
A100 / H100 all work), container runtime with GPU access, model + tokenizer
baked into image (outbound network optional).

| target | shipped at v1 |
|---|---|
| AWS Batch / GCP Batch / k8s / Modal / Ray / Slurm / local | ✅ |
| Azure Batch / Databricks / Argo | community PRs welcome (~30 LOC each via env-var contract) |

## Differentiation vs DIY

A competent engineer could build this from public components in 2-4 weeks.
The framework collapses that to 30 minutes:

| concern | DIY | krunch |
|---|---|---|
| Pick base model | research + benchmark | RWKV-4-Pile-169M (validated) |
| Pick inference path | trial-and-error | BlinkDL/RWKV-LM (HF transformers is a trap) |
| Compile WKV kernel | apt+nvcc+ninja debug loop | image is pre-built |
| Pick AC coder | choose between 5 libraries | range coder built-in (GPU + bit-exact) |
| Chunking strategy | guess | 64 KB validated default |
| GPU lifecycle / scaling | manual | `krunch plan` emits config for any batch system |
| Model version safety | break old archives on upgrade | version-pinned blob format, migration tooling |

Plus the Spike-6 traps documented in `CLAUDE.md` — the framework hides every
one behind a working default.

## License: Apache-2.0

Maximum adoption (enterprise legal teams approve faster than GPL/AGPL/SSPL),
compatible with v2 hosted business, patent grant + attribution prevents
trivial whitewash forks. Trade-off: "AWS can't fork us" defense (vs SSPL/BUSL)
is given up; mitigation is the v2 vertical-LoRA registry as moat.

## Distribution

- **GitHub**: `github.com/dmatth1/Krunch` (private; flips to public at v1 launch)
- **Container image**: `ghcr.io/dmatth1/krunch:v1`
- **Docs site**: GitHub Pages or Vercel from `docs/`
- **Discoverability**: deferred until artifact ships

## Roadmap to 1.0

| Tier | scope | status |
|---|---|---|
| **T1: Quick checks** (free) | unit tests, plan-template lint, `--dry-run` CI | ✅ |
| **T2: Local CPU integration** (free) | full neural path on CPU, byte-exact roundtrip on 224 B sample | ✅ |
| **T3: Single-instance speed** (~$0.10–$5, hours) | `tests/gpu.sh` on g5 (A10G): ratio ≤ 0.11, byte-exact, compress AND decompress ≥ 200 KB/s | 🚧 **close — plan in place (see below)** |
| **T4: `krunch plan` end-to-end** (~$2-5, ~30 min) | scaffolding shipped (templates, worker entry, CI dry-run); blocked on T3 green for live AWS Batch run | 🚧 next |
| Docs / Polish | README, architecture.md, tuning.md, benchmarks.md, typed Python SDK, SECURITY.md, CONTRIBUTING.md | ⏳ |

No paid customers in v1 — that's v2's job.

## Tier 3: single-instance compress + decompress speed

**Gate (tightened 2026-04-30 per Dan):** ratio ≤ 0.11, byte-exact roundtrip,
compress AND decompress avg ≥ 200 KB/s on A10G.

### Current state (2026-05-02, A10G g5.xlarge on-demand, 10 MB WildChat, 161 chunks)

| | Pre-fix baseline | T3.7 fix | + N-padding | + 3way_async (M≥256) | Target | Gap |
|---|---|---|---|---|---|---|
| Ratio | 0.1164 | 0.1164 | 0.1164 | 0.1164 | ≤ 0.11 | +6% |
| Compress | 111.27 KB/s | 137.04 KB/s | 141.47 KB/s | **151.59 KB/s** | ≥ 200 | **1.32×** |
| Decompress | 42.82 KB/s | 46.91 KB/s | 46.72 KB/s | **46.74 KB/s** | ≥ 200 | 4.28× |
| Roundtrip | ✅ | ✅ | ✅ | ✅ | ✅ | — |

**Cumulative session-to-session: 1.36× compress, 1.09× decompress.**
Gate gap: compress closed from 1.85× → 1.32×; decompress closed from
4.7× → 4.28×.

**Cumulative this session: 1.27× compress, 1.10× decompress.**
Compress gap closed from 1.85× to 1.41×. Decompress gap closed from
4.7× to 4.3×.

**T3.7 routing-bug fix was the unlock.** Phase profile (cudaEvents
inside `rwkv4_layer_step_cpp`) revealed that levers 1+2 + T3.6
cp.async kernel never actually executed for layer matmuls in
production. Layer-matmul `gemm_fp16` had no `KRUNCH_HEAD_ASYNC`
routing — fell through to the original 1-warp 16×16 `det_matmul_tc`.
Adding async routing to `gemm_fp16` made layer matmuls actually use
the async kernel: per-call layer time 1.75 → 1.21 ms (1.45× layer
speedup). Per-phase: ffn_K matmul 0.45 → 0.22 ms (2.0×), ffn_V
0.44 → 0.26 (1.7×), Ow 0.15 → 0.087 (1.7×), ffn_R 0.13 → 0.069 (1.9×).

**Head matmul N-padding** lifted compress further (137 → 141): pad
W to next multiple of 8 (50277 → 50280), run async on padded shape,
slice output. Decompress unchanged (head matmul is a small share at
B=128).

**Lessons learned (~5 sessions of empirical work):**
- Multiple optimizations validated bit-exact + microbench but failed
  to translate to E2E for ~3 sessions. Root cause was a single
  routing bug (gemm_fp16 never picking up async). Profile data
  exposed it in ~30 minutes.
- Always profile E2E before assuming microbench wins translate.
- Layer matmul on A10G is **88% of layer cost** (not 18% as initially
  estimated from T4 ratios) — A10G's faster compute makes matmul a
  bigger relative share within the layer step than expected.

(A10G baseline of 115/43 supersedes the V1_PLAN's earlier 66/35 figure
which was stale. The new baseline is the prior production code on a
fresh A10G run with the current kernel set.)

A10G E2E on 1 MB WildChat (16 chunks, B=16 — small-batch artifact):

| | baseline | lever 1 alone | lever 2 alone | both |
|---|---|---|---|---|
| compress KB/s | 98.36 | 122.03 | 122.20 | 121.87 |
| decompress KB/s | 13.59 | 13.67 | 13.63 | 13.65 |

Levers 1 and 2 each give ~1.24× compress on A10G, but **don't compound**
— same ~122 KB/s ceiling. They both reduce launch overhead in the same
forward path. Decompress unchanged because at B=16 (1 MB / 64 KB chunk
size), launch overhead per timestep dominates regardless.

The bit-exact GPU AC path is the only correct path on GPU
(`KRUNCH_CPP_PATH=1`, `KRUNCH_DETERMINISTIC_MATMUL=1` default ON). Stock
BlinkDL forward + GPU AC fails roundtrip and is kept in tree only as an
opt-out for benchmark comparison.

### Plan to pass T3 (sequenced by impact-per-day)

Four arch-agnostic, ratio-preserving levers identified in 2026-05-01 code
review of `krunch_ac/cuda/` + `cpp_path.py`. Try the sub-day items BEFORE
committing 1-2 weeks to the persistent-kernel rewrite — two of them are
already half-built in tree. Measure on A10G first; instance bump (L40S /
A100 / H100) is independent and composes with all four.

#### 1. Fix the broken CUDA Graph capture — ~~~0.5–1 day~~ landed 2026-05-01

**Root cause confirmed:** `rwkv::wkv_forward` (BlinkDL's WKV via the c10
dispatcher) is graph-unsafe even with deterministic matmul. Captured
graph diverges step 0 (out_diff = 0.18 → 21 by step 2). Diagnostic:
`scripts/test_t31_graph_diagnostic.py --regime=A`.

**Fix landed:** new `wkv_kernel.cu` — graph-safe drop-in replacement
(one thread per (b,c), sequential over T → bit-identical regardless of
T, no dispatcher, no workspaces). Wired into `layer_cpp.cpp` (both T=1
stepped and T>=1 packed paths) behind `KRUNCH_OWN_WKV=1` env var.
With own-WKV, one-layer graph capture+replay now produces 0.0 diff
across 3 replay steps (regime B, T4 g4dn.xlarge spot 2026-05-01).

**E2E AC roundtrip preserved on T4 1 MB WildChat:** ratio 0.1151
identical to baseline across (baseline / own_wkv / own_wkv+graph),
byte-exact PASS in all three. So WKV math matches BlinkDL exactly
(after fixing initial mis-formulation: `ww = u + k`, `e1 = exp(pp-p)`,
state update uses `time_decay` as-passed since rwkv loader already
applies `-exp(raw)`).

**Speedup measurement DEFERRED** — `KRUNCH_CPP_GRAPH=1` only gates the
single-chunk path (`_decompress_chunk_cpp`); the production
`_decompress_chunks_batched_cpp` (B=64-256) has no graph wrapper.
Adding `forward_stepped_batched_graphed` is the actual KB/s deliverable;
captured as a follow-up. The hard part — graph-safety prerequisites —
is done.

- **Files:** `krunch_ac/cuda/wkv_kernel.cu` (new), `setup.py`,
  `layer_cpp.cpp`, `scripts/krunch` (KRUNCH_* env passthrough),
  `scripts/test_t31_graph_diagnostic.py` (new).
- **Toggle:** `KRUNCH_OWN_WKV=1` (default off; pre-launch consider
  flipping to default-on once batched graph wrapper lands and gate is
  measured).
- **Risk:** bitstream changes when KRUNCH_OWN_WKV flips because WKV
  math is fp16-noise-equivalent but not bit-equal to BlinkDL's. Pre-
  launch acceptable; encoder + decoder must use the same value.

#### 2. Multi-warp `det_matmul_tc` for head matmul — landed 2026-05-01

**Profile result (T4, T=1024):** forward_packed_window 98.7 ms total;
head det_matmul **27.9% of wall** at 27.5 ms (M=1024, K=768, N=50277,
old kernel: 1 warp/block, 16×16 tiles).

**Built:** `krunch_ac/cuda/det_matmul_tc_mw.cu` — 64×64 output tile per
block, 4 warps (128 threads), 2×2 WMMA-frag layout per warp, shared-mem
K-staging. Bit-stable across M by construction. T4 microbench:
**1.78× speedup** at M=1024 (15.7 ms vs 28.0 ms); 0.0 abs M-invariance
diff verified across K∈{768,3072}.

**Routing:** `gemm_fp16` auto-routes to MW when K∈{768,3072} AND N≥16384
AND M≥256 (`KRUNCH_HEAD_MW_M_MIN`, default 256). The M-floor avoids a
decompress regression: at M=B≤128 (decompress stepped batched) the
64×64 tile pads 50-75% of A rows and runs slower than 16×16 single-warp.
Compress packed (M=SEQ_BATCH=1024) clears the threshold; decompress
(M=B=64-128) stays on old kernel.

**T4 E2E result (1 MB WildChat, KRUNCH_HEAD_MW=0 vs default):**

| | HEAD_MW=0 | HEAD_MW=1 (default) | speedup |
|---|---|---|---|
| compress | 37.69 KB/s | **41.33 KB/s** | **1.10×** |
| decompress | 13.61 KB/s | 13.63 KB/s | neutral |
| ratio | 0.1151 | 0.1151 | identical |
| byte_exact | True | True | preserved |

End-to-end compress 1.10× < kernel 1.78× because head is 27.9% of wall
(saving 12 ms of 99 ms = 12% wall reduction). A10G share likely higher
(faster layer loop = larger head share) → A10G E2E TBD.

- **Files:** `det_matmul_tc_mw.cu` (new), `layer_cpp.cpp`, `setup.py`,
  `scripts/test_det_matmul_tc_mw.py`, `scripts/profile_compress_breakdown.py`.
- **Toggle:** `KRUNCH_HEAD_MW=0` disables; `KRUNCH_HEAD_MW_M_MIN=N`
  tunes the M-floor.

#### 3. Warp-cooperative `decode_step_batched` + uint16 CDF — ~2 days

`decode_kernel.cu:159-216` + `det_softmax_cdf.cu`. One thread per stream
binary-searches a 50K-entry CDF with random global loads (~16 probes ×
~200 ns latency = ~3 µs of pointer chasing per token per stream, serialized
inside the warp). Two cheap wins:

- 32-thread warp-cooperative scan replaces binary search with coalesced
  strided reads + warp ballot on the interval containing target. ~5–10×
  faster on the search path.
- CDF values fit in uint16 (`CDF_T = 65536`), currently stored as int32.
  Halves HBM write/read traffic on softmax_cdf → decode_step (~25 MB/step
  at B=128 → ~12 MB).

Also: `det_softmax_cdf_kernel` does two passes over logits (max, then
sum-of-exp). Online softmax (single pass) halves logit-read traffic.

- **Predicted gain:** ~10–20% decompress at B=128. Compounds with items 1, 4.
- **Risk:** very low. **Bit-exact:** must verify uint16 + cumsum
  produces byte-identical bitstreams.

#### 4. Finish `rwkv_step.cu` v2 — first-cut landed 2026-05-01, redesign needed

**Status:** v2 stepped scaffold built (`krunch_ac/cuda/rwkv_step_v2.cu`,
~250 lines) — single cooperative-launch kernel covering all 10 phases of
one layer (LN1, premix, KVR, WKV, Ow, LN2, ffn-premix, ffn_R, ffn_K relu²,
ffn_V residual). 24 blocks × 32 threads, grid.sync() between phases over a
24 KB per-call HBM scratch.

**Correctness (T4):** single-layer fresh-state v1 vs v2 diffs are fp16
noise — x_out 0.008, aa 0.001, pp 0.017, ffn_xx 0.0002. Multi-layer
multi-token compounding amplifies (4 tokens × 12 layers → x_out 1.0,
aa 31.5) but per-call kernel math is right.

**Speed (T4, 12-layer token forward):**

| path | ms/token | vs v1 | vs cpp_path |
|---|---|---|---|
| v1 (rwkv_step.cu, single-block) | 11.17 | 1.00× | 0.34× |
| v2 (rwkv_step_v2.cu, multi-block) | 9.22 | **1.21×** | 0.42× |
| cpp_path.forward_stepped (production) | **3.84** | 2.91× | 1.00× |

**v2 first-cut DOES NOT beat cpp_path.** Tried THREE iterations on T4:

| variant | layout | ms/token | vs cpp_path |
|---|---|---|---|
| v1 baseline (1 block, 768 threads) | 1×768 | 11.19 | 0.34× |
| v2.0 first-cut (scalar per-thread GEMV) | 24×32 | 9.22 | 0.42× |
| v2.1 + shared-mem coop input load | 24×32 | 9.14 | 0.42× |
| v2.2 + 4 warps per block (more SM warps) | 6×128 | 9.36 | 0.41× |
| **cpp_path.forward_stepped (production)** | (multi-launch) | **3.84** | **1.00×** |

All three v2 layouts plateau at ~9 ms regardless of architecture
choice. The remaining 2.4× gap to cpp_path is genuinely scalar-fp32
vs Tensor-Cores. Empirical conclusions:
- 24×32 vs 6×128 layouts: same wall. Neither memory-bw nor
  warp-per-SM-occupancy is the binding constraint.
- Shared-mem input load: zero help (L2 was already warming the reads).
- Scalar-fp32 throughput on T4 is ~200 GFLOPS/SM. 7.7M muladds/layer ×
  12 = 92M; theoretical compute floor ~40 µs/token at 24-SM full
  utilization. We're 225× over that floor — bound by HBM weight reads
  (170 MB / 320 GB/s = 530 µs floor, 17× over) plus grid.sync overhead
  (6 syncs/layer × 12 = 72/token, ~5-10 µs each = 360-720 µs).

**No layout change reaches cpp_path's regime without using Tensor
Cores.** cpp_path's `det_matmul_tc` runs at fp16 TC = 65 TFLOPS peak
(8× scalar fp32) — that's the perf class needed.

**Genuine fix is WMMA-based + batched (~1-2 day focused session, not
random chunks):**
1. **WMMA-based GEMV** — 16×16×16 frags (fp16 inputs, fp32 accum).
   - At B=1 stepped: pad M=1→16 → 94% compute waste, net ~0.5× scalar
     (slower than current). **Therefore lever 4 stepped variant alone is
     a bad bet — must combine with batching.**
   - At B≥16 batched (production decompress at B=64-256): TC fully used,
     8× raw fp16 throughput pays off.
2. **B>1 batch dim** — production decompress is B=64-256. Current v2
   hardcodes B=1. Extension means scratch buffers grow to [B, C], state
   tensors [B, C], thread→(B, channel) mapping. Sub-redesign on top of
   the WMMA work.
3. **Restructured warp→tile mapping** — tile-per-warp instead of
   channel-per-thread. New phase decomposition: each warp owns a 16×16
   output sub-tile of one matmul, all warps in block cooperate on input
   staging. Per-channel ops (LN apply, premix, sigmoid, WKV) refactored
   to thread-per-(B,channel) for B-batched.

**These are interlocking** — can't pick off independently. Together
they're the V1_PLAN's "~2-week" scope. Empirical evidence: 3 layout
iterations on scalar GEMV all converge to 9 ms; 4× speedup to beat
cpp_path needs the perf-class change (Tensor Cores).

**Files in tree (3 iterations):** `rwkv_step_v2.cu` (current = 6×128
4-warps with shared-mem cooperative loads), pybind in `main.cpp`,
`scripts/test_rwkv_step_v2.py`, `scripts/test_v2_one_layer.py`.
Empirical perf data is the load-bearing artifact — phase decomposition,
cooperative launch + grid-sync orchestration, shared-mem cooperative
load helper (`coop_load_f32`), block-wide reduction (`blk_red_sum`),
and per-block scratch layout are all reusable scaffolding for the
WMMA+batched redesign. Only the per-thread GEMV inner loops need
replacement — the surrounding kernel structure stays.

**Microbench (T4, 2026-05-01) reframes the lever:** `det_matmul_tc` (and
its `det_matmul_tc_mw` multi-warp variant) is **1.3–12× slower than
cuBLAS HGEMM** across all v2-relevant shapes — both use Tensor Cores;
gap is missing CUTLASS-level optimizations (async `cp.async`, double-
buffered K loop, shape-tuned tile schedules, vectorized loads).

| M | K=768, N=768 (det_mm vs cuBLAS) | K=3072, N=768 (det_mm vs cuBLAS) |
|---|---|---|
| 1 | 0.019 / 0.015 ms (1.3×) | 0.115 / 0.020 ms (5.7×) |
| 128 | 0.055 / 0.017 ms (3.3×) | 0.229 / 0.049 ms (4.7×) |
| 1024 | 0.425 / 0.051 ms (**8.4×**) | 2.077 / 0.179 ms (**11.6×**) |

**Implication:** even a perfectly-implemented WMMA-based v3 lands in
the det_matmul_tc speed class (since it shares the same WMMA kernel
implementation strategy). That's ~2× over cpp_path at best.

**The bigger lever** is making `det_matmul_tc` itself fast — close the
8× gap to cuBLAS while keeping M-invariance (cuBLAS isn't M-invariant,
hence we wrote our own; can't drop in cuBLAS for this reason). If
det_matmul_tc became 4× faster (half the cuBLAS gap):
- Compress packed forward (M=1024): head matmul drops to ~7 ms (from 28),
  layer matmuls scale similarly → forward ~25 ms instead of 99 = **4×
  compress** = 264 KB/s on A10G → CLEARS GATE.
- Decompress batched (M=128): layer matmul 3.3× faster → forward maybe
  2× faster → ~70 KB/s on A10G. Combined with lever 1's batched-graph
  plumbing + lever 3's decode opts, plausibly clears gate.

**Replace lever 4 v3 with "det_matmul_tc fast path" (T3.6):**
1. Add `cp.async` (PTX intrinsic, sm_75+ supports cp.async.cg) for
   async A/B-tile loads → overlap memory with compute.
2. Double-buffer the K loop — issue load N+1 while computing N.
3. Per-shape tile schedules (split-K when N small, larger M tiles when
   M big, etc.). Maintain bit-stability across M by fixing tile
   schedule per (K, N) combination.
4. Verify M-invariance via `test_det_matmul_tc_mw.py`-style unit test.

**Effort:** ~1 week of focused CUDA work. Same scope as lever 4 v3 but
~5× higher impact (helps compress AND decompress simultaneously) and
lower risk (no big architectural redesign — just inner-loop work on
existing kernel).

**Files:** `det_matmul_tc.cu`, `det_matmul_tc_mw.cu`, new bench
`scripts/bench_wmma_vs_scalar.py` is the load-bearing artifact pinning
down the cuBLAS gap.

**Original plan re-stated:**

The lever-D persistent kernel ("3–5 days, ~3× decompress / ~2× compress")
in earlier versions undercounted what's already in tree:

- `rwkv4_layer_step_kernel` (v1, `rwkv_step.cu:106-217`) is a complete
  fully-fused per-token layer (LN1 + premix + KVR + WKV + Ow + LN2 +
  ffn-premix + relu² + ffn-V), bit-correct vs reference, **already
  shipping** via `main.cpp:65-78`. Slow because `<<<1, 768>>>` (1 SM out of
  80; 11.2 ms/token vs BlinkDL's 7.5 on T4).
- v2 is sketched but not implemented (`rwkv_step.cu:240-249`):
  `<<<24, 32>>>` so 24 SMs read HBM in parallel, multi-block GEMV per
  sub-component, atomic-add LN reductions. The `mb_gemv.cu` 768×768 /
  768×3072 / 3072×768 kernels are already in tree (exposed via
  `main.cpp:342-353`) but unused.

HBM-bandwidth floor: 600 GB/s on A10G with 170 MB weights = 0.28 ms/token
theoretical. v2 realistic target: 1.5–3 ms/token, **4–7× over BlinkDL =
decompress aggregate ~150–250 KB/s at B=128** (closes the decompress gate
alone).

Bit-exactness wedge: v1 is "within fp16 noise" of reference (not bit-exact).
Encoder must reproduce v2's exact bits via a packed-T variant of v2 (same
arithmetic, T=N at WMMA tiles instead of single-token GEMV). Encoder and
decoder share one fused code path → bit-exact by construction.

- **Effort:** v2 stepped ~3 days + cooperative LN reductions ~2 days +
  bit-exact packed-T variant ~5 days = ~2 weeks total.
- **Predicted gain:** 4–7× decompress (alone hits gate); ~2× compress when
  packed variant lands.
- **Risk:** medium — mitigation is a unit test that asserts
  `step(t) == packed(t)` for t ∈ {1, 16, 64, 1024} before enabling the
  encoder switch.

### Sequence

Items 1, 2 (profile-gated), 3 — sub-week total. Re-measure A10G with
`tests/gpu.sh`. If both gates green, ship. If still short, commit to item 4.

### Honest A10G ceiling math (kept for accountability)

- **Compress ceiling on A10G with this model: ~80–90 KB/s.** Per-step
  forward is ~3.7 ms = 16 ops × ~230 µs (50 µs launch + 180 µs compute).
  Eliminating ALL launch overhead saves ~22% per-step → ~80 KB/s. Not 200.
- **Decompress with persistent kernel: ~150–200 KB/s** at B=128.
  ~750 µs compute floor at B=128 (12 layers × 7 matmul + WKV + premix + LN
  + sigmoid + add); current 15 ms/step gap is ATen launch overhead.

If compress at 80–90 KB/s after item 2 lands isn't enough for the gate, the
remaining options are: (a) accept the gate as 80–90 KB/s compress, (b) bump
hardware (L40S / A100 / H100), (c) shrink the model. Decision deferred to
post-measurement.

### Lessons learned (one-liners)

History from the T3 journey, kept short. Full detail in git log.

**Architecture / what shipped**
- **Char model BlinkDL/RWKV-LM + WKV CUDA kernel** is the substrate. HF transformers RWKV is a trap (kernel only in training mode; ~1000× slower in eval).
- **GPU AC encode/decode kernels** in tree (`krunch_ac/cuda/`); 11/11 unit tests pass; byte-identical roundtrip vs CPU reference.
- **Range coder, not ANS.** Symmetric forward; lets compress interleave forward + AC encode batch-by-batch (peak mem O(SEQ_BATCH × vocab) ≈ 200 MB regardless of chunk size).
- **64 KB default chunks** (was 1 MB). +0.08% ratio cost, 16× more chunks for the worker pool.
- **Multi-process decompress worker pool** (`krunch/worker_pool.py`); 2.56× on T4 (N=4); composable with cpp_path.
- **`krunch warmup`** — pre-bakes WKV ninja-build + warm cache into a docker volume so first compress doesn't pay 60 s of one-time costs.
- **Per-GPU auto-tune for cross-chunk batch B** at worker startup (T4=64, A10G=128, A100/L40S=256, H100=512).
- **Bit-exact GPU AC path** required custom shape-invariant matmul (`det_matmul.cu` + `det_matmul_tc.cu`) because cuBLAS auto-selects different algos at different M (~10⁻³ drift per algo/M, breaks AC roundtrip).
- **Encoder softmax must run row-by-row** (mirror decoder's [1,V] shape) — `torch.softmax([T,V])` and `softmax([1,V])` of the same row are NOT bit-identical (different reduction strategy). Now done by `det_softmax_cdf` kernel.
- **Premix kernels** (`launch_premix_3`, `launch_premix_2`, `launch_relu_sq`) shipped — fp32-internal, used by both stepped and packed paths so they produce bit-identical kx/vx/rx.
- **UTF-8 chunk boundary slicing** — `chunking._split_utf8_safe` snaps boundaries off continuation bytes; 2 of 160 WildChat chunks were producing replacement chars before the fix.

**Tried, didn't help / dead ends**
- **cuBLAS algo pinning** for encoder fusion — DEAD END. The enum is a hint, not a binding; cuBLAS picks actual kernel by shape regardless. ZERO of 16 algos bit-stable across M.
- **`det_matmul_tc_v2` (32×32 tile)** — bit-stable, microbench 1.7–2× faster, **end-to-end neutral** because matmul isn't the layer-step bottleneck. Kept as `KRUNCH_TC_V2=1`.
- **Custom `layer_norm` kernel** — bit-exact but **3% slower end-to-end** (LN is <1% of wall). Kept as `KRUNCH_LAYERNORM_CUSTOM=1`.
- **CUTLASS GEMM** — skipped, same diagnosis as TC v2.
- **Mid-layer / head logit quantization** as fix for AC roundtrip — math is hopeless: drift d=0.19 with step=1.0 → 19% round-flip per element × 50K vocab × 64 positions → AC decodes diverge immediately. Step finer than fp16 needed.
- **Naive multi-block GEMV** (`mb_gemv.cu`, single-warp tiles) — cuBLAS beats by 1.6–11× on RWKV-4 shapes; matching needs `mma.sync` MMA + cooperative K reduction. Code retained as reference.
- **`__ldg(const __half*)`** for read-only weight loads — broke kernel correctness (x_out diff jumped 0.016 → 0.75). Reverted.
- **ThreadPool decompress** with per-thread CUDA streams — 1.7× slower than sequential (per-token `.item()` + GIL contention). Replaced by multi-process pool.
- **Naive batched decompress at 64 KB chunks (B=46)** — 3.8× wall reduction on T4 but byte-different (compress's `rwkv.forward(packed)` vs decompress's `forward_batched` differ ~12 abs in logits → AC roundtrip breaks).
- **`forward_batched(B=1, T=1)` to bypass BlinkDL** — slower than BlinkDL (12 vs 7.5 ms/token), drifts 3–5 abs.
- **`torch.compile(reduce-overhead)`** on per-step forward — silently falls back to eager (BlinkDL has Python control flow that breaks graph capture).
- **Compile-friendly forward_batched (plain `@`, pure-torch WKV)** — compile finally works but 14.3 ms/token vs BlinkDL's 7.5 (2× slower). Pure-torch WKV at T=1024 is 200× slower than the kernel.
- **Worker-pool decode for cpp_path** — 4-worker aggregate at 0.9 KB/s vs single-stream 1.3 KB/s; cpp_path saturates the GPU, workers contend.
- **CUDA graph around per-token C++ stepped path** — both wrappers diverge on step 0 (out_diff = 0.18, then explodes). `rwkv::gemm_fp16_cublas` and `rwkv::wkv_forward` don't replay deterministically in this PyTorch+CUDA build. Fix is multi-day (replace with graph-safe equivalents) — captured as a sub-task of item 1 in the T3 plan above.
- **Symmetric `compress_chunks_batched`** (force compress to lockstep B=N, T=1) — correct but 5× slower compress; off-table per "don't slow compress." Code retained, env-toggle `KRUNCH_COMPRESS_BATCHED=1`.
- **Self-speculative greedy lookahead** — structurally bounded at ~1/(1-p_top1) ≈ 1.5×. Composes with fused kernel but not enough alone.
- **Speculative decoding from LLM literature, Jacobi/Lookahead, SSM parallel scan** — AC-incompatible (trajectory pinned by bitstream, not sampled).
- **Embedded full-state checkpoints in bitstream (CABAC-style)** — RWKV state is ~150 KB per checkpoint vs CABAC's 16 bytes; one checkpoint > a chunk's compressed size.

**Levers held in reserve (compose with fused kernel if needed)**
- **Parallel rANS streams in bitstream.** ~1–2% ratio cost, ~4–8× decode speedup at K=8. Bitstream format change.
- **Quantized-probability AC.** 16-bit fixed-point CDFs tolerate ~5 abs drift between encoder and decoder. ~2–5% ratio cost. Unlocks alternative inference engines (web-rwkv, rwkv.cpp).

### Compress backlog (Tier 3 secondary if item 2 is enough)

- **Fused CDF construction kernel** — already shipped as `det_softmax_cdf`; removed the per-row Python loop.
- **Top-k probability truncation** (~30× win on CDF, bitstream change). Encode top-k tokens (k≈1024) plus an "escape" symbol. Real LM puts >99.99% mass in top-1024 on text. Save for v1.2.

## What to steal from peer codecs (2026-05-01)

Cross-checked krunch's current numbers (137 KB/s compress, 47 KB/s
decompress, 0.93 bpb on WildChat) against ts_zip and Nacrith via
`l3tc-prod/docs/COMPARISON.md` (primary-source landscape doc).

**Why we can't just use ts_zip.** Bellard's binary is non-commercial-
only proprietary; LibNC is GPL non-commercial. Neither is
redistributable in our Apache-2.0 OSS image, neither is callable as a
library, neither documents its blob format, and neither integrates
with batch frameworks for fleet-scale compression. ts_zip serves
single-user GPU enthusiasts; krunch serves enterprise OSS commercial
+ batch-deployable + library-shaped + format-stable. **Different
buyers, different product, no overlap that can be papered over by
wrapping the binary.**

**What we can steal: the engineering ideas.** Algorithms, numerical
layouts, kernel-organization patterns, op-fusion choices, memory
layouts — none of those are copyrightable. Read LibNC + ts_zip's
documented design notes, take notes, translate the *ideas* into our
own Apache-licensed C++/CUDA. This is how systems-software fields
have always cross-pollinated.

**What we cannot do**: copy code verbatim, redistribute his binary,
link against LibNC. So no `git clone bellard/libnc` shortcut. Our
implementation is original code informed by his published design.

### Hard constraint: no bitstream changes for v1

v1 ships one codec; new techniques must produce byte-identical output
to the current encoder. (Future model_id values can ship different
bitstreams in v1.x / v2 — that's what the `model_id` header field is
for — but v1 itself is one codec, one bytes-on-disk format.)

This filter splits the technique list cleanly:

- **Engineering** on the same fp16 forward + AC pipeline (ts_zip-shaped
  ideas) → byte-identical by construction → **for v1**.
- **Algorithmic** changes that pick a different CDF at any token
  position (Nacrith-shaped ideas) → produce different bytes → **for
  v2 model_ids**.

### The v1 plan, four steps

1. **Read LibNC source + ts_zip's documented design notes** (1 day).
   ~~Take notes on: kernel boundaries, state-buffer layout, int8 layout,
   op-fusion patterns, how Bellard organizes per-token state in HBM,
   how he avoids host syncs.~~ **DONE 2026-05-02 → `docs/PEER_CODEC_NOTES.md`.**
   **Finding: Bellard's published material is not a useful primary
   source** — pages document WHAT (model, ratio, 1 MB/s on RTX 4090
   for ts_zip) and WHY, never HOW. LibNC + ts_zip are closed-source.
   Substituted: read `harrisonvanderbyl/rwkv-cpp-accelerated`
   (Apache-2.0, 729-line CUDA, same RWKV-4 model class). Captured
   actionable patterns: uint8 weights with per-input-channel scale +
   offset, inline dequant in matmul, 3-way fused KVR launch, fp64 WKV
   state. Notes also identify three cheap bit-identical experiments to
   do BEFORE int8 work: bf16 matmul swap (~1 day), fp64 WKV state
   precision check (~1 day), persistent kernel via own design (~1 week).

2. **Translate the persistent-kernel discipline into our own
   C++/CUDA**, building on `layer_cpp.cpp` + the per-layer graph work
   already in tree. One launch per token, fixed shapes, state mutated
   in HBM. This is V1_PLAN's primary T3 lever already. Bellard
   confirms it's the right architecture; we implement in our own
   Apache-licensed code. Byte-identical by construction (same math,
   different scheduling).

3. **Replicate Bellard's int8 layout in our quantization spike** (the
   one already on the plan). Per-channel scales, the saturation /
   rounding choices he documents, fp16 accumulation. Run encoder
   fp16 vs int8 on the same WildChat sample and compare emitted
   bytes. If byte-identical → port to `layer_cpp.cpp` and ship
   (~1.5–2× speedup on top of the persistent kernel). If bytes
   differ → park for v2 with a new model_id; do not ship a different
   encoder output silently as v1.

4. **Skip the Nacrith algorithmic ideas for v1.** n-gram skip,
   adaptive bias head, top-k truncation — all change CDFs at some
   token positions regardless of how they're implemented, so all
   produce different bytes than today's encoder. Real wins, just not
   byte-identical wins. Park for v2 alongside the LoRA adapter work,
   where new model_ids carry the new bitstreams.

### Honest framing on the speed gap

We may not match ts_zip's absolute speed in v1 even after this. Bellard
has spent years on it; we're spending weeks. **That's fine.** The
pitch is "the open, library-shaped, batch-deployable one that's
roughly competitive on a single instance and scales linearly across
workers." A user with 10 TB to compress doesn't care if one A10G hits
200 KB/s vs 500 KB/s when 16 A10Gs hit 3.2 MB/s vs 8 MB/s — at fleet
scale we both finish before lunch. Multi-worker scaling is the actual
UX advantage and is the part **ts_zip structurally can't provide**
because it's a single-stream binary with no batch-framework
integration.

### v2 / v1.x parking lot (require bitstream changes)

For record, captured here so we don't re-derive them when v2 starts:

- **Confidence-based n-gram skip (Nacrith)** — online n-gram + entropy
  threshold + AC against n-gram CDF when LM is unconfident. ~1.4–2×
  speedup on krunch's text mix. Best transferable algorithmic idea
  on the table. v2 model_id `RWKV169M+ngram_v1`.
- **Adaptive log-space bias head (Nacrith)** — per-document online
  gradient correction. Nacrith reports −0.39 bpb on enwik8; estimated
  −0.05 to −0.15 bpb on WildChat. v2 model_id ratio-polish.
- **Top-k probability truncation** — ~30× win on the CDF stage with
  escape-symbol fallback. Already noted in compress backlog. v2
  model_id.
- **Hybrid text+binary segmentation (Nacrith NC06)** — content-detected
  segments encoded with LZMA/gzip/raw for binary regions. Format-level
  change; defer indefinitely (current "use zstd for non-text" framing
  is the honest answer).

## Tier 4: `krunch plan` end-to-end

`tests/gpu.sh` covers single-instance T3. Tier 4 builds `krunch plan` itself
and validates the contract by running at least one target end-to-end.

### Shipped (2026-04-30 unless noted)

- ✅ `krunch/plan/` module — render() + validate(); 7 targets ship (aws-batch, gcp-batch, k8s, modal, ray, slurm, local), all render + schema-validate cleanly.
- ✅ `krunch/plan_cli.py` — in-image entry point. Host wrapper docker-runs it so the user only needs docker (no `pip install krunch`).
- ✅ `scripts/krunch plan ...` host-side wrapper subcommand. `--dry-run` validates schema; otherwise emits to stdout.
- ✅ `krunch/job.py` reshaped around the v1 env-var contract. Handles compress / decompress / finalize. Per-target templates map orchestrator-specific vars (`AWS_BATCH_JOB_ARRAY_INDEX`, `JOB_COMPLETION_INDEX`, `SLURM_ARRAY_TASK_ID`) → `KRUNCH_PART_INDEX`.
- ✅ `tests/test_plan.py` — Tier-1 CI test renders + validates every target on every PR.
- ✅ Per-GPU auto-tune for cross-chunk batch B at worker startup (heuristic table by `nvidia-smi` GPU name; wired into `_decompress_chunks_batched_cpp`).
- ✅ **Dynamic compress chunk size (2026-05-02).** `compute_chunk_size(total)` in `krunch/chunking.py`: floors at 64 KB, otherwise `ceil(total / (4 × target_B))` so a file produces ~4× target_B chunks (≥16). All workers see the same `KRUNCH_INPUT_LEN` → all converge on the same chunk size; `_byte_range` in `krunch/job.py` uses the same function → byte ranges align across workers without coordination. `KRUNCH_TARGET_B` overrides the planning hint (default 128, A10G class); `KRUNCH_CHUNK_SIZE` still pins a static size for back-compat. Compress workers pass `total_size=KRUNCH_INPUT_LEN` so per-worker `len(raw)` doesn't redirect to a different chunk size. Tests in `tests/test_blob.py`: `test_compute_chunk_size` (floor / scale / pin / target_B override) + `test_dynamic_chunking_roundtrip` (auto-derive + total_size override roundtrip byte-exact).

### Remaining

- ⏳ **AWS Batch e2e validation run** (the actual T4 gate). Blocked on T3
  green so we know per-instance numbers before pinning the queue config.
  Run on the same WildChat sample as T3:
  ```
  krunch plan --target aws-batch --source s3://… --dest s3://… --workers 2 > job.json
  aws batch submit-job --cli-input-json file://job.json
  ```
  Gates: same ratio as T3, ~2× wall reduction at N=2 (within cold-start
  tolerance), parts cleaned up, finalize task succeeds, Batch cold-start
  time recorded.
- ⏳ **Microbench-based auto-tune** (replace heuristic table with startup
  probe) — backlog; current table is fine for shipped GPU classes.

Modal / k8s / Ray / Slurm targets are validated to the artifact level in
T1 (template renders, schema-valid output), but full e2e on each is
**deferred to post-launch**. Real users will surface friction faster than
synthetic exercises across 5 targets.

**Required before v1 ships** — the "any batch system" claim in the README
rests on the env-var contract + `krunch plan`, and has never been validated
end-to-end. T3 green unblocks the AWS Batch run.

## Backlog (post-T3, deferred)

- **Bigger model option** — `RWKV-4-Pile-1B5`. Better ratio (~0.07–0.09 on chat, matches ts_zip) at 3–5× slower forward. Add `KRUNCH_MODEL` env var + multi-image publish.
- **fp8 / int8 quantized forward** for H100/L40S. Defer until customer hardware mix is known.
- **Custom AMI with pre-pulled image** for cold Batch workers. Eliminates ~30–90 sec docker pull.
- **Streaming inputs > GPU VRAM.** v1 reads whole input into memory; v2 will stream.
- **fp16-probs CDF** (skip `to(fp32)` cast). Likely off-by-1 count errors at boundaries; defer until measured demand.

## Success criteria for v1 → v2 transition

Move to v2 (hosted offering) when:
- **GitHub stars ≥ 500** within 2 months of public launch (category interest)
- **At least 3 independent users** report production deployments (framework actually works)
- **At least 1 inbound** "would you host this?" inquiry (demand for managed)

If none trigger by month 3 post-launch, OSS bet didn't work; revisit
positioning before investing in hosted v2.

## Versioning

Every `.krunch` blob is self-describing via a fixed header:

```
magic:           KRNC (4 bytes)
blob_version:    u8
model_id:        u32
tokenizer_id:    u32
adapter_id:      u32
adapter_version: u16
flags:           u16
original_len:    u64
n_chunks:        u32
crc32:           u32
```

- **Forward compat forever** within major version — newer images always read older blobs
- **Backward compat** is configurable via `KRUNCH_WRITE_BLOB_VERSION` env var
- **Images bundle all readable model versions** — upgrade without breaking existing archives
- **Migration**: `krunch recompress --target-model v2` re-encodes old blobs without decompress-to-disk

## What's NOT in v1

- ❌ Hosted offering / managed API (v2)
- ❌ Vertical-specialized LoRA adapters (v2)
- ❌ Per-tenant fine-tuning (v3)
- ❌ Storage-as-a-service (defer indefinitely)
- ❌ Multi-region / HA out of the box (single-region; customer's orchestrator handles HA)
- ❌ Authentication (OSS framework runs in trusted networks; auth is customer's responsibility)
- ❌ Streaming for inputs > GPU VRAM (v2)

## Risks (ranked)

1. **Nobody adopts.** Mitigation: public launch + benchmarking case studies + targeted reach-out to logging vendors / archival storage SREs.
2. **Adoption but no upsell.** Mitigation: vertical-LoRA registry is paid even for self-host (subscription for adapter access).
3. **AWS/GCP fork it.** Real long-term risk. Mitigation: build adapter registry as the moat in v2.
4. **Apache-2.0 too permissive.** If cannibalization risk becomes acute, relicense future versions to BUSL (Hashicorp pattern). v1 stays Apache forever.

## Immediate next step (2026-05-01 — revised after T3.6 attempt landed)

A10G measurement showed levers 1+2 don't move the gate at production
scale. T3.6 cp.async kernel built + tested — also doesn't move the
gate. Findings forced a re-prioritization.

1. ~~**T3.6: cp.async + double-buffered K loop in `det_matmul_tc`**~~
   **PARTIAL — kernel built, bit-stable, 1.7-2× microbench, but
   1.02× E2E on A10G.** Why: matmul is only ~18% of compress wall on
   A10G (vs ~60% on T4). A10G's faster compute reduces matmul share;
   bottleneck is non-matmul ops + small-kernel launch overhead. cp.async
   optimizes the wrong thing. Keep kernel in tree (`det_matmul_tc_async.cu`,
   default ON) for any future workload where matmul share is bigger;
   adds correctness, no regression. **Next investigation: profile A10G
   forward_packed_window phase-by-phase to identify what actually eats
   the 99 ms / window.**
2. **T3.3: warp-cooperative decode + uint16 CDF** — ~2 days, low risk,
   ~10–20% decompress at B=128. Useful regardless of T3.6 outcome.
3. **T3.1b: plumb `forward_stepped_batched_graphed`** into batched
   decompress — converts the lever-1 graph-safety prereq into actual
   KB/s. ~2 hours, but only single-digit % at B=128 (production size).
4. **Lever 4 v3 (WMMA + B>1 batched persistent kernel)** — ~2 weeks.
   Defer until T3.6 + T3.3 measured; only commit if those don't close
   the gate.
5. **Instance bump** (L40S / A100 / H100) — independent path. L40S
   has 4.5× compute over A10G; on-paper plausibly clears gate without
   any kernel work. Worth measuring once T3.6 lands.
5. **T4 AWS Batch e2e** once T3 green. Validates the headline "any batch system" claim.
6. **Real-corpus ratio benchmarks** for README (logs / chat / support tickets / wiki / code).
7. **Polish + soft launch + Show HN.**

Keep this file under ~600 lines.
