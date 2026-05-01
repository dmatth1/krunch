# Krunch v1 — open-source neural codec for text (library + image + format spec)

**Decided 2026-04-24**: ship as an open-source framework, not a
hosted service. Hosted is **v2** (`V2_PLAN.md`).

## What v1 is

An open-source neural codec for text. Ships as a Python library, a
Docker image, and a documented blob format. Customers run it on their
own infrastructure (any NVIDIA GPU + container runtime) and parallelize
it across machines using whatever batch system they already use —
`krunch plan` emits the config for AWS Batch, GCP Batch, k8s, Modal,
Ray, Slurm, or local. Customer's data never leaves their network.

Two ways to run, same Docker image:

**Single-shot** — container starts, processes one input, exits.
Good for one-off compression on any GPU host. The complete user
experience is two commands:

```bash
# 1. Install (~5-10 min one-time — downloads CLI + pulls 3.5 GB image)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash

# 2. Compress (or decompress) — instant, image is cached
krunch compress   < data.jsonl  > data.krunch
krunch decompress < data.krunch > data.jsonl
```

**Single-machine parallel — automatic.** `krunch compress`
autodetects available GPUs / cores and uses the multi-process worker
pool by default. The user doesn't see a knob: one beefy GPU box,
`krunch compress data.jsonl -o data.krunch`, all of it gets used.
(Internal escape hatch via env var if the autodetect ever gets it
wrong on a weird host.)

**Distributed across machines** — `krunch plan` emits a runnable
artifact for the user's existing batch system. We don't run it; we
generate the config and they execute it with their own credentials:
```bash
krunch plan --target aws-batch --source s3://… --dest s3://… --workers 16 > job.json
krunch plan --target k8s       --source s3://… --dest s3://… --workers 16 > job.yaml
krunch plan --target modal     --source s3://… --dest s3://… --workers 16 > run.py
krunch plan --target ray       --source s3://… --dest s3://… --workers 16 > run.py
krunch plan --target slurm     --source s3://… --dest s3://… --workers 16 > run.sbatch
# user then runs: aws batch submit-job --cli-input-json file://job.json
#                 kubectl apply -f job.yaml
#                 modal run run.py  /  ray job submit ...  /  sbatch run.sbatch
```

The artifact contains both the worker tasks (each computes its byte
range from a framework-injected index) and a finalize task that
stitches partial blobs into the final `.krunch`. No krunch control
plane, no krunch credentials — the user's framework runs the tasks,
we just emit the right config.

`krunch compress` / `decompress` shell out to
`docker run --gpus all -i ghcr.io/dmatth1/krunch:v1 …`. The CLI
is a thin Python wrapper; the Docker image is the real work.

That's the whole product surface for v1. A library + Docker image,
a CLI (`compress` / `decompress` / `plan` / `bench`), a documented
blob format spec, and a gallery of examples for popular batch
frameworks. No always-on server, no HTTP API, no hosted SaaS — that's
v2.

## Repositioning (2026-04-30): codec, not framework

Earlier framing called krunch a "distributed neural compression
framework." On reflection, we're not building a batch system —
"distributed batched compression" is just batch inference where the
inference happens to be compression. Modal/Ray/Spark/Batch already do
that part well, and reinventing it is not where the value is.

What we actually ship:

1. **A neural codec library** — `from krunch import Codec; codec.compress(bytes)`.
   `pip install krunch` (CPU works) + `pip install krunch[gpu]` (CUDA).
   Pools model loading; integrates into existing batch workers without
   forking a container per shard.
2. **A Docker image** — the GPU-baked packaging of the same library,
   for environments that prefer container-per-task.
3. **A documented blob format spec** (`docs/format.md`, RFC-style).
   Anyone can implement a decoder. Format-stable across major versions.
4. **A container env-var contract** (`KRUNCH_INPUT_URL`,
   `KRUNCH_OUTPUT_URL`, `KRUNCH_PART_INDEX`, `KRUNCH_PART_COUNT`, …) —
   the integration interface for any batch system.
5. **`krunch plan`** — emits ready-to-run artifacts for AWS Batch,
   GCP Batch, k8s, Modal, Ray, Slurm, local. Templates over the
   contract above. Community PRs add new targets.
6. **`krunch bench`** — `krunch bench --corpus path` reports ratio +
   compress/decompress KB/s on the user's actual data. Lets buyers
   qualify krunch in 30 s before integrating.
7. **An examples gallery** — short (~30–100 LOC) snippets for each
   batch framework. Maintained as examples, not products.

What we **stop** owning:

- The AWS Batch deployer is no longer the headline. It stays as one
  example under `examples/aws-batch/` (demoted from `deploy/aws-cdk/`).
- `krunch submit` is **deprecated** in favor of `krunch plan` +
  user-side execution. It stays for one minor release as a
  thin wrapper (`plan --target aws-batch | aws batch submit-job`)
  for users on the reference CDK stack.
- The "we own multi-cloud orchestration" implication. We own the
  container contract; targets are templates.

### Container env-var contract (the real interface)

Workers (any target) get these env vars and behave identically:

```
KRUNCH_INPUT_URL    s3://…  |  gs://…  |  https://…  |  file://…
KRUNCH_OUTPUT_URL   destination prefix; worker writes to <prefix>.parts/<index>
KRUNCH_PART_INDEX   0-based, injected by the framework
KRUNCH_PART_COUNT   total worker count
KRUNCH_MODE         compress | decompress | finalize
```

On startup the worker:
1. `HEAD`s the source for total size.
2. Computes its byte range: `[i * size / N, (i+1) * size / N)`.
3. Ranged GET → compress chunks → write partial blob to `parts/<index>`.

A separate `finalize` task (also emitted by `krunch plan`) stitches
the partial blobs into the final `.krunch` (sums `n_chunks`,
`original_len`, recomputes CRC).

**Format tweak under consideration:** move the variable header fields
to a small footer at end-of-blob (16–32 bytes: total chunks, total
original length, CRC). Then `finalize` degenerates to ordered concat
+ footer write, no header rewrite needed. Bitstream change only;
codec untouched. Removes the "k8s Jobs don't natively chain" friction
since finalize becomes trivial enough to inline. **Decision: defer
until first user hits the assembly friction; current header-rewrite
finalize is fine for AWS Batch + Modal + Ray + Slurm which all
support task dependencies cleanly.**

### Splitting + stitching across plan targets

| target | worker indexing | finalize dependency |
|---|---|---|
| AWS Batch | `AWS_BATCH_JOB_ARRAY_INDEX` (array job) | `dependsOn: [{type: SEQUENTIAL}]` |
| GCP Batch | `BATCH_TASK_INDEX` | `taskGroups` ordering |
| K8s | `JOB_COMPLETION_INDEX` (Indexed Job) | second Job; init-container wait, or Argo |
| Modal | function param `i` | call sites: `for_each(shards); assemble()` |
| Ray | `ray.remote()` task index | `ray.get([…]); assemble.remote(parts)` |
| Slurm | `SLURM_ARRAY_TASK_ID` | `sbatch --dependency=afterok:$JOBID` |
| local | bash `for i in $(seq …)` | sequential after worker loop |

Caveat: requires a seekable, ranged-readable source (S3, GCS, HTTP
with Range, local FS). Stdin can't be split — distributed mode
requires URL input. Already-sharded inputs (a prefix of files) get a
file-list mode where worker `i` takes shards `[i::N]` — same contract,
different splitting math.

## Why parallelize

Compress and decompress both parallelize linearly across machines:

- **Compress**: input chunks are independent. Each chunk is one model
  forward pass + arithmetic-encode. N workers = ~N× throughput. A
  customer with 10 TB of archives spins up 10 workers, finishes ~10×
  faster.
- **Decompress**: chunks decode independently. Within a chunk the RNN
  is sequential (token-by-token), but **across chunks** it's fully
  parallel. Same N× scaling.

We don't ship a batch system — Modal, Ray, AWS Batch, k8s, GCP Batch,
Slurm already do that part well. Krunch ships the **container env-var
contract** (one set of vars, identical worker behavior across all
targets) and `krunch plan` to emit the right config for each.

How distributed scaling actually works (krunch plan):

```
  krunch plan --target <framework> --source s3://… --dest s3://… --workers N  >  artifact
        │
        ▼
   user runs the artifact with their own creds:
   aws batch submit-job  /  kubectl apply  /  modal run  /  ray job submit  /  sbatch
        │
        ▼  framework injects KRUNCH_PART_INDEX into each worker
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ worker 0 │   │ worker 1 │   │ worker N │   each: 1 GPU container
  │ range    │   │ range    │   │ range    │   reads byte range from URL
  │ 0..1GB   │   │ 1..2GB   │   │ N..EOF   │   writes partial blob to URL
  └──────────┘   └──────────┘   └──────────┘
        │              │              │
        └──────────────┴──────────────┘
                       ▼
              ┌────────────────┐
              │ finalize task  │   stitches partial blobs into final
              │ (1 container)  │   .krunch blob, cleans up parts
              └────────────────┘
```

Workers don't talk to each other and don't proxy data through any
orchestrator — they read their byte range directly from object storage
and write their partial blob back. The job is **embarrassingly parallel
+ map-only**, so it maps onto any batch processing framework. Krunch's
job is to emit the right plan artifact and provide a worker container
that obeys the env-var contract.

## Multi-target, not AWS-only

We ship one thing: a Docker image plus a Python library. The same
image runs anywhere NVIDIA GPUs + a container runtime exist, and
`krunch plan` emits a runnable artifact for whichever batch system
the user already uses:

| target | shipped at v1 | how it's distributed |
|---|---|---|
| AWS Batch | ✅ | `krunch plan --target aws-batch` + reference CDK in `examples/aws-batch/` |
| GCP Batch | ✅ | `krunch plan --target gcp-batch` |
| Kubernetes (Indexed Jobs) | ✅ | `krunch plan --target k8s` |
| Modal | ✅ | `krunch plan --target modal` |
| Ray | ✅ | `krunch plan --target ray` |
| Slurm | ✅ | `krunch plan --target slurm` |
| Local (single machine) | ✅ | `krunch plan --target local` (also: `krunch compress` autodetects parallelism) |
| Azure Batch / Databricks / Argo / others | ❌ | community PRs welcome; the env-var contract makes them ~30 LOC each |

We don't maintain cloud-specific deployers as products. The reference
AWS CDK stack lives under `examples/aws-batch/` — same care as any
other example, none of the "this is the canonical way" framing.

The contract we publish:
- NVIDIA GPU with CUDA 12.x driver
- ≥16 GB VRAM (A10G / L4 / A100 / H100 all work)
- Container runtime with GPU access (`--gpus all` or k8s
  nvidia-device-plugin)
- Outbound network optional (model + tokenizer baked into image)
- Worker env-var contract: `KRUNCH_INPUT_URL`, `KRUNCH_OUTPUT_URL`,
  `KRUNCH_PART_INDEX`, `KRUNCH_PART_COUNT`, `KRUNCH_MODE`

Any customer environment meeting that contract runs the codec.

## What's in the v1 framework

```
krunch/
├── Dockerfile              # pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel base
├── install.sh              # public install script (curl-fetched by users)
├── krunch/                 # the Python package — codec library + CLI
│   ├── __init__.py         # public API: Codec, compress, decompress
│   ├── cli.py              # CLI: compress | decompress | plan | bench
│   ├── inference.py        # RWKV-4-Pile-169M wrapper + AC coder + blob format
│   ├── chunking.py         # chunk splitter (neural-only, no fallback)
│   ├── worker_pool.py      # multi-process pool for --workers N (single machine)
│   ├── job.py              # in-container per-worker entry: range → partial blob
│   ├── plan/               # krunch plan templates (jinja)
│   │   ├── aws_batch.json.j2
│   │   ├── gcp_batch.json.j2
│   │   ├── k8s.yaml.j2
│   │   ├── modal.py.j2
│   │   ├── ray.py.j2
│   │   ├── slurm.sbatch.j2
│   │   └── local.sh.j2
│   └── url_io.py           # generic URL read/write (s3://, http://, file://)
├── docs/
│   └── format.md           # blob format spec (RFC-style, implementable)
├── scripts/
│   ├── krunch              # the user-facing CLI wrapper (Python)
│   └── entrypoint.sh       # container ENTRYPOINT (worker | finalize | compress | decompress)
├── tests/
│   ├── test_blob.py        # unit tests (header, AC, chunking, CRC, tokenizer)
│   ├── quick.sh            # CI-equivalent local sweep
│   ├── integration.sh      # CPU end-to-end with the real model
│   ├── gpu.sh              # GPU smoke on a g5.xlarge spot
│   └── README.md           # describes the four above
├── examples/
│   ├── aws-batch/          # CDK reference (was deploy/aws-cdk/)
│   ├── modal/              # ~30 lines: distributed compress on Modal
│   ├── ray/                # ~30 lines: ray.data shard map
│   ├── k8s/                # Indexed Job + finalize Job manifests
│   ├── spark-udf/          # PySpark UDF wrapping the library
│   └── single-machine/     # bash + xargs -P over a directory
├── .github/workflows/      # ci.yml (test_blob + cdk type-check), publish-image.yml
├── README.md               # public intro: codec library + image + format spec
└── LICENSE                 # Apache-2.0
```

Total surface: ~1500 LOC of Python (codec library + CLI + plan
templates) + Dockerfile + a few short shell scripts. The reference AWS
CDK in `examples/aws-batch/` is additional but not on the maintenance
hot-path.

## Differentiation vs DIY

A competent engineer could build this from public components in 2-4
weeks. The framework collapses that to 30 minutes:

| concern | DIY | krunch |
|---|---|---|
| Pick base model | research + benchmark | RWKV-4-Pile-169M (validated) |
| Pick inference path | trial-and-error | BlinkDL/RWKV-LM (HF transformers is a trap) |
| Compile WKV kernel | apt+nvcc+ninja debug loop | image is pre-built |
| Pick AC coder | choose between 5 libraries | constriction (mature, Rust-backed) |
| Chunking strategy | guess | 1 MB validated default (Spike 6) |
| Classical fallback | wire it yourself if needed | none — krunch is neural-only by design |
| GPU lifecycle / scaling | manual | `krunch plan` emits config for any batch system |
| Observability | build dashboards | EMF metrics + ratio histogram out of box |
| Model version safety | break old archives on upgrade | version-pinned blob format, migration tooling |

Plus the 7+ Spike-6 traps already documented in `CLAUDE.md` (DLAMI
version, ninja-build, kernel-load-only-in-training-mode, etc.) — the
framework hides every one of them behind a working default.

## License: Apache-2.0

Reasoning:
- **Maximum adoption** — enterprise legal teams approve Apache-2.0
  faster than GPL/AGPL/SSPL
- **Compatible with v2 hosted business** — we can run our own
  Apache-2.0 code as a hosted service without dual-licensing
  gymnastics
- **Reasonable defense** — patent grant + attribution requirement
  prevents trivial whitewash forks

We trade away "AWS can't fork us" defense (vs SSPL/BUSL). That's a
real risk but mitigated by:
1. Vertical-LoRA registry is the v2 moat, not the runtime
2. By the time AWS/GCP forks the runtime, we'd have customer
   relationships, hosted offering, and adapter ecosystem
3. Hashicorp's BUSL relicense came AFTER they'd built a $5B company
   on Apache-2.0. We can do the same if we get there.

## Distribution

- **GitHub repo**: `github.com/dmatth1/Krunch` (already exists,
  currently private; flip to public at v1 launch)
- **Container image**: `ghcr.io/dmatth1/krunch:v1`
- **Docs site**: simple GitHub Pages or Vercel-hosted docs from
  `docs/`
- **Discoverability**: TBD (deferred until artifact ships)

## Roadmap to 1.0

Validation tiers (cheapest → most expensive, gate each before the next):

| Tier | scope | status |
|---|---|---|
| **T1: Quick checks** (free) | unit tests (`test_blob.py`), `krunch plan --target … --dry-run` artifact-validation for every supported target, plan-template lint. CI-equivalent. | ✅ |
| **T2: Local CPU integration** (free) | `tests/integration.sh`: full neural path on CPU, byte-exact roundtrip on 224 B sample. Validates the single-shot CLI before spending GPU $. | ✅ |
| **T3: Single-instance speed** (~$0.10–$5, hours) | `tests/gpu.sh` on g4dn (T4 cheap) or g5 (A10G full): ratio + **compress AND decompress** wall on a real WildChat sample. **Gate (tightened 2026-04-30 per Dan): ratio ≤ 0.11, byte-exact roundtrip, compress AND decompress avg ≥ 200 KB/s on A10G.** Cross-chunk batched stepped forward landed; correctness bug at multi-chunk decompress remains (see "T3 first-run findings" below). | 🚧 in progress |
| **T4: `krunch plan` end-to-end** (~$2-5, ~30 min) | Build `krunch plan` for AWS Batch + k8s + Modal + local. Validate each emits a syntactically-valid artifact, then run **at least one target end-to-end** on the WildChat sample (likely AWS Batch since we already have the reference CDK + spike-6 validated path). Gates: same ratio, ~N× wall reduction at N=2, partial-blob cleanup happens, finalize task succeeds. Validates the "any batch system" README claim by exercising the env-var contract through one real scheduler. | 🚧 scaffolding shipped 2026-04-30 (templates + CLI + worker + CI dry-run); blocked on T3 green for e2e run |
| Docs | README, architecture.md, tuning.md, operations.md, benchmarks.md. 3-4 ratio benchmarks on public corpora (chat, support tickets, wiki, code) | ⏳ |
| Polish | typed Python client SDK, CI (lint + smoke), SECURITY.md, CONTRIBUTING.md | ⏳ |

No paid customers in v1 — that's v2's job.

### Notable design changes since the original plan

- **`krunch submit` deprecated → `krunch plan` (2026-04-30).** Reframed
  krunch from "distributed compression framework" to "codec library +
  Docker image + format spec." Distributing tasks across workers is
  what Modal/Ray/AWS Batch/k8s already do well — we don't reinvent
  that. Instead, `krunch plan --target <framework>` emits a
  ready-to-run artifact (job JSON, k8s YAML, Modal script, etc.) that
  the user runs with their own tooling and credentials. The container
  env-var contract (`KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`,
  `KRUNCH_PART_COUNT`, …) is the integration interface; targets are
  templates over it. `krunch submit` stays for one minor release as a
  thin shim around `plan --target aws-batch | aws batch submit-job`,
  then removed. `deploy/aws-cdk/` demoted to `examples/aws-batch/`.
- **Single-machine `--workers N` is the primary path for non-fleet
  users.** Most users have one beefy GPU box, not a cluster. `krunch
  compress --workers N` exposes the existing multi-process worker pool
  for compress (it already exists for decompress). One command, no
  scheduler, handles the ~70% case in the same one-line UX as
  `pigz`/`pbzip2`.
- **No always-on orchestrator instance.** Original plan had a custom EC2
  orchestrator that fanned out HTTP requests across worker EC2s. Dropped
  in favor of "user's own batch system runs the tasks; krunch emits the
  config." Workers read byte ranges directly from object storage — no
  data flows through any krunch-owned component. Scales to 100s of
  workers, costs $0 idle, and works on any framework via the env-var
  contract; `krunch plan` adapts the contract to per-target syntax.
- **Hot path is just `docker run`.** No always-warm server — if a
  customer wants always-warm, that's a one-line `docker run` on
  whatever GPU host they already operate. The reference AWS CDK
  (`examples/aws-batch/`) is opinionated for batch archival, the
  primary use case.
- **Kubernetes is a `krunch plan` target, not a separate manifest
  artifact in the repo.** Plan emits an Indexed Job + finalize Job.
  Users can also write their own ~30-line Job spec against the
  env-var contract if they want full control.
- **No FastAPI server.** Compression is a one-shot transform — input goes
  in, output comes out, done. Always-warm HTTP server adds operational
  complexity (lifespan, healthchecks, ports) and pays GPU cost while idle.
  Single-shot CLI replaces it; the user-facing UX is `krunch compress`,
  Docker is an implementation detail. Hosted HTTP API is v2.
- **No per-chunk dispatcher / classical fallback.** Original design had
  per-chunk neural-vs-zstd "shortest wins" to never lose to classical.
  Tier 3 measurements showed the chunking + format overhead made the
  dispatcher *worse* than whole-file zstd-22 by 4%, and capped neural's
  gains. Dropped — krunch is neural-only. If the LM doesn't compress your
  data, krunch isn't the right tool; use zstd directly. Documented in
  README's "When *not* to use krunch" section.
- **Chunk size: 1 MB → 64 KB** (revised 2026-04-30 after T3 sweep on
  a 3 MB WildChat sample): ratio cost of going to 64 KB chunks is
  +0.08% (essentially noise) and you get 16× more chunks → 16× more
  parallelism for the multi-process decompress pool. Compress wall is
  flat across chunk sizes 16K–4M. Defaults flipped. Override via
  `KRUNCH_CHUNK_SIZE`.
- **Package directory: `server/` → `krunch/`.** Standard Python convention
  (package = project name) and not actually a server anymore.
- **Test scripts reorganized into `tests/`.** Renamed Tier{1,2,3}_check.sh
  → quick.sh / integration.sh / gpu.sh. Smoke test → test_blob.py. CI
  runs test_blob.py on every push.
- **Streaming compress (range coder, not ANS).** First Tier 3 with the
  full neural-only path OOM-killed at 15 GB on g5.xlarge: the per-chunk
  logits buffer (`tokens × vocab × 4 bytes`) was 50 GB at 1 MB chunks
  for English chat (256K tokens × 50K vocab — my earlier math was off
  16× because I conflated Spike 6's `--seq-len 1024` with tokens-per-chunk).
  Fix: switched ANS → range coder, which is symmetric (encode/decode
  in same forward order). compress_chunk now interleaves the forward
  pass with range encoding batch-by-batch, so peak memory is
  O(SEQ_BATCH × vocab) ≈ 200 MB regardless of chunk size. Ratio is
  unchanged (both ANS and range hit Shannon entropy bound). Bitstream
  format changed (incompatible with any pre-streaming-era blobs;
  pre-launch, no users yet).
- **Cross-chunk parallel decompress: ThreadPool → multi-process** (revised
  2026-04-30). The v1.0 `KRUNCH_DECOMPRESS_BATCH=16` ThreadPool win
  evaporated when AC moved to GPU — every per-token step ends in a
  `.item()` GPU sync, so 16 threads serialize on one CUDA context + one
  Python GIL (measured **1.7× SLOWER** than sequential on T4). Default
  flipped to 1. Replaced by `DecompressWorkerPool` in
  `krunch/worker_pool.py`: N independent processes (default
  `KRUNCH_DECOMPRESS_WORKERS=4` on GPU), each with its own CUDA context
  → real GPU-level overlap. Measured **2.56× on T4 (N=4)**, expect 4-7×
  on A10G. No bitstream change.
- **Tier 3 always pulls `:latest` from ghcr.io.** Never `LOCAL_BUILD=1`
  except for local-diff sanity checks. The published image is the
  artifact users get; that's what we test. Per CLAUDE.md.

### Tier 3 status: single-instance compress + decompress speed

**Goal:** ratio holds, byte-exact roundtrip, **decompress within 2-3× of
compress wall on a single A10G**. The 30× compress/decompress asymmetry
is the gate.

**T3 status (2026-05-01):**

Bug fixed and pushed: **CRC mismatch was UTF-8 chunk boundary
slicing**, not a batched-decompress bug. `chunking._split_utf8_safe`
now snaps each chunk boundary back from UTF-8 continuation bytes
(0x80-0xBF) so multi-byte codepoints aren't cut. ASCII content
(my earlier synthetic tests) never hit this; WildChat hits it ~1%
of the time, which is why 158/160 chunks bit-exact while 2 chunks
got `\xef\xbf\xbd` (replacement char) where original was `\xe2\x80\x9c`.

Verified bit-exact on T4 with full 10 MB WildChat:
  compress 29 KB/s, decompress 18 KB/s, ratio 0.1166, BYTE_EXACT=PASS.
T4 numbers extrapolate to A10G ~3× → ~87 / 54 KB/s.

**Remaining T3 gaps (vs ratio ≤0.11, both speeds ≥200, byte-exact):**

| | A10G projection | Target | Gap |
|---|---|---|---|
| Ratio | 0.1166 | ≤0.11 | +6% |
| Compress | ~87 KB/s | ≥200 | 2.3× |
| Decompress | ~54 KB/s | ≥200 | 3.7× |
| Roundtrip | ✅ | ✅ | — |

**Plan (in order):**

1. ~~Encoder fusion via cuBLAS algo pin~~ **DEAD END (2026-05-01).**
   Swept all 16 `CUBLAS_GEMM_ALGO*_TENSOR_OP` enums on A10G; ZERO
   are bit-stable across M (all show ~10⁻³ abs diff between M=1
   and M=N for the same algo enum). The enum is a HINT to cuBLAS,
   not a binding — cuBLAS picks the actual kernel based on M
   regardless. Encoder fusion via cuBLAS is impossible without
   per-arch tuning + risking shape-dep on next cuBLAS version.
   See `scripts/test_cublas_pinned.py` SWEEP=1 output. Live
   det_matmul_cublas + KRUNCH_CUBLAS_PINNED env are kept in tree
   as opt-in for benchmark comparison only.

   **Compress speed honest state (2026-05-01):**
   - Measured A10G: 66 KB/s (10 MB WildChat, bit-exact path).
   - Gate: ≥200 KB/s. **Gap: 3.0× short.**
   - There's no direct apples-to-apples prior compress
     measurement; "stock 200 KB/s pre-GPU-AC" was on a DIFFERENT
     code path with CPU AC + constriction, never decoded back
     under our current bit-exact contract — so we don't have a
     credible "regression" or "improvement" claim against it.
   - Compress speed gap matters as much as decompress for the user
     UX: if decompress catches up to ≥200 via persistent kernel,
     compress at 66 becomes the new asymmetry users feel
     (decompress 3× faster than compress).

   **Arch-agnostic compress levers (in order):**

   1. ~~Tune the existing `det_matmul_tc`~~ **Tried 2026-05-01.**
      Built `det_matmul_tc_v2` (32×32 tile, 4 WMMA frags per block)
      — bit-stable across M (verified at M={1,8,16,32,64,128,1024,
      4096} on T4), microbench 1.7-2× faster. **End-to-end neutral**:
      compress 374s vs 356s baseline on T4 (~5% slower, in noise),
      66 KB/s on A10G — same as v1. Reason: profile shows forward =
      98% of compress wall and matmul is NOT the dominant cost in
      forward. Premix kernel + WKV + layer_norm + sigmoid + add
      add up to most of the per-step time. v2 kept in tree as
      `KRUNCH_TC_V2=1` opt-in for future use (e.g., bigger models
      where matmul dominates more).

   2. ~~CUTLASS-based GEMM~~ **Skipped.** Same diagnosis as (1):
      better matmul doesn't move forward wall when matmul isn't the
      bottleneck. Won't help here.

   **Real compress lever (= same as decompress):** persistent kernel
   that fuses many ops per launch + reduces ATen overhead. Compress
   forward at A10G is 530ms / chunk = 35 µs / token. To hit 200 KB/s
   = ~12 µs / token (~3× faster), need to fold the ~15 ATen ops per
   layer into one or two larger custom kernels. This is the same
   fused-kernel work needed for decompress; should design once and
   apply to both paths.

   **Persistent / fused layer kernel — design plan:**

   Current per-layer ATen op chain inside `rwkv4_layer_step_cpp`
   (~15 launches per layer × 12 layers = 180 per token):

     1. `at::layer_norm(x)`                — LN1
     2. `launch_premix_3` (custom)         — kx, vx, rx
     3. `gemm_fp16(rx, Rw)` → `at::sigmoid` — r
     4. `gemm_fp16(kx, Kw)` (fp32 out)     — k
     5. `gemm_fp16(vx, Vw)` (fp32 out)     — v
     6. `wkv_op` (rwkv custom)             — y_flat
     7. `r * y` (ATen mul)
     8. `gemm_fp16(ry, Ow)`                — out_att
     9. `x + out_att` (ATen add)           — x_after_att
    10. `at::layer_norm(x_after_att)`      — LN2
    11. `launch_premix_2` (custom)         — ffn_kx, ffn_rx
    12. `gemm_fp16(ffn_rx, ffn_Rw)` → sigmoid → r_ffn
    13. `gemm_fp16(ffn_kx, ffn_Kw)` → `launch_relu_sq` → k_ffn
    14. `gemm_fp16(k_ffn, ffn_Vw)`         — v_ffn
    15. `x_after_att + r_ffn * v_ffn`      — x_final

   Fusion targets (highest leverage first):

     A. ~~Replace at::layer_norm with custom layer_norm kernel.~~
        **Tried 2026-05-01.** Built `layer_norm.cu` (one block per
        row, fp32 reductions); enabled via KRUNCH_LAYERNORM_CUSTOM=1.
        Bit-exact T3 roundtrip PASS. End-to-end **3% slower** on T4
        (compress 366s vs 356s baseline; decompress 584s vs 572s).
        Diagnosis: at::layer_norm is already efficient + LN time is
        <1% of compress wall (~23K LN calls × ~50µs = ~1s out of 358s
        total compress). Single-op replacement doesn't help here.
        Kept in tree as KRUNCH_LAYERNORM_CUSTOM=1 opt-in.

     B. **Fuse `at::sigmoid` + multiply into one kernel.** sigmoid
        of an [n] tensor + multiplied by another [n] is one block
        per element-pair. 2 calls per layer × 12 = 24 ATen launches
        eliminated. ~half-day.

     C. **Fuse residual add into the matmul epilogue.** Ow output
        adds to x; ffn_V output adds to (x_after_att + r_ffn *).
        WMMA epilogue can do `acc + bias` for free. ~1 day to add
        epilogue support to det_matmul_tc.

     D. **Persistent kernel covering the whole layer step.** One
        `cudaLaunchKernel` does premix + matmuls + WKV + r*y + Ow.
        Single-block-per-batch design with explicit grid scheduling.
        ~3-5 days, biggest win (~3× decompress, ~2× compress).

   Cumulative (A+B+C): ~2 days, ~1.5-2× both sides.
   D alone: ~3-5 days, ~2-3× both sides. May overlap A/B/C effort.

   **Honest 200 KB/s reachability assessment (2026-05-01 after
   lever A measurement + careful math):**

   Compress (currently 66 KB/s on A10G):
   - Per-layer-step time: 3.7 ms = 16 ops × ~230 µs each
   - Of that 230 µs: ~50 µs launch overhead, ~180 µs actual GPU compute
   - Eliminating ALL launch overhead (full persistent kernel) saves
     16 × 50 = 800 µs / 3.7 ms = 22% per-layer-step → ~80 KB/s
   - Intermediate global memory r/w between fused ops is 1.5-12 KB
     per intermediate; total savings ~1 MB at 600 GB/s = negligible
   - **Compress ceiling on A10G with this model: ~80-90 KB/s.**
     Not 200. The model's flops/token are not enough to keep the
     TC pipeline full at small batches.

   Decompress (currently 35 KB/s on A10G with B=128):
   - Per-token cost is 142 µs of which most is per-step overhead
     from ~16 launches across many small operations
   - Persistent kernel ELIMINATES that per-step launch chain →
     1 launch per layer instead of 16
   - Plus M=B=128 keeps TC matmul saturated
   - Realistic decompress with persistent kernel: ~150-200 KB/s
     (true variance depends on WKV implementation efficiency
     inside the persistent kernel — sequential per-channel chain
     can't be parallelized further but lives in registers, no
     global memory r/w per step)

   **Verdict:** persistent kernel gets DECOMPRESS to gate (~150-200
   KB/s) but COMPRESS caps around 80-90 KB/s on a single A10G with
   this 169M model. Reaching 200 compress requires bigger GPU or
   smaller model.

   **Recommendation:** commit to persistent kernel for decompress
   (high ROI — closes the bigger gap). For compress, either accept
   ~80-90 KB/s OR shift the speed gate to acknowledge the
   model+hardware floor. Don't pretend single-A10G can hit 200
   compress with RWKV-4-Pile-169M; the math doesn't support it.

   **Persistent kernel decompress — sharper math (2026-05-01):**

   Measured A10G decompress: 35 KB/s aggregate at B≈128 = ~15 ms
   per timestep. Theoretical compute floor at B=128 from per-step
   ops (12 layers × 7 matmul + WKV + premix + LN + sigmoid + add):
   ~750 µs. Gap: 20× over theoretical.

   Persistent kernel attacks the 20× gap by collapsing 12 × 16 ≈
   190 ATen launches per step into ~10 fused-kernel launches.
   Realistic recovery: 2-3× (5-7 ms / step instead of 15 ms) →
   **~80-100 KB/s aggregate decompress on A10G**.

   **Not 200 KB/s.** The remaining 5-7 ms / step is real GPU
   compute — matmul, WKV recurrence, layer_norm reductions — that
   no amount of fusion eliminates.

   **Fused kernels in tree (2026-05-01):**
   - `fused_pre_attn.cu` — LN1 + premix3 fused, T=1 only (decompress).
     Skeleton built; not yet wired into rwkv4_layer_step_cpp_t1.
     Bit-stability vs the LN+premix chain needs verification before
     enable: requires both compress + decompress to use compatible
     LN arithmetic. Compress would use `launch_layer_norm` (already
     in tree) + `launch_premix_3` — same arithmetic, two launches
     instead of one.

   **Decision needed before continuing:**
   - (a) Build out remaining 3 fused kernels (sigmoid+WKV+mul,
        residual+LN2+premix2, sigmoid+mul+add), wire into engine,
        measure on A10G. ~3 more days for 2-3× decompress (target
        ~80-100 KB/s).
   - (b) Accept that single-A10G with 169M can't hit 200 either way;
        re-spec gate to ~80-100 KB/s decompress + ~80 KB/s compress.
        Ship what we have.
   - (c) Switch hardware target to A100/L40S/H100 — bigger TFlops
        + bigger B fit; 200 KB/s likely reachable with current code.
   - (d) Switch model to a smaller one (e.g., RWKV-4-Pile-1.5B is
        OPPOSITE direction; would need a sub-100M variant). Big
        product change.

2. **Bigger-sample T3 measurement** — re-run on 100 MB+ WildChat
   slice with the encoder fusion live. Decompress gate may be
   reachable without the persistent kernel because larger files
   support higher B without dropping chunk size (the ratio-cost
   tradeoff that caps a 10 MB file).

3. **Persistent kernel for decompress** — only if (2) shows
   decompress still <200 KB/s. 1-2 weeks of CUDA work. Sized
   from prior measurements: A10G with B=128 plateaus around
   120-150 KB/s; persistent kernel removes per-step launch
   overhead floor.

4. **Ratio gap (+6%)** — minor. If everything else lands and ratio
   is the only gap, can revisit chunk size / model numerics.

**Shipped (in tree, exercised by tests/gpu.sh):**

- **GPU AC encode kernel** (`krunch_ac/cuda/encode_kernel.cu`). 200 KB/s
  on T4, ~155 KB/s on A10G. Compress is launch+forward bound, not AC
  bound — see "compress backlog" below for further wins.
- **GPU AC decode kernel** (`decode_kernel.cu`). 11/11 unit tests pass,
  byte-identical roundtrip vs CPU reference.
- **Batched encode/decode kernels** for B independent streams in one
  launch — used by the dormant chunk-batched compress path. 4 unit
  tests pass at B=2..8, V=32..50277.
- **64 KB default chunks** (was 1 MB). 0.08% ratio cost on a 3 MB
  WildChat sample, gives 16× more chunks for the worker pool.
- **Multi-process decompress worker pool** (`krunch/worker_pool.py`).
  N independent CUDA contexts via `mp.spawn`. Default
  `KRUNCH_DECOMPRESS_WORKERS=4` on GPU. Measured 2.56× on T4 (N=4),
  expect 4-7× on A10G.
- **`krunch warmup`** post-install command. Materializes the WKV
  ninja-build + warm filesystem cache into a docker volume so the
  first real `compress` doesn't pay ~60 s of one-time costs.

**Current numbers (T4 measured, A10G extrapolated):**

| | T4 measured | A10G estimate |
|---|---|---|
| Compress, 1 MB | ~38 s (~26 KB/s) | ~21 s (~48 KB/s) |
| Decompress sequential, 1 MB / 16 chunks | extrapolated 37 m | ~28 m |
| Decompress with worker pool N=4, 1 MB | 14 m 36 s | ~5-9 m (N=4-8) |
| **Asymmetry compress vs pool decompress** | ~25× | ~10-15× |

**Open work — custom fused single-step CUDA kernel (chosen 2026-04-30).**

After ruling out cheaper paths via measurement, the only viable lever
left for closing the remaining ~10× asymmetry under the constraints
"don't slow compress, don't change bitstream, don't add a model" is
a custom CUDA kernel that bypasses BlinkDL's Python wrapper.

Cost model on T4:
- Per-token wall via BlinkDL forward: **7.5 ms/token**
- Memory-bandwidth floor (180 MB weights / 320 GB/s): **0.56 ms/token**
- Realistic kernel target: **1-2 ms/token = 4-7× single-process**
- Composes with worker pool (~2-4× more on top): **8-30× total**

Scaffolding in `krunch_ac/cuda/rwkv_step.cu`. Plan, ~1-2 weeks:
1. **Pure-torch single-step reference — DONE 2026-04-30.**
   `scripts/rwkv4_step_ref.py` implements one-token-at-a-time forward
   in plain torch (~150 LOC). Verified on T4: drift vs
   `forward_batched(T=1)` is **0.0-0.04 max abs across 6 token
   positions** (well within fp16 noise). This is the math the CUDA
   kernel must implement. Provides:
   - A simple, debuggable ground-truth for kernel correctness tests.
   - The exact tensor/state layout the kernel inherits.
   - A torch.compile-friendly path (it doesn't use BlinkDL's custom
     ops) — useful as a fallback if the kernel hits issues.
2. **Single-layer fused CUDA kernel — CORRECT BUT SLOW 2026-04-30.**
   Wrote `krunch_ac/cuda/rwkv_step.cu` (`rwkv4_layer_step_kernel`) +
   bindings + correctness test (`scripts/test_rwkv_step_kernel.py`).
   Single-block 768-thread design with per-thread GEMV reductions
   over n_embd=768.

   **Correctness on T4 (vs `_layer_step` reference):**

   | output | max abs diff |
   |--------|-------------:|
   | x_out  | 0.01562 |
   | att_xx | 0.00000 |
   | aa     | 0.00305 |
   | bb     | 0.00000 |
   | pp     | 0.00975 |
   | ffn_xx | 0.00062 |

   All well within fp16 noise. PASS.

   **Speed on T4 (12 layers per token):**
   - kernel:  11.187 ms/token
   - BlinkDL:  7.671 ms/token
   - **kernel/BlinkDL = 0.69× — kernel is 1.45× SLOWER**.

   **Root cause of the slowness:** my hand-written GEMVs use scalar
   `fp16→fp32 multiply-add`. BlinkDL's `gemm_fp16_cublas` uses **tensor
   cores** (T4: ~65 TFLOPS Tensor Core vs ~16 TFLOPS scalar fp32 =
   4× compute headroom missed). Also, single-block-per-layer launch
   uses only 1/40 of T4's SMs — terrible utilization for an already
   memory-bound workload.

   **Bottleneck diagnosed (refined estimate 2026-04-30):**
   Per-layer wall is 11.2 ms / 12 = **0.9 ms/layer**. Per-layer memory
   read = 15 MB. T4 HBM bandwidth = 320 GB/s → mem-bw floor 0.05 ms.
   We're **18× over the floor**. Compute isn't the issue — at B=1, the
   per-token forward is 96M muladds total (~6 ms scalar at full T4
   compute, but we're using 1/40 of SMs per launch). The dominant
   cost is **single-block-per-SM HBM bandwidth**: with only 1 block
   per kernel launch, only 1/40 of T4's HBM read capacity is in
   flight. Tensor cores would help compute but compute isn't the wall.

   **The actual optimization needed:** multi-block per layer (24+
   blocks distributing the 768 channels across SMs), with grid-level
   sync between layers via `cooperative_groups::this_grid().sync()`
   on a persistent kernel, OR materialize per-layer outputs to HBM
   and launch sequentially with multi-block layers. Both routes
   ~3-5 days of careful CUDA. Tensor-core MMA can be added later as
   a polish; it's secondary to the multi-SM utilization fix.

   The kernel scaffolding + correctness test is the foundation for
   the optimization work. The redesign keeps the verified math and
   replaces only the dispatch / block-decomposition.

   **Tried & reverted (2026-04-30):** `__ldg(const __half*)` for
   read-only weight loads — broke correctness (x_out diff jumped to
   0.75 vs ref's 0.016). Either the CUDA `__ldg` overload for `__half`
   has subtle precision behavior, or the compiler emitted misaligned
   loads. Plain pointer reads stay; revisit with explicit
   `unsigned short` reinterpret cast if `__ldg` is needed.

   **V2 multi-block GEMV — TESTED & ABANDONED 2026-04-30.**
   Wrote `krunch_ac/cuda/mb_gemv.cu` with multi-block GEMV kernels for
   the three RWKV-4 shapes. <<<N/32, 32>>>: each block handles 32
   contiguous outputs, 24+ blocks across SMs. **Correctness ✓** (max
   diff ≤ 0.0002). **Speed (T4):**

   | shape | kernel μs | torch `@` (cuBLAS) μs | kernel/torch |
   |-------|-----------|------------------------|-------------:|
   | 768×768   | 33.87  | 20.89 | 0.62× |
   | 768×3072  | 74.16  | 30.79 | 0.42× |
   | 3072×768  | 274.75 | 23.67 | **0.09×** (11× slower) |

   **Conclusion: cuBLAS already beats naive multi-block CUDA on these
   shapes by 1.6-11×.** cuBLAS uses tensor cores + optimized memory
   patterns; matching it requires `mma.sync` MMA + cooperative
   reduction over K — real CUDA expert work, ~1-2 weeks just for the
   GEMV. Not a productive path.

   **C++ orchestration validated 2026-04-30:** wrote
   `krunch_ac/cuda/layer_cpp.cpp` — one C++ call per layer,
   `gemm_fp16_cublas` for matmuls (same op BlinkDL uses) + `at::layer_norm`
   + WKV via dispatcher. Bench on T4:
   - C++ full forward (12 layers): **6.32 ms/token**
   - BlinkDL: 7.55 ms/token
   - **C++ is 1.19× faster than BlinkDL — first time we've beaten it**

   Drift vs BlinkDL T=1: 2.4-4.5 abs (unchanged by gemm_fp16_cublas use).
   BlinkDL's `forward_one` (T=1 path) uses different op orchestration
   than `forward_seq` (T>1) — different intermediate dtype rounding.
   Same drift signature as `forward_batched` vs BlinkDL.

   Composed with multi-process pool: 1.19× × 2.56× ≈ **3× over current
   sequential decompress on T4**. On A10G expect ~5-7×.

   **Numerical match BREAKTHROUGH 2026-04-30.** Wrote
   `rwkv4_layer_step_cpp` (the T>1 packed counterpart to
   `rwkv4_layer_step_cpp_t1`). Both share identical ATen op orchestration.
   Result on T4 over a 16-token sequence:

   | path comparison | drift |
   |-----------------|-------|
   | C++ packed T=N vs C++ stepped T=1 | **0.016-0.063 max abs** ✓ fp16 noise |

   **AC roundtrips cleanly between the two paths — no quantized AC
   needed, no ratio cost.**

   **Compress: turned out C++ BEATS BlinkDL when comparing apples-to-apples.**
   Earlier numbers compared C++ `full_output=True` vs BlinkDL's
   default `forward(tok_list)` (last-position logits only). But
   `compress_chunk` calls `forward(batch, state, full_output=True)` —
   it needs one logit vector per token. When both paths compute the
   same workload (T4 bench):

   | path | per-call ms | vs BlinkDL `full_output=True` |
   |------|-------------|-------------------------------|
   | BlinkDL packed `full_output=True` (compress workload)| **19.97** | 1.00× |
   | BlinkDL packed last-only (apples-to-oranges)         | 13.47 | n/a |
   | C++ packed unwrapped                                  | 17.66 | **1.13× FASTER** |
   | **C++ packed full-forward CUDA graph**                | **15.15** | **1.32× FASTER** |

   **Compress speeds up 1.32× on the actual workload — no regression,
   it's an improvement.**

   Engineering that landed this:
   - Fused `premix_3` + `premix_2` + `relu_sq` custom kernels in
     `krunch_ac/cuda/premix_kernels.cu` (saves ~3 ms vs pure ATen).
   - Full-forward CUDA graph capture via Python `torch.cuda.graph` over
     all 12 layers + final LN + head (saves another ~2.5 ms).
   - `rwkv4_layer_step_cpp_graphed` per-layer graph cache also in tree
     (smaller win, full-forward graph dominates).

   **AC roundtrip blocker — fundamental drift problem 2026-04-30.**
   End-to-end byte-exact AC roundtrip (compress with C++ packed →
   decompress with C++ stepped) **FAILS** at step 2-7 of any 32-64
   token sequence. Root cause: logit drift between C++ packed (T=1024)
   and C++ stepped (T=1) is **0.19 max abs on a 64-token sequence**
   (grows with sequence length). Source: cuBLAS auto-selects different
   GEMM algorithms for different T shapes; fp16 accumulation order
   differs across algos.

   **Logit quantization at the head doesn't fix it.** Sweep over
   SCALE ∈ {64, 32, 16, 8, 4, 2, 1} all show roundtrip FAIL at 0% ratio
   cost (the AC bitstream is identical because compressed sizes are
   byte-equal, so quantization itself isn't lossy — but the rounding
   boundary problem dominates).

   The math: for drift d=0.19 and step=1.0, P(round-flip per element)
   = 0.19 / 1.0 = 19%. Across 50K vocab × 64 positions, ~600K of the
   ~3.2M CDF entries differ → AC decode diverges as soon as one
   high-probability token flips at step 2 → cascade.

   To get round-flip rate to ~0% requires step ≪ d/100 = 0.002, which
   is finer than fp16 resolution. Logit quantization at the head is
   architecturally unable to fix this.

   **Mid-layer activation quantization also does NOT fix it.** Tested
   2026-04-30: applying `round(x * SCALE) / SCALE` to the residual
   activations after EVERY layer (so encoder/decoder re-synchronize at
   each layer boundary) fails roundtrip across SCALE ∈ {256, 128, 64,
   32, 16, 8, 4}. Same root cause: drift across cuBLAS shape choices
   compounds non-uniformly, and quantization's discontinuity at
   rounding boundaries is hit constantly across the 50K-vocab × 64-step
   surface.

   **Path forward — three options, all multi-day:**
   1. **Custom shape-invariant matmul kernel** — own GEMV/GEMM with
      deterministic accumulation order, bit-identical regardless of T.
      Loses ~2-10× of cuBLAS speedup → compress regresses to ~50-100s
      on A10G. ~1-2 weeks.
   2. **Sync codes in bitstream** — encoder emits a checkpoint byte
      every K tokens; decoder uses it to recover from drift. Bitstream
      change. ~1-2% overhead. ~3-5 days.
   3. **Slow encoder via C++ stepped** — encoder uses same T=1 path as
      decoder. Bit-identical numerics, no quantization. But compress
      drops to ~7-11 min/MB on T4. Untenable.

   Options (1) and (2) can ship without violating "no compress
   regression + no ratio cost" only marginally — both have small
   real costs.

   **End-to-end projection** on 1 MB / 16 × 64 KB chunks:
   - Compress (T4): 38s → ~29s (1.32× faster)
   - Compress (A10G): 21s → ~16s (projected)
   - Decompress (C++ × multi-process pool): ~3× T4 / ~5-7× A10G
   - **Asymmetry: 30× → ~2-3×** on T4, ~1.5× expected on A10G
   - No bitstream change, no extra model, no ratio cost

   **Recast (2026-04-30): the right architecture is C++ orchestration
   over cuBLAS, not a custom CUDA fused kernel.** `forward_batched`
   already uses cuBLAS-equivalent (`gemm_fp16_cublas`) for matmuls but
   is **slower than BlinkDL** (12 ms vs 7.5 ms) due to ~120 Python
   torch op calls × ~75 us each = **9-13 ms of pure Python overhead
   per token**. The path that actually beats BlinkDL:
   1. Rewrite `forward_batched` orchestration in C++ as a single
      `forward_batched_cpp` extension that takes the model state +
      input token and runs all 12 layers' worth of cuBLAS calls
      directly without Python round-trips.
   2. Fuse only the small elementwise + WKV ops as custom kernels
      (V1 kernel's LN + time-mix premix + WKV portions are already
      written; reuse).
   3. Expected speedup: cuBLAS time + small custom = clearly under
      BlinkDL's 7.5 ms/token. Possibly 2-3× faster.

   Estimated effort: 3-5 days of careful C++ + binding work. Lower
   risk than custom fused CUDA. Current V1 kernel + mb_gemv code
   stay as references and for the elementwise-fusion subset.

   **Update 2026-04-30 (late) — model side now bit-exact.** Built and
   landed:
   - `det_matmul.cu` — shape-invariant deterministic GEMM. One thread
     per output element, sequential fp32 accumulation. **Verified
     M-invariant (max abs diff = 0.0)** across (M=64, K=768, N∈{768,
     3072}) and K=3072. cuBLAS comparison drifts ~0.002–0.004 abs.
   - Toggle via `KRUNCH_DETERMINISTIC_MATMUL=1` env. `gemm_fp16` in
     `layer_cpp.cpp` switches paths at static init.
   - **Unified `rwkv4_layer_step_cpp_t1` premix with the packed
     path**: stepped path now also uses `launch_premix_3`,
     `launch_premix_2`, `launch_relu_sq` (was using ATen elementwise
     ops, which use fp16 arithmetic → drift vs fp32-internal kernel).
   - Direct pybind binding `det_matmul` exposed for testing.

   **Verified results (real RWKV-4-Pile-169M weights, 12 layers,
   T=31):**
   - **Layer-stack output: bit-identical** between T=31 packed and
     T=1×31 stepped, every timestep, max_abs = 0.000.
   - **All 12-layer state: bit-identical** (att_xx, aa, bb, pp,
     ffn_xx) at every layer.
   - **ln_out + head logits: bit-identical** (per-row vs batched).
   - **CDFs: bit-identical** through `probs_to_cdf_gpu` (per-row vs
     batched), 0/31 row mismatches.

   **AC roundtrip status: PASS — bit-exact byte-for-byte 2026-04-30.**
   Final root cause: PyTorch's CUDA `torch.softmax(., dim=-1)` is
   shape-dependent — `softmax([T,V])` and `softmax([1,V])` of the same
   row don't produce bit-identical outputs (different reduction
   strategy). Fix: encoder runs softmax row-by-row to mirror the
   decoder's [1,V] invocation shape. Verified with synthetic CDF
   roundtrip that the AC kernels themselves are bit-symmetric
   (`scripts/test_ac_only_roundtrip.py`). End-to-end roundtrip on
   32-token sample: 31 bytes (~7.75 bits/token), all tokens decoded
   exact, 0 CDF mismatches.

   **Cost:** per-row Python loop in encoder for softmax + CDF. For 32
   tokens this is negligible; for large chunks it adds Python
   overhead. Follow-up: write a deterministic batched softmax kernel
   (per-row max/exp/sum in fp32 with explicit serial reduction) to
   restore batched encoder throughput without breaking bit-exactness.

   **End-to-end engine roundtrip via cpp_path (T4, 2026-04-30, latest):**
   ```
   8 KB  : C++  enc 44.7 KB/s  dec 1.3 KB/s  ratio 0.055  PASS bit-exact
   8 KB  : stck enc 44.5 KB/s  dec 0.6 KB/s  ratio 0.044  (no roundtrip)
   32 KB : C++  enc 33.2 KB/s  dec 1.3 KB/s  ratio 0.084  PASS bit-exact
   32 KB : stck enc 44.9 KB/s  dec 0.7 KB/s  ratio 0.070  (no roundtrip)
   ```
   Encoder matches stock at 8 KB and is within 1.35× at 32 KB.
   Decoder is **2× faster than stock decode** at every size and is
   bit-exact. Asymmetry collapsed to 26-35× from cpp's actual numbers.

   Per-token decode profile (T4): forward 3.90 ms (96.4%), cdf 0.12,
   decode 0.01, sync 0.02 — total 4.05 ms/token. Forward is ~96% of
   decode time and dominated by per-launch kernel overhead from
   ~180 op launches per token (60 matmuls + 120 ATen ops).

   **A10G validation (g5.xlarge, 2026-04-30):**
   ```
   8 KB  : C++  enc 132.2 KB/s  dec 1.2 KB/s  ratio 0.055  PASS
   8 KB  : stck enc   0.7 KB/s  dec 0.7 KB/s  ratio 0.044  FAIL
   32 KB : C++  enc  98.7 KB/s  dec 1.2 KB/s  ratio 0.084  PASS
   32 KB : stck enc  52.8 KB/s  dec 0.7 KB/s  ratio 0.070  FAIL
   ```
   Encoder scales ~3× from T4 to A10G (33→99 KB/s, 45→132 KB/s) —
   compute-bound, benefits from A10G's TFLOP advantage. **cpp_path
   beats stock encode at 32 KB on A10G** (98.7 vs 52.8 KB/s, 1.87×
   faster) — the deterministic kernels apparently scale better than
   BlinkDL's JIT-scripted forward at this size.

   Decoder is **identical T4 vs A10G** (1.2 KB/s on both, per-token
   forward 4.07 ms on A10G vs 3.90 ms on T4 — within noise). This
   confirms the decode bottleneck is CPU-side launch overhead, not
   GPU compute. **A10G cannot help decode** until the per-token
   forward is collapsed into one launch (persistent kernel).

   Distance to "decompress = compress at 200 KB/s" goal:
   - Compress: 132 KB/s on A10G — need 1.5× more (close).
   - Decompress: 1.2 KB/s on A10G — need 167× more.
   No single-stream optimization closes this gap. **Cross-chunk
   batched decode is the only viable lever.**

   **Path forward (2026-04-30) — cross-chunk batched stepped
   forward.** Per-timestep launch overhead (~4 ms) is independent
   of B (matmul work scales but launches don't), so processing B
   chunks in parallel through one batched layer call gives near-
   linear throughput scaling until TC matmul utilization saturates.

   Math at B=128 chunks in parallel:
   - 4 ms/timestep × ~6000 timesteps/chunk = 24 s for one whole
     batch
   - 128 chunks × 32 KB = 4 MB in 24 s = **170 KB/s decode** on T4
   - On A10G with better TC utilization at M=128: extrapolate
     ~250-400 KB/s. Comfortably over the 200 target.

   Implementation plan:
   1. Drop B==1 assert in rwkv4_layer_step_cpp; verify it works
      at B>1, T=1 (the kernel already takes (B, T, C) flat
      [B*T, C] internally).
   2. Add cpp_path.forward_stepped_batched(weights, last_tokens
      [B], states[B]) wrapper.
   3. Verify bit-exact: B chunks decoded batched produces the
      same per-chunk tokens as B chunks decoded sequentially.
   4. Wire decompress_all to dispatch all chunks of a file as
      one batched stream (or in batches of MAX_B if memory
      caps).
   5. Bench WildChat sample on A10G end-to-end. Target: avg
      compress ≥ 200 KB/s, avg decompress ≥ 200 KB/s, ratio
      ≤ 0.11, byte-exact roundtrip.

   Estimated effort: 2-3 days focused.

   **Compress speedup plan — DEFERRED past T3 gate (per Dan,
   2026-04-30).** T3 only requires decompress to hit 200 KB/s; once
   that lands and T3 passes, T4 work begins and compress
   optimization can be sequenced after. Plan retained below for
   reference.

   ----
   **Compress speedup plan (post-T3, to hit ≥200 KB/s on A10G;
   stretch 700 KB/s).**

   Current A10G measured: 132 KB/s @ 8 KB chunks, 99 KB/s @ 32 KB
   (memory-bandwidth bound on the head matmul at large T). Need
   ~1.5-2× to hit 200, ~5× for 700.

   Levers ranked by leverage:
   | Lever | Est. gain | Effort |
   |---|---|---|
   | Pin `cublasGemmEx` algo + replace det_matmul_tc on layer
     matmuls (still shape-stable because algo is fixed) | 2-4× layer
     matmuls | 1 day |
   | Bigger TC tiles (128×128 + async copy + double buffer) in
     det_matmul_tc | 2-3× | 2 days |
   | Encoder ATen → custom kernels (layer_norm + sigmoid + residual
     add fused, kills dispatch overhead) | 20-40% | 2 days |
   | Stream pipelining (forward / softmax / AC encode overlap) |
     15-25% | 1 day |
   | Cross-chunk batched encoder for small chunks (amortize
     per-launch overhead at small T) | 1.5-2× | 2 days |

   Plan to 200 KB/s on A10G: cuBLAS-algo-pin + stream pipelining.
   Cumulative ~2-2.5×. **3-4 day estimate.**

   Plan to 700 KB/s: all five stacked. Cumulative ~5-7×. ~1.5-2
   weeks. A10G memory bandwidth on the head matmul likely caps
   real-world around 400-500 KB/s; 700 KB/s is more comfortable on
   A100/H100. If 700 is hard requirement, recommend a bigger GPU
   for v1 ship vs further encoder work.

   Order of attack: starting with cuBLAS-algo-pinning since it's the
   highest-impact, lowest-risk single change.

   **Worker-pool decode is dead as a lever for cpp_path.** Measured
   4-worker aggregate at 0.9 KB/s vs single-stream 1.3 KB/s — workers
   contend for the same SMs since cpp_path now saturates the GPU.
   The earlier worker-pool gains (1.81-2.12×) were when cpp_path
   was Python-bound; not the case anymore.

   **Remaining decode levers:**
   - More multi-way matmul fusion (ffn_Rw + ffn_Kw stacked, even
     with different N).
   - Epilogue fusion: sigmoid into 3-way output, relu_sq into
     ffn_Kw output, residual-add into output matmul. ~24 launches
     saved per token (~10% decode win each).
   - Persistent kernel covering ln+premix+attn+wkv+r*y+Ow+ln+premix+
     ffn block in one launch. 5-7 day item; would gain 5-10×.
   - Half-precision WKV with TC: WKV is currently a serial fp32
     kernel; could co-locate with surrounding matmuls.
   Encoder matches stock at 8 KB and is within 1.28× at 32 KB.
   Decoder beats stock at every size and is bit-exact. Asymmetry
   collapsed from the original 30-68× to ~42-57× of cpp's actual
   (working) numbers — the residual gap is the per-token forward+
   sync floor, not the model arithmetic.
   Ratio_dec/enc collapsed from 30–68× → 12×. But two regressions
   surfaced:
   1. **Encode 4× slower than stock** at 32 KB (10.4 vs 44.1 KB/s).
      Was 7× slower; closed somewhat by batched det_softmax_cdf
      kernel (`krunch_ac/cuda/det_softmax_cdf.cu`, +60% encoder
      speedup, bit-stable across [T,V] vs [1,V] invocation).

      **Profiling (KRUNCH_CPP_PROFILE=1) at T=6360 / 32 KB chunk:**
      ```
      forward    2969.7 ms  (97%)
      cdf          75.1 ms
      ac            5.1 ms
      total      3051.0 ms
      ```
      The C++ packed forward is the bottleneck, not anything around
      it. Cause: `det_matmul.cu` uses one thread per output element
      with a serial fp32 K-loop — no Tensor Cores. At M=6360 K=768,
      this is dramatically slower than cuBLAS's TC-accelerated GEMM.
      cuBLAS gives the speed but is shape-dependent (different
      algo for M=1 vs M=6360 → fp16 drift breaks AC roundtrip).

      **Fix:** rewrite `det_matmul` to use Tensor Cores via the
      WMMA / `mma.sync` API. Bit-invariant across M is achievable
      by always using the same block tile schedule + the same
      reduction order; this is what makes WMMA a viable
      replacement (unlike cuBLAS which auto-selects). Estimated 2-4
      days of careful CUDA work + verification it stays bit-stable.
   2. ~~**Ratio degraded ~20%**~~ — initially flagged as a
      regression, but the "stock" baseline number is meaningless: the
      stock path FAILS roundtrip (the comparison row in the bench
      output explicitly says FAIL), so its compressed size is what
      the encoder produced, not what's recoverable. cpp_path is the
      first GPU-AC path that actually decodes back. Independent
      precision check (`scripts/test_det_matmul_precision.py`) shows
      det_matmul matches cuBLAS to within fp16 quantization for all
      our shapes — abs_mean diff vs fp64 ground truth is identical
      to cuBLAS for K∈{768,3072}, N∈{768,3072,50277}, M∈{1,32,1024}.
      Real cpp_path ratio (0.055 @ 8 KB, 0.084 @ 32 KB) is the
      faithful compression number for this model + this text.

   Decompress speed unchanged from stock (0.5 vs 0.7 KB/s) — the
   per-token forward+sync floor is what actually bounds decode, and
   we haven't attacked that yet (CUDA graphs around the stepped C++
   path + worker pool are the next levers).

   **Next concrete steps:**
   1. Investigate ratio gap: does dropping fp32-accumulate to a
      different scheme (e.g., Kahan, or pairwise reduction) close
      it, or is fp16-output the floor? Try fp32 output + post-cast
      after the head only (head computes more carefully than mid
      layers).
   2. Deterministic batched softmax+CDF kernel — kill the per-row
      Python loop, restore batched encoder speed.
   3. CUDA graph the per-token C++ stepped path so decode latency
      drops below ~200 us/token. **Tried, blocked.** Both v1 (C++
      wrapper) and v2 (Python torch.cuda.graph with
      snapshot/restore around capture) FAIL roundtrip. Single-layer
      reproducer (`scripts/test_graph_one_layer.py`) shows the
      graph diverges from a ground-truth direct call immediately
      on step 0 (out_diff = 0.18, then explodes). The layer
      forward calls two ops that don't appear to capture
      deterministically in this PyTorch+CUDA build:
      `rwkv::gemm_fp16_cublas` (rwkv's custom op via the
      Dispatcher) and `rwkv::wkv_forward`. Likely either uses
      cuBLAS workspaces that aren't graph-safe, or stream-ordered
      memory that doesn't replay. Fix is multi-day: replace those
      two custom ops with graph-safe equivalents (bare cuBLAS API
      with explicit workspace, our own WKV kernel inline). Until
      then decode is bounded by per-token Python+kernel-launch
      latency.

   Files added/modified:
   - NEW `krunch_ac/cuda/det_matmul.cu`
   - MOD `krunch_ac/cuda/layer_cpp.cpp` (env-toggle gemm + unified t1
     premix + `det_matmul` pybind)
   - MOD `krunch_ac/cuda/setup.py` (sources)
   - NEW `scripts/test_det_matmul.py`
   - NEW `scripts/test_layer_packed_vs_stepped.py`
   - NEW `scripts/test_full_model_packed_vs_stepped.py`
   - MOD `scripts/test_e2e_ac_roundtrip.py` (uses det_matmul on head)
3. **Whole-step fused CUDA kernel** (~3-5 days). Persistent kernel
   across 12 layers + final LN + head GEMV. One launch per token.
   State buffers in HBM, mutated in place. Wired into
   `decompress_chunk` as `_compiled_step_forward`.
4. **Compress-side compatibility decision.** Pure-torch packed T=1024
   is ~60% slower than BlinkDL's T=1024 path (measured: 21.6 ms vs
   13.5 ms/call) — so we can't simply switch compress to the
   reference. Three options:
   - (a) Match BlinkDL's T=1024 numerics in the kernel by using
     identical fp16 cuBLAS algorithms + reduction order. Doable but
     fragile (tied to library version).
   - (b) Use lever (f) quantized-prob AC: 16-bit fixed-point CDFs
     tolerate ~5 abs drift between encoder and decoder paths. ~2-5%
     ratio cost.
   - (c) Accept the compress regression — switch compress to
     forward_batched packed T=1024 too. Kernel and forward_batched
     match within 0.04 abs, AC roundtrips. Compress drops from
     ~155 KB/s to ~95 KB/s on A10G. **Fails the user's "don't slow
     compress" constraint.**
   Plan: (b) is the safest and lowest-risk; revisit (a) if the ratio
   cost matters.
5. **End-to-end validation:** byte-exact compress→decompress on a
   text sample, decompress wall on T4 cheap, then A10G full T3.

**Levers held in reserve (compose with the fused kernel if needed):**

- (e) **Parallel rANS streams in the bitstream.** ~1-2% ratio cost,
  ~4-8× decode speedup at K=8. Bitstream format change required.
- (f) **Quantized-probability AC.** Quantize CDFs to 16-bit fixed-point
  so encoder/decoder don't need bit-exact float match. ~2-5% ratio
  cost. Unlocks alternative inference engines (web-rwkv Rust+WGPU,
  rwkv.cpp ggml) without numerical-reproducibility bugs.
- (g) **Self-speculative greedy lookahead.** No extra model, no
  bitstream change. Top-1 acceptance measured 33.6% on prose / 27.2%
  on code → ~1.5× max because chain speculation is structurally
  sequential. Composes multiplicatively with the fused kernel.

**Dead ends (one-line summaries; full detail in git history):**

- Symmetric `compress_chunks_batched` (force compress to lockstep
  B=N, T=1 to match decompress) — correct but **slows compress 5×**;
  off-table per "don't slow compress" constraint. Code retained,
  env-toggle `KRUNCH_COMPRESS_BATCHED=1`.
- ThreadPool decompress with per-thread CUDA streams — no win; per-token
  `.item()` sync + GIL contention. Replaced by multi-process pool.
- Naive batched decompress (B=46) at 64 KB chunks — **3.8× wall
  reduction** on T4 but byte-different output (compress's
  `rwkv.forward(packed)` and decompress's `forward_batched` differ by
  ~12 abs in logits → AC roundtrip breaks).
- `forward_batched(B=1, T=1)` to bypass BlinkDL Python wrapper —
  measured **slower** than BlinkDL (12.0 vs 7.5 ms/token on T4); also
  drifts 3-5 abs vs BlinkDL at the same shape.
- `torch.compile(mode="reduce-overhead")` on decompress's per-step
  forward — silently falls back to eager (BlinkDL has Python control
  flow that breaks graph capture). 0% measured speedup.
- **Compile-friendly forward_batched (round 3, 2026-04-30).** Replaced
  `gemm_fp16_cublas` with plain `@`, replaced `wkv_forward` custom op
  with pure-torch WKV (env-toggles `KRUNCH_PLAIN_MATMUL=1`,
  `KRUNCH_PURE_WKV=1`). torch.compile **finally works** under both
  modes (default + reduce-overhead). Result: compiled fb at T=1 is
  **14.3 ms/token vs BlinkDL's 7.5** — **2× slower**. Eager fb is also
  slower (13.4 ms). Pure-torch WKV at T=1024 packed is 2.6 s/call
  (**200× slower than the WKV kernel**), so a hybrid pure-T=1 + kernel-
  T>1 path doesn't help compress either. **No pure-torch path beats
  BlinkDL** — the Python overhead of iterating 12 layers in
  forward_batched is itself the bottleneck. Confirms the only viable
  decompress speedup ≥5× is the custom CUDA fused kernel.
- Speculative decoding from LLM literature, Jacobi/Lookahead, SSM
  parallel scan — surveyed and confirmed AC-incompatible (trajectory
  pinned by bitstream, not sampled).
- Embedded full-state checkpoints in the bitstream (Qualcomm CABAC-style)
  — RWKV state is ~150 KB per checkpoint vs CABAC's 16 bytes; one
  checkpoint exceeds a chunk's compressed size. Doesn't transfer to
  LM codecs.
- Self-speculative greedy lookahead at K=4 — speculative chain is
  sequential by construction → effective speedup is structurally
  bounded at ~1/(1-p_top1) ≈ 1.5×.
- Batched WKV via `batched_rwkv4.py` at B=8 (compress side) — only
  1.2× scaling (WKV kernel is recurrence-bound within a (batch, channel)
  thread). Code retained for the fused-kernel design as the reference
  forward.

### Compress backlog (Tier 3 secondary — gate is decompress)

Compress already passes the revised gate (≥150 KB/s commit / ~200 KB/s
observed on T4, expected ~155 KB/s on A10G). Further wins available if
needed:

- **Fused CDF construction kernel** (~7× win on the CDF stage, ~1 day).
  One kernel reads probs, multiplies/floors, finds argmax, writes
  counts. Eliminates 4-5 intermediate (1024, 50K) tensors torch
  allocates per call. Estimated CDF construction drops from ~15 ms
  to ~2 ms on T4 → end-to-end ~340 KB/s on A10G, **likely past the
  original 300 KB/s gate.**
- **Top-k probability truncation** (~30× win on CDF, bitstream change).
  Encode only top-k tokens (k≈1024) plus an "escape" symbol; CDF
  shrinks from (B, 50277) to (B, 1025). Real LM puts >99.99% mass in
  top-1024 on text. Save for v1.2.

### Tier 4: `krunch plan` end-to-end

`tests/gpu.sh` covers single-instance T3. Tier 4 builds `krunch plan`
itself and validates the contract by running at least one target end-
to-end. Blocked on T3 green.

**Build (in `krunch/plan/`):**

- Worker entry (`krunch/job.py` reshaped around the env-var contract):
  reads `KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`, `KRUNCH_PART_COUNT`;
  computes byte range; compress / decompress chunks to
  `<KRUNCH_OUTPUT_URL>.parts/<index>`.
- Finalize entry: reads the parts in index order, stitches them into
  the final `.krunch` (sums `n_chunks`, `original_len`, recomputes
  `crc32` against the assembled bitstream), deletes parts.
- Plan templates (jinja, one per target): `aws_batch.json.j2`,
  `gcp_batch.json.j2`, `k8s.yaml.j2`, `modal.py.j2`, `ray.py.j2`,
  `slurm.sbatch.j2`, `local.sh.j2`. Each emits both the worker tasks
  and the finalize task with the right dependency primitive for the
  target (see Splitting + stitching table).
- `krunch plan` CLI: parses `--target`, `--source`, `--dest`,
  `--workers`, optional `--image` / `--queue` / `--memory` etc.;
  renders the matching template to stdout. `--dry-run` validates
  template against schema without printing.

**Per-GPU saturation (NEW 2026-04-30 — required for the "any GPU"
claim).** `--workers N` controls machine count; per-GPU saturation
must happen INSIDE the worker container. Compress already saturates
because each chunk is a packed-T matmul (large M). Decompress
saturation depends on the cross-chunk batch size B (number of chunks
in flight on one GPU at once), and **optimal B is GPU-specific**:

  - T4 (16 GB):  B ≈ 64
  - A10G (24 GB): B ≈ 128–256
  - A100/H100:   B ≈ 512+

`krunch plan` itself stays GPU-agnostic — the artifact only describes
the worker count, not internal batching. The Docker image picks B
internally via one of (in order of preference):

  1. **Auto-tune at worker startup** — probe VRAM, run a ~1 sec
     microbench at B=8/64/256/512, pick the B that maximizes
     KB/s within the VRAM budget. Best UX, robust to new GPUs.
     Adds ~2-5 sec to worker cold start.
  2. **Heuristic table** keyed on `nvidia-smi --query-gpu=name`
     — fast, but every new GPU needs a table entry.
  3. **Env override** `KRUNCH_DECOMPRESS_BATCH=N` for power users
     and CI repeatability.

**T4 ships with all three:** auto-tune by default, heuristic table as
fallback if probing fails, env override for explicit control. Document
in `tuning.md`. The contract stays simple: "give me a byte range and
a GPU, I'll saturate it."

**T4 progress (2026-04-30):**
- ✅ `krunch/plan/` module — render() + validate(), 7 targets shipped
  (aws-batch, k8s, modal, ray, slurm, gcp-batch, local). All
  render + schema-validate cleanly.
- ✅ `krunch/plan_cli.py` — in-image entry point. Host wrapper
  docker-runs it so user only needs docker (no `pip install krunch`).
- ✅ `scripts/krunch plan ...` host-side wrapper subcommand.
  `--dry-run` validates schema; otherwise emits to stdout.
- ✅ `krunch/job.py` reshaped around v1 env-var contract (KRUNCH_MODE,
  KRUNCH_INPUT_URL, KRUNCH_OUTPUT_URL, KRUNCH_PART_INDEX,
  KRUNCH_PART_COUNT, KRUNCH_INPUT_LEN, KRUNCH_FINALIZE_OF). Handles
  compress, decompress, finalize modes. Same contract works on
  every orchestrator; per-target templates map orchestrator-specific
  vars (AWS_BATCH_JOB_ARRAY_INDEX, JOB_COMPLETION_INDEX,
  SLURM_ARRAY_TASK_ID) → KRUNCH_PART_INDEX in the launch artifact.
- ✅ `tests/test_plan.py` — Tier-1 CI test renders + validates every
  target on every PR. Catches template breakage before push to
  ghcr.io.
- ✅ Per-GPU auto-tune for cross-chunk batch B at worker startup —
  heuristic table by `nvidia-smi` GPU name (T4=64, A10G=128,
  A100/L40S=256, H100=512). Wired into
  `_decompress_chunks_batched_cpp` so a multi-thousand-chunk file
  splits into B-sized groups instead of OOM-ing.
- ⏳ Dynamic compress chunk size — currently fixed at 64 KB
  default. 64 KB is wrong for two regimes: tiny files (<1 MB,
  fewer chunks than B → wasted ratio cost from cold-starts that
  buy no extra parallelism) and very large files on big GPUs (1 GB+
  on H100 with B=512 capacity → fewer, bigger chunks would lower
  per-chunk bookkeeping overhead).

  Right design (works single-machine + distributed):
  ```
  total = int(os.environ["KRUNCH_INPUT_LEN"])  # always known
  target_B = int(os.environ.get("KRUNCH_TARGET_B", "128"))  # decompress
                                                            # GPU's B
  target_chunks = max(target_B * 4, 16)
  chunk_size = max(64KB, ceil(total / target_chunks))
  ```
  Every worker reads `KRUNCH_INPUT_LEN` from the same env contract,
  so all N workers pick the SAME chunk_size regardless of how the
  file is split across them. The aggregate blob has uniform chunk
  sizing; decompress (single-host or another distributed pass)
  picks its own B independently from that chunk count.

  `KRUNCH_TARGET_B` env override lets the user pin chunk size for a
  known decompress GPU class. Default (128) is the A10G assumption
  — conservative for H100 (still fills B=512 if file is large
  enough), suboptimal-but-functional for T4 (will pick B=64 from
  available chunks).
- ⏳ Microbench-based auto-tune (replace heuristic table with
  startup probe) — backlog; current table is good enough for
  shipped GPU classes.
- ⏳ AWS Batch e2e validation run (the actual T4 gate). Blocked on
  T3 green so we know the per-instance numbers before pinning a
  Batch queue config.

**Edge cases that don't fully saturate (acceptable):**
- File smaller than `B × chunk_size` — runs at lower B, graceful.
- Heterogeneous chunk lengths — batch shrinks toward end of file.
- Weaker-than-expected GPU — auto-tune picks lower B.

**Validate:**

- T1 gains: `--dry-run` for every target on every PR (free, in CI). No
  artifact ever fails schema-validation in published releases.
- T4 e2e (one target, ~$2-5, ~30 min): pick **AWS Batch** as the
  exercise target since we have the reference CDK + spike-6 validated
  path. Run:
  ```
  krunch plan --target aws-batch --source s3://… --dest s3://… --workers 2 > job.json
  aws batch submit-job --cli-input-json file://job.json
  ```
  on the same WildChat sample as T3.
- Gates: same ratio as T3, ~2× wall reduction at N=2 (within
  cold-start tolerance), parts cleaned up, finalize task succeeds,
  Batch cold-start time recorded.

Modal / k8s / Ray / Slurm targets are validated to the artifact level
in T1 (template renders, schema-valid output), but full e2e on each is
**deferred to post-launch**. Real users will surface friction faster
than synthetic exercises across 5 targets.

**Required before v1 ships** — the "any batch system" claim in the
README rests on the env-var contract + `krunch plan`, and has never
been validated end-to-end. Blocked on T3 green.

### Backlog (post-Tier-3, deferred)

- **Bigger model option** — `RWKV-4-Pile-1B5`. Better ratio (~0.07-0.09
  on chat, matching ts_zip) at 3-5× slower forward. Add `KRUNCH_MODEL`
  env var + multi-image publish.
- **fp8 / int8 quantized forward** for H100/L40S (A10G doesn't support
  fp8 natively). Defer until customer hardware mix is known.
- **Custom AMI with pre-pulled image** for cold Batch workers.
  Eliminates ~30-90 sec docker pull. The reference AWS CDK in
  `examples/aws-batch/` already supports `imageId` override. Worth
  pursuing once real workloads are running.
- **Streaming inputs > GPU VRAM.** v1 reads the whole input into memory;
  v2 will stream.
- **fp16-probs CDF** (skip the to(fp32) cast). Marginal on precision
  at T=2^16 scaling — likely off-by-1 count errors at boundaries.
  Defer until measured demand.

## Success criteria for v1 → v2 transition

Move to v2 (hosted offering) when:

- **GitHub stars ≥ 500** within 2 months of public launch (signals
  category interest)
- **At least 3 independent users** report production deployments
  (signals the framework actually works for someone besides us)
- **At least 1 inbound** "would you host this for us?" inquiry
  (signals demand for the managed tier)

If none of these trigger by month 3 post-launch, the OSS bet didn't
work; revisit positioning before investing in the hosted v2.

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

Rules:
- **Forward compat forever** within major version — newer images always read older blobs
- **Backward compat** is configurable via `KRUNCH_WRITE_BLOB_VERSION` env var — pin to an older version if older images need to read new output
- **Images bundle all readable model versions** — upgrade the image without breaking existing archives
- **Migration**: `krunch recompress --target-model v2` re-encodes old blobs without decompress-to-disk

## What's NOT in v1

- ❌ Hosted offering / managed API (that's v2)
- ❌ Vertical-specialized LoRA adapters (v2)
- ❌ Per-tenant fine-tuning (v3)
- ❌ Storage-as-a-service (defer indefinitely)
- ❌ Multi-region / HA out of the box (single-region, customer
   handles HA via their own orchestrator)
- ❌ Authentication (the OSS framework runs in trusted networks; auth
  is the customer's responsibility for self-host. Hosted in v2 has
  API keys.)
- ❌ Streaming for inputs > GPU VRAM (v1 reads whole input into
  memory; v2 will stream)

## Risks (ranked)

1. **Nobody adopts.** Possible if "neural compression" is too niche
   a category. Mitigation: public launch + benchmarking case studies
   + targeted reach-out to likely-interested teams (logging vendors,
   archival storage SREs).
2. **Adoption but no upsell.** Some users self-host forever and v2
   gets no customers. Mitigation: vertical-LoRA registry is paid even
   for self-host (subscription for adapter access). Hosted is the
   convenient option, LoRAs are the value.
3. **AWS/GCP fork it.** Real long-term risk. Mitigation: build
   adapter registry as the moat, build customer relationships in
   v2, brand association with the category.
4. **Apache-2.0 too permissive.** If the cannibalization risk
   becomes acute, relicense future versions to BUSL (à la Hashicorp).
   v1 stays Apache forever.

## Immediate next step (2026-04-30)

1. **Custom fused single-step CUDA kernel** for decompress —
   scaffolding in `krunch_ac/cuda/rwkv_step.cu`, plan in "Tier 3 status"
   above. ~1-2 weeks. Closes the remaining ~10× compress/decompress
   asymmetry on a single instance. **This is the gate for T3.**
2. **T3 retest on A10G** once the kernel lands. Byte-exact roundtrip,
   compress ≥150 KB/s, decompress within 2-3× of compress wall.
3. **T4 `krunch plan`** — build the worker/finalize entries against
   the env-var contract, build the plan templates, then run
   `krunch plan --target aws-batch | aws batch submit-job` end-to-end
   on the WildChat sample at N=2. Validates the headline "any batch
   system" claim through one real scheduler. Other targets validated
   to the artifact level in CI; full e2e deferred to post-launch.
4. **Real-corpus ratio benchmarks** for README placeholders (logs /
   chat / support tickets / wiki / code).
5. **Polish + soft launch + Show HN.**
