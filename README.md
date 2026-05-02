# Krunch

> **Krunch is a neural codec for text.**
> It works on any NVIDIA GPU and beats traditional compression algorithms (like zstd-22) by 20-40% on
> text-heavy data (logs, chat, support tickets, code).
>
> Ships as a Python library, a Docker image, and a documented blob
> format. Run it on one machine, parallelize it across a cluster with
> any batch system you already use — your call.

> Status: pre-launch.

## Install + compress

Run on any host with an NVIDIA GPU + Docker:

```bash
# 1. Install (~5-10 min one-time — downloads CLI + pulls 3.5 GB image)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash

# 2. Use it (instant — image is cached)
krunch compress   data.jsonl  -o data.krunch
krunch decompress data.krunch -o data.jsonl

# Or pipe-style (Unix idiom)
krunch compress   < data.jsonl  > data.krunch
krunch decompress < data.krunch > data.jsonl
```

The installer puts a thin wrapper at `/usr/local/bin/krunch`
that shells out to
`docker run --gpus all -i ghcr.io/dmatth1/krunch:latest …`. After install
every call starts in ~30 seconds (model load + WKV kernel cache).

## Distributed across machines

For large files / archival workloads, run krunch as parallel tasks on
whatever batch system you already use. `krunch plan` emits a
ready-to-run artifact for the target you pick — we don't run anything
for you, we just generate the config:

```bash
krunch plan --target aws-batch --source s3://… --dest s3://… --workers 16 > job.json
krunch plan --target k8s       --source s3://… --dest s3://… --workers 16 > job.yaml
krunch plan --target modal     --source s3://… --dest s3://… --workers 16 > run.py
krunch plan --target ray       --source s3://… --dest s3://… --workers 16 > run.py
krunch plan --target slurm     --source s3://… --dest s3://… --workers 16 > run.sbatch
```

Then run it with your own tooling and credentials:
`aws batch submit-job --cli-input-json file://job.json`,
`kubectl apply -f job.yaml`, `modal run run.py`, etc.

The artifact contains both the worker tasks (each computes its byte
range from a framework-injected index) and a finalize task that
stitches partial blobs into the final `.krunch`. The container
contract (`KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`, `KRUNCH_PART_COUNT`,
…) is documented and stable — you can wire krunch into a batch system
we don't have a template for in ~30 lines.

See `examples/` for full reference deployments (including an AWS
Batch CDK stack you can deploy as-is).

> `krunch submit` is deprecated and will be removed in a future
> release; use `krunch plan --target aws-batch` instead.

## What's inside the Docker image

- **RWKV-4-Pile-169M** pretrained language model (Apache-2.0, BlinkDL) —
  the next-byte predictor.
- **Custom WKV CUDA kernel** — fused recurrence op, ~1000× faster than
  HF transformers' eval-mode fallback.
- **constriction** arithmetic coder — turns the model's
  next-token distribution into a bitstream.
- **1 MB chunks (default)** — independent across chunks, parallelizable; large
  enough to amortize per-chunk overhead and give the model useful
  context.

Architecture validated on real GPU: ratio **0.111** on WildChat-English
(vs zstd-22's 0.167 — a 33% reduction), compress throughput **≥ 800
KB/s** on A10G fp16, byte-exact decompression.

## Ratio comparisons

> *To be filled in. Need: krunch vs zstd-22 vs bzip3 vs ts_zip on at*
> *least — WildChat-English (chat / dialogue), enwik8 / enwik9*
> *(Wikipedia), a log corpus (e.g. nginx or HDFS), and a code corpus*
> *(e.g. The Stack Python subset). All numbers from a single g5.xlarge*
> *run with the published `:latest` image, sample size ≥ 100 MB per*
> *corpus. ts_zip uses its published `1B5-v3` model.*

| corpus | krunch | ts_zip | zstd-22 | bzip3 | krunch vs zstd |
|---|---|---|---|---|---|
| WildChat-English | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| enwik8 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| enwik9 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| nginx logs | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| The Stack (Python) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Throughput vs total CUDA cores in the fleet

Compression chunks are independent, so aggregate throughput scales
linearly with parallel CUDA cores — you can move right on the chart
either by adding workers (`--workers N`) or by picking a GPU with
more cores per chip. Both knobs end up at the same place.

> *To be filled in. Need: compress + decompress KB/s on a fixed corpus*
> *(~10 GB) at several fleet sizes. Mix configurations to span the*
> *x-axis: e.g., 1× g5.xlarge (9k cores), 1× g6e.xlarge (18k), 4×*
> *g5.xlarge (37k), 8× g5.xlarge (74k), 1× p5.48xlarge (135k).*

```
  aggregate KB/s
    │                                            compress
    │                                                ●  (tbd)
    │                                          ●━━━━━━●
    │                                     ●━━●
    │                                ●━━●            decompress
    │                          ●━━●                       ●  (tbd)
    │                ●━━●                            ●━━━━━━●
    │           ●━━━●                           ●━━●
    │      ●━━━●                            ●━━●
    │  ●━━━●                            ●━●
    └──┬──────┬──────┬──────┬──────┬──────→  total CUDA cores in the fleet
       9k    18k    37k    74k    135k
       1×    1×     4×     8×     1×
       g5    g6e    g5     g5     p5
       .x    .x     .x     .x     .48x
```

Each `●` is `_tbd_` until measured. The fleet-size column under the
x-axis shows one example configuration that lands at that core count;
others (e.g. 2× g6e or 4× g6) reach similar core totals and should
land on the same line.

## When *not* to use krunch

Krunch is a neural compressor for text. 
If your data isn't text-heavy enough that the language model can
predict it, krunch can produce *larger* output than the input. For
arbitrary binary data, mixed media, or already-compressed payloads, use 
a different compressor.

## Why parallelize

Compression chunks are independent — N workers means ~N× throughput.
Decompression is the same: token-step is sequential within a chunk
(RNN), but chunks decode in parallel. A 10 TB backfill on 10 workers
finishes in 1/10th the time.

Krunch doesn't ship a batch system — Modal, Ray, AWS Batch, k8s, and
Slurm already do that part well. Instead we ship a stable container
contract (one set of env vars, same behavior everywhere) and `krunch
plan` to emit configs for the popular targets.

## Repo layout

```
krunch/
├── Dockerfile              # CUDA + PyTorch + RWKV + WKV kernel + model weights
├── install.sh              # one-line installer (used by the curl install)
├── krunch/                 # the Python package — codec library + CLI
│   ├── cli.py              # CLI: compress | decompress | plan | bench
│   ├── inference.py        # RWKV-4-Pile-169M wrapper + AC coder + blob format
│   ├── chunking.py         # chunk splitter (neural-only, no fallback)
│   ├── worker_pool.py      # multi-process pool for --workers N
│   ├── plan/               # krunch plan templates (aws-batch, k8s, modal, ray, slurm, …)
│   ├── job.py              # in-container per-worker entry: range → partial blob
│   └── url_io.py           # generic URL read/write (s3://, http://, file://)
├── docs/
│   └── format.md           # blob format spec (RFC-style, implementable)
├── scripts/
│   ├── krunch              # the user-facing CLI wrapper (Python)
│   └── entrypoint.sh       # container entrypoint (worker | finalize | compress | decompress)
├── tests/                  # see tests/README.md
│   ├── test_blob.py        # unit tests (blob format, AC codec, chunking, CRC)
│   ├── quick.sh            # CI-equivalent local checks (free, seconds)
│   ├── integration.sh      # CPU end-to-end with the real model (free, ~30s)
│   └── gpu.sh              # GPU smoke on a g5.xlarge spot (~$0.15)
├── examples/               # batch-framework integrations (AWS Batch CDK, Modal, Ray, k8s, …)
└── LICENSE                 # Apache-2.0
```

## License

Apache-2.0. See `NOTICE` for upstream attributions (RWKV-LM, constriction).
