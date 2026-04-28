# Krunch

> **Krunch is a distributed neural compression framework.**
> It works on any NVIDIA GPU and beats traditional compression algorithms (like zstd-22) by 20-40% on
> text-heavy data (logs, chat, support tickets, code).

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

## Distributed batch jobs

For large files / archival workloads, run the same image as an array job
under any batch scheduler (AWS Batch, k8s Jobs, Spark, Dask, Ray) —
workers read byte ranges directly from object storage and process them
in parallel.

```bash
# One-time AWS setup: deploy the Batch infra
cd deploy/aws-cdk && npm install && npx cdk deploy

# Submit a distributed job
krunch submit \
  --source s3://my-bucket/logs/data.jsonl \
  --dest   s3://my-bucket/logs/data.krunch \
  --workers 8
```

See `deploy/aws-cdk/README.md` for the AWS-specific setup. The job
contract (read byte range from URL, compress, write partial blob) is
generic — the same Docker image runs under any scheduler that can pass
env vars and grant S3 access.

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

## Why "distributed"

Compression chunks are independent — N workers means ~N× throughput.
Decompression is the same: token-step is sequential within a chunk
(RNN), but chunks decode in parallel. A 10 TB backfill on 10 workers
finishes in 1/10th the time, with no orchestration code on the
customer's side.

The compression task is map-only and parallel, so it
ships as a generic "run this container with these env vars" contract
that fits any batch framework.

## Repo layout

```
krunch/
├── Dockerfile              # CUDA + PyTorch + RWKV + WKV kernel + model weights
├── install.sh              # one-line installer (used by the curl install)
├── krunch/                 # the Python package — core compression code
│   ├── cli.py              # single-shot CLI entrypoint
│   ├── inference.py        # RWKV-4-Pile-169M wrapper + AC coder + blob format
│   ├── chunking.py         # 1 MB chunk splitter (neural-only, no fallback)
│   ├── job.py              # Batch job runner: compress (array) + assemble
│   └── url_io.py           # generic URL read/write (s3://, http://, file://)
├── scripts/
│   ├── krunch              # the user-facing CLI wrapper (Python)
│   └── entrypoint.sh       # container entrypoint (compress | decompress | job)
├── tests/                  # see tests/README.md
│   ├── test_blob.py        # unit tests (blob format, AC codec, chunking, CRC)
│   ├── quick.sh            # CI-equivalent local checks (free, seconds)
│   ├── integration.sh      # CPU end-to-end with the real model (free, ~30s)
│   └── gpu.sh              # GPU smoke on a g5.xlarge spot (~$0.15)
├── deploy/aws-cdk/         # AWS Batch deployer (compute envs, job queue, S3)
└── LICENSE                 # Apache-2.0
```

## License

Apache-2.0. See `NOTICE` for upstream attributions (RWKV-LM, constriction).
