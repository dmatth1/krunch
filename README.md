# Krunch

> **Distributed neural compression as an open-source framework.**
> One Docker image. Any NVIDIA GPU. Beats zstd by 30-40% on
> text-heavy data (logs, chat, support tickets, code).

> Status: pre-launch (private repo). v1 launch target: 6 weeks.
> See `V1_PLAN.md` for the roadmap.

## Install + compress

On any host with an NVIDIA GPU + Docker:

```bash
# 1. Install (~5-10 min one-time — downloads CLI + pulls 3.5 GB image)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash

# 2. Use it (instant — image is cached)
krunch compress   < data.jsonl  > data.krunch
krunch decompress < data.krunch > data.jsonl
```

That's it. The installer puts a thin Python wrapper at
`/usr/local/bin/krunch` that shells out to
`docker run --gpus all -i ghcr.io/dmatth1/krunch:v1 …`. After install
every call starts in ~30 seconds (model load + WKV kernel cache).

For large files / archival workloads, run as a distributed batch job.
The same Docker image runs as an array job under any scheduler
(AWS Batch, k8s Jobs, Spark, Dask, Ray) — workers read byte ranges
directly from object storage and run in parallel.

```bash
# One-time setup: deploy the AWS Batch infra
cd deploy/aws-cdk && npm install && npx cdk deploy

# Submit a distributed job
krunch submit \
  --source s3://my-bucket/logs/data.jsonl \
  --dest   s3://my-bucket/logs/data.krunch \
  --workers 8
```

> Override the Docker image for `compress` / `decompress` via
> `KRUNCH_IMAGE=ghcr.io/me/krunch:dev krunch compress < in > out`.

## What's inside the image

- **RWKV-4-Pile-169M** pretrained language model (Apache-2.0,
  BlinkDL) — the predictor.
- **Custom WKV CUDA kernel** — fused recurrence op,
  ~1000× faster than HF transformers' eval-mode fallback.
- **constriction** arithmetic coder — encodes the next-token
  distribution into a bitstream.
- **64 KB chunks** — independent across chunks, parallelizable.
- **Dispatcher** — neural vs zstd per-chunk, shortest output wins.
  Ensures we never lose to classical on chunks where it does.

Architecture validated in `SPIKE_6_LOG.md`: ratio 0.111 on
WildChat-English (vs zstd 0.166), 330–430 KB/s compress on A10G fp16.

## Why "distributed"

Compression chunks are independent — N workers means ~N× throughput.
Decompression is the same: token-step is sequential within a chunk
(RNN), but chunks decode in parallel. A 10 TB backfill on 10 workers
finishes in 1/10th the time, with no orchestration code on the
customer's side.

The compression task is map-only and embarrassingly parallel, so it
ships as a generic "run this container with these env vars" contract
that fits any batch framework.

## What v1 ships

```
krunch/
├── Dockerfile              # CUDA + PyTorch + RWKV-LM, single-shot + job entrypoints
├── server/                 # core compression code
│   ├── cli.py              # single-shot entrypoint: compress | decompress
│   ├── inference.py        # RWKV-4-Pile-169M wrapper + AC coder + blob format
│   ├── chunking.py         # 64KB chunk dispatcher (neural vs zstd, shortest wins)
│   ├── job.py              # Batch job runner: compress (array) + assemble (single)
│   └── url_io.py           # generic URL read/write (s3://, http://, file://)
├── scripts/                # entrypoint.sh, krunch_cli.py (submit), test scripts
├── deploy/aws-cdk/         # AWS Batch deployer (compute envs, job queue, S3 bucket)
├── V1_PLAN.md, V2_PLAN.md, SPIKE_6_LOG.md, CLAUDE.md
└── LICENSE                 # Apache-2.0
```

See `V1_PLAN.md` for the launch roadmap and `V2_PLAN.md` for the
planned hosted offering + vertical LoRA adapter registry.

## License

Apache-2.0. Maximum adoption, compatible with the planned v2 hosted
offering, patent grant + attribution prevents trivial whitewash forks.
See `NOTICE` for upstream attributions (RWKV-LM, constriction).
