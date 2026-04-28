# Krunch

> **Distributed neural compression as an open-source framework.**
> One Docker image. Any NVIDIA GPU. Beats zstd by 30-40% on
> text-heavy data (logs, chat, support tickets, code).

> Status: pre-launch (private repo). v1 launch target: 6 weeks.
> See `V1_PLAN.md` for the roadmap.

## Quick start

```bash
# Server mode — always-warm REST API, good for interactive / streaming
docker run --gpus all -p 8080:8080 ghcr.io/dmatth1/krunch:v1

curl -X POST http://localhost:8080/compress   -T data.jsonl  > data.krunch
curl -X POST http://localhost:8080/decompress -T data.krunch > data.jsonl
```

For large files / archival workloads, use **job mode**: the same
Docker image runs as an array job under any batch scheduler (AWS
Batch, k8s Jobs, Spark, Dask, Ray). Workers read byte ranges
directly from object storage and run in parallel.

```bash
# Reference deployer: AWS Batch
cd deploy/aws-cdk && npm install && npx cdk deploy

python3 scripts/krunch_cli.py submit \
  --source s3://my-bucket/logs/data.jsonl \
  --dest   s3://my-bucket/logs/data.krunch \
  --workers 8
```

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
├── Dockerfile              # CUDA + PyTorch + RWKV-LM, server + job modes
├── server/                 # FastAPI server, RWKV inference, AC codec, blob format
├── scripts/                # entrypoint.sh, krunch_cli.py, smoke + roundtrip tests
├── deploy/aws-cdk/         # AWS Batch deployer (compute envs, job queue, S3 bucket)
├── deploy/docker-compose.yml  # local CPU-mode dev
├── V1_PLAN.md, V2_PLAN.md, SPIKE_6_LOG.md, CLAUDE.md
└── LICENSE                 # Apache-2.0
```

See `V1_PLAN.md` for the launch roadmap and `V2_PLAN.md` for the
planned hosted offering + vertical LoRA adapter registry.

## License

Apache-2.0. Maximum adoption, compatible with the planned v2 hosted
offering, patent grant + attribution prevents trivial whitewash forks.
See `NOTICE` for upstream attributions (RWKV-LM, constriction).
