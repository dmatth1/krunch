# Krunch — project guide

Open-source distributed neural compression framework. One Docker image,
runs on any NVIDIA GPU. Text-heavy data (logs, chat, support tickets,
code) at ratios beating zstd by 30-40%.

Public repo: `github.com/dmatth1/krunch` (private, pre-launch)

## What it is

Single-shot Docker CLI + RWKV-4-Pile-169M + WKV CUDA kernel + constriction AC coder.
Customer runs `docker run --gpus all -i ghcr.io/dmatth1/krunch:v1 compress < in > out`.
For distributed/archival workloads: `krunch submit` fans out to AWS Batch
(or any batch scheduler — the container also supports a `job` mode driven by env vars).

No HTTP server in v1. Compression is a one-shot transform; an always-warm
service makes no sense at this layer. Hosted offering is v2.

Distributed: chunks are independent, N workers = ~N× throughput.

v2 (hosted API + vertical LoRA adapters) starts after v1 adoption signals.
See `V1_PLAN.md` and `V2_PLAN.md`.

## Architecture validated

SPIKE_6_LOG.md: RWKV-4-Pile-169M + BlinkDL WKV kernel = ratio 0.111 on
WildChat-English, 330-430 KB/s on A10G fp16. Both gates met.

## Hard-won GPU gotchas

- **Always `apt install -y ninja-build`** before any CUDA kernel script.
  `pip install ninja` installs a Python wrapper that torch can't find.
  This ate ~$3 of Spike 6 debugging.
- **HF transformers RWKV is a trap.** `model.eval()` silently disables
  the WKV kernel, falling back to ~1000× slower Python loop. Use the
  `pip install rwkv` package (BlinkDL, Apache-2.0).
- **AMI**: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)`
  — use `/opt/pytorch/bin/python3` directly, don't activate the venv.
- **Bundle weights + tokenizer to S3 as a tarball** — don't git clone or
  HF download at runtime (auth + rate limits).

## Docker base + torch+cu124 pairing (Tier 3 lessons)

- **`requirements.txt` `--extra-index-url` must be on its own line**, not
  appended after a package spec. Pip silently ignores it inline → falls
  back to PyPI → can't find +cu124 wheels.
- **PyTorch only publishes `+cu124` wheels from 2.4.0 onwards.** 2.3.x is
  cu121-only. For CUDA 12.4 base images, pin `torch==2.4.1+cu124` or later.
- **Better base for fewer moving parts**: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
  has torch + CUDA pre-baked, avoiding the wheel-version juggling. Worth
  switching to this from `nvidia/cuda:*-devel-ubuntu22.04` once stable.
- **Tier 3 fail-fast on `docker build` errors.** Earlier script piped
  through `tail` and lost the exit code, marching on to compress against
  a non-existent image. Use `set -o pipefail` + check `${PIPESTATUS[0]}`.

## v1 directory shape

```
krunch/
├── Dockerfile
├── server/          # cli.py (single-shot), inference.py, chunking.py, job.py, url_io.py
├── scripts/         # entrypoint.sh, krunch_cli.py (submit), tests
├── deploy/aws-cdk/  # AWS Batch deployer
├── benchmarks/      # ratio + throughput docs (W3)
├── docs/            # architecture, tuning, operations (W3)
├── examples/        # CLI + Batch usage examples (W4)
├── V1_PLAN.md, V2_PLAN.md, SPIKE_6_LOG.md
└── (no HTTP server, no docker-compose, no always-on instance)
```

## Conventions

- **Image builds**: shake down locally with `docker run --entrypoint python`
  before pushing. Catches import bugs in 20s vs 10min/retry CodeBuild cycle.
- **Blob format**: self-describing header (KRNC magic, model_id, tokenizer_id,
  adapter_id, original_len, n_chunks, crc32). Newer images always read older
  blobs. See V1_PLAN.md for full spec.
- **No per-tenant training in v1.** Zero-shot RWKV-4-Pile-169M only.
  LoRA adapters are v2.

## Tier 3 testing policy

- **`tests/gpu.sh` is the canonical end-to-end user-UX test.** Spins up
  a fresh g5.xlarge spot, pulls the published image, runs `krunch
  compress` + `krunch decompress` on the WildChat sample. Same exact
  CLI path a real user runs, plus the gates: ratio, compress KB/s,
  decompress KB/s, byte-exact roundtrip. Use this — not ad-hoc bench
  scripts — to validate any change that touches compress / decompress
  speed or the bit-exact path.
- **Always pull `ghcr.io/dmatth1/krunch:latest`** for GPU validation runs
  (`tests/gpu.sh` defaults to pull mode). Never pass `KRUNCH_LOCAL_BUILD=1`
  except for local-diff sanity checks before pushing — the published
  image is the artifact users get; that's what we test.
- **Push fix → wait for `Publish image to ghcr.io` to succeed → re-run
  `tests/gpu.sh`.** Don't try to shortcut by building on the GPU
  instance; it diverges from what users run and accumulates drift.
- **Build typically takes ~5 minutes** — short enough to wait, no need to
  schedule long sleeps. Poll every minute or so.
- Monitor publish via:
  `gh api repos/dmatth1/krunch/actions/runs --jq '.workflow_runs[0:3] | .[] | "\(.status)/\(.conclusion // "running") \(.head_sha[0:7])"'`

## Performance validation on cloud

- Every cloud development session that touches the compress or decompress
  hot path **must measure ratio AND wall-clock for both directions** on a
  representative sample, not just the one being optimized. Improvements to
  one side that silently regress the other (e.g., GPU AC kernel landed
  decompress threading regression at 1.7×) are easy to ship and hard to
  unwind once published.
- Iterate to optimize: when a number moves the wrong way, **call it out
  in V1_PLAN.md with the measurement** before continuing — don't paper
  over the regression with a different optimization. Address it directly
  (revert / fix / explicitly accept and document) before adding more.
- Compare against the most recent committed measurement, not against the
  pre-session baseline only. Drift accumulates one "small" regression at
  a time.

Keep this file under 200 lines.
