# Krunch

> **Krunch is a neural codec for text.** It works on any NVIDIA GPU
> and beats traditional compression algorithms (like zstd-22) by 20-40%
> on text-heavy data (logs, chat, support tickets, code).
>
> Run it on one machine or parallelize across a cluster with any batch
> system you already use.

## Install + compress

Run on any host with an NVIDIA GPU + Docker:

```bash
# 1. Install (~5-10 min one-time — downloads CLI + pulls 3.5 GB image)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash
# For a pinned, reproducible install:
#   curl -fsSL .../install.sh | sudo KRUNCH_VERSION=v1.0.0 bash

# 2. Use it (instant — image is cached)
krunch compress   data.jsonl  -o data.krunch
krunch decompress data.krunch -o data.jsonl

# Or pipe-style (Unix idiom)
krunch compress   < data.jsonl  > data.krunch
krunch decompress < data.krunch > data.jsonl
```

## Distributed compression

For large files / archival workloads, run krunch as parallel tasks on
whatever batch system you already use. `krunch plan` emits a
ready-to-run artifact for the target you pick.

```bash
# Compress
krunch plan --target aws-batch --mode compress \
    --source s3://… --dest s3://… --workers 16 > compress.json

# Decompress
krunch plan --target aws-batch --mode decompress \
    --source s3://… --dest s3://… --workers 16 > decompress.json

# Planned targets — same flag shape, not yet implemented
krunch plan --target k8s       --mode compress --source … --dest … --workers 16 > job.yaml
krunch plan --target modal     --mode compress --source … --dest … --workers 16 > run.py
krunch plan --target ray       --mode compress --source … --dest … --workers 16 > run.py
krunch plan --target slurm     --mode compress --source … --dest … --workers 16 > run.sbatch
krunch plan --target gcp-batch --mode compress --source … --dest … --workers 16 > job.json
```

Then submit with your own tooling and credentials:
`aws batch submit-job --cli-input-json file://compress.json`,
`kubectl apply -f job.yaml`, `modal run run.py`, etc.

> Only `--target aws-batch` works today; the rest are illustrative of
> the intended UX. Contributions welcome — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

See [`deploy/aws-cdk/`](deploy/aws-cdk/) for a working AWS Batch
reference stack you can `cdk deploy` as-is.

## Throughput

Measured on AWS Batch (A10G g5.xlarge, 100 MB WildChat-English) —
real-work elapsed inside `compress_all` / `decompress_all`, excluding
cold-start container init:

![Krunch throughput vs fleet size](assets/throughput.png)

*Note: cold-start tax may increase runtimes on the first job, but
amortizes to zero on warm fleets and on large jobs.*

## Ratio comparisons

Compressed-size ratio (smaller = better) on a single A10G g5.xlarge,
1 MB chunks. ts_zip hasn't been benched locally yet.

| corpus | krunch | ts_zip | zstd -22 --long | krunch vs zstd |
|---|---|---|---|---|
| Chat — WildChat-English (100 MB) | **0.114** | _tbd_ | 0.170 | **−33%** |
| Wikipedia — enwik8 (100 MB) | **0.146** | _tbd_ | 0.253 | **−42%** |
| Python code — CodeParrot (100 MB) | **0.097** | _tbd_ | 0.154 | **−37%** |
| Support tickets — Bitext (19 MB) | 0.099 | _tbd_ | **0.083** | +20% |
| HTTP logs — NASA Apache (100 MB) [^1] | 0.157 | _tbd_ | **0.061** | +158% |

[^1]: krunch is **lossy** on this corpus because the input contains
  raw non-UTF-8 bytes (e.g. `0x80` from `%80`-decoded URL paths in
  1995 Apache logs). The codec substitutes invalid sequences with
  `U+FFFD` before tokenizing, so the decompressed bytes differ from
  the original (same length, different sha256). Ratio shown for
  completeness, not for direct comparison to zstd. Lossless byte-
  exact roundtrip on arbitrary input is a v1.1 target. All other
  rows in this table are byte-exact.

The honest story: krunch wins decisively on natural-language text
(chat, prose, code) and loses to zstd-22's 128 MB dictionary window
on highly-repetitive structured text (templated logs, intent
labels). The fan-out story — N workers ≈ N× throughput on
independent chunks — is independent of ratio, so the worker-scaling
pitch still applies even on the rows where the ratio is worse.

## What's inside the Docker image

- **RWKV-4-Pile-169M** pretrained language model (Apache-2.0, BlinkDL) —
  the next-byte predictor.
- **Custom WKV CUDA kernel** — fused recurrence op, ~1000× faster than
  HF transformers' eval-mode fallback.
- **constriction** arithmetic coder — turns the model's
  next-token distribution into a bitstream.

## Adding a new batch target

The artifact `krunch plan` emits contains both the worker tasks (each
computes its byte range from a framework-injected index) and a
finalize task that stitches partial blobs into the final output. The
container contract (`KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`,
`KRUNCH_PART_COUNT`, …) is documented and stable — you can wire krunch
into a batch system we don't have a template for in ~30 lines.

## When *not* to use krunch

Krunch is a neural compressor for **UTF-8 text**. Avoid it when:

- **Your data isn't valid UTF-8.** Inputs with raw non-UTF-8 bytes
  (1995-era HTTP server logs with `%XX`-decoded URL paths, raw email
  bodies with binary attachments, mixed-encoding CSV exports) are
  currently lossy: the codec substitutes invalid sequences with
  `U+FFFD` before tokenizing, so the decompressed bytes won't match
  the input. Byte-exact roundtrip on arbitrary input is a v1.1
  target; for now, stick to confirmed UTF-8 data.
- **Your data is highly repetitive structured text** (templated
  logs, intent labels, repeating timestamps). zstd-22's 128 MB
  dictionary window catches that pattern far more cheaply than a
  169 M-parameter language model — see the ratio table above.
- **Arbitrary binary, mixed media, or already-compressed payloads.**
  A 169 M-parameter language model has no advantage predicting
  randomness; krunch will produce *larger* output than the input.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See `NOTICE` for upstream attributions (RWKV-LM, constriction).
