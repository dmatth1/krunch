# Tests

```
tests/
├── unit/         # pure Python, no GPU, no AWS, runs in seconds
├── integration/  # GPU + krunch_ac extension, or AWS resources
└── gpu/          # standalone kernel-correctness scripts cited inline
                  #   from production code (run inside Docker image,
                  #   not pytest-collected)
```

A tiered test ladder, cheapest to most-expensive.

## `unit/` — fast Python tests (free, seconds)

Validates the blob format, arithmetic codec, chunking machinery, CRC,
plan-template rendering, and UTF-8 byte-range correctness. No GPU,
no model weights, no AWS.

```bash
pip install constriction tokenizers numpy zstandard pytest jsonschema jinja2 pyyaml
PYTHONPATH=. pytest tests/unit/ -v
```

CI runs this on every push.

## `integration/quick.sh` — local fast checks (free, seconds)

What CI runs end-to-end. Unit tests + CDK type-check + CDK synth +
`krunch plan --target aws-batch --dry-run` schema validation.

```bash
tests/integration/quick.sh
```

## `integration/integration.sh` — CPU end-to-end with the real model (free, ~30s)

Loads RWKV-4-Pile-169M into CPU memory and runs `krunch compress` +
`krunch decompress` on a tiny (~200 byte) sample. Verifies the full
neural pipeline byte-exact. CPU is too slow for large inputs but fast
enough for correctness. Catches encode/decode symmetry bugs before
spending GPU dollars.

Prereqs (one-time):
```bash
python3 -m venv /tmp/krunch-venv
/tmp/krunch-venv/bin/pip install constriction tokenizers numpy boto3 \
    rwkv torch --index-url https://download.pytorch.org/whl/cpu
# Place RWKV-4-Pile-169M-20220807-8023.pth and 20B_tokenizer.json in models/
```

Then:
```bash
tests/integration/integration.sh
```

## `integration/test_ac_*_gpu.py` — GPU AC-codec correctness (skipped without CUDA)

Pytest-runnable on a CUDA host with the `krunch_ac` extension built.
Skips at module import if either is missing.

```bash
PYTHONPATH=. pytest tests/integration/ -v
```

## `integration/gpu.sh` — GPU smoke on a g5.xlarge spot (~$0.15, ~10 min)

Provisions one g5.xlarge spot instance, runs `curl install.sh | bash`
followed by `krunch compress` + `krunch decompress` on a 100 MB
WildChat sample, validates ratio + throughput + byte-exact roundtrip,
self-terminates. Uses your AWS account.

Required env vars (no defaults — must point at your own resources):

| var | what |
|---|---|
| `KRUNCH_KEY_PAIR` | EC2 key pair name |
| `KRUNCH_SG` | security group name |
| `KRUNCH_S3_BUCKET` | S3 bucket the test reads/writes |
| `KRUNCH_INSTANCE_PROFILE` | IAM profile granting S3 access |

By default uses `ghcr.io/dmatth1/krunch:latest`. Set `KRUNCH_LOCAL_BUILD=1`
to build the image from source on the instance instead.

```bash
KRUNCH_KEY_PAIR=my-key \
KRUNCH_SG=my-sg \
KRUNCH_S3_BUCKET=my-bucket \
KRUNCH_INSTANCE_PROFILE=my-profile \
tests/integration/gpu.sh
```

## `integration/batch.sh` — AWS Batch end-to-end (~$1, ~15 min)

Multi-worker compress + decompress through the deployed CDK stack.
See script header for full setup; requires the stack to be deployed
(`cd deploy/aws-cdk && npx cdk deploy`).

## `gpu/` — kernel-level correctness (run inside Docker)

Standalone scripts (not pytest) that back kernel invariants cited
inline from `krunch/cpp_path.py`, `krunch/inference.py`, and the CUDA
sources. See `tests/gpu/README.md`.
