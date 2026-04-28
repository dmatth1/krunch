# Tests

A tiered test ladder, cheapest to most-expensive.

## `test_blob.py` — unit tests

Validates the blob format, arithmetic codec, chunking machinery, CRC,
and tokenizer loading. No GPU required, no model weights required.
Runs in seconds.

```bash
pip install constriction tokenizers numpy zstandard
python tests/test_blob.py
```

CI runs this on every push.

## `quick.sh` — local fast checks (free, seconds)

What CI runs. `test_blob.py` + CDK type-check + `krunch submit --dry-run`
plumbing.

```bash
tests/quick.sh
```

## `integration.sh` — CPU end-to-end with the real model (free, ~30s)

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
tests/integration.sh
```

## `gpu.sh` — GPU smoke on a g5.xlarge spot (~$0.15, ~10 min)

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
to build the image from source on the instance instead (useful for
testing local changes before publishing).

```bash
KRUNCH_KEY_PAIR=my-key \
KRUNCH_SG=my-sg \
KRUNCH_S3_BUCKET=my-bucket \
KRUNCH_INSTANCE_PROFILE=my-profile \
tests/gpu.sh
```
