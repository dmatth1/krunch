# Releases

Versioning policy + compatibility guarantees for `krunch`.

## How to install a specific version

```bash
# Latest tagged release
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash

# Pin to a specific version (recommended for production deployments)
curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh \
  | sudo KRUNCH_VERSION=v0.1.0 bash
```

`KRUNCH_VERSION` controls both the wrapper script (`scripts/krunch`)
and the Docker image tag (`ghcr.io/dmatth1/krunch:v0.1.0`). Both are
fetched from the matching git tag, so the install is deterministic.

To pin only the image (e.g., to test a release candidate against an
already-installed wrapper):

```bash
KRUNCH_IMAGE=ghcr.io/dmatth1/krunch:v0.1.0 krunch compress < in > out
```

## Versioning policy

We follow [semver](https://semver.org). Pre-1.0 (the v0.x series), the
contract is slightly looser — see the pre-1.0 caveat below.

| What changed | Version bump | Compat impact |
|---|---|---|
| Bug fix, no behavior change | **patch** (`v0.1.0 → v0.1.1`) | none — same bits in, same bits out |
| New flag, new orchestrator template, faster kernel that produces SAME compressed bits | **minor** (`v0.1.0 → v0.2.0`) | none — same bits |
| New `model_id` / new `tokenizer_id` / AC contract change → blob byte change | **major** (`v0.x → v1.0`) | OLD blobs still decode (image bundles old model); NEW blobs only decoded by ≥ v1.0 |

**Pre-1.0 (v0.x) caveat:** breaking changes are possible between minor
versions while we're stabilizing the format. Pin
`KRUNCH_VERSION=v0.1.0` (or whatever stable tag you've validated) for
production deployments. We'll guarantee strict semver from v1.0
onward.

## Compatibility guarantee — old blobs always decode

The blob format includes a self-describing header
(`krunch/inference.py:48-57`):

```
magic           4 bytes (KRNC)
blob_version    u8
model_id        u32
tokenizer_id    u32
adapter_id      u32
adapter_version u16
flags           u16
original_len    u64
n_chunks        u32
crc32           u32
```

**The hard contract:** every published krunch image bundles the
weights + tokenizer for every `model_id` it has ever shipped. So a
v3.0.0 client can decompress a v0.1.0 blob — the model
(`model_id=1`) used to encode it is still present in the v3.0.0
image. Forward-compatibility is preserved indefinitely; you don't
have to keep the old client around to read old archives.

The corollary: **bumping `model_id` is the conscious "we made a
breaking decode change" signal,** which always rolls a major
version. Within a major version, the model + tokenizer + AC contract
are frozen, so compress + decompress produce bit-exact output
regardless of which patch/minor version did the work. (We pinned
this property in the T4 100 MB / 4-worker run — see `docs/TIER_4.md`.)

What invalidates a blob (= forces a major bump):

- New `model_id` (e.g., swap to RWKV-7 / RWKV-8 / a fine-tuned variant)
- New `tokenizer_id` (different BPE vocab)
- AC coder contract change (different constriction model family,
  different softmax → CDF transform, different range-coder params)
- Chunker input → tokens mapping change (e.g., turning NFC
  normalization back on, switching to byte-level tokenization)

What does NOT invalidate a blob (no version bump needed for compat):

- Faster kernels that produce the same compressed bytes (W8A8,
  cp.async, persistent kernels)
- Bug fixes in the chunker / batcher / orchestration layer
- New `--target` plan templates (k8s, Modal, etc.)
- Different chunk sizes (`compute_chunk_size` is determined by the
  total input size, baked into the blob via `n_chunks`)

## Cutting a release

For maintainers — the four-step ritual:

```bash
# 1. Tag (annotated, with notes)
git tag v0.1.0 -a -m "First pre-launch — AWS Batch fan-out validated"
git push origin v0.1.0

# 2. Wait for the publish workflow (~5 min)
gh run list --workflow=publish-image.yml --limit 1
# Image lands at:
#   ghcr.io/dmatth1/krunch:v0.1.0
#   ghcr.io/dmatth1/krunch:0.1.0
#   ghcr.io/dmatth1/krunch:0
#   (and :latest, since the tag landed on main)

# 3. Cut a GitHub Release with notes
gh release create v0.1.0 \
  --title "v0.1.0 — first pre-launch tag" \
  --notes-file RELEASE_NOTES_v0.1.0.md \
  --prerelease   # drop this once we're at v1.0+

# 4. Smoke-test the pin actually works on a fresh host
#    (~$0.05, ~5 min on a g4dn.xlarge spot — see CLAUDE.md AWS access)
KRUNCH_VERSION=v0.1.0 KRUNCH_INSTANCE_TYPE=g4dn.xlarge \
  KRUNCH_SAMPLE_MB=1 bash tests/integration/gpu.sh
```

## Released versions

_None yet — pre-launch. v0.1.0 will be the first tag once we're
satisfied with the post-T4 state._
