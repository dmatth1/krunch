# Changelog

All notable changes to krunch are recorded here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning policy: see `docs/TIER_3_CLEANUP.md` §5.4.

Two distinct version axes:

- **`krunch.__version__`** — software/CLI version. Bump on any
  user-visible CLI / API / build change.
- **`MODEL_ID`** (in `krunch/inference.py`) — bitstream version. Bump
  ONLY when a code change makes blobs produced by older images
  unreadable, or vice versa. A `MODEL_ID` bump is a **breaking
  change** for stored archives.

When in doubt: would an old blob in S3 still decode under this
release? If no, bump `MODEL_ID` and document migration here.

---

## [Unreleased]

### Added
- `IncompatibleBlobError` raised by `decode_header` when blob's
  `model_id` / `tokenizer_id` / `blob_version` don't match the running
  image. Replaces the previous silent "produce garbage, fail on CRC"
  failure mode. (`docs/TIER_3_CLEANUP.md` §5.2)
- `tests/unit/test_blob_format_versioning.py` — pins the on-disk
  header layout. Any future change to `HEADER_FMT` triggers this
  test, forcing a `BLOB_VERSION` bump.
- `tests/unit/test_result_json_schema.py` — pins the schema of
  `result.json` produced by `tests/integration/gpu.sh`.
- `CHANGELOG.md` (this file).

### Changed
- `cli.py` CRC32 mismatch error now includes the blob's `model_id`,
  `tokenizer_id`, `blob_version`, and the running image's
  `krunch.__version__`. Lets users self-diagnose version-skew issues
  without reading source.

### Notes
- `MODEL_ID` and `BLOB_VERSION` unchanged — every existing blob in S3
  remains readable.

---

## [0.1.0] — initial pre-launch

### Bitstream
- `BLOB_VERSION = 1`
- `MODEL_ID = 1` (RWKV-4-Pile-169M)
- `TOKENIZER_ID = 1` (GPT-NeoX 20B BPE)

Anything older than this baseline is not in scope.
