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

## [0.1.1] — 2026-05-11

### Fixed
- **Bug #3 — Codec is lossy on non-UTF-8 input** (correctness bug).
  Compress preprocessing used `bytes.decode("utf-8", errors="replace")`
  which substituted invalid byte sequences with `U+FFFD`; decompress
  reproduced *the substitution*, not the original. Affected HTTP logs,
  mixed-encoding CSVs, raw email bodies — any input containing non-UTF-8
  bytes. **Fix:** new `_bytes_to_text` / `_text_to_bytes` helpers that
  escape invalid bytes as PUA codepoints `U+E000..U+E0FF`. Encoder
  unconditionally uses the new path; decoder dispatches on the blob's
  `tokenizer_id`. See `docs/Bugs.md` #3 for the limitation around input
  literally containing valid UTF-8 sequences for U+E0XX codepoints.

### Added
- **`TOKENIZER_ID_LEGACY = 1`** + **`TOKENIZER_ID = 2`** —
  `SUPPORTED_TOKENIZER_IDS = (1, 2)`. New compress writes
  `tokenizer_id=2`. Existing `tokenizer_id=1` blobs continue to decode
  via the legacy `text.encode("utf-8")` path that the image still
  bundles. Bump driven by Bug #3 fix above.
- `tests/unit/test_bytes_to_text_pua.py` (13 tests) — pins the
  byte-safe encoding contract, the documented limitation, and the
  legacy converter that stays around for old-blob compat.

### Added
- **`MODEL_ID_ADAPTIVE = 2`** — second supported model_id covering
  blobs produced with `KRUNCH_ADAPTIVE_HEAD=1` (NEXT-3, per-document
  online bias correction in log-prob space). Existing `MODEL_ID = 1`
  blobs continue to round-trip unchanged. `SUPPORTED_MODEL_IDS` is
  now `(1, 2)`. Image must agree on encoder + decoder side via the
  env flag; the blob header carries the chosen model_id so a
  mismatched decoder fails cleanly via `IncompatibleBlobError`.
- `krunch/codec/adaptive_head.py` — fp64 CPU reference for the
  adaptive bias head (Apache-2.0 port from
  [Nacrith-GPU](https://github.com/robtacconelli/Nacrith-GPU)).
- `krunch/codec/adaptive_head_gpu.py` — torch fp64 batched
  implementation, byte-exact with the CPU reference at every step.
- `krunch/codec/gpu_encode.py::probs_to_cdf_gpu_fp64` — fp64-input
  CDF builder for the adaptive head path; mirrors `cdf.probs_to_cdf`
  bit-for-bit.
- `tests/unit/test_adaptive_head.py` (15 tests).
- `tests/unit/gpu/test_adaptive_head_parity.py` — CPU↔GPU
  bit-exactness pin.
- `tests/unit/test_blob_format_versioning.py::test_accepts_adaptive_head_model_id`.

### Changed
- `decode_header` now accepts any `model_id` in `SUPPORTED_MODEL_IDS`
  rather than only `MODEL_ID`. The image still rejects unknown ids
  with `IncompatibleBlobError`.
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
