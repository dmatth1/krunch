"""Tier-1 test: the tokenizer must not apply Unicode normalization.

The 20B_tokenizer ships with `"normalizer": {"type": "NFC"}` in its
JSON config, which silently maps compatibility-equivalent codepoints to
their canonical form on encode — e.g. U+2126 OHM SIGN → U+03A9 GREEK
CAPITAL OMEGA. Same glyph, different bytes (3 → 2). That breaks
byte-exact roundtrip on any text containing such characters.

`InferenceEngine.load()` disables the normalizer immediately after
loading the tokenizer to recover byte-exact roundtrip. This test pins
that behavior so a future tokenizer reload doesn't reintroduce the
NFC pass without us noticing.

Discovered 2026-05-07 in the T4 100 MB / 4-worker AWS Batch run:
WildChat content includes "0.20 Ω" (the Ohm sign for resistance), and
the codec lost 2 bytes per occurrence × dozens of occurrences = 125
bytes across the 100 MB corpus.
"""
from pathlib import Path

import pytest
from tokenizers import Tokenizer

# Same path the production code uses (krunch/inference.py:TOKENIZER_PATH).
# The tokenizer JSON is gitignored (downloaded into models/ at setup time
# alongside the .pth weights), so tests skip cleanly if it's not present
# — e.g. in CI without model artifacts.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKENIZER_PATH = _REPO_ROOT / "models" / "20B_tokenizer.json"

_skip_if_missing = pytest.mark.skipif(
    not _TOKENIZER_PATH.exists(),
    reason=f"{_TOKENIZER_PATH} not present (gitignored model artifact)",
)


# Bytes containing several compatibility-equivalent codepoints that would
# get rewritten by NFC normalization:
#   - U+2126 OHM SIGN          (\xe2\x84\xa6)  → U+03A9 GREEK CAP. OMEGA  (2 bytes)
#   - U+FB00 LATIN SMALL LIG.  (\xef\xac\x80)  → "ff"                     (2 bytes)
#   - U+212B ANGSTROM SIGN     (\xe2\x84\xab)  → U+00C5 LATIN A W/ RING   (2 bytes)
_NORMALIZATION_TRAPS = (
    "Resistance: 0.20 Ω\n"
    "Ligature: diﬀerence\n"
    "Angstrom: 1.5 Å\n"
)


@_skip_if_missing
def test_tokenizer_config_has_nfc_normalizer():
    """Sanity check: confirm the on-disk tokenizer DOES have NFC. If
    this ever changes, the disable line in inference.py becomes
    unnecessary (but should still be safe as a no-op)."""
    import json
    with open(_TOKENIZER_PATH) as f:
        cfg = json.load(f)
    assert cfg.get("normalizer") == {"type": "NFC"}, \
        f"tokenizer normalizer changed: {cfg.get('normalizer')!r}"


@_skip_if_missing
def test_default_tokenizer_breaks_byte_exact_on_traps():
    """Negative control — without disabling the normalizer, the
    tokenizer roundtrip is lossy on compatibility-equivalent Unicode."""
    tok = Tokenizer.from_file(str(_TOKENIZER_PATH))
    orig = _NORMALIZATION_TRAPS
    back = tok.decode(tok.encode(orig).ids)
    assert orig != back, \
        "default tokenizer didn't apply NFC — was the JSON config changed?"
    assert orig.encode("utf-8") != back.encode("utf-8")


@_skip_if_missing
def test_disabling_normalizer_gives_byte_exact_roundtrip():
    """Setting `tokenizer.normalizer = None` after load (what
    `InferenceEngine.load()` does) recovers byte-exact roundtrip on
    text containing compatibility-equivalent codepoints."""
    tok = Tokenizer.from_file(str(_TOKENIZER_PATH))
    tok.normalizer = None

    orig_text = _NORMALIZATION_TRAPS
    back_text = tok.decode(tok.encode(orig_text).ids)
    assert orig_text == back_text, \
        f"disabled-normalizer roundtrip mismatch: {orig_text!r} → {back_text!r}"
    assert orig_text.encode("utf-8") == back_text.encode("utf-8")


def test_inference_engine_disables_normalizer_on_load():
    """InferenceEngine.load() must include `self._tokenizer.normalizer
    = None` (otherwise compress is silently lossy)."""
    import inspect

    from krunch.inference import InferenceEngine

    src = inspect.getsource(InferenceEngine.load)
    assert "_tokenizer.normalizer" in src and "= None" in src, (
        "InferenceEngine.load() must explicitly disable the tokenizer's "
        "NFC normalizer (see comment block in inference.py near "
        "Tokenizer.from_file). If this assertion failed, the lossy-codec "
        "regression has reappeared.")


if __name__ == "__main__":
    test_tokenizer_config_has_nfc_normalizer()
    test_default_tokenizer_breaks_byte_exact_on_traps()
    test_disabling_normalizer_gives_byte_exact_roundtrip()
    test_inference_engine_disables_normalizer_on_load()
    print("all 4 tests passed")
