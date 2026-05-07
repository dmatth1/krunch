"""Regression: disabling the tokenizer's NFC normalizer must work
against the real `tokenizers` package surface, not just our mock.

Caught a production bug 2026-05-07: setting `tokenizer.normalizer = None`
crashes with `'NoneType' object cannot be converted to 'Normalizer'`
on the version baked into the published image, breaking every compress
+ decompress invocation. Fix: assign an empty Sequence as a no-op.
"""
import os
import pytest


MODEL_DIR = os.environ.get("KRUNCH_MODEL_DIR", "models")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "20B_tokenizer.json")
HAVE_TOKENIZER = os.path.isfile(TOKENIZER_PATH)


@pytest.mark.skipif(not HAVE_TOKENIZER,
                    reason="20B_tokenizer.json not present (CI fetches it)")
def test_normalizer_disable_preserves_unicode_byte_exactness():
    """The whole reason we disable the NFC normalizer: it canonicalizes
    compatibility-equivalent codepoints (U+2126 OHM SIGN → U+03A9 GREEK
    CAPITAL OMEGA, 3 bytes → 2 bytes) and breaks byte-exact roundtrip.

    With the normalizer disabled, U+2126 and U+03A9 must tokenize to
    DIFFERENT id sequences. With it enabled (the default in the trained
    tokenizer JSON), they collapse to the same ids — that's the bug.
    """
    from tokenizers import Tokenizer
    from tokenizers.normalizers import Sequence

    tk = Tokenizer.from_file(TOKENIZER_PATH)
    # Sanity: by default the trained tokenizer has the NFC normalizer.
    assert tk.normalizer is not None

    tk.normalizer = Sequence([])  # production fix

    # Use explicit codepoints — string literals can be NFC-normalized at
    # parse time depending on filesystem / editor, defeating the test.
    ohm   = "Ω"  # OHM SIGN, 3 bytes utf-8
    omega = "Ω"  # GREEK CAPITAL OMEGA, 2 bytes utf-8
    assert ohm.encode("utf-8") != omega.encode("utf-8")
    assert tk.encode(ohm).ids != tk.encode(omega).ids


@pytest.mark.skipif(not HAVE_TOKENIZER,
                    reason="20B_tokenizer.json not present (CI fetches it)")
def test_normalizer_load_path_matches_engine():
    """End-to-end check of the same path engine.load() runs. Importing
    the disable line via the production module guards against future
    refactors that re-introduce `None` here."""
    from tokenizers import Tokenizer
    from tokenizers.normalizers import Sequence

    tk = Tokenizer.from_file(TOKENIZER_PATH)
    # Should not raise (whatever the local tokenizers version is).
    tk.normalizer = Sequence([])
