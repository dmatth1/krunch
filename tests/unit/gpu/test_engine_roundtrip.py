"""End-to-end byte-exact roundtrip via InferenceEngine on small samples.

The fastest test that catches the kind of regression we hit with the
tokenizer normalizer crash: anything that breaks compress() or
decompress() at all surfaces here in ~30s on T4.

Three samples (~360-1100 bytes) covering English prose, code-y text,
and ASCII-only repetitive text. Each round-trips byte-exact.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest


SAMPLES = {
    "english_prose": (
        b"The quick brown fox jumps over the lazy dog. " * 8
    ),
    "code_python": (
        b"def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
        * 16
    ),
    "neural_codec_blurb": (
        b"This is a test of the krunch neural compression codec running with "
        b"the C++ orchestration path on both compress and decompress sides. "
        b"We expect a byte-exact roundtrip. "
    ) * 4,
}


# `engine` fixture is provided by conftest.py (session-scoped).


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_roundtrip_byte_exact(engine, name):
    data = SAMPLES[name]
    encoded = engine.compress_chunk(data)
    decoded = engine.decompress_chunk(encoded)
    assert decoded == data, _diff_msg(data, decoded)
    assert len(encoded) < len(data), (
        f"compressed bigger than original ({len(encoded)} >= {len(data)})"
    )


def _diff_msg(expected: bytes, actual: bytes) -> str:
    if len(expected) != len(actual):
        return f"length mismatch: got {len(actual)} expected {len(expected)}"
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            ctx_e = expected[max(0, i - 10):i + 10]
            ctx_a = actual[max(0, i - 10):i + 10]
            return (f"first diff at byte {i}: got 0x{actual[i]:02x} "
                    f"expected 0x{expected[i]:02x}\n"
                    f"  expected ctx: {ctx_e!r}\n"
                    f"  actual   ctx: {ctx_a!r}")
    return "buffers equal length but assert failed (this should not happen)"
