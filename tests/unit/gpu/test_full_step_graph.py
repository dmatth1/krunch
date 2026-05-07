"""KRUNCH_OWN_WKV=1 + KRUNCH_DECOMPRESS_GRAPH=full path.

Production decompress mode when own-WKV is enabled (the production
default for full-graph capture). Captures the entire per-token decode
step into ONE CUDA graph that's replayed n_tokens times.

Exercises `cpp_path.forward_step_full_graphed_v3` + the buffer cache
+ the eager fallback path. Without this test, those ~150 lines of
graph capture / replay infrastructure carry no coverage.
"""
import os
# These env vars must be set BEFORE krunch.cpp_path imports — it auto-
# disables some kernels on sm<8 at module load. Override per-test
# rather than via conftest because most other tests want eager mode.
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest


@pytest.fixture(autouse=True)
def _own_wkv_full_graph(monkeypatch):
    """Force the own-WKV + full-step graph path for every test in
    this file."""
    monkeypatch.setenv("KRUNCH_OWN_WKV", "1")
    monkeypatch.setenv("KRUNCH_DECOMPRESS_GRAPH", "full")
    yield


SAMPLE = b"The quick brown fox jumps over the lazy dog. " * 10  # ~450 B


def test_full_step_graph_roundtrip(engine):
    """Compress a chunk with the standard path, then decompress with
    KRUNCH_OWN_WKV=1 + KRUNCH_DECOMPRESS_GRAPH=full. Output must match
    the original bytes — graph capture + replay shouldn't introduce
    any divergence vs eager."""
    encoded = engine.compress_chunk(SAMPLE)
    decoded = engine.decompress_chunk(encoded)
    assert decoded == SAMPLE, (
        "full-step graph decompress diverged from original — graph "
        "capture/replay is broken"
    )


def test_full_step_graph_replay_is_repeatable(engine):
    """Two consecutive decompress calls against the SAME engine must
    produce the same output. Graphs are pointer-bound; the second
    call replays the graph captured by the first."""
    encoded = engine.compress_chunk(SAMPLE)
    decoded_1 = engine.decompress_chunk(encoded)
    decoded_2 = engine.decompress_chunk(encoded)
    assert decoded_1 == decoded_2 == SAMPLE
