"""KRUNCH_DECOMPRESS_GRAPH=per_layer mode.

Captures one CUDA graph per layer (12 graphs total) instead of one
graph for the whole step. Exercises `forward_stepped_graphed_v2` (B=1)
and `forward_stepped_batched_graphed_v2` (B>=2) plus their per-layer
capture helpers (`_capture_one_layer_graph`,
`_capture_one_layer_graph_batched`, `_get_stepped_bufs`,
`_get_stepped_batched_bufs`, `_snapshot_state`, `_restore_state`).

These ~150 lines of cpp_path.py are otherwise uncovered.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest


@pytest.fixture(autouse=True)
def _per_layer_graph(monkeypatch):
    monkeypatch.setenv("KRUNCH_OWN_WKV", "1")
    monkeypatch.setenv("KRUNCH_DECOMPRESS_GRAPH", "per_layer")
    yield


SAMPLE = b"def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n" * 6


def test_per_layer_graph_single_chunk_roundtrip(engine):
    """Single-chunk decompress with per-layer graphs (B=1) — exercises
    forward_stepped_graphed_v2 + 12-layer-graph capture."""
    encoded = engine.compress_chunk(SAMPLE)
    decoded = engine.decompress_chunk(encoded)
    assert decoded == SAMPLE


CHUNKS = [
    b"chunk-A " * 40,
    b"chunk-B " * 50,
    b"chunk-C " * 30,
]


def test_per_layer_graph_batched_roundtrip(engine):
    """Cross-chunk batched decompress with per-layer graphs (B>=2) —
    exercises forward_stepped_batched_graphed_v2 + the batched
    capture path."""
    encs = [engine.compress_chunk(c) for c in CHUNKS]
    decoded = engine.decompress_chunks_batched(encs)
    for orig, got in zip(CHUNKS, decoded):
        assert got == orig
