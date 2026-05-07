"""GPU test collection guards.

Skip the whole tests/unit/gpu/ tree when pytest is run without a GPU
(collect_ignore_glob, evaluated at collection time before module
imports — needed because the test modules import torch + krunch_ac
at top level and would crash the collector otherwise).
"""
import os

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

if not _HAS_CUDA:
    collect_ignore_glob = ["test_*.py", "bench_*.py"]


# ---------------------------------------------------------------------------
# Session-scoped engine + weights fixtures.
#
# Tests that need the model share ONE engine for the whole pytest session.
# Per-module engine fixtures load multiple RWKV models into GPU memory and,
# more importantly, leak `cpp_path._WEIGHTS_CACHE` / `_BATCHED_STATE_CACHE` /
# `_FULL_STEP_GRAPH_CACHE` entries that key on `id(weights)`. When an old
# model gets garbage-collected, Python can reuse the address; stale CUDA-
# graph captures then replay against unrelated memory and tests start
# silently corrupting each other's output.
# ---------------------------------------------------------------------------

if _HAS_CUDA:
    import pytest

    @pytest.fixture(scope="session")
    def engine():
        from krunch.inference import InferenceEngine
        eng = InferenceEngine()
        eng.load()
        return eng

    @pytest.fixture(scope="session")
    def weights(engine):
        from krunch import cpp_path
        return cpp_path.init_weights(engine._model, "cuda")

    @pytest.fixture(scope="session")
    def model(engine):
        return engine._model

    @pytest.fixture(autouse=True)
    def _reset_state_between_tests():
        """Reset everything that could carry across tests:

        1. Env vars that some tests toggle (W8A8, INT8_WEIGHTS, BF16).
           Pop them so the next test sees the production default
           (cpp_path's import-time auto-detect — W8A8 ON for sm_80+,
           OFF for sm_75).
        2. cpp_path graph + state caches. CUDA graphs captured by
           `decompress_chunks_batched` (B>=2) contaminate subsequent
           `decompress_chunk` (B=1) calls in the same process. Production
           cli.py never mixes the two; pytest does.
        """
        yield
        for k in ("KRUNCH_INT8_W8A8", "KRUNCH_INT8_WEIGHTS", "KRUNCH_BF16"):
            os.environ.pop(k, None)
        from krunch import cpp_path
        cpp_path._BATCHED_STATE_CACHE.clear()
        cpp_path._FULL_STEP_BUFS_CACHE.clear()
        cpp_path._FULL_STEP_GRAPH_CACHE.clear()
