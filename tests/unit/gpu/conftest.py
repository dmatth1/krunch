"""GPU test collection guards + global env defaults.

Two things this conftest does:

1. Skip the whole tests/unit/gpu/ tree when pytest is run without a GPU
   (collect_ignore_glob, evaluated at collection time before module
   imports — needed because the test modules import torch + krunch_ac
   at top level and would crash the collector otherwise).

2. Default `KRUNCH_INT8_W8A8=0` for every test in this directory.
   Reason: docs/Bugs.md #1 — small-chunk codec break on sm_75 with W8A8=1.
   Tests that explicitly want to exercise W8A8 dispatch toggle the
   var inside the test (and clear the init_weights cache).
"""
import os

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

if not _HAS_CUDA:
    collect_ignore_glob = ["test_*.py", "bench_*.py", "rwkv4_step_ref.py"]


# Set BEFORE any test module imports cpp_path / krunch.inference; pytest
# imports conftest.py before collecting test modules, so this is the
# earliest hook available for env-var defaults.
os.environ["KRUNCH_INT8_W8A8"] = "0"


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

        1. Env vars some tests toggle (W8A8, INT8_WEIGHTS, BF16). Keeps
           the conftest's W8A8=0 default from getting clobbered.
        2. cpp_path graph + state caches. CUDA graphs captured by
           `decompress_chunks_batched` (B>=2) contaminate subsequent
           `decompress_chunk` (B=1) calls in the same process. Production
           cli.py never mixes the two; pytest does.
        """
        yield
        os.environ["KRUNCH_INT8_W8A8"] = "0"
        os.environ.pop("KRUNCH_INT8_WEIGHTS", None)
        os.environ.pop("KRUNCH_BF16", None)
        from krunch import cpp_path
        cpp_path._BATCHED_STATE_CACHE.clear()
        cpp_path._FULL_STEP_BUFS_CACHE.clear()
        cpp_path._FULL_STEP_GRAPH_CACHE.clear()
