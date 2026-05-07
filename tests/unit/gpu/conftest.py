"""GPU test collection guards + cross-test isolation.

Three things this conftest does:

1. Skip the whole tests/unit/gpu/ tree when pytest is run without a
   GPU (collect_ignore_glob, evaluated at collection time before
   module imports — the test modules import torch + krunch_ac at top
   level and would crash the collector otherwise).

2. Provide session-scoped engine / weights / model fixtures so all
   tests that need the model share ONE RWKV load. Per-module fixtures
   would leak `cpp_path._WEIGHTS_CACHE` / `_BATCHED_STATE_CACHE` etc.
   entries keyed on id(weights); when an old weights dict is GC'd,
   Python can reuse the address and stale captured CUDA graphs replay
   against unrelated memory.

3. Reset every piece of process-global state between tests (autouse
   fixture). Keeping captured CUDA graphs across tests was the cause
   of docs/Bugs.md #2 — tests that passed in isolation failed in the
   full suite.
"""
import os

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

if not _HAS_CUDA:
    collect_ignore_glob = ["test_*.py", "bench_*.py"]


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

    # cpp_path graph + pointer-bound state caches. CUDA graphs are
    # tied to specific tensor data pointers; replaying a captured
    # graph against a different tensor (e.g., from a previous test)
    # silently corrupts output.
    #
    # Crucially: do NOT clear `_WEIGHTS_CACHE`. The session-scoped
    # `weights` fixture holds a reference to the cached weights dict.
    # Clearing the cache forces init_weights to rebuild fresh weights
    # the next time engine.compress_chunk is called → tests using
    # `weights` (fixture) get the OLD dict; tests using `engine` get
    # the NEW dict; the two dicts have different tensor pointers, and
    # any graph captured against one fails against the other.
    _CPP_CACHES = (
        "_STEPPED_BUFS_CACHE",
        "_GRAPH_CACHE",
        "_STEPPED_BATCHED_BUFS_CACHE",
        "_GRAPH_BATCHED_CACHE",
        "_FULL_STEP_BUFS_CACHE",
        "_FULL_STEP_GRAPH_CACHE",
        "_BATCHED_STATE_CACHE",
    )

    # Codec env vars that some tests toggle (test_w8a8_dispatch,
    # test_init_weights_*). Pop them between tests so the next test
    # sees the production default (W8A8=1, BF16=0, INT8_WEIGHTS=0).
    _CODEC_ENV_VARS = (
        "KRUNCH_INT8_W8A8",
        "KRUNCH_INT8_WEIGHTS",
        "KRUNCH_BF16",
    )

    @pytest.fixture(autouse=True)
    def _reset_state_between_tests():
        yield
        for k in _CODEC_ENV_VARS:
            os.environ.pop(k, None)
        from krunch import cpp_path
        for name in _CPP_CACHES:
            getattr(cpp_path, name).clear()
        # cuBLAS pinned algo is global C++ state — set to default
        # (-1 = CUBLAS_GEMM_DEFAULT). Without this, test_cublas_pinned's
        # algo 99 leaks into every subsequent test.
        try:
            import krunch_ac_cuda
            krunch_ac_cuda.set_cublas_pinned_algo(-1)
        except Exception:
            pass
