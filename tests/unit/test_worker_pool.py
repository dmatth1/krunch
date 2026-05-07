"""CPU-testable surface of krunch.worker_pool.

The DecompressWorkerPool spawns processes that import torch + the
inference engine — that path needs a GPU and lives in the integration
suite. Here we cover the parts that don't touch CUDA."""
import pytest
from unittest.mock import patch

from krunch import worker_pool


# ---- default_worker_count --------------------------------------------------

def test_default_worker_count_respects_env(monkeypatch):
    monkeypatch.setenv("KRUNCH_DECOMPRESS_WORKERS", "8")
    assert worker_pool.default_worker_count() == 8


def test_default_worker_count_clamps_to_at_least_one(monkeypatch):
    monkeypatch.setenv("KRUNCH_DECOMPRESS_WORKERS", "0")
    assert worker_pool.default_worker_count() == 1


def test_default_worker_count_cpu_default(monkeypatch):
    monkeypatch.delenv("KRUNCH_DECOMPRESS_WORKERS", raising=False)
    fake_torch = type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: False)})})
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert worker_pool.default_worker_count() == 1


def test_default_worker_count_gpu_default(monkeypatch):
    monkeypatch.delenv("KRUNCH_DECOMPRESS_WORKERS", raising=False)
    fake_torch = type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: True)})})
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert worker_pool.default_worker_count() == 4


def test_default_worker_count_no_torch_falls_back_to_one(monkeypatch):
    """On a stranger's machine that ran `pip install krunch` without
    torch (CPU-only sanity), the import should swallow + return 1."""
    monkeypatch.delenv("KRUNCH_DECOMPRESS_WORKERS", raising=False)
    import builtins
    real_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=reject_torch):
        assert worker_pool.default_worker_count() == 1


# ---- DecompressWorkerPool constructor validation ---------------------------

def test_pool_rejects_zero_workers():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        worker_pool.DecompressWorkerPool(0)


def test_pool_rejects_negative_workers():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        worker_pool.DecompressWorkerPool(-3)
