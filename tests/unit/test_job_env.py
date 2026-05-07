"""Env-var contract tests for krunch.job.

Covers:
  - _resolve_part_index fallback chain (the AWS Batch Ref:: workaround)
  - run() mode dispatch (compress|decompress|finalize|invalid)
  - _parts_prefix shape

The actual compress/decompress/finalize bodies need a GPU + the model
image; they're exercised by the integration suite. This file pins the
env-var contract that every batch system depends on."""
import os
import pytest
from unittest.mock import patch

from krunch import job


# ---- _resolve_part_index ---------------------------------------------------

def _clear_env(monkeypatch):
    for var in ("KRUNCH_PART_INDEX", "AWS_BATCH_JOB_ARRAY_INDEX",
                "JOB_COMPLETION_INDEX", "SLURM_ARRAY_TASK_ID",
                "BATCH_TASK_INDEX", "MODAL_TASK_ID"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_uses_krunch_part_index_when_set(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("KRUNCH_PART_INDEX", "3")
    assert job._resolve_part_index() == 3


def test_resolve_falls_back_when_value_is_unsubstituted_ref(monkeypatch):
    _clear_env(monkeypatch)
    # Real bug: AWS Batch's Ref::AWS_BATCH_JOB_ARRAY_INDEX substitution
    # does NOT apply to env values, only to command-line params. The
    # literal string reaches the worker; we must fall back.
    monkeypatch.setenv("KRUNCH_PART_INDEX", "Ref::AWS_BATCH_JOB_ARRAY_INDEX")
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "5")
    assert job._resolve_part_index() == 5


@pytest.mark.parametrize("native_var,native_val", [
    ("AWS_BATCH_JOB_ARRAY_INDEX", "0"),
    ("JOB_COMPLETION_INDEX", "1"),
    ("SLURM_ARRAY_TASK_ID", "7"),
    ("BATCH_TASK_INDEX", "2"),
    ("MODAL_TASK_ID", "9"),
])
def test_resolve_falls_back_to_native_orchestrator_var(monkeypatch, native_var, native_val):
    _clear_env(monkeypatch)
    monkeypatch.setenv(native_var, native_val)
    assert job._resolve_part_index() == int(native_val)


def test_resolve_raises_when_nothing_set(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(KeyError, match="KRUNCH_PART_INDEX"):
        job._resolve_part_index()


def test_resolve_prefers_krunch_var_over_native(monkeypatch):
    """If KRUNCH_PART_INDEX is set to a real number, it wins even if
    a native orchestrator var is also present."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("KRUNCH_PART_INDEX", "10")
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "99")
    assert job._resolve_part_index() == 10


# ---- run() mode dispatch ---------------------------------------------------

def test_run_invalid_mode_exits_nonzero(monkeypatch):
    monkeypatch.setenv("KRUNCH_MODE", "garbage")
    with pytest.raises(SystemExit) as exc:
        job.run()
    assert exc.value.code != 0


def test_run_missing_mode_exits_nonzero(monkeypatch):
    monkeypatch.delenv("KRUNCH_MODE", raising=False)
    with pytest.raises(SystemExit) as exc:
        job.run()
    assert exc.value.code != 0


@pytest.mark.parametrize("mode,target", [
    ("compress",   "_run_compress_worker"),
    ("decompress", "_run_decompress_worker"),
    ("finalize",   "_run_finalize"),
])
def test_run_dispatches_to_correct_worker(monkeypatch, mode, target):
    """Mode dispatch is the public contract every batch template fills.
    Mock the actual worker bodies so this test stays CPU-only."""
    monkeypatch.setenv("KRUNCH_MODE", mode)
    with patch.object(job, target) as m:
        job.run()
        m.assert_called_once()


# ---- _parts_prefix ---------------------------------------------------------

def test_parts_prefix_shape():
    assert job._parts_prefix("s3://bucket/out.krunch") == "s3://bucket/out.krunch.parts"
    # Trailing slash stripped — finalize loops over .parts/<i> consistently.
    assert job._parts_prefix("file:///tmp/x.krunch/") == "file:///tmp/x.krunch.parts"
