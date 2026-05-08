"""
Pin the KRUNCH_* env-var contract.

Every KRUNCH_* env var read by `krunch/` Python code MUST be registered
in `krunch/_env.py`. New flags get a row in the table, removed flags
come out of the table. This test detects drift in either direction.

Out of scope for this test (registered separately):
- Flags read only in tests/, scripts/, and tests/integration/*.sh —
  those are deploy/test-driver settings, not runtime contract.
- C++/CUDA-only flags read in `krunch/kernels/` (e.g., layer_cpp.cpp).
  Those are routing tunables; this test focuses on the Python read
  surface where flags are most likely to drift.

Per docs/TIER_3_CLEANUP.md §2.2.
"""

import os
import re
from pathlib import Path

import pytest

from krunch import _env


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = REPO_ROOT / "krunch"

# Match KRUNCH_* names that appear in a context that *reads* an env var:
# `os.environ["KRUNCH_X"]`, `os.environ.get("KRUNCH_X")`,
# `os.getenv("KRUNCH_X")`, `os.environ.setdefault("KRUNCH_X", ...)`,
# or `os.environ.pop("KRUNCH_X")`. This skips bare string literals
# used as log message prefixes etc.
_READ_RE = re.compile(
    r"""
    (?:
        os\.environ\s*\[\s*["'](KRUNCH_[A-Z0-9_]+)["']
      | os\.environ\s*\.\s*(?:get|setdefault|pop)\s*\(\s*["'](KRUNCH_[A-Z0-9_]+)["']
      | os\.getenv\s*\(\s*["'](KRUNCH_[A-Z0-9_]+)["']
    )
    """,
    re.VERBOSE,
)
_FLAG_RE = re.compile(r"\bKRUNCH_[A-Z0-9_]+")


def _scan_python_sources() -> set[str]:
    """Collect every KRUNCH_* env var actually read from krunch/*.py."""
    found: set[str] = set()
    for p in PY_ROOT.rglob("*.py"):
        # _env.py is the registry — its mentions are part of the spec,
        # not a usage to validate.
        if p.name == "_env.py":
            continue
        text = p.read_text()
        for m in _READ_RE.finditer(text):
            for g in m.groups():
                if g is not None:
                    found.add(g)
    return found


def test_every_python_flag_is_registered():
    used = _scan_python_sources()
    registered = set(_env.BY_NAME.keys())
    unregistered = used - registered
    assert not unregistered, (
        f"KRUNCH_* env vars referenced in krunch/*.py but not registered "
        f"in krunch/_env.py: {sorted(unregistered)}. Add them to "
        f"_env.ENV_FLAGS or remove the read site."
    )


def test_no_orphan_registrations_for_python_only_flags():
    """Heuristic: a flag tagged tier='internal'/'tuning'/'debug' should
    appear at least once in krunch/*.py OR in krunch/kernels/. ('ops'
    flags may live entirely in job.py, which is fine.)
    """
    used_py = _scan_python_sources()
    cpp_text = ""
    for p in (PY_ROOT / "kernels").rglob("*"):
        if p.suffix in (".cpp", ".cu", ".cuh", ".h", ".hpp"):
            cpp_text += p.read_text()
    used_cpp = set(_FLAG_RE.findall(cpp_text))
    used = used_py | used_cpp
    orphans = []
    for f in _env.ENV_FLAGS:
        if f.name not in used:
            orphans.append(f.name)
    assert not orphans, (
        f"Registered in krunch/_env.py but not referenced in any source: "
        f"{orphans}. Either restore a usage or remove from _env.ENV_FLAGS."
    )


def test_bitstream_flags_documented():
    """Every flag with bitstream_impact='yes' MUST have a non-trivial
    summary mentioning 'roundtrip', 'bytes', 'encoder', 'decoder',
    'codec', or 'blob' — anything that signals the operator that
    flipping the flag has cross-version implications."""
    keywords = ("roundtrip", "byte", "encoder", "decoder", "codec",
                "blob", "chunk", "weight", "structure")
    bad = []
    for f in _env.ENV_FLAGS:
        if f.bitstream_impact != "yes":
            continue
        if not any(k in f.summary.lower() for k in keywords):
            bad.append(f.name)
    assert not bad, (
        f"Bitstream-affecting flags with weak summary (none of {keywords} "
        f"mentioned): {bad}. Strengthen their summary in _env.py to warn "
        f"future operators."
    )


def test_names_helper_is_sorted():
    assert _env.names() == sorted(_env.BY_NAME.keys())


def test_is_bitstream_flag_for_known_flags():
    assert _env.is_bitstream_flag("KRUNCH_INT8_W8A8") is True
    assert _env.is_bitstream_flag("KRUNCH_PHASE_PROFILE") is False
    assert _env.is_bitstream_flag("KRUNCH_DOES_NOT_EXIST") is False


def test_no_duplicate_registrations():
    seen = set()
    for f in _env.ENV_FLAGS:
        assert f.name not in seen, f"duplicate registration: {f.name}"
        seen.add(f.name)
