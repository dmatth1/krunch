"""Repo-root conftest: put `src/` on sys.path so tests can `import krunch`
without callers having to set `PYTHONPATH=src` first. There's no
pyproject.toml today (pre-1.0), so this is the lightest-weight way
to make `pytest tests/unit/` work out of the box from a fresh clone.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
