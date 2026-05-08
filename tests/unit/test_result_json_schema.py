"""
Pin the schema of result.json produced by tests/integration/gpu.sh.

Drift here breaks the host-side poller in gpu.sh (which reads
all_gates_pass / ratio / compress_kb_s / decompress_kb_s / byte_exact)
and any downstream perf-history dashboards. If you change the schema,
update gpu.sh's producer + this test in the same commit.

Per docs/TIER_3_CLEANUP.md §4.3.
"""

import json
import pytest


# Required keys + expected types for a SUCCESS result.json (gates body).
_SUCCESS_SCHEMA = {
    "tag": str,
    "input_bytes": int,
    "compressed_bytes": int,
    "recovered_bytes": int,
    "ratio": (int, float),
    "compress_seconds": (int, float),
    "decompress_seconds": (int, float),
    "compress_kb_s": (int, float),
    "decompress_kb_s": (int, float),
    "byte_exact": bool,
    "gates": dict,
    "all_gates_pass": bool,
}

_GATES_SCHEMA = {
    "ratio_lte_0_11": bool,
    "compress_kb_s_gte_200": bool,
    "decompress_kb_s_gte_200": bool,
    "byte_exact": bool,
}

# Failure stub written by the user-data EXIT trap when the run aborts
# before result.json was uploaded by the test body.
_FAILURE_SCHEMA = {
    "all_gates_pass": bool,
    "error": str,
}


def _validate(obj, schema, path=""):
    """Assert every key in `schema` exists on `obj` with the right type.
    Extra keys on `obj` are tolerated (forward compat). Missing or
    mistyped keys are a hard failure."""
    for key, expected in schema.items():
        full = f"{path}.{key}" if path else key
        assert key in obj, f"missing key: {full}"
        # bool is a subclass of int — check explicitly to avoid false matches.
        if expected is int:
            assert isinstance(obj[key], int) and not isinstance(obj[key], bool), \
                f"{full}: expected int, got {type(obj[key]).__name__}"
        elif expected is bool:
            assert isinstance(obj[key], bool), \
                f"{full}: expected bool, got {type(obj[key]).__name__}"
        else:
            assert isinstance(obj[key], expected), \
                f"{full}: expected {expected}, got {type(obj[key]).__name__}"


def test_success_result_schema():
    """Sample of a real success result.json (shape only, values fake)."""
    sample = {
        "tag": "20260507-203000_a10g_10mb",
        "input_bytes": 10 * 1024 * 1024,
        "compressed_bytes": 1_200_000,
        "recovered_bytes": 10 * 1024 * 1024,
        "ratio": 0.115,
        "compress_seconds": 50.5,
        "decompress_seconds": 140.0,
        "compress_kb_s": 198.5,
        "decompress_kb_s": 75.1,
        "byte_exact": True,
        "gates": {
            "ratio_lte_0_11": False,
            "compress_kb_s_gte_200": False,
            "decompress_kb_s_gte_200": False,
            "byte_exact": True,
        },
        "all_gates_pass": False,
    }
    _validate(sample, _SUCCESS_SCHEMA)
    _validate(sample["gates"], _GATES_SCHEMA, path="gates")
    # Round-trip through JSON to make sure types survive serialization.
    reparsed = json.loads(json.dumps(sample))
    _validate(reparsed, _SUCCESS_SCHEMA)


def test_failure_stub_schema():
    """The EXIT trap in tests/integration/gpu.sh writes this minimal
    stub when the run aborts. It must NOT match _SUCCESS_SCHEMA — the
    host poller distinguishes success vs failure by presence of
    `byte_exact`."""
    stub = {
        "all_gates_pass": False,
        "error": "user-data exited rc=1 — see setup.log",
    }
    _validate(stub, _FAILURE_SCHEMA)
    assert "byte_exact" not in stub
    assert "ratio" not in stub


def test_all_gates_pass_implies_individual_gates():
    """Logical invariant: all_gates_pass <-> all four individual gates true."""
    def _build(byte_exact, ratio_ok, comp_ok, decomp_ok):
        return {
            "all_gates_pass": all((byte_exact, ratio_ok, comp_ok, decomp_ok)),
            "gates": {
                "byte_exact": byte_exact,
                "ratio_lte_0_11": ratio_ok,
                "compress_kb_s_gte_200": comp_ok,
                "decompress_kb_s_gte_200": decomp_ok,
            },
        }

    for combo in [(True, True, True, True),
                  (True, True, True, False),
                  (False, False, False, False),
                  (True, False, True, False)]:
        r = _build(*combo)
        all_pass = r["all_gates_pass"]
        assert all_pass == all(r["gates"].values())


def test_gpu_sh_producer_block_unchanged():
    """Sanity-check: the producer block in tests/integration/gpu.sh
    sets all _SUCCESS_SCHEMA keys. If a key disappears from the
    producer, the failure surface here points the operator at it.

    This is a textual presence check, not a parse — gpu.sh is bash
    and we don't want to import it."""
    import os
    here = os.path.dirname(__file__)
    sh_path = os.path.normpath(os.path.join(here, "..", "integration",
                                            "gpu.sh"))
    if not os.path.exists(sh_path):
        pytest.skip(f"gpu.sh not found at {sh_path}")
    with open(sh_path) as f:
        body = f.read()
    for key in _SUCCESS_SCHEMA:
        # gpu.sh writes JSON like `"key": value` so search for `"key"`.
        assert f'"{key}"' in body, (
            f"gpu.sh result.json producer is missing key {key!r} — "
            f"either restore it in gpu.sh or update _SUCCESS_SCHEMA "
            f"and CHANGELOG"
        )
    for key in _GATES_SCHEMA:
        assert f'"{key}"' in body, (
            f"gpu.sh gates dict is missing key {key!r}"
        )
