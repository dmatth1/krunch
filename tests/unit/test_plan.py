"""Tier-1 test: every `krunch plan` target renders + validates.

Free-to-run in CI (no GPU, no AWS). Catches template syntax errors
+ schema regressions before they get pushed to ghcr.io.
"""
import pytest

from krunch import plan as krunch_plan


CTX_BASE = {
    "mode":              "compress",
    "input_url":         "s3://krunch-test/sample.bin",
    "output_url":        "s3://krunch-test/out.krunch",
    "input_len":         1_000_000,
    "n_workers":         4,
    "n_workers_minus_1": 3,
    "image":             "ghcr.io/dmatth1/krunch:latest",
    "queue":             "krunch-queue",
    "job_definition":    "krunch-job",
    "cpus":              4,
    "gpus":              1,
    "memory_mb":         16384,
    "timeout_s":         3600,
    "slurm_time":        "01:00:00",
    "run_id":            "test-run",
}


@pytest.mark.parametrize("target", sorted(krunch_plan.TARGETS.keys()))
def test_render_and_validate(target: str):
    ctx = {**CTX_BASE, "target": target}
    rendered = krunch_plan.render(target, ctx)
    krunch_plan.validate(target, rendered)
    assert "test-run" in rendered  # run_id substituted
    assert "{{ " not in rendered    # no unfilled placeholders


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        krunch_plan.render("unknown-target", {**CTX_BASE, "target": "x"})


def test_missing_var_raises():
    ctx = {k: v for k, v in CTX_BASE.items() if k != "queue"}
    ctx["target"] = "aws-batch"
    with pytest.raises(ValueError, match="unfilled placeholders"):
        krunch_plan.render("aws-batch", ctx)
