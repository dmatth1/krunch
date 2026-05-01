"""
`krunch plan` — render a runnable artifact for the user's batch system.

We don't run distributed jobs ourselves. Instead, `krunch plan
--target <name>` emits a syntactically-valid artifact (job definition,
manifest, script) that the user submits with their own credentials
to whichever orchestrator they already operate.

All targets share the same env-var contract (see `krunch/job.py`).
The per-target template only handles the orchestrator-specific
plumbing (how an array index propagates, how a finalize task
depends on the worker batch, etc.).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


# Each template fills in these variables; missing keys are the
# user's responsibility to supply via CLI flags.
COMMON_CONTEXT_KEYS = (
    "target", "mode", "input_url", "output_url", "input_len",
    "n_workers", "image",
    # orchestrator-specific
    "queue", "memory_mb", "cpus", "gpus",
    # advanced
    "timeout_s", "image_pull_policy",
)

# Targets shipped at v1. Each must have <name>.<ext>.j2 in
# krunch/plan/templates/.
TARGETS = {
    "aws-batch": {
        "ext": "json",
        "template": "aws_batch.json.j2",
        "validator": "_validate_json",
    },
    "k8s": {
        "ext": "yaml",
        "template": "k8s.yaml.j2",
        "validator": "_validate_yaml",
    },
    "modal": {
        "ext": "py",
        "template": "modal.py.j2",
        "validator": "_validate_py",
    },
    "ray": {
        "ext": "py",
        "template": "ray.py.j2",
        "validator": "_validate_py",
    },
    "slurm": {
        "ext": "sbatch",
        "template": "slurm.sbatch.j2",
        "validator": "_validate_text",
    },
    "gcp-batch": {
        "ext": "json",
        "template": "gcp_batch.json.j2",
        "validator": "_validate_json",
    },
    "local": {
        "ext": "sh",
        "template": "local.sh.j2",
        "validator": "_validate_text",
    },
}


def render(target: str, context: dict[str, Any]) -> str:
    """Render the named target's template with the given context.
    Raises ValueError if target is unknown or required keys are missing.
    """
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; supported: {sorted(TARGETS.keys())}")
    info = TARGETS[target]
    template_path = Path(__file__).parent / "templates" / info["template"]
    if not template_path.exists():
        raise ValueError(
            f"template not found for target {target!r}: {template_path}")
    raw = template_path.read_text()
    return _simple_render(raw, context)


def validate(target: str, rendered: str) -> None:
    """Schema-check the rendered artifact. Raises ValueError on failure."""
    info = TARGETS[target]
    validator = globals()[info["validator"]]
    validator(rendered)


def _simple_render(template: str, ctx: dict[str, Any]) -> str:
    """Tiny `{{ var }}` substitution. We use this instead of jinja2
    to keep the runtime dep footprint to zero. Templates use only
    `{{ varname }}` — no control flow. Anything more complex goes
    in Python before render()."""
    out = template
    for k, v in ctx.items():
        out = out.replace("{{ " + k + " }}", str(v))
    # Detect unfilled placeholders so the user gets a clear error
    # instead of an unrunnable artifact.
    import re
    leftovers = re.findall(r"\{\{\s*(\w+)\s*\}\}", out)
    if leftovers:
        raise ValueError(
            f"template {ctx.get('target')!r} has unfilled placeholders: "
            f"{sorted(set(leftovers))}")
    return out


def _validate_json(text: str) -> None:
    json.loads(text)


def _validate_yaml(text: str) -> None:
    # YAML lib isn't a runtime dep; do a light text check that the
    # file is non-empty and starts with `apiVersion:` (k8s-shape).
    if not text.strip():
        raise ValueError("k8s manifest is empty")
    if "apiVersion:" not in text:
        raise ValueError("k8s manifest missing apiVersion:")


def _validate_py(text: str) -> None:
    compile(text, "<plan>", "exec")


def _validate_text(text: str) -> None:
    if not text.strip():
        raise ValueError("artifact is empty")
