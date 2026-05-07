"""Coverage of the per-target validators + render error paths in
krunch.plan. The active TARGETS dict only registers aws-batch (JSON);
the YAML/Python/text validators are still kept for the planned
targets (see CONTRIBUTING.md) and need their own coverage."""
import pytest

from krunch import plan as krunch_plan


# ---- Validators ------------------------------------------------------------

def test_validate_json_accepts_well_formed():
    krunch_plan._validate_json('{"a": 1}')


def test_validate_json_rejects_malformed():
    with pytest.raises(Exception):
        krunch_plan._validate_json("{not json}")


def test_validate_yaml_accepts_k8s_shape():
    krunch_plan._validate_yaml("apiVersion: batch/v1\nkind: Job\n")


def test_validate_yaml_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        krunch_plan._validate_yaml("")


def test_validate_yaml_rejects_missing_apiversion():
    with pytest.raises(ValueError, match="apiVersion"):
        krunch_plan._validate_yaml("kind: Job\n")


def test_validate_py_accepts_valid_python():
    krunch_plan._validate_py("def main():\n    return 0\n")


def test_validate_py_rejects_syntax_error():
    with pytest.raises(SyntaxError):
        krunch_plan._validate_py("def main(:\n")


def test_validate_text_accepts_nonempty():
    krunch_plan._validate_text("#!/bin/bash\necho hi\n")


def test_validate_text_rejects_whitespace_only():
    with pytest.raises(ValueError, match="empty"):
        krunch_plan._validate_text("   \n\t\n")


# ---- render() error paths --------------------------------------------------

def test_render_unknown_target_lists_supported():
    with pytest.raises(ValueError, match="unknown target"):
        krunch_plan.render("nope", {})


def test_render_detects_unfilled_placeholder():
    """The {{ var }} substituter walks the rendered text afterwards
    looking for any leftover `{{ ... }}` — a missing context key
    surfaces as a clear ValueError instead of an unrunnable artifact."""
    # Use the real aws-batch template but pass an empty context.
    with pytest.raises(ValueError, match="unfilled placeholders"):
        krunch_plan.render("aws-batch", {})
