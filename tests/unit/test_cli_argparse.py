"""argparse smoke for krunch.cli.main and krunch.plan_cli.main.

These exercise the user-facing flag surface end-to-end without loading
the model. The bodies of compress / decompress / plan are covered
elsewhere; here we just pin the CLI contract — so a stranger running
`krunch compress --help` doesn't get a stack trace, and adding a flag
doesn't accidentally remove an existing one.
"""
import sys
import pytest


# ---- krunch.cli (the in-image dispatcher) ---------------------------------

def test_cli_help_exits_zero(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["krunch", "--help"])
    from krunch import cli
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "compress" in out
    assert "decompress" in out


@pytest.mark.parametrize("subcommand", ["compress", "decompress"])
def test_cli_subcommand_help_exits_zero(capsys, monkeypatch, subcommand):
    monkeypatch.setattr(sys, "argv", ["krunch", subcommand, "--help"])
    from krunch import cli
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--in" in out
    assert "--out" in out


def test_cli_no_subcommand_errors(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["krunch"])
    from krunch import cli
    with pytest.raises(SystemExit) as exc:
        cli.main()
    # argparse exits 2 for missing required argument
    assert exc.value.code == 2


def test_cli_unknown_subcommand_errors(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["krunch", "nope"])
    from krunch import cli
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


# ---- krunch.plan_cli (renderer for batch artifacts) -----------------------

def _plan_args(*extra):
    return ["krunch.plan_cli",
            "--target", "aws-batch",
            "--mode", "compress",
            "--source", "s3://test/in",
            "--dest", "s3://test/out",
            "--workers", "4",
            "--image", "ghcr.io/dmatth1/krunch:latest",
            "--input-len", "1048576",
            *extra]


def test_plan_cli_renders_aws_batch(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", _plan_args())
    from krunch import plan_cli
    rc = plan_cli.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert '"jobName"' in out  # AWS Batch spec keys
    assert "krunch-compress-" in out


def test_plan_cli_dry_run_validates_silently(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", _plan_args("--dry-run"))
    from krunch import plan_cli
    rc = plan_cli.main()
    assert rc == 0
    # --dry-run intentionally emits nothing; CI consumes the exit code.
    assert capsys.readouterr().out == ""


def test_plan_cli_unknown_target_rejected(monkeypatch):
    argv = ["krunch.plan_cli", "--target", "k8s",  # not implemented today
            "--mode", "compress",
            "--source", "s3://x/y", "--dest", "s3://x/z",
            "--workers", "1", "--image", "x", "--input-len", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    from krunch import plan_cli
    with pytest.raises(SystemExit) as exc:
        plan_cli.main()
    assert exc.value.code == 2


def test_plan_cli_missing_required_mode_rejected(monkeypatch):
    """`--mode` is required (was defaultable; tightened so users can't
    accidentally render a compress plan when they meant decompress)."""
    argv = ["krunch.plan_cli", "--target", "aws-batch",
            "--source", "s3://x/y", "--dest", "s3://x/z",
            "--workers", "1", "--image", "x", "--input-len", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    from krunch import plan_cli
    with pytest.raises(SystemExit) as exc:
        plan_cli.main()
    assert exc.value.code == 2
