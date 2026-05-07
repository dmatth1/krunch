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


def test_plan_cli_input_len_zero_triggers_url_io_size(monkeypatch, capsys):
    """--input-len 0 means 'go ask the source URL'. Mock url_io.size so
    we don't hit the network."""
    monkeypatch.setattr(sys, "argv", _plan_args("--input-len", "0"))
    from krunch import plan_cli, url_io
    monkeypatch.setattr(url_io, "size", lambda url: 12345)
    rc = plan_cli.main()
    assert rc == 0
    # Rendered spec should contain the resolved size as KRUNCH_INPUT_LEN.
    assert '"12345"' in capsys.readouterr().out


def test_plan_cli_input_len_zero_warns_on_size_failure(monkeypatch, capsys):
    """If url_io.size() raises (e.g. private bucket, no creds), emit a
    warn-and-continue rather than crashing — caller can supply --input-len."""
    monkeypatch.setattr(sys, "argv", _plan_args("--input-len", "0"))
    from krunch import plan_cli, url_io

    def boom(_):
        raise RuntimeError("creds missing")
    monkeypatch.setattr(url_io, "size", boom)

    rc = plan_cli.main()
    assert rc == 0  # rendered with input_len=0 + warning on stderr
    err = capsys.readouterr().err
    assert "couldn't stat" in err
    assert "creds missing" in err


def test_plan_cli_render_failure_returns_one(monkeypatch, capsys):
    """When render() raises ValueError (e.g. broken template), main()
    should print the error to stderr and return 1, not crash."""
    monkeypatch.setattr(sys, "argv", _plan_args())
    from krunch import plan_cli, plan_cli as _pc
    from krunch import plan as krunch_plan

    def boom(*_a, **_kw):
        raise ValueError("bad template")
    monkeypatch.setattr(krunch_plan, "render", boom)

    rc = plan_cli.main()
    assert rc == 1
    assert "bad template" in capsys.readouterr().err
