"""I/O helpers in krunch.cli that don't need the model.

Covers _read_input / _write_output (file vs stdin/stdout dispatch) and
_stdout_to_stderr (FD-level redirect that protects binary stdout from
ninja build noise during model load)."""
import io
import os
import sys
import pytest

from krunch import cli


# ---- _read_input -----------------------------------------------------------

def test_read_input_from_file(tmp_path):
    p = tmp_path / "in.bin"
    p.write_bytes(b"hello\x00\x01")
    assert cli._read_input(str(p)) == b"hello\x00\x01"


def test_read_input_from_stdin(monkeypatch):
    fake = io.BytesIO(b"piped bytes")
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": fake})())
    assert cli._read_input(None) == b"piped bytes"


def test_read_input_dash_means_stdin(monkeypatch):
    fake = io.BytesIO(b"via dash")
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": fake})())
    assert cli._read_input("-") == b"via dash"


# ---- _write_output ---------------------------------------------------------

def test_write_output_to_file(tmp_path):
    p = tmp_path / "out.bin"
    cli._write_output(str(p), b"\x00\x01\x02")
    assert p.read_bytes() == b"\x00\x01\x02"


def test_write_output_to_stdout(monkeypatch):
    captured = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": captured})())
    cli._write_output(None, b"piped out")
    assert captured.getvalue() == b"piped out"


def test_write_output_dash_means_stdout(monkeypatch):
    captured = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": captured})())
    cli._write_output("-", b"dashed")
    assert captured.getvalue() == b"dashed"


# ---- _stdout_to_stderr -----------------------------------------------------

def test_stdout_to_stderr_redirects_at_fd_level(capfd):
    """Subprocesses (e.g. ninja during the WKV kernel build) inherit
    FD 1 — Python-level sys.stdout reassignment doesn't reach them.
    _stdout_to_stderr must dup FD 2 onto FD 1 for the duration."""
    with cli._stdout_to_stderr():
        # Bypass Python buffering and write straight to FD 1.
        os.write(1, b"to-stderr\n")
    captured = capfd.readouterr()
    assert "to-stderr" in captured.err
    assert "to-stderr" not in captured.out


def test_stdout_to_stderr_restores_fd_on_exit(capfd):
    with cli._stdout_to_stderr():
        os.write(1, b"during\n")
    os.write(1, b"after\n")
    captured = capfd.readouterr()
    assert "after" in captured.out
    assert "during" in captured.err
