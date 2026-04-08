"""Compressor wrappers for the benchmark harness.

Each compressor is a subclass of `Compressor` that knows how to:
  1. Compress a file to an output path
  2. Decompress an output back to a file
  3. Report what preset it's using

The wrapper's job is to shell out to the real compressor and return
timing information. It does not measure anything itself — the harness
in `bench.py` handles measurement so all compressors are measured by
the same code.

Design notes:

- Uses only the Python standard library. No numpy, no pandas, no click.
  See DECISIONS.md D7 for why.
- Each compressor is tested via the same round-trip interface, so adding
  a new compressor means adding one subclass.
- L3TC is present as a stub; the full implementation is wired up in
  Phase 0 step "L3TC wrapper".
- Compression and decompression are separate methods because we want to
  measure them independently (real workloads are often asymmetric).
"""
from __future__ import annotations

import abc
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CompressorResult:
    """What a single compress or decompress operation returns.

    Kept deliberately small — the harness adds timing and memory
    information around these.
    """

    ok: bool
    command: list[str]
    input_bytes: int
    output_bytes: int
    stderr: str = ""


class Compressor(abc.ABC):
    """Abstract base class for every compressor we benchmark.

    Subclasses implement `compress_cmd` and `decompress_cmd` which return
    the argv to shell out. The base class handles running them, capturing
    stderr, and returning a `CompressorResult`.
    """

    #: Short name used in JSON output and logs (e.g. "gzip-9", "zstd-22").
    name: str = "abstract"

    #: Whether the compressor is expected to be available on the system.
    #: False for things like L3TC until setup.sh has been run.
    expected_available: bool = True

    #: File extension the compressor produces, without leading dot.
    extension: str = "bin"

    def available(self) -> bool:
        """Return True if the compressor's binary is on PATH."""
        return shutil.which(self.binary) is not None

    @property
    @abc.abstractmethod
    def binary(self) -> str:
        """Name of the executable. Used to check availability."""

    @abc.abstractmethod
    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        """Return the command to compress input_path to output_path."""

    @abc.abstractmethod
    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        """Return the command to decompress input_path to output_path."""

    def run(self, cmd: list[str]) -> tuple[int, str]:
        """Run a command, return (returncode, stderr)."""
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            return proc.returncode, proc.stderr
        except FileNotFoundError as e:
            return -1, f"binary not found: {e}"

    def compress(self, input_path: Path, output_path: Path) -> CompressorResult:
        """Compress input_path -> output_path. Return result metadata."""
        cmd = self.compress_cmd(input_path, output_path)
        rc, stderr = self.run(cmd)
        ok = rc == 0 and output_path.exists()
        return CompressorResult(
            ok=ok,
            command=cmd,
            input_bytes=input_path.stat().st_size if input_path.exists() else 0,
            output_bytes=output_path.stat().st_size if output_path.exists() else 0,
            stderr=stderr,
        )

    def decompress(self, input_path: Path, output_path: Path) -> CompressorResult:
        """Decompress input_path -> output_path. Return result metadata."""
        cmd = self.decompress_cmd(input_path, output_path)
        rc, stderr = self.run(cmd)
        ok = rc == 0 and output_path.exists()
        return CompressorResult(
            ok=ok,
            command=cmd,
            input_bytes=input_path.stat().st_size if input_path.exists() else 0,
            output_bytes=output_path.stat().st_size if output_path.exists() else 0,
            stderr=stderr,
        )


# -------- Classical compressors -------- #


class Gzip(Compressor):
    """gzip at a specified preset level (1-9)."""

    def __init__(self, level: int = 9):
        self.level = level
        self.name = f"gzip-{level}"
        self.extension = "gz"

    @property
    def binary(self) -> str:
        return "gzip"

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        # gzip writes to stdout when given -c, which lets us control the
        # output filename instead of relying on gzip's in-place rewrite.
        return [
            "sh",
            "-c",
            f"gzip -{self.level} -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"gzip -d -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]


class Bzip2(Compressor):
    """bzip2 at a specified preset level (1-9)."""

    def __init__(self, level: int = 9):
        self.level = level
        self.name = f"bzip2-{level}"
        self.extension = "bz2"

    @property
    def binary(self) -> str:
        return "bzip2"

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"bzip2 -{self.level} -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"bzip2 -d -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]


class Xz(Compressor):
    """xz at a specified preset level (0-9), optional extreme mode."""

    def __init__(self, level: int = 9, extreme: bool = True):
        self.level = level
        self.extreme = extreme
        suffix = "e" if extreme else ""
        self.name = f"xz-{level}{suffix}"
        self.extension = "xz"

    @property
    def binary(self) -> str:
        return "xz"

    def _flags(self) -> str:
        flag = f"-{self.level}"
        if self.extreme:
            flag += "e"
        return flag

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"xz {self._flags()} -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"xz -d -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]


class Zstd(Compressor):
    """zstd at a specified preset level.

    Levels 1-19 are standard presets. 20-22 are "ultra" presets that
    require the --ultra flag. We pass --ultra automatically for levels
    above 19.
    """

    def __init__(self, level: int = 19):
        self.level = level
        self.name = f"zstd-{level}"
        self.extension = "zst"

    @property
    def binary(self) -> str:
        return "zstd"

    def _flags(self) -> str:
        if self.level > 19:
            return f"--ultra -{self.level}"
        return f"-{self.level}"

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"zstd -q {self._flags()} -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        return [
            "sh",
            "-c",
            f"zstd -q -d -c {shell_quote(input_path)} > {shell_quote(output_path)}",
        ]


# -------- Neural compressors -------- #


class L3TC(Compressor):
    """L3TC wrapper — stub until scripts/setup.sh has been run.

    This wrapper will shell out to the L3TC Python reference
    implementation once it's been cloned under vendor/L3TC/ and its
    dependencies installed in a local venv.

    We deliberately treat L3TC as a subprocess rather than importing its
    Python API because:
      1. It decouples L3TC's Python environment from the harness's.
         Harness runs on system Python; L3TC runs in its own venv with
         pinned PyTorch.
      2. Subprocess timing is what classical compressors report too, so
         comparisons are apples-to-apples (no hidden Python startup cost
         that we'd ignore for a C binary).
      3. L3TC's reference code does not expose a clean CLI today, so we
         build one in `vendor/L3TC/wrapper.py` and shell out to it.
    """

    def __init__(self, variant: str = "3.2M", batch_size: int = 1, device: str = "cpu"):
        self.variant = variant
        self.batch_size = batch_size
        self.device = device
        self.name = f"l3tc-{variant}-b{batch_size}-{device}"
        self.extension = "l3tc"
        self.expected_available = False  # until setup.sh has run

    @property
    def binary(self) -> str:
        # "python3" is always available; real availability is checked
        # via the presence of vendor/L3TC/wrapper.py
        return "python3"

    def available(self) -> bool:
        wrapper = _repo_root() / "vendor" / "L3TC" / "wrapper.py"
        return wrapper.exists()

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        wrapper = _repo_root() / "vendor" / "L3TC" / "wrapper.py"
        venv_python = _repo_root() / "vendor" / "L3TC" / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else "python3"
        return [
            python,
            str(wrapper),
            "compress",
            "--variant",
            self.variant,
            "--batch-size",
            str(self.batch_size),
            "--device",
            self.device,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        wrapper = _repo_root() / "vendor" / "L3TC" / "wrapper.py"
        venv_python = _repo_root() / "vendor" / "L3TC" / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else "python3"
        return [
            python,
            str(wrapper),
            "decompress",
            "--variant",
            self.variant,
            "--batch-size",
            str(self.batch_size),
            "--device",
            self.device,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]


# -------- Factory -------- #


def default_classical() -> list[Compressor]:
    """Return the classical compressor suite we benchmark by default.

    We include multiple preset levels for each because the ratio/speed
    tradeoff matters — comparing L3TC only to gzip -9 would hide the
    fact that zstd -19 is a much more reasonable competitor.
    """
    return [
        Gzip(level=6),  # default
        Gzip(level=9),  # max
        Bzip2(level=9),  # max
        Xz(level=6, extreme=False),  # default
        Xz(level=9, extreme=True),  # max
        Zstd(level=3),  # default
        Zstd(level=19),  # max standard
        Zstd(level=22),  # ultra max
    ]


def default_neural() -> list[Compressor]:
    """Return the neural compressor suite. Availability depends on setup.sh."""
    return [
        L3TC(variant="3.2M", batch_size=1, device="cpu"),
        L3TC(variant="3.2M", batch_size=128, device="cpu"),
        L3tcRust(),
    ]


class L3tcRust(Compressor):
    """l3tc-rust: Phase 2 Rust port of L3TC-200K.

    Shells out to the release binary at
    `l3tc-rust/target/release/l3tc`. Built from
    `l3tc-rust/src/bin/l3tc.rs`. Requires the converted 200K
    checkpoint at `l3tc-rust/checkpoints/l3tc_200k.bin` (produced
    by `l3tc-rust/scripts/convert_checkpoint.py`) and the SPM
    tokenizer at `vendor/L3TC/dictionary/.../*.model`.
    """

    name = "l3tc-rust-200k"
    extension = "l3tc"
    expected_available = True

    @property
    def binary(self) -> str:
        return "l3tc-rust-200k"  # placeholder — we check via absolute path

    def _rust_bin(self) -> Path:
        return _repo_root() / "l3tc-rust" / "target" / "release" / "l3tc"

    def available(self) -> bool:
        return self._rust_bin().exists()

    def compress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        bin_path = self._rust_bin()
        return [
            str(bin_path),
            "compress",
            str(input_path),
            "-o",
            str(output_path),
            "--model",
            str(_repo_root() / "l3tc-rust" / "checkpoints" / "l3tc_200k.bin"),
            "--tokenizer",
            str(
                _repo_root()
                / "vendor"
                / "L3TC"
                / "dictionary"
                / "vocab_enwik8_bpe_16384_0.999"
                / "spm_enwik8_bpe_16384_0.999.model"
            ),
        ]

    def decompress_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        bin_path = self._rust_bin()
        return [
            str(bin_path),
            "decompress",
            str(input_path),
            "-o",
            str(output_path),
            "--model",
            str(_repo_root() / "l3tc-rust" / "checkpoints" / "l3tc_200k.bin"),
            "--tokenizer",
            str(
                _repo_root()
                / "vendor"
                / "L3TC"
                / "dictionary"
                / "vocab_enwik8_bpe_16384_0.999"
                / "spm_enwik8_bpe_16384_0.999.model"
            ),
        ]


def all_compressors() -> list[Compressor]:
    return default_classical() + default_neural()


# -------- Helpers -------- #


def shell_quote(path: Path | str) -> str:
    """Single-quote a path for shell use, escaping any embedded quotes.

    We use sh -c to handle stdout redirection in the compress commands.
    This keeps the Python code simple but means we have to quote paths
    ourselves.
    """
    s = str(path)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _repo_root() -> Path:
    """Absolute path to the l3tc-prod repository root."""
    # bench/compressors.py -> bench/ -> repo root
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    # Quick sanity check: list available compressors.
    print("Classical compressors:")
    for c in default_classical():
        avail = "✓" if c.available() else "✗"
        print(f"  [{avail}] {c.name}")
    print()
    print("Neural compressors:")
    for c in default_neural():
        avail = "✓" if c.available() else "✗"
        print(f"  [{avail}] {c.name}")
