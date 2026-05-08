"""
Single-shot CLI: compress / decompress stdin → stdout (or --in/--out files).

Container starts, loads model, processes one input, exits.

Usage:
  krunch compress   [--in PATH] [--out PATH]
  krunch decompress [--in PATH] [--out PATH]

If --in is omitted, reads from stdin. If --out is omitted, writes to stdout.
"""

import os
import sys
import zlib
import argparse
import contextlib
import logging

from .inference import engine, encode_header, decode_header, HEADER_SIZE
from .chunking import compress_all, decompress_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect FD 1 → FD 2 for the duration of the block.

    Needed when loading the model: the first runtime invocation triggers
    torch.utils.cpp_extension.load() which compiles the WKV CUDA kernel via
    ninja. ninja writes its progress (`[1/4] c++ -MMD ...`) to stdout, which
    would pollute our binary blob output. Reassigning sys.stdout in Python
    doesn't affect subprocesses — this dups the file descriptor at the OS
    level so ninja inherits a stderr-pointing FD 1.
    """
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def cmd_compress(args):
    raw = _read_input(args.input)
    with _stdout_to_stderr():
        engine.load()
    # Multi-chunk encode MUST go through compress_chunks_batched
    # (forward_stepped_batched, M=1 lockstep) to stay symmetric with
    # the decoder, which uses decompress_chunks_batched. The
    # per-chunk compress_chunk path uses forward_packed_window at
    # M=1024, and the W8A8 layer-step kernel produces 1-LSB-different
    # fp16 output at M=1024 vs M=1; softmax+CDF rounds those to
    # different CDFs and AC roundtrip breaks (CRC32 mismatch on
    # A10G 10MB / 161 chunks). Single-chunk inputs keep the packed
    # path — compress_chunk is symmetric with decompress_chunk
    # (forward_stepped, B=1) for AC purposes when there's no batched
    # decoder lockstep.
    chunk_entries, n_chunks = compress_all(
        raw, engine.compress_chunk,
        neural_batch_fn=engine.compress_chunks_batched,
    )
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    blob = encode_header(len(raw), n_chunks, crc) + b"".join(chunk_entries)
    _write_output(args.output, blob)
    ratio = len(blob) / len(raw) if raw else 0
    logger.info("compress %d → %d bytes (ratio=%.4f, %d chunks)",
                len(raw), len(blob), ratio, n_chunks)


def cmd_decompress(args):
    blob = _read_input(args.input)
    with _stdout_to_stderr():
        engine.load()
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]
    if hdr["n_chunks"] > 1:
        # Cross-chunk batched stepped forward saturates the GPU by
        # processing B chunks in parallel through one batched forward
        # per timestep. Bit-exact roundtrip.
        raw = decompress_all(
            entries_bytes, hdr["n_chunks"], engine.decompress_chunk,
            neural_batch_fn=engine.decompress_chunks_batched,
        )
    else:
        raw = decompress_all(
            entries_bytes, hdr["n_chunks"], engine.decompress_chunk,
        )

    # CRC check — skipped if header flags say "no CRC" (assembled-from-parts blobs)
    if not hdr["flags"] & 0x01:
        actual = zlib.crc32(raw) & 0xFFFFFFFF
        if actual != hdr["crc32"]:
            raise SystemExit(
                f"CRC32 mismatch: got {actual:#010x}, expected {hdr['crc32']:#010x}"
            )
    if len(raw) != hdr["original_len"]:
        raise SystemExit(
            f"length mismatch: got {len(raw)}, expected {hdr['original_len']}"
        )

    _write_output(args.output, raw)
    logger.info("decompress %d → %d bytes", len(blob), len(raw))


def _read_input(path: str | None) -> bytes:
    if path and path != "-":
        with open(path, "rb") as f:
            return f.read()
    return sys.stdin.buffer.read()


def _write_output(path: str | None, data: bytes) -> None:
    if path and path != "-":
        with open(path, "wb") as f:
            f.write(data)
    else:
        sys.stdout.buffer.write(data)


def main():
    ap = argparse.ArgumentParser(prog="krunch",
                                 description="Krunch neural compression")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="Compress bytes")
    pc.add_argument("--in", dest="input", help="Input file (default: stdin)")
    pc.add_argument("--out", dest="output", help="Output file (default: stdout)")

    pd = sub.add_parser("decompress", help="Decompress a .krunch blob")
    pd.add_argument("--in", dest="input", help="Input file (default: stdin)")
    pd.add_argument("--out", dest="output", help="Output file (default: stdout)")

    args = ap.parse_args()
    if args.cmd == "compress":
        cmd_compress(args)
    elif args.cmd == "decompress":
        cmd_decompress(args)


if __name__ == "__main__":
    main()
