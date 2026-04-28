"""
Single-shot CLI: compress / decompress stdin → stdout (or --in/--out files).

Container starts, loads model, processes one input, exits.

Usage:
  krunch compress   [--in PATH] [--out PATH]
  krunch decompress [--in PATH] [--out PATH]

If --in is omitted, reads from stdin. If --out is omitted, writes to stdout.
"""

import sys
import zlib
import argparse
import logging

from .inference import engine, encode_header, decode_header, HEADER_SIZE
from .chunking import compress_all, decompress_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)


def cmd_compress(args):
    raw = _read_input(args.input)
    engine.load()
    chunk_entries, n_chunks = compress_all(raw, engine.compress_chunk)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    blob = encode_header(len(raw), n_chunks, crc) + b"".join(chunk_entries)
    _write_output(args.output, blob)
    ratio = len(blob) / len(raw) if raw else 0
    logger.info("compress %d → %d bytes (ratio=%.4f, %d chunks)",
                len(raw), len(blob), ratio, n_chunks)


def cmd_decompress(args):
    blob = _read_input(args.input)
    engine.load()
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]
    raw = decompress_all(entries_bytes, hdr["n_chunks"], engine.decompress_chunk)

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
