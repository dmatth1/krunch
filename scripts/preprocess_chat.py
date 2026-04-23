"""Schema-aware preprocessor for chat NDJSON.

Losslessly splits `{conversation_id, turn, role, content, timestamp, model}`
NDJSON into three streams:

  * `header` — JSON manifest (field order, role/model enum tables,
    base timestamp, record count, SHA-256 of reconstructed output for
    roundtrip verification).
  * `meta`   — per-record compact binary structural fields.
  * `content` — raw `content` bytes concatenated, no delimiters.

Meta record layout (binary, per input line):

  [uuid: 16 bytes]             # conversation_id hex-decoded
  [turn: varint]
  [role_idx: u8]               # index into header.role_table
                               # (255 = literal — string follows, varint-len)
  [ts_delta_s: svarint]        # signed delta from header.base_timestamp
  [model_idx: u8]              # index into header.model_table
                               # (255 = literal — string follows, varint-len)
  [content_len: varint]        # bytes of content in content stream

Varints are little-endian base-128 (same as protobuf). Signed varints
use zigzag encoding.

Design notes:

  * UUIDs in hex form cost 32 bytes per record; hex-decoded to 16
    bytes is a 50% structural win on that field alone.
  * Role/model as u8 indices is a 1-byte win per record vs the raw
    `"role":"user",` ~15-byte JSON fragment.
  * Content is length-prefixed (via meta), not delimiter-separated,
    so no escape mechanism and no ambiguity on binary content.
  * Header carries a roundtrip SHA-256 — decoder re-serializes and
    verifies before returning.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# -----------------------------------------------------------------------------
# Varint primitives
# -----------------------------------------------------------------------------

def _vwrite(buf: bytearray, n: int) -> None:
    if n < 0:
        raise ValueError(f"varint must be non-negative, got {n}")
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n & 0x7F)


def _vread(data: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")


def _svwrite(buf: bytearray, n: int) -> None:
    # Zigzag
    _vwrite(buf, (n << 1) ^ (n >> 63))


def _svread(data: memoryview, pos: int) -> tuple[int, int]:
    n, pos = _vread(data, pos)
    # Inverse zigzag
    return (n >> 1) ^ -(n & 1), pos


# -----------------------------------------------------------------------------
# Encode
# -----------------------------------------------------------------------------

BASE_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"


def _parse_ts(s: str) -> int:
    """Parse ISO8601 timestamp (no tz) as epoch seconds (UTC assumed)."""
    dt = datetime.strptime(s, BASE_TIMESTAMP_FMT).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _format_ts(epoch_s: int) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime(BASE_TIMESTAMP_FMT)


def encode(ndjson: bytes) -> tuple[bytes, bytes, bytes]:
    """Encode NDJSON bytes → (header_bytes, meta_bytes, content_bytes)."""
    meta = bytearray()
    content = bytearray()

    role_table: list[str] = []
    model_table: list[str] = []
    role_idx: dict[str, int] = {}
    model_idx: dict[str, int] = {}
    n_records = 0
    base_ts: int | None = None

    # First pass: find base timestamp (minimum), build role/model tables in
    # order of first appearance. Not strictly necessary but gives stable
    # output and lets us use small indices.
    first_pass_records = []
    for line in ndjson.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ts = _parse_ts(r["timestamp"])
        if base_ts is None or ts < base_ts:
            base_ts = ts
        if r["role"] not in role_idx:
            role_idx[r["role"]] = len(role_table)
            role_table.append(r["role"])
        if r["model"] not in model_idx:
            model_idx[r["model"]] = len(model_table)
            model_table.append(r["model"])
        first_pass_records.append(r)
        n_records += 1
    if base_ts is None:
        base_ts = 0

    # Second pass: emit meta + content
    for r in first_pass_records:
        cid = r["conversation_id"]
        # Must be 32-char hex. If not, we'd need a literal-UUID escape;
        # fail fast for now — any real chat corpus has canonical hex UUIDs.
        try:
            uuid_bytes = bytes.fromhex(cid)
            if len(uuid_bytes) != 16:
                raise ValueError(f"conversation_id not 16 bytes: {cid!r}")
        except ValueError:
            raise ValueError(f"non-hex conversation_id: {cid!r}")
        meta.extend(uuid_bytes)

        _vwrite(meta, int(r["turn"]))

        ri = role_idx.get(r["role"], 255)
        meta.append(ri if ri < 255 else 255)
        if ri == 255:
            role_bytes = r["role"].encode("utf-8")
            _vwrite(meta, len(role_bytes))
            meta.extend(role_bytes)

        ts_delta = _parse_ts(r["timestamp"]) - base_ts
        _svwrite(meta, ts_delta)

        mi = model_idx.get(r["model"], 255)
        meta.append(mi if mi < 255 else 255)
        if mi == 255:
            model_bytes = r["model"].encode("utf-8")
            _vwrite(meta, len(model_bytes))
            meta.extend(model_bytes)

        content_bytes = r["content"].encode("utf-8")
        _vwrite(meta, len(content_bytes))
        content.extend(content_bytes)

    # Build header
    header_obj = {
        "version": 1,
        "format": "chat-v1",
        "fields": ["conversation_id", "turn", "role", "content", "timestamp", "model"],
        "role_table": role_table,
        "model_table": model_table,
        "base_timestamp": _format_ts(base_ts),
        "n_records": n_records,
    }
    # SHA-256 of the original ndjson bytes (canonical, verifies roundtrip).
    header_obj["sha256"] = hashlib.sha256(ndjson).hexdigest()

    header_bytes = json.dumps(header_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header_bytes, bytes(meta), bytes(content)


# -----------------------------------------------------------------------------
# Decode
# -----------------------------------------------------------------------------

def decode(header_bytes: bytes, meta_bytes: bytes, content_bytes: bytes,
           verify_sha: bool = True) -> bytes:
    """Decode (header, meta, content) → original NDJSON bytes."""
    header = json.loads(header_bytes.decode("utf-8"))
    if header.get("format") != "chat-v1":
        raise ValueError(f"unknown format: {header.get('format')!r}")

    role_table: list[str] = header["role_table"]
    model_table: list[str] = header["model_table"]
    base_ts = _parse_ts(header["base_timestamp"])
    n_records: int = header["n_records"]

    meta_mv = memoryview(meta_bytes)
    content_pos = 0
    out = io.BytesIO()

    pos = 0
    for _ in range(n_records):
        cid = meta_bytes[pos : pos + 16].hex()
        pos += 16

        turn, pos = _vread(meta_mv, pos)

        ri = meta_bytes[pos]
        pos += 1
        if ri == 255:
            rlen, pos = _vread(meta_mv, pos)
            role = meta_bytes[pos : pos + rlen].decode("utf-8")
            pos += rlen
        else:
            role = role_table[ri]

        ts_delta, pos = _svread(meta_mv, pos)
        timestamp = _format_ts(base_ts + ts_delta)

        mi = meta_bytes[pos]
        pos += 1
        if mi == 255:
            mlen, pos = _vread(meta_mv, pos)
            model = meta_bytes[pos : pos + mlen].decode("utf-8")
            pos += mlen
        else:
            model = model_table[mi]

        clen, pos = _vread(meta_mv, pos)
        content = content_bytes[content_pos : content_pos + clen].decode("utf-8")
        content_pos += clen

        record = {
            "conversation_id": cid,
            "turn": turn,
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "model": model,
        }
        out.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        out.write(b"\n")

    result = out.getvalue()
    if verify_sha:
        got = hashlib.sha256(result).hexdigest()
        want = header.get("sha256")
        if want and got != want:
            raise ValueError(f"roundtrip SHA mismatch: got {got}, want {want}")
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cmd_encode(args: argparse.Namespace) -> None:
    raw = Path(args.input).read_bytes()
    header, meta, content = encode(raw)
    prefix = Path(args.output_prefix)
    (prefix.parent / (prefix.name + ".header.json")).write_bytes(header)
    (prefix.parent / (prefix.name + ".meta.bin")).write_bytes(meta)
    (prefix.parent / (prefix.name + ".content.bin")).write_bytes(content)
    print(
        f"input={len(raw)}  header={len(header)}  meta={len(meta)}  "
        f"content={len(content)}  total_out={len(header)+len(meta)+len(content)}  "
        f"ratio={(len(header)+len(meta)+len(content))/len(raw):.4f}",
        file=sys.stderr,
    )


def _cmd_decode(args: argparse.Namespace) -> None:
    prefix = Path(args.input_prefix)
    header = (prefix.parent / (prefix.name + ".header.json")).read_bytes()
    meta = (prefix.parent / (prefix.name + ".meta.bin")).read_bytes()
    content = (prefix.parent / (prefix.name + ".content.bin")).read_bytes()
    out = decode(header, meta, content, verify_sha=not args.no_verify)
    Path(args.output).write_bytes(out)
    print(f"output={len(out)} bytes", file=sys.stderr)


def _cmd_measure(args: argparse.Namespace) -> None:
    """End-to-end: encode → zstd each stream → report ratios."""
    import zstandard as zstd
    raw = Path(args.input).read_bytes()
    header, meta, content = encode(raw)

    # Baseline: zstd level 22 on raw input.
    zbase = zstd.ZstdCompressor(level=args.zstd_level)
    baseline_compressed = zbase.compress(raw)
    baseline_ratio = len(baseline_compressed) / len(raw)

    # Preprocess+zstd: zstd each stream independently. Header is tiny,
    # store uncompressed.
    meta_c = zbase.compress(meta)
    content_c = zbase.compress(content)
    total_pp = len(header) + len(meta_c) + len(content_c)
    pp_ratio = total_pp / len(raw)

    # Decode round-trip check (before reporting).
    roundtripped = decode(header, meta, content, verify_sha=True)
    if roundtripped != raw:
        raise SystemExit("FATAL: roundtrip mismatch (byte-level)")

    result = {
        "input_bytes": len(raw),
        "raw_content_bytes": len(content),
        "raw_meta_bytes": len(meta),
        "raw_header_bytes": len(header),
        "zstd_level": args.zstd_level,
        "baseline_zstd_bytes": len(baseline_compressed),
        "baseline_ratio": round(baseline_ratio, 5),
        "pp_header_bytes": len(header),
        "pp_meta_zstd_bytes": len(meta_c),
        "pp_content_zstd_bytes": len(content_c),
        "pp_total_bytes": total_pp,
        "pp_ratio": round(pp_ratio, 5),
        "improvement_pct": round(100.0 * (1.0 - pp_ratio / baseline_ratio), 2),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="NDJSON → (header, meta, content) streams")
    e.add_argument("input", help="Input NDJSON path")
    e.add_argument("--output-prefix", required=True, help="Output prefix (writes .header.json, .meta.bin, .content.bin)")
    e.set_defaults(func=_cmd_encode)

    d = sub.add_parser("decode", help="(header, meta, content) → NDJSON")
    d.add_argument("--input-prefix", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--no-verify", action="store_true", help="Skip SHA-256 roundtrip check")
    d.set_defaults(func=_cmd_decode)

    m = sub.add_parser("measure", help="End-to-end zstd ratio comparison")
    m.add_argument("input", help="Input NDJSON path")
    m.add_argument("--zstd-level", type=int, default=22)
    m.set_defaults(func=_cmd_measure)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
