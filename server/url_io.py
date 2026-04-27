"""
Generic URL read/write abstraction.

Supported schemes:
  s3://bucket/key          — AWS S3 (boto3, uses instance role or env creds)
  http:// / https://       — HTTP GET (read-only; dest writes not supported)
  file:///absolute/path    — local filesystem

Workers use this to read their byte range from source and write results to dest.
The orchestrator uses it to read file size, assemble final blob, clean up parts.
"""

import io
import os
import urllib.request
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

def size(url: str) -> int:
    """Return the total byte size of the object at url."""
    scheme, rest = _split(url)
    if scheme == "s3":
        bucket, key = _s3_parts(rest)
        return _s3().head_object(Bucket=bucket, Key=key)["ContentLength"]
    elif scheme in ("http", "https"):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            cl = r.headers.get("Content-Length")
            if cl is None:
                raise ValueError(f"no Content-Length on {url}")
            return int(cl)
    elif scheme == "file":
        return Path(_file_path(rest)).stat().st_size
    else:
        raise ValueError(f"unsupported scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_range(url: str, start: int, end: int) -> bytes:
    """Read bytes [start, end) from url."""
    scheme, rest = _split(url)
    if scheme == "s3":
        bucket, key = _s3_parts(rest)
        resp = _s3().get_object(
            Bucket=bucket, Key=key,
            Range=f"bytes={start}-{end - 1}"
        )
        return resp["Body"].read()
    elif scheme in ("http", "https"):
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end - 1}"}
        )
        with urllib.request.urlopen(req) as r:
            return r.read()
    elif scheme == "file":
        with open(_file_path(rest), "rb") as f:
            f.seek(start)
            return f.read(end - start)
    else:
        raise ValueError(f"unsupported scheme: {scheme!r}")


def read_all(url: str) -> bytes:
    """Read entire object at url into memory."""
    scheme, rest = _split(url)
    if scheme == "s3":
        bucket, key = _s3_parts(rest)
        resp = _s3().get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    elif scheme in ("http", "https"):
        with urllib.request.urlopen(url) as r:
            return r.read()
    elif scheme == "file":
        return Path(_file_path(rest)).read_bytes()
    else:
        raise ValueError(f"unsupported scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write(url: str, data: bytes) -> None:
    """Write data to url. Overwrites if already exists."""
    scheme, rest = _split(url)
    if scheme == "s3":
        bucket, key = _s3_parts(rest)
        _s3().put_object(Bucket=bucket, Key=key, Body=data)
    elif scheme == "file":
        p = Path(_file_path(rest))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    else:
        raise ValueError(f"write not supported for scheme: {scheme!r}")


def delete(url: str) -> None:
    """Delete object at url (best-effort, no error if missing)."""
    scheme, rest = _split(url)
    if scheme == "s3":
        bucket, key = _s3_parts(rest)
        try:
            _s3().delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass
    elif scheme == "file":
        try:
            Path(_file_path(rest)).unlink()
        except FileNotFoundError:
            pass
    # http/https delete not supported — silently skip


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split(url: str) -> tuple[str, str]:
    if "://" not in url:
        raise ValueError(f"not a URL (missing scheme): {url!r}")
    scheme, rest = url.split("://", 1)
    return scheme.lower(), rest


def _s3_parts(rest: str) -> tuple[str, str]:
    bucket, _, key = rest.partition("/")
    if not key:
        raise ValueError(f"S3 URL missing key: s3://{rest}")
    return bucket, key


def _file_path(rest: str) -> str:
    # file:///absolute/path → /absolute/path
    return "/" + rest.lstrip("/") if rest.startswith("/") else rest


_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3")
    return _s3_client
