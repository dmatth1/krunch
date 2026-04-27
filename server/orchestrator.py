"""
Krunch orchestrator — coordinates compression across a pool of GPU workers.

Accepts the same /compress and /decompress API as a single worker, but fans
out across N workers in parallel. Handles both direct bytes (small files) and
URL-reference mode (large files, no memory ceiling).

Workers are passed via KRUNCH_WORKERS env var:
  KRUNCH_WORKERS=http://10.0.1.1:8080,http://10.0.1.2:8080

URL mode: POST /compress with JSON body {"source": "<url>", "dest": "<url>"}
          POST /decompress with JSON body {"source": "<url>", "dest": "<url>"}

Direct mode: POST /compress with raw bytes (Content-Type: application/octet-stream)
             POST /decompress with raw bytes
"""

import os
import zlib
import struct
import logging
import asyncio
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import Response as FastAPIResponse

from . import url_io
from .chunking import CHUNK_SIZE
from .inference import encode_header, decode_header, HEADER_SIZE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

def _load_workers() -> list[str]:
    raw = os.environ.get("KRUNCH_WORKERS", "")
    workers = [w.strip().rstrip("/") for w in raw.split(",") if w.strip()]
    if not workers:
        raise RuntimeError(
            "KRUNCH_WORKERS env var is required (comma-separated worker URLs)"
        )
    return workers


WORKERS: list[str] = []

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_metrics = {
    "compress_requests_total": 0,
    "decompress_requests_total": 0,
    "compress_bytes_in_total": 0,
    "compress_bytes_out_total": 0,
    "compress_errors_total": 0,
    "decompress_errors_total": 0,
}
_metrics_lock = threading.Lock()


def _inc(key: str, val: int = 1):
    with _metrics_lock:
        _metrics[key] += val


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global WORKERS
    WORKERS = _load_workers()
    logger.info("Worker pool: %s", WORKERS)
    # Health-check all workers at startup
    async with httpx.AsyncClient(timeout=10) as client:
        results = await asyncio.gather(
            *[client.get(f"{w}/healthz") for w in WORKERS],
            return_exceptions=True,
        )
    for w, r in zip(WORKERS, results):
        if isinstance(r, Exception):
            logger.warning("Worker %s not reachable at startup: %s", w, r)
        else:
            logger.info("Worker %s: %s", w, r.json())
    yield


app = FastAPI(title="Krunch Orchestrator", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not WORKERS:
        raise HTTPException(503, detail="no workers configured")
    async with httpx.AsyncClient(timeout=5) as client:
        results = await asyncio.gather(
            *[client.get(f"{w}/readyz") for w in WORKERS],
            return_exceptions=True,
        )
    ready = [w for w, r in zip(WORKERS, results)
             if not isinstance(r, Exception) and r.status_code == 200]
    if not ready:
        raise HTTPException(503, detail="no workers ready")
    return {"status": "ready", "workers_ready": len(ready), "workers_total": len(WORKERS)}


@app.get("/metrics")
def metrics():
    with _metrics_lock:
        snap = dict(_metrics)
    lines = ["# TYPE krunch_orchestrator counter"]
    for k, v in snap.items():
        lines.append(f"krunch_orchestrator_{k} {v}")
    return FastAPIResponse(content="\n".join(lines) + "\n",
                           media_type="text/plain; version=0.0.4")


@app.post("/compress")
async def compress(request: Request):
    content_type = request.headers.get("content-type", "")
    _inc("compress_requests_total")

    try:
        if "application/json" in content_type:
            payload = await request.json()
            source = payload.get("source")
            dest = payload.get("dest")
            if not source or not dest:
                raise HTTPException(400, detail="JSON body requires 'source' and 'dest'")
            result = await _compress_url(source, dest)
            return result
        else:
            body = await request.body()
            if not body:
                raise HTTPException(400, detail="empty request body")
            _inc("compress_bytes_in_total", len(body))
            blob = await _compress_bytes(body)
            _inc("compress_bytes_out_total", len(blob))
            return FastAPIResponse(content=blob,
                                   media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        _inc("compress_errors_total")
        logger.exception("compress error")
        raise HTTPException(500, detail=str(e))


@app.post("/decompress")
async def decompress(request: Request):
    content_type = request.headers.get("content-type", "")
    _inc("decompress_requests_total")

    try:
        if "application/json" in content_type:
            payload = await request.json()
            source = payload.get("source")
            dest = payload.get("dest")
            if not source or not dest:
                raise HTTPException(400, detail="JSON body requires 'source' and 'dest'")
            result = await _decompress_url(source, dest)
            return result
        else:
            body = await request.body()
            if not body:
                raise HTTPException(400, detail="empty request body")
            raw = await _decompress_bytes(body)
            return FastAPIResponse(content=raw,
                                   media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        _inc("decompress_errors_total")
        logger.exception("decompress error")
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Direct bytes fan-out (small files, in-memory)
# ---------------------------------------------------------------------------

async def _compress_bytes(raw: bytes) -> bytes:
    """Split raw bytes across workers, reassemble blob."""
    segments = _split_bytes(raw, len(WORKERS))
    async with httpx.AsyncClient(timeout=300) as client:
        responses = await asyncio.gather(*[
            client.post(f"{worker}/compress",
                        content=seg,
                        headers={"Content-Type": "application/octet-stream"})
            for worker, seg in zip(WORKERS, segments)
        ])
    for i, r in enumerate(responses):
        if r.status_code != 200:
            raise RuntimeError(f"worker {WORKERS[i]} returned {r.status_code}: {r.text[:200]}")

    return _assemble_blobs([r.content for r in responses], len(raw))


async def _decompress_bytes(blob: bytes) -> bytes:
    """Split blob chunk entries across workers, reassemble raw bytes."""
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]
    groups, orig_lens = _split_entries(entries_bytes, hdr["n_chunks"], len(WORKERS))

    async with httpx.AsyncClient(timeout=300) as client:
        responses = await asyncio.gather(*[
            client.post(f"{worker}/decompress",
                        content=_wrap_blob(group, orig_len),
                        headers={"Content-Type": "application/octet-stream"})
            for worker, group, orig_len in zip(WORKERS, groups, orig_lens)
        ])
    for i, r in enumerate(responses):
        if r.status_code != 200:
            raise RuntimeError(f"worker {WORKERS[i]} returned {r.status_code}: {r.text[:200]}")
    return b"".join(r.content for r in responses)


# ---------------------------------------------------------------------------
# URL fan-out (large files — workers read/write directly)
# ---------------------------------------------------------------------------

async def _compress_url(source: str, dest: str) -> dict:
    """
    Workers each read their byte range from source, compress, write partial
    blob to a temp URL. Orchestrator assembles and writes to dest.
    """
    file_size = await asyncio.get_event_loop().run_in_executor(
        None, url_io.size, source
    )
    logger.info("compress_url %s → %s (%d bytes, %d workers)",
                source, dest, file_size, len(WORKERS))

    ranges = _split_ranges(file_size, len(WORKERS))
    part_urls = [f"{dest}.part.{i:04d}" for i in range(len(ranges))]

    # Fan out: workers read their range, compress, write partial blob
    async with httpx.AsyncClient(timeout=600) as client:
        responses = await asyncio.gather(*[
            client.post(
                f"{worker}/compress_range",
                json={"source": source, "byte_range": list(rng), "dest": part_url},
                timeout=600,
            )
            for worker, rng, part_url in zip(WORKERS, ranges, part_urls)
        ])
    for i, r in enumerate(responses):
        if r.status_code != 200:
            raise RuntimeError(
                f"worker {WORKERS[i]} /compress_range returned {r.status_code}: {r.text[:200]}"
            )
    worker_meta = [r.json() for r in responses]

    # Read partial blobs and assemble master blob
    loop = asyncio.get_event_loop()
    partial_blobs = await asyncio.gather(*[
        loop.run_in_executor(None, url_io.read_all, pu) for pu in part_urls
    ])
    total_orig = sum(m["original_len"] for m in worker_meta)
    master_blob = _assemble_blobs(list(partial_blobs), total_orig)

    await loop.run_in_executor(None, url_io.write, dest, master_blob)

    # Clean up parts
    for pu in part_urls:
        await loop.run_in_executor(None, url_io.delete, pu)

    total_chunks = sum(m["n_chunks"] for m in worker_meta)
    ratio = len(master_blob) / total_orig if total_orig else 0
    logger.info("compress_url done: ratio=%.3f n_chunks=%d", ratio, total_chunks)
    return {"dest": dest, "original_bytes": total_orig,
            "compressed_bytes": len(master_blob),
            "ratio": round(ratio, 4), "n_chunks": total_chunks}


async def _decompress_url(source: str, dest: str) -> dict:
    """
    Orchestrator reads blob header + entries from source, splits entries
    across workers. Workers decompress their share and write parts to dest.
    Orchestrator concatenates parts to dest.
    """
    loop = asyncio.get_event_loop()
    blob = await loop.run_in_executor(None, url_io.read_all, source)
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]

    groups, orig_lens = _split_entries(entries_bytes, hdr["n_chunks"], len(WORKERS))
    part_urls = [f"{dest}.part.{i:04d}" for i in range(len(groups))]

    async with httpx.AsyncClient(timeout=600) as client:
        responses = await asyncio.gather(*[
            client.post(
                f"{worker}/decompress_range",
                json={"blob_b64": _b64(group_blob), "dest": part_url},
                timeout=600,
            )
            for worker, group_blob, part_url in zip(
                WORKERS,
                [_wrap_blob(g, ol) for g, ol in zip(groups, orig_lens)],
                part_urls,
            )
        ])
    for i, r in enumerate(responses):
        if r.status_code != 200:
            raise RuntimeError(
                f"worker {WORKERS[i]} /decompress_range returned {r.status_code}: {r.text[:200]}"
            )

    # Reassemble decompressed parts in order
    parts = await asyncio.gather(*[
        loop.run_in_executor(None, url_io.read_all, pu) for pu in part_urls
    ])
    raw = b"".join(parts)
    await loop.run_in_executor(None, url_io.write, dest, raw)

    for pu in part_urls:
        await loop.run_in_executor(None, url_io.delete, pu)

    return {"dest": dest, "decompressed_bytes": len(raw)}


# ---------------------------------------------------------------------------
# Blob assembly helpers
# ---------------------------------------------------------------------------

def _assemble_blobs(partial_blobs: list[bytes], total_orig_len: int) -> bytes:
    """Strip headers from partial blobs, write one master header + all entries."""
    all_entries = []
    total_chunks = 0
    for blob in partial_blobs:
        hdr = decode_header(blob)
        all_entries.append(blob[HEADER_SIZE:])
        total_chunks += hdr["n_chunks"]

    entries_bytes = b"".join(all_entries)
    crc = zlib.crc32(b"")  # placeholder — full-data CRC not available in URL mode
    header = encode_header(total_orig_len, total_chunks, crc)
    return header + entries_bytes


def _wrap_blob(entries_bytes: bytes, orig_len: int) -> bytes:
    """Wrap a subset of chunk entries with a synthetic valid blob header."""
    # Count chunks: each entry starts with codec_tag(1) + orig_len(4) + comp_len(4)
    n = 0
    pos = 0
    while pos < len(entries_bytes):
        _, _, comp_len = struct.unpack(">BII", entries_bytes[pos:pos + 9])
        pos += 9 + comp_len
        n += 1
    crc = 0
    header = encode_header(orig_len, n, crc)
    return header + entries_bytes


def _split_entries(entries_bytes: bytes, n_chunks: int,
                   n_workers: int) -> tuple[list[bytes], list[int]]:
    """
    Split chunk entries into n_workers groups of roughly equal chunk count.
    Returns (groups_bytes, orig_len_per_group).
    """
    chunks_per_worker = max(1, n_chunks // n_workers)
    groups = []
    orig_lens = []
    pos = 0
    remaining = n_chunks
    for i in range(n_workers):
        count = chunks_per_worker if i < n_workers - 1 else remaining
        if count <= 0:
            break
        group_start = pos
        group_orig = 0
        for _ in range(count):
            tag, orig_len, comp_len = struct.unpack(">BII", entries_bytes[pos:pos + 9])
            group_orig += orig_len
            pos += 9 + comp_len
        groups.append(entries_bytes[group_start:pos])
        orig_lens.append(group_orig)
        remaining -= count
    return groups, orig_lens


def _split_bytes(raw: bytes, n: int) -> list[bytes]:
    """Split raw bytes into n roughly equal segments aligned to CHUNK_SIZE."""
    total = len(raw)
    base = (total // n // CHUNK_SIZE) * CHUNK_SIZE
    base = max(base, CHUNK_SIZE)
    segments = []
    pos = 0
    for i in range(n):
        end = pos + base if i < n - 1 else total
        segments.append(raw[pos:end])
        pos = end
        if pos >= total:
            break
    return [s for s in segments if s]


def _split_ranges(total_size: int, n: int) -> list[tuple[int, int]]:
    """Split [0, total_size) into n byte ranges aligned to CHUNK_SIZE."""
    base = (total_size // n // CHUNK_SIZE) * CHUNK_SIZE
    base = max(base, CHUNK_SIZE)
    ranges = []
    pos = 0
    for i in range(n):
        end = pos + base if i < n - 1 else total_size
        if pos < total_size:
            ranges.append((pos, end))
        pos = end
        if pos >= total_size:
            break
    return ranges


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()
