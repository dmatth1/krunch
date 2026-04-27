"""
Krunch v1 — FastAPI compression worker.

Endpoints (public):
  POST /compress        — raw bytes in, .krunch blob out
  POST /decompress      — .krunch blob in, raw bytes out
  GET  /healthz         — liveness (process alive)
  GET  /readyz          — readiness (model loaded, kernel compiled)
  GET  /metrics         — Prometheus-compatible text metrics

Endpoints (orchestrator-internal):
  POST /compress_range  — read byte range from URL, compress, write blob to URL
  POST /decompress_range — decompress base64-encoded blob, write raw bytes to URL
"""

import base64
import time
import zlib
import struct
import logging
import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import Response as FastAPIResponse

from .inference import engine, encode_header, decode_header, HEADER_SIZE
from .chunking import compress_all, decompress_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_metrics = {
    "compress_requests_total": 0,
    "decompress_requests_total": 0,
    "compress_bytes_in_total": 0,
    "compress_bytes_out_total": 0,
    "decompress_bytes_in_total": 0,
    "decompress_bytes_out_total": 0,
    "compress_errors_total": 0,
    "decompress_errors_total": 0,
}
_metrics_lock = threading.Lock()


def _inc(key: str, val: int = 1):
    with _metrics_lock:
        _metrics[key] += val


# ---------------------------------------------------------------------------
# Lifespan: load model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    yield


app = FastAPI(title="Krunch", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if not engine.ready:
        raise HTTPException(503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    with _metrics_lock:
        snap = dict(_metrics)
    lines = ["# HELP krunch_* Krunch compression server metrics",
             "# TYPE krunch_compress_requests_total counter"]
    for k, v in snap.items():
        lines.append(f"krunch_{k} {v}")
    return FastAPIResponse(content="\n".join(lines) + "\n",
                           media_type="text/plain; version=0.0.4")


@app.post("/compress")
async def compress(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, detail="empty request body")

    _inc("compress_requests_total")
    _inc("compress_bytes_in_total", len(body))

    try:
        t0 = time.perf_counter()
        chunk_entries, n_chunks = compress_all(body, engine.compress_chunk)
        entries_bytes = b"".join(chunk_entries)
        crc = zlib.crc32(body) & 0xFFFFFFFF
        header = encode_header(len(body), n_chunks, crc)
        blob = header + entries_bytes
        elapsed = time.perf_counter() - t0

        _inc("compress_bytes_out_total", len(blob))
        ratio = len(blob) / len(body)
        throughput_kb = len(body) / 1024 / elapsed if elapsed > 0 else 0
        logger.info("compress %d → %d bytes (ratio=%.3f, %.0f KB/s)",
                    len(body), len(blob), ratio, throughput_kb)
        return FastAPIResponse(content=blob,
                               media_type="application/octet-stream")
    except Exception as e:
        _inc("compress_errors_total")
        logger.exception("compress error")
        raise HTTPException(500, detail=str(e))


@app.post("/decompress")
async def decompress(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, detail="empty request body")

    _inc("decompress_requests_total")
    _inc("decompress_bytes_in_total", len(body))

    try:
        t0 = time.perf_counter()
        hdr = decode_header(body)
        entries_bytes = body[HEADER_SIZE:]
        raw = decompress_all(entries_bytes, hdr["n_chunks"],
                             engine.decompress_chunk)

        # Verify integrity
        actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
        if actual_crc != hdr["crc32"]:
            raise ValueError(
                f"CRC32 mismatch: got {actual_crc:#010x}, expected {hdr['crc32']:#010x}"
            )
        if len(raw) != hdr["original_len"]:
            raise ValueError(
                f"length mismatch: got {len(raw)}, expected {hdr['original_len']}"
            )

        elapsed = time.perf_counter() - t0
        _inc("decompress_bytes_out_total", len(raw))
        throughput_kb = len(raw) / 1024 / elapsed if elapsed > 0 else 0
        logger.info("decompress %d → %d bytes (%.0f KB/s)",
                    len(body), len(raw), throughput_kb)
        return FastAPIResponse(content=raw,
                               media_type="application/octet-stream")
    except Exception as e:
        _inc("decompress_errors_total")
        logger.exception("decompress error")
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Orchestrator-internal endpoints
# ---------------------------------------------------------------------------

@app.post("/compress_range")
async def compress_range(request: Request):
    """
    Read a byte range from a URL, compress it, write blob to a dest URL.
    Called by the orchestrator for large-file fan-out.

    Body (JSON):
      source      — URL to read from (s3://, http://, file://)
      byte_range  — [start, end) inclusive of start, exclusive of end
      dest        — URL to write the compressed blob to
    """
    payload = await request.json()
    source = payload["source"]
    start, end = payload["byte_range"]
    dest = payload["dest"]

    try:
        from . import url_io
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, url_io.read_range, source, start, end)

        t0 = time.perf_counter()
        chunk_entries, n_chunks = compress_all(raw, engine.compress_chunk)
        entries_bytes = b"".join(chunk_entries)
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        header = encode_header(len(raw), n_chunks, crc)
        blob = header + entries_bytes

        await loop.run_in_executor(None, url_io.write, dest, blob)
        elapsed = time.perf_counter() - t0
        logger.info("compress_range [%d,%d) → %d bytes (%.0f KB/s)",
                    start, end, len(blob), len(raw) / 1024 / elapsed)
        return {"original_len": len(raw), "n_chunks": n_chunks,
                "compressed_len": len(blob)}
    except Exception as e:
        logger.exception("compress_range error")
        raise HTTPException(500, detail=str(e))


@app.post("/decompress_range")
async def decompress_range(request: Request):
    """
    Decompress a base64-encoded blob and write raw bytes to a dest URL.
    Called by the orchestrator for large-file fan-out.

    Body (JSON):
      blob_b64  — base64-encoded .krunch blob
      dest      — URL to write decompressed bytes to
    """
    payload = await request.json()
    blob = base64.b64decode(payload["blob_b64"])
    dest = payload["dest"]

    try:
        from . import url_io
        loop = asyncio.get_event_loop()
        hdr = decode_header(blob)
        entries_bytes = blob[HEADER_SIZE:]
        raw = decompress_all(entries_bytes, hdr["n_chunks"], engine.decompress_chunk)
        await loop.run_in_executor(None, url_io.write, dest, raw)
        logger.info("decompress_range → %d bytes written to %s", len(raw), dest)
        return {"decompressed_len": len(raw)}
    except Exception as e:
        logger.exception("decompress_range error")
        raise HTTPException(500, detail=str(e))
