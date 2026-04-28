"""
Batch job runner — called when the container starts in job mode.

Two job types, selected by KRUNCH_JOB_TYPE env var:

  compress  — reads one byte range from source URL, compresses it,
              writes a partial blob to S3. Array job: each task uses
              AWS_BATCH_JOB_ARRAY_INDEX to compute its own byte range.

  assemble  — reads all partial blobs from S3 prefix, stitches into
              one master blob, writes to dest URL. Single (non-array) job.

Environment variables:

  compress job:
    KRUNCH_JOB_TYPE=compress
    KRUNCH_SOURCE          URL to read from (s3://, http://, file://)
    KRUNCH_TOTAL_SIZE      Total byte size of source (int)
    KRUNCH_TOTAL_TASKS     Number of array tasks (= array size)
    KRUNCH_PARTS_PREFIX    S3 prefix to write partial blobs (e.g. s3://b/parts/job-abc/)
    AWS_BATCH_JOB_ARRAY_INDEX  injected by Batch

  assemble job:
    KRUNCH_JOB_TYPE=assemble
    KRUNCH_PARTS_PREFIX    Same prefix used by compress jobs
    KRUNCH_N_PARTS         Number of parts to assemble (= KRUNCH_TOTAL_TASKS)
    KRUNCH_DEST            Final output URL
    KRUNCH_ORIGINAL_LEN    Total original byte count (int)
"""

import os
import sys
import zlib
import logging
import struct

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run():
    job_type = os.environ.get("KRUNCH_JOB_TYPE", "").lower()
    if job_type == "compress":
        _run_compress()
    elif job_type == "assemble":
        _run_assemble()
    else:
        logger.error("KRUNCH_JOB_TYPE must be 'compress' or 'assemble', got %r", job_type)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Compress job
# ---------------------------------------------------------------------------

def _run_compress():
    from . import url_io
    from .chunking import compress_all, CHUNK_SIZE
    from .inference import encode_header, engine

    source = os.environ["KRUNCH_SOURCE"]
    total_size = int(os.environ["KRUNCH_TOTAL_SIZE"])
    total_tasks = int(os.environ["KRUNCH_TOTAL_TASKS"])
    parts_prefix = os.environ["KRUNCH_PARTS_PREFIX"].rstrip("/")
    task_idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))

    start, end = _byte_range(task_idx, total_tasks, total_size)
    part_url = f"{parts_prefix}/part-{task_idx:06d}"

    logger.info("compress task %d/%d: bytes [%d, %d) → %s",
                task_idx, total_tasks, start, end, part_url)

    raw = url_io.read_range(source, start, end)
    logger.info("read %d bytes from %s", len(raw), source)

    engine.load()
    chunk_entries, n_chunks = compress_all(raw, engine.compress_chunk)
    entries_bytes = b"".join(chunk_entries)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    blob = encode_header(len(raw), n_chunks, crc) + entries_bytes

    url_io.write(part_url, blob)
    ratio = len(blob) / len(raw) if raw else 0
    logger.info("wrote %d bytes (ratio=%.3f, %d chunks) to %s",
                len(blob), ratio, n_chunks, part_url)


# ---------------------------------------------------------------------------
# Assemble job
# ---------------------------------------------------------------------------

def _run_assemble():
    from . import url_io
    from .inference import encode_header, decode_header, HEADER_SIZE

    parts_prefix = os.environ["KRUNCH_PARTS_PREFIX"].rstrip("/")
    n_parts = int(os.environ["KRUNCH_N_PARTS"])
    dest = os.environ["KRUNCH_DEST"]
    original_len = int(os.environ["KRUNCH_ORIGINAL_LEN"])

    logger.info("assemble %d parts from %s → %s", n_parts, parts_prefix, dest)

    all_entries = []
    total_chunks = 0

    for i in range(n_parts):
        part_url = f"{parts_prefix}/part-{i:06d}"
        blob = url_io.read_all(part_url)
        hdr = decode_header(blob)
        all_entries.append(blob[HEADER_SIZE:])
        total_chunks += hdr["n_chunks"]
        logger.info("  part %d: %d chunks, %d bytes", i, hdr["n_chunks"], len(blob))

    entries_bytes = b"".join(all_entries)
    # CRC32 of the full original is unavailable here without re-reading source;
    # flag=0x01 in the header signals "no CRC" — decompress skips the check.
    master = encode_header(original_len, total_chunks, crc32=0, flags=0x01) + entries_bytes

    url_io.write(dest, master)
    ratio = len(master) / original_len if original_len else 0
    logger.info("assembled %d bytes → %s (ratio=%.3f)", len(master), dest, ratio)

    # Clean up parts
    for i in range(n_parts):
        url_io.delete(f"{parts_prefix}/part-{i:06d}")
    logger.info("cleaned up %d part files", n_parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _byte_range(task_idx: int, total_tasks: int,
                total_size: int) -> tuple[int, int]:
    """Compute [start, end) for this task, aligned to CHUNK_SIZE."""
    from .chunking import CHUNK_SIZE
    per_task = (total_size // total_tasks // CHUNK_SIZE) * CHUNK_SIZE
    per_task = max(per_task, CHUNK_SIZE)
    start = task_idx * per_task
    end = start + per_task if task_idx < total_tasks - 1 else total_size
    return start, min(end, total_size)
