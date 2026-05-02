"""
Worker job runner — entry point for distributed compress/decompress.

Container starts in worker mode by setting `KRUNCH_MODE`. The runner
reads the env-var contract that every `krunch plan` template
populates, computes its byte range from `KRUNCH_PART_INDEX` /
`KRUNCH_PART_COUNT`, performs compress or decompress on its slice,
and writes the partial result to `<KRUNCH_OUTPUT_URL>.parts/<index>`.
A subsequent `finalize` invocation stitches all parts into the final
output.

The same contract works on every batch system because it doesn't
depend on the orchestrator-specific env vars (AWS_BATCH_JOB_ARRAY_INDEX,
JOB_COMPLETION_INDEX in k8s, MODAL_TASK_ID, etc). `krunch plan` per
target maps the orchestrator var → KRUNCH_PART_INDEX in the launch
artifact.

Env-var contract (read by the runner):

  KRUNCH_MODE          compress | decompress | finalize
  KRUNCH_INPUT_URL     URL of the source object (s3://, http://, file://)
  KRUNCH_OUTPUT_URL    URL of the final output. Parts written to
                        <KRUNCH_OUTPUT_URL>.parts/<index>
  KRUNCH_PART_INDEX    0-based index of this worker (compress + decompress)
  KRUNCH_PART_COUNT    Total worker count (compress + decompress)
  KRUNCH_INPUT_LEN     Total byte size of input (compress only — workers
                        compute their byte range from this; finalize
                        passes through to the master header)

The orchestrator template is responsible for setting KRUNCH_PART_INDEX
correctly per task. Common mappings:

  AWS Batch:    KRUNCH_PART_INDEX=$AWS_BATCH_JOB_ARRAY_INDEX
  k8s Indexed:  KRUNCH_PART_INDEX=$JOB_COMPLETION_INDEX
  Slurm array:  KRUNCH_PART_INDEX=$SLURM_ARRAY_TASK_ID
  Modal:        KRUNCH_PART_INDEX=$MODAL_TASK_ID  (or pass explicitly)
  Ray:          set explicitly per task
  Local:        bash loop sets KRUNCH_PART_INDEX=$i
"""

import os
import sys
import zlib
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    mode = os.environ.get("KRUNCH_MODE", "").lower()
    if mode == "compress":
        _run_compress_worker()
    elif mode == "decompress":
        _run_decompress_worker()
    elif mode == "finalize":
        _run_finalize()
    else:
        logger.error("KRUNCH_MODE must be 'compress'|'decompress'|'finalize', "
                     "got %r", mode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Compress worker
# ---------------------------------------------------------------------------

def _run_compress_worker():
    from . import url_io
    from .chunking import compress_all
    from .inference import encode_header, engine

    src = os.environ["KRUNCH_INPUT_URL"]
    dst = os.environ["KRUNCH_OUTPUT_URL"]
    total_size = int(os.environ["KRUNCH_INPUT_LEN"])
    part_count = int(os.environ["KRUNCH_PART_COUNT"])
    part_index = int(os.environ["KRUNCH_PART_INDEX"])
    parts_prefix = _parts_prefix(dst)
    part_url = f"{parts_prefix}/part-{part_index:06d}"

    start, end = _byte_range(part_index, part_count, total_size)
    logger.info("compress part %d/%d: bytes [%d, %d) → %s",
                part_index, part_count, start, end, part_url)

    raw = url_io.read_range(src, start, end)
    logger.info("read %d bytes", len(raw))

    engine.load()
    # Pass total_size so all workers pick the SAME chunk_size as
    # _byte_range did (deterministic from KRUNCH_INPUT_LEN). Without this,
    # a worker on a small slice would derive a smaller chunk_size from
    # its own len(raw) and emit chunks that don't line up with the
    # byte-range alignment chosen above.
    chunk_entries, n_chunks = compress_all(
        raw, engine.compress_chunk, total_size=total_size,
    )
    entries_bytes = b"".join(chunk_entries)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    blob = encode_header(len(raw), n_chunks, crc) + entries_bytes

    url_io.write(part_url, blob)
    ratio = len(blob) / len(raw) if raw else 0.0
    logger.info("wrote %d bytes (ratio=%.3f, %d chunks) to %s",
                len(blob), ratio, n_chunks, part_url)


# ---------------------------------------------------------------------------
# Decompress worker
# ---------------------------------------------------------------------------

def _run_decompress_worker():
    """Decompress a chunk-index range from the input .krunch blob.

    Each worker handles roughly `n_chunks / part_count` chunks. Reads
    only its own chunk-index range from the input (no need to download
    the whole file), decompresses to raw bytes, and writes a partial
    raw output to the parts prefix. Finalize concatenates them.
    """
    from . import url_io
    from .chunking import decompress_all
    from .inference import decode_header, HEADER_SIZE, engine
    import struct

    src = os.environ["KRUNCH_INPUT_URL"]
    dst = os.environ["KRUNCH_OUTPUT_URL"]
    part_count = int(os.environ["KRUNCH_PART_COUNT"])
    part_index = int(os.environ["KRUNCH_PART_INDEX"])
    parts_prefix = _parts_prefix(dst)
    part_url = f"{parts_prefix}/part-{part_index:06d}"

    # Read only the header first, then compute which chunk-byte range
    # this worker owns. We can't seek inside the chunk-stream without
    # reading the variable-length entries, so for v1 we read the whole
    # blob — small overhead given decompress is dominated by the model
    # forward, not download. v2 will add a chunk-index sidecar so
    # workers can seek directly to their byte range.
    blob = url_io.read_all(src)
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]

    # Slice into per-chunk (orig_len, encoded) tuples so we can pick
    # this worker's range without repeating the parse.
    pos = 0
    chunks: list[tuple[int, bytes]] = []
    for _ in range(hdr["n_chunks"]):
        orig_len, comp_len = struct.unpack(">II", entries_bytes[pos:pos + 8])
        pos += 8
        chunks.append((orig_len, entries_bytes[pos:pos + comp_len]))
        pos += comp_len

    n_chunks = len(chunks)
    chunks_per = (n_chunks + part_count - 1) // part_count
    lo = part_index * chunks_per
    hi = min(lo + chunks_per, n_chunks)
    my_chunks = chunks[lo:hi]
    logger.info("decompress part %d/%d: chunks [%d, %d) of %d → %s",
                part_index, part_count, lo, hi, n_chunks, part_url)

    engine.load()
    # Re-encode my chunk slice as a self-contained entries blob so we
    # can reuse decompress_all.
    my_entries = b"".join(struct.pack(">II", ol, len(enc)) + enc
                           for ol, enc in my_chunks)
    raw = decompress_all(my_entries, len(my_chunks),
                         engine.decompress_chunk,
                         neural_batch_fn=engine.decompress_chunks_batched)
    url_io.write(part_url, raw)
    logger.info("wrote %d raw bytes (%d chunks) to %s",
                len(raw), len(my_chunks), part_url)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def _run_finalize():
    """Stitch partial blobs/raw into the single final output, then clean up
    parts. Compress finalize → produces a .krunch master blob (with
    flag=0x01 indicating no full-file CRC). Decompress finalize →
    produces the raw output bytes (concatenation in part-index order).
    """
    from . import url_io
    from .inference import encode_header, decode_header, HEADER_SIZE

    submode = os.environ.get("KRUNCH_FINALIZE_OF", "compress").lower()
    dst = os.environ["KRUNCH_OUTPUT_URL"]
    part_count = int(os.environ["KRUNCH_PART_COUNT"])
    parts_prefix = _parts_prefix(dst)

    if submode == "compress":
        original_len = int(os.environ["KRUNCH_INPUT_LEN"])
        all_entries = []
        total_chunks = 0
        for i in range(part_count):
            blob = url_io.read_all(f"{parts_prefix}/part-{i:06d}")
            hdr = decode_header(blob)
            all_entries.append(blob[HEADER_SIZE:])
            total_chunks += hdr["n_chunks"]
            logger.info("  part %d: %d chunks, %d bytes",
                        i, hdr["n_chunks"], len(blob))
        master = (encode_header(original_len, total_chunks, crc32=0, flags=0x01)
                  + b"".join(all_entries))
        url_io.write(dst, master)
        logger.info("assembled %d bytes (%d chunks) → %s",
                    len(master), total_chunks, dst)
    elif submode == "decompress":
        # Concatenate raw parts in index order
        out = bytearray()
        for i in range(part_count):
            blob = url_io.read_all(f"{parts_prefix}/part-{i:06d}")
            out.extend(blob)
            logger.info("  part %d: %d bytes", i, len(blob))
        url_io.write(dst, bytes(out))
        logger.info("assembled %d raw bytes → %s", len(out), dst)
    else:
        logger.error("KRUNCH_FINALIZE_OF must be 'compress'|'decompress', "
                     "got %r", submode)
        sys.exit(1)

    # Clean up parts.
    for i in range(part_count):
        try:
            url_io.delete(f"{parts_prefix}/part-{i:06d}")
        except Exception as e:  # noqa: BLE001
            logger.warning("part cleanup failed: %s", e)
    logger.info("cleaned up %d parts", part_count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parts_prefix(output_url: str) -> str:
    """parts/<i> live alongside the final output URL."""
    return output_url.rstrip("/") + ".parts"


def _byte_range(part_index: int, part_count: int,
                total_size: int) -> tuple[int, int]:
    """Compute [start, end) for this part, aligned to the dynamic chunk
    size derived from total_size. All N workers see the same total_size
    (KRUNCH_INPUT_LEN) and so pick the same chunk_size — byte ranges
    align across workers without coordination."""
    from .chunking import compute_chunk_size
    chunk_size = compute_chunk_size(total_size)
    per_part = (total_size // part_count // chunk_size) * chunk_size
    per_part = max(per_part, chunk_size)
    start = part_index * per_part
    end = start + per_part if part_index < part_count - 1 else total_size
    return start, min(end, total_size)
