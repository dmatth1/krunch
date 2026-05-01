"""
Input chunking: splits input into fixed-size chunks, neural-compresses
each, assembles blob.

Krunch is a neural compressor — every chunk goes through the RWKV path.
There is no per-chunk classical fallback: if your data isn't text-heavy
enough that the language model can compress it, krunch isn't the right
tool — use zstd instead.

Chunking serves three purposes: cross-worker parallelism (each Batch
worker takes a byte range), bounded memory per forward pass, and
cross-chunk parallelism on a single GPU during decompress (RNN decode
is inherently sequential within a chunk; running B chunks concurrently
keeps the GPU busy and overlaps Python + AC + kernel-launch latency).
"""

import os
import struct
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default 1 MB. Chunk size is no longer memory-bound (compress streams range-
# encoded bytes batch by batch, so peak memory is O(SEQ_BATCH × vocab) ≈
# 200 MB regardless of chunk size). Bigger chunks → better ratio (more model
# context, smaller cold-start fraction); smaller chunks → more cross-chunk
# parallelism for the distributed Batch path. 1 MB matches Spike 6's tradeoff.
# Override via KRUNCH_CHUNK_SIZE.
CHUNK_SIZE = int(os.environ.get("KRUNCH_CHUNK_SIZE", 1048576))  # 1 MB

# Number of chunks decompressed concurrently on a single worker. Each
# concurrent stream maintains its own RNN state and AC decoder; threads
# overlap their forward calls so the GPU isn't idle between Python steps.
# Larger = more GPU utilization but more memory for state + logits.
# Default 1 (sequential): with the GPU AC kernel landing in v1.1, each
# decode step ends in a `.item()` GPU sync that blocks the calling thread,
# and 8 threads contending for one GPU + one Python GIL ran ~1.7× SLOWER
# than sequential on T4 (32m vs 19m on a 3 MB / 3-chunk sample, 2026-04-28).
# Per-thread CUDA streams didn't fix it — bottleneck is per-token
# Python+launch overhead, not GPU SM occupancy. Set >1 only for the older
# CPU-AC code path or for a future batched-WKV decode redesign.
DECOMPRESS_BATCH = int(os.environ.get("KRUNCH_DECOMPRESS_BATCH", 1))

# Per-chunk entry: orig_len(4) + comp_len(4) + neural_compressed_data


def compress_all(raw: bytes,
                 neural_fn: Callable[[bytes], bytes],
                 neural_batch_fn: Optional[Callable[[list], list]] = None
                 ) -> tuple[list[bytes], int]:
    """Compress raw bytes chunk by chunk via the neural codec.
    If `neural_batch_fn` is supplied (the chunks-batched encode path),
    invoke it once with all chunks; otherwise per-chunk via `neural_fn`.
    """
    chunks = [raw[i:i + CHUNK_SIZE] for i in range(0, len(raw), CHUNK_SIZE)]
    if neural_batch_fn is not None and len(chunks) > 1:
        compressed = neural_batch_fn(chunks)
        entries = [
            struct.pack(">II", len(c), len(z)) + z
            for c, z in zip(chunks, compressed)
        ]
        return entries, len(entries)
    entries = [_compress_chunk(chunk, neural_fn) for chunk in chunks]
    return entries, len(entries)


def _compress_chunk(chunk: bytes,
                    neural_fn: Callable[[bytes], bytes]) -> bytes:
    compressed = neural_fn(chunk)
    orig_len = len(chunk)
    comp_len = len(compressed)
    logger.debug("chunk %d bytes → %d bytes (ratio %.3f)",
                 orig_len, comp_len, comp_len / orig_len)
    return struct.pack(">II", orig_len, comp_len) + compressed


def decompress_all(entries_bytes: bytes, n_chunks: int,
                   neural_fn: Callable[[bytes], bytes],
                   neural_batch_fn: Optional[Callable[[list], list]] = None
                   ) -> bytes:
    """
    Decompress all chunks. If `neural_batch_fn` is supplied (the
    GPU-batched chunk path), invoke it once with all chunks; otherwise
    fall back to per-chunk via `neural_fn` (sequential or, when
    `KRUNCH_DECOMPRESS_BATCH > 1`, ThreadPoolExecutor — only useful for
    the legacy CPU-AC path; see chunking.py module docstring + V1_PLAN
    for why we default sequential under GPU AC).
    """
    # Pre-parse all chunks (cheap — just slicing + 8 bytes per entry)
    pos = 0
    chunks: list[tuple[int, bytes]] = []  # (orig_len, encoded)
    for _ in range(n_chunks):
        orig_len, comp_len = struct.unpack(">II", entries_bytes[pos:pos + 8])
        pos += 8
        chunks.append((orig_len, entries_bytes[pos:pos + comp_len]))
        pos += comp_len

    # neural_batch_fn signature: list[encoded_bytes] -> list[decoded_bytes].
    # Two implementations:
    #   - DecompressWorkerPool.decompress_chunks (multi-process, recommended)
    #   - InferenceEngine.decompress_chunks_batched (single-process lockstep)
    if neural_batch_fn is not None and len(chunks) > 1:
        encs = [enc for _, enc in chunks]
        decoded_full = neural_batch_fn(encs)
        return b"".join(d[:ol] for (ol, _), d in zip(chunks, decoded_full))

    if DECOMPRESS_BATCH <= 1 or len(chunks) <= 1:
        # Sequential path — used for tiny inputs and for tests that mock
        # neural_fn (no real model = no GIL release = threading just
        # adds overhead).
        return b"".join(neural_fn(enc)[:orig_len] for orig_len, enc in chunks)

    workers = min(DECOMPRESS_BATCH, len(chunks))
    logger.debug("decompress: %d chunks across %d threads", len(chunks), workers)

    def _one(item: tuple[int, bytes]) -> bytes:
        orig_len, enc = item
        return neural_fn(enc)[:orig_len]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        decoded = list(ex.map(_one, chunks))
    return b"".join(decoded)
