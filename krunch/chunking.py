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
from typing import Callable

logger = logging.getLogger(__name__)

# Per-chunk peak memory during compress is dominated by the logits buffer:
# `tokens × vocab × 4 bytes`. For RWKV-4-Pile-169M that's tokens × ~200KB.
# A 1 MB chunk of English text ≈ 256K tokens → ~50 GB of logits. We auto-
# tune the chunk size at import time to fit comfortably on the host's RAM.
# Override via KRUNCH_CHUNK_SIZE for explicit control.
_VOCAB = 50277          # RWKV-4-Pile-169M
_BYTES_PER_TOKEN = 4    # English chat / code, GPT-NeoX BPE — conservative
_LOGITS_BYTES_PER_TOKEN = _VOCAB * 4
_OVERHEAD_BYTES = 4 * 1024 ** 3  # model + raw input + torch allocator + python


def _autotune_chunk_size() -> int:
    """Pick the largest chunk size that fits comfortably in available RAM.
    Capped at 4 MB (parallelism), floored at 64 KB (chunk-header overhead)."""
    avail = _available_ram_bytes()
    budget_for_logits = max(int((avail - _OVERHEAD_BYTES) * 0.6), 256 * 1024 * 1024)
    max_tokens = budget_for_logits // _LOGITS_BYTES_PER_TOKEN
    max_chunk_bytes = max_tokens * _BYTES_PER_TOKEN
    return max(64 * 1024, min(4 * 1024 * 1024, int(max_chunk_bytes)))


def _available_ram_bytes() -> int:
    """Read MemAvailable from /proc/meminfo (Linux). Fallback: 8 GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * 1024 ** 3


CHUNK_SIZE = int(os.environ.get("KRUNCH_CHUNK_SIZE", _autotune_chunk_size()))
logger.info("CHUNK_SIZE = %d bytes (autotuned from /proc/meminfo)", CHUNK_SIZE)

# Number of chunks decompressed concurrently on a single worker. Each
# concurrent stream maintains its own RNN state and AC decoder; threads
# overlap their forward calls so the GPU isn't idle between Python steps.
# Larger = more GPU utilization but more memory for state + logits.
DECOMPRESS_BATCH = int(os.environ.get("KRUNCH_DECOMPRESS_BATCH", 16))

# Per-chunk entry: orig_len(4) + comp_len(4) + neural_compressed_data


def compress_all(raw: bytes,
                 neural_fn: Callable[[bytes], bytes]) -> tuple[list[bytes], int]:
    """Compress raw bytes chunk by chunk via the neural codec."""
    entries = []
    for i in range(0, len(raw), CHUNK_SIZE):
        chunk = raw[i:i + CHUNK_SIZE]
        entries.append(_compress_chunk(chunk, neural_fn))
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
                   neural_fn: Callable[[bytes], bytes]) -> bytes:
    """
    Decompress all chunks. Up to DECOMPRESS_BATCH chunks run concurrently
    via a ThreadPoolExecutor — each thread runs its own AC decoder + RNN
    forward loop. The rwkv package's forward() releases the GIL during
    CUDA work, so threads overlap Python (AC + bookkeeping) and CUDA
    (kernel launches) latency across chunks. Output is byte-identical
    to the sequential path.
    """
    # Pre-parse all chunks (cheap — just slicing + 8 bytes per entry)
    pos = 0
    chunks: list[tuple[int, bytes]] = []  # (orig_len, encoded)
    for _ in range(n_chunks):
        orig_len, comp_len = struct.unpack(">II", entries_bytes[pos:pos + 8])
        pos += 8
        chunks.append((orig_len, entries_bytes[pos:pos + comp_len]))
        pos += comp_len

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
