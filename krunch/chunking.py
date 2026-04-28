"""
Input chunking: splits input into fixed-size chunks, neural-compresses
each, assembles blob.

Krunch is a neural compressor — every chunk goes through the RWKV path.
There is no per-chunk classical fallback: if your data isn't text-heavy
enough that the language model can compress it, krunch isn't the right
tool — use zstd instead.

Chunking still serves a purpose: parallelism (workers can compress chunks
independently) and bounded memory (each forward pass is sized to one
chunk, not the whole input).
"""

import struct
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# 1 MB matches Spike 6's measured tradeoff. Smaller chunks amplify per-chunk
# header overhead and shrink the model's effective context; larger chunks
# delay parallelism wins and risk OOM during the neural sequence-forward
# pass. Override with KRUNCH_CHUNK_SIZE for experiments.
CHUNK_SIZE = int(__import__("os").environ.get("KRUNCH_CHUNK_SIZE", 1048576))  # 1 MB

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
    """Decompress all chunks from the packed entries bytes."""
    pos = 0
    parts = []
    for _ in range(n_chunks):
        orig_len, comp_len = struct.unpack(">II", entries_bytes[pos:pos + 8])
        pos += 8
        compressed = entries_bytes[pos:pos + comp_len]
        pos += comp_len
        parts.append(neural_fn(compressed)[:orig_len])
    return b"".join(parts)
