"""
Input chunking + dispatcher: splits input into fixed-size chunks,
compresses each, picks the winner (neural vs classical), assembles blob.
"""

import zlib
import struct
import logging
from typing import Callable

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(__import__("os").environ.get("KRUNCH_CHUNK_SIZE", 65536))  # 64 KB

# Chunk entry in blob: codec_tag(1) + compressed_len(4) + data
CODEC_NEURAL = 0x01
CODEC_ZSTD = 0x02
CODEC_BZIP3 = 0x03


def _zstd_compress(data: bytes) -> bytes:
    import zstandard as zstd
    ctx = zstd.ZstdCompressor(level=3)
    return ctx.compress(data)


def _zstd_decompress(data: bytes) -> bytes:
    import zstandard as zstd
    ctx = zstd.ZstdDecompressor()
    return ctx.decompress(data)


def _bzip3_compress(data: bytes) -> bytes:
    try:
        import bzip3
        return bzip3.compress(data)
    except ImportError:
        import bz2
        return bz2.compress(data)


def _bzip3_decompress(data: bytes) -> bytes:
    try:
        import bzip3
        return bzip3.decompress(data)
    except ImportError:
        import bz2
        return bz2.decompress(data)


def compress_all(raw: bytes,
                 neural_fn: Callable[[bytes], bytes]) -> tuple[list[bytes], int]:
    """
    Compress raw bytes chunk by chunk.
    Returns (chunk_entries, n_chunks) where each entry is:
      codec_tag(1) + original_chunk_len(4) + compressed_len(4) + compressed_data
    """
    entries = []
    for i in range(0, len(raw), CHUNK_SIZE):
        chunk = raw[i:i + CHUNK_SIZE]
        entries.append(_compress_chunk(chunk, neural_fn))
    return entries, len(entries)


def _compress_chunk(chunk: bytes,
                    neural_fn: Callable[[bytes], bytes]) -> bytes:
    candidates = {}

    # Neural
    try:
        candidates[CODEC_NEURAL] = neural_fn(chunk)
    except Exception as e:
        logger.debug("Neural compress failed for chunk: %s", e)

    # zstd fallback
    try:
        candidates[CODEC_ZSTD] = _zstd_compress(chunk)
    except Exception as e:
        logger.debug("zstd compress failed: %s", e)

    if not candidates:
        raise RuntimeError("all codecs failed for chunk")

    # Pick shortest output
    tag, compressed = min(candidates.items(), key=lambda kv: len(kv[1]))
    orig_len = len(chunk)
    comp_len = len(compressed)
    logger.debug("chunk %d bytes → %s %d bytes (ratio %.3f)",
                 orig_len, _tag_name(tag), comp_len, comp_len / orig_len)
    return struct.pack(">BII", tag, orig_len, comp_len) + compressed


def decompress_all(entries_bytes: bytes, n_chunks: int,
                   neural_fn: Callable[[bytes], bytes]) -> bytes:
    """Decompress all chunks from the packed entries bytes."""
    pos = 0
    parts = []
    for _ in range(n_chunks):
        tag, orig_len, comp_len = struct.unpack(">BII", entries_bytes[pos:pos + 9])
        pos += 9
        compressed = entries_bytes[pos:pos + comp_len]
        pos += comp_len
        parts.append(_decompress_chunk(tag, compressed, orig_len, neural_fn))
    return b"".join(parts)


def _decompress_chunk(tag: int, data: bytes, orig_len: int,
                      neural_fn: Callable[[bytes], bytes]) -> bytes:
    if tag == CODEC_NEURAL:
        return neural_fn(data)[:orig_len]
    elif tag == CODEC_ZSTD:
        return _zstd_decompress(data)[:orig_len]
    elif tag == CODEC_BZIP3:
        return _bzip3_decompress(data)[:orig_len]
    else:
        raise ValueError(f"unknown codec tag: {tag:#x}")


def _tag_name(tag: int) -> str:
    return {CODEC_NEURAL: "neural", CODEC_ZSTD: "zstd",
            CODEC_BZIP3: "bzip3"}.get(tag, f"unknown({tag:#x})")
