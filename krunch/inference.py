"""
RWKV-4-Pile-169M inference core.

Uses BlinkDL's `rwkv` pip package (NOT HF transformers): the WKV CUDA
kernel only engages from BlinkDL's path; HF's RWKV implementation
silently falls back to a ~1000× slower Python loop in eval mode.
"""

import os
import gc
import codecs
import time
import struct
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Byte-safe input encoding (Bugs.md #3 fix)
# ---------------------------------------------------------------------------
# Up to tokenizer_id=1 the codec preprocessed input via
# `bytes.decode("utf-8", errors="replace")`, which substitutes any
# invalid UTF-8 sequence with U+FFFD. Decompress reproduced *the
# substitution*, not the original byte — lossy on HTTP logs, mixed-
# encoding CSVs, arbitrary binary.
#
# tokenizer_id=2 fixes it by escaping invalid bytes into a Private-Use
# Area codepoint range (U+E000+byte). Decode reverses. Clean UTF-8
# inputs go through the encoder unchanged (no escape triggered).
# Inputs that legitimately contain U+E000-U+E0FF codepoints are
# decoded via tokenizer_id=1's text.encode("utf-8") path; new encodes
# always write tokenizer_id=2 so the right reverse step is applied.

_PUA_BASE = 0xE000

def _krunch_pua_handler(error: UnicodeError):
    """codecs error handler — invalid byte b -> chr(U+E000 + b)."""
    if not isinstance(error, UnicodeDecodeError):
        raise error
    obj = error.object
    chars = [chr(_PUA_BASE + obj[i]) for i in range(error.start, error.end)]
    return ("".join(chars), error.end)

codecs.register_error("krunch_pua", _krunch_pua_handler)


def _bytes_to_text(data: bytes) -> str:
    """Byte-safe input preprocessing for the tokenizer. ASCII +
    valid-UTF-8 multibyte sequences pass through unchanged; invalid
    bytes become PUA codepoints in U+E080..U+E0FF. Roundtrip is
    `_text_to_bytes(_bytes_to_text(b)) == b` for any bytes."""
    return data.decode("utf-8", errors="krunch_pua")


def _text_to_bytes(text: str) -> bytes:
    """Inverse of `_bytes_to_text`. PUA codepoints in U+E000..U+E0FF
    map back to the corresponding raw byte; everything else encodes
    as UTF-8. Use this for tokenizer_id=2 blobs."""
    out = bytearray()
    for ch in text:
        cp = ord(ch)
        if _PUA_BASE <= cp < _PUA_BASE + 0x100:
            out.append(cp - _PUA_BASE)
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def _text_to_bytes_legacy(text: str) -> bytes:
    """tokenizer_id=1 reverse step — direct UTF-8 encode. PUA
    codepoints in input were already mangled at compress time
    (errors="replace" substituted them with U+FFFD); we just round-
    trip the tokenizer's output as-is."""
    return text.encode("utf-8")

# ---------------------------------------------------------------------------
# Model + tokenizer paths (baked into the Docker image)
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("KRUNCH_MODEL_DIR", "/models"))
MODEL_PATH = MODEL_DIR / "RWKV-4-Pile-169M-20220807-8023.pth"
TOKENIZER_PATH = MODEL_DIR / "20B_tokenizer.json"

# ---------------------------------------------------------------------------
# Lazy RWKV-LM import (BlinkDL's model_run.py, vendored in /opt/rwkv-lm)
# ---------------------------------------------------------------------------

def _load_rwkv():
    """Load BlinkDL's RWKV (pip install rwkv).
    The package's RWKV class accepts a strategy string ('cpu fp32', 'cuda fp16')
    and uses the custom WKV CUDA kernel automatically when 'cuda' is in the strategy.
    """
    from rwkv.model import RWKV
    return RWKV


# ---------------------------------------------------------------------------
# Blob format constants
# ---------------------------------------------------------------------------

BLOB_MAGIC = b"KRNC"
BLOB_VERSION = 1
# Model IDs for RWKV-4-Pile-169M:
#   1 = baseline (fp16, with or without W8A8 — W8A8 is a kernel-level
#                 dtype swap that produces identical bytes)
#   2 = adaptive bias head (NEXT-3, KRUNCH_ADAPTIVE_HEAD=1)
MODEL_ID = 1
MODEL_ID_ADAPTIVE = 2
SUPPORTED_MODEL_IDS = (MODEL_ID, MODEL_ID_ADAPTIVE)
# Tokenizer IDs for GPT-NeoX 20B BPE + bytes-to-text preprocessing:
#   1 = legacy: bytes.decode("utf-8", errors="replace"). Lossy on
#       non-UTF-8 input (Bugs.md #3). Read-only for backward compat.
#   2 = byte-safe PUA escape: invalid bytes → U+E000+b codepoints.
#       Lossless on arbitrary bytes. New compress writes this.
TOKENIZER_ID_LEGACY = 1
TOKENIZER_ID = 2
SUPPORTED_TOKENIZER_IDS = (TOKENIZER_ID_LEGACY, TOKENIZER_ID)


def _model_id_for_run() -> int:
    """Pick the MODEL_ID this image will write into the blob header,
    based on the active codec env settings."""
    if os.environ.get("KRUNCH_ADAPTIVE_HEAD") == "1":
        return MODEL_ID_ADAPTIVE
    return MODEL_ID

# Header: magic(4) + blob_version(1) + model_id(4) + tokenizer_id(4) +
#         adapter_id(4) + adapter_version(2) + flags(2) +
#         original_len(8) + n_chunks(4) + crc32(4) = 37 bytes
# Pinned by tests/unit/test_blob_format_versioning.py — any change here
# requires a BLOB_VERSION bump and a CHANGELOG entry.
HEADER_FMT = ">4sBIIIHHQII"  # big-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Defense-in-depth caps on header-declared sizes. A malformed or
# malicious blob can declare absurd n_chunks (u32 → up to 4B) or
# original_len (u64 → up to 2^64-1) that would OOM the decoder loop
# long before CRC32 ever catches the corruption. These caps are
# orders of magnitude above any realistic v1 workload — krunch v1's
# canonical inputs are 1 MB – 10 GB.
MAX_CHUNKS = 100_000_000        # 100M chunks (would need a 6.4 TB input at 64 KB/chunk)
MAX_ORIGINAL_LEN = 1 << 40      # 1 TiB
MAX_CHUNK_BYTES = 16 << 20      # 16 MiB per chunk (default chunk size is 64 KiB)


def encode_header(original_len: int, n_chunks: int, crc32: int,
                  adapter_id: int = 0, adapter_version: int = 0,
                  flags: int = 0, model_id: int | None = None) -> bytes:
    """Encode the blob header. ``model_id`` defaults to whatever this
    runtime currently writes (per env settings), so callers don't have
    to thread the codec choice through every call site."""
    if model_id is None:
        model_id = _model_id_for_run()
    return struct.pack(
        HEADER_FMT,
        BLOB_MAGIC, BLOB_VERSION,
        model_id, TOKENIZER_ID,
        adapter_id, adapter_version, flags,
        original_len, n_chunks, crc32
    )


class IncompatibleBlobError(ValueError):
    """Blob was produced by a krunch image with a model_id / tokenizer_id
    / blob_version this image cannot read. Bump the local image to match,
    or recompress the input with this image."""


def decode_header(data: bytes, *, strict: bool = True) -> dict:
    """Parse a blob header. With ``strict=True`` (default), raise
    ``IncompatibleBlobError`` if the blob's model_id / tokenizer_id /
    blob_version don't match what this image can read. With
    ``strict=False``, return the parsed fields without compatibility
    checks (test/inspection use).
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"blob too short: {len(data)} < {HEADER_SIZE}")
    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic, bv, mid, tid, aid, av, flags, orig_len, n_chunks, crc = fields
    if magic != BLOB_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    parsed = {
        "blob_version": bv, "model_id": mid, "tokenizer_id": tid,
        "adapter_id": aid, "adapter_version": av, "flags": flags,
        "original_len": orig_len, "n_chunks": n_chunks, "crc32": crc,
    }
    if strict:
        # Cheap bound checks BEFORE the version checks so a malicious
        # blob with absurd sizes can't tie up the decoder even on a
        # codec-incompatible image.
        if n_chunks > MAX_CHUNKS:
            raise IncompatibleBlobError(
                f"n_chunks {n_chunks} exceeds MAX_CHUNKS {MAX_CHUNKS}; "
                f"either the blob is malformed or v1 caps need raising "
                f"(see krunch/inference.py)"
            )
        if orig_len > MAX_ORIGINAL_LEN:
            raise IncompatibleBlobError(
                f"original_len {orig_len} exceeds MAX_ORIGINAL_LEN "
                f"{MAX_ORIGINAL_LEN}; same as above"
            )
        if bv != BLOB_VERSION:
            raise IncompatibleBlobError(
                f"blob_version {bv} not supported by this image "
                f"(expected {BLOB_VERSION})"
            )
        if mid not in SUPPORTED_MODEL_IDS:
            raise IncompatibleBlobError(
                f"model_id {mid} not supported by this image "
                f"(supported: {SUPPORTED_MODEL_IDS}); recompress with a "
                f"matching image"
            )
        if tid not in SUPPORTED_TOKENIZER_IDS:
            raise IncompatibleBlobError(
                f"tokenizer_id {tid} not supported by this image "
                f"(supported: {SUPPORTED_TOKENIZER_IDS})"
            )
    return parsed


# ---------------------------------------------------------------------------
# Arithmetic coder (constriction)
# ---------------------------------------------------------------------------

import constriction  # noqa: E402


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


BOS_TOKEN = 0  # initial seed for both encode and decode — must match


def _softmax_clip_normalize(logits: np.ndarray) -> np.ndarray:
    """Vectorized softmax → clip → renormalize. Works on (V,) or (N, V).
    Returns float32 probabilities suitable for constriction's batched-params
    `Categorical` model_family API. Same numerical recipe on encode + decode
    sides so the bitstream is byte-exact."""
    arr = logits.astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr - arr.max()
        np.exp(arr, out=arr)
        arr /= arr.sum()
        np.clip(arr, 1e-9, 1.0, out=arr)
        arr /= arr.sum()
    else:
        arr = arr - arr.max(axis=1, keepdims=True)
        np.exp(arr, out=arr)
        arr /= arr.sum(axis=1, keepdims=True)
        np.clip(arr, 1e-9, 1.0, out=arr)
        arr /= arr.sum(axis=1, keepdims=True)
    return arr.astype(np.float32)


def ac_encode(tokens: list[int], logits_seq: np.ndarray) -> bytes:
    """
    Range-encode tokens[0..N-1] using logits_seq[0..N-1] as distributions.
    Uses constriction's Option-3 batched-params API: one Rust call for the
    whole sequence, instead of per-token Python iterations.
    """
    assert len(tokens) == logits_seq.shape[0], \
        f"tokens ({len(tokens)}) vs logits ({logits_seq.shape[0]}) length mismatch"
    enc = constriction.stream.queue.RangeEncoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    probs = _softmax_clip_normalize(logits_seq)  # (N, V) float32
    enc.encode(np.asarray(tokens, dtype=np.int32), model_family, probs)
    return np.array(enc.get_compressed(), dtype=np.uint32).tobytes()


def ac_decode(bitstream: bytes, n_tokens: int, logits_fn) -> list[int]:
    """
    Range-decode n_tokens. logits_fn(state, last_input) -> (logits, new_state)
    is called n_tokens times — autoregressive, so per-step (each step's
    distribution depends on the previous decoded token). Uses model_family
    + per-step probs to skip Categorical re-construction overhead.
    """
    compressed = np.frombuffer(bitstream, dtype=np.uint32)
    dec = constriction.stream.queue.RangeDecoder(compressed)
    model_family = constriction.stream.model.Categorical(perfect=False)
    tokens = []
    state = None
    last_input = BOS_TOKEN
    for _ in range(n_tokens):
        logits, state = logits_fn(state, last_input)
        probs = _softmax_clip_normalize(logits).reshape(1, -1)  # (1, V)
        tok = int(dec.decode(model_family, probs)[0])
        tokens.append(tok)
        last_input = tok
    return tokens


# ---------------------------------------------------------------------------
# InferenceEngine: model + tokenizer, loaded once per process
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
        self._model = None
        self._tokenizer: Optional[Tokenizer] = None
        self._device = "cpu"  # resolved in load() after torch is imported
        self._ready = False
        self._load_start: Optional[float] = None
        # Cached compress buffers reused across _compress_chunk_cpp calls.
        # Sized to max chunk seen so far; reset in place each call (zero
        # output_buf head + reset ac_state).
        self._compress_output_buf = None
        self._compress_ac_state = None

    def load(self):
        """Load model + tokenizer. Blocks until ready."""
        import torch
        self._load_start = time.time()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
        self._tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        # Disable the tokenizer's NFC normalizer. The 20B_tokenizer was
        # trained with `"normalizer": {"type": "NFC"}` baked in, which
        # silently maps compatibility-equivalent Unicode codepoints to
        # their canonical form on encode (e.g. U+2126 OHM SIGN → U+03A9
        # GREEK CAPITAL OMEGA, 3 bytes → 2 bytes), breaking byte-exact
        # roundtrip on text with such characters. BPE byte-level
        # tokenization handles the originals fine without the pass.
        # The tokenizers package rejects `None` as a normalizer value;
        # use an empty Sequence as a no-op replacement.
        from tokenizers.normalizers import Sequence as _NormSeq
        self._tokenizer.normalizer = _NormSeq([])

        logger.info("Loading RWKV-4-Pile-169M from %s (device=%s)",
                    MODEL_PATH, self._device)
        RWKV = _load_rwkv()

        # rwkv.model.RWKV(model_path_without_ext, strategy)
        # strategy: 'cpu fp32' for CPU, 'cuda fp16' for GPU with WKV kernel
        strategy = "cuda fp16" if self._device == "cuda" else "cpu fp32"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1" if self._device == "cuda" else "0"

        # rwkv expects the path without .pth extension. verbose=False keeps
        # the layer table off stdout — important because CLI mode writes the
        # binary blob to stdout.
        model_path_no_ext = str(MODEL_PATH).removesuffix(".pth")
        self._model = RWKV(model=model_path_no_ext, strategy=strategy,
                           verbose=False)
        self._ready = True
        elapsed = time.time() - self._load_start
        logger.info("Model loaded in %.1fs", elapsed)

    @property
    def ready(self) -> bool:
        return self._ready

    def compress_chunks(self, chunks: list[bytes]) -> list[bytes]:
        """Batch-tokenize then per-chunk compress. ~10-20× faster
        tokenization than calling compress_chunk in a list comprehension
        — `tokenizer.encode_batch` parallelizes in Rust. Equivalent
        bytes per chunk to compress_chunk(c) called individually.
        """
        if len(chunks) == 0:
            return []
        if len(chunks) == 1:
            return [self.compress_chunk(chunks[0])]
        # Decode all to text once, then a single batch-encode call.
        # tokenizer.encode_batch is the Rust tokenizers library's
        # parallel path — much faster than 16× sequential encode().
        # tokenizer_id=2: byte-safe PUA escape. Clean UTF-8 inputs
        # produce identical tokens to the legacy errors="replace" path,
        # so this is a no-op for the existing corpora and only kicks in
        # on inputs with raw non-UTF-8 bytes (Bugs.md #3).
        texts = [_bytes_to_text(d) for d in chunks]
        encodings = self._tokenizer.encode_batch(texts)
        all_tokens = [e.ids for e in encodings]
        return [self._compress_chunk_with_tokens(d, t)
                for d, t in zip(chunks, all_tokens)]

    def _compress_chunk_with_tokens(self, data: bytes,
                                      tokens: list[int]) -> bytes:
        """Inner compress that takes pre-tokenized input. Used by
        compress_chunks (after batch-tokenize) and by compress_chunk
        (after single tokenize)."""
        if len(tokens) < 1:
            raise ValueError("chunk has no tokens after tokenization")
        return self._compress_chunk_cpp(data, tokens)

    def compress_chunk(self, data: bytes) -> bytes:
        """Compress a single chunk to a range-coded bitstream + mini-header.

        Probs stay on GPU, CDF is computed via torch.compile-d
        probs_to_cdf_gpu, then the custom CUDA range coder kernel encodes
        batch-by-batch. AC state persists across batches in a (4,) uint32
        GPU tensor — no prob transfer to CPU per batch.

        For multi-chunk callers, prefer `compress_chunks([…])` which
        batch-tokenizes — same per-chunk bytes, faster overall via the
        Rust tokenizer's parallel path.
        """
        text = _bytes_to_text(data)
        tokens = self._tokenizer.encode(text).ids
        return self._compress_chunk_with_tokens(data, tokens)

    def _compress_chunk_cpp(self, data: bytes, tokens: list[int]) -> bytes:
        """Bit-exact C++ orchestration path. Encoder runs all 12 layers
        packed (one shot), then per-row softmax+CDF + batched GPU AC
        encode. Output bitstream is byte-identical to what
        `_decompress_chunk_cpp` reproduces stepped."""
        import torch
        import krunch_ac_cuda
        from krunch import cpp_path

        prof = os.environ.get("KRUNCH_CPP_PROFILE") == "1"
        if prof:
            import time as _time
            torch.cuda.synchronize()
            t0 = _time.time()

        weights = cpp_path.init_weights(self._model, self._device)
        if prof:
            torch.cuda.synchronize(); t1 = _time.time()
        full_input = [BOS_TOKEN] + tokens[:-1]
        T = len(full_input)
        state = cpp_path.fresh_state(weights)
        full_input_t = torch.as_tensor(full_input, dtype=torch.long,
                                        device=self._device)
        if prof:
            torch.cuda.synchronize(); t2 = _time.time()

        # Stream forward → cdf → encode in SEQ_BATCH-sized windows so
        # peak VRAM is bounded by one window's logits + cdfs (~600 MB
        # at SEQ_BATCH=4096). State + AC state carry forward between
        # windows naturally. Bit-identical to running each stage all
        # at once (when memory allows).
        SEQ_BATCH = int(os.environ.get("KRUNCH_FORWARD_BATCH", "1024"))
        cap = max(len(data) * 2, 64 << 10)
        # Reuse engine-level buffers across compress calls. Grow
        # `output_buf` to max cap seen; ac_state is fixed-size [4].
        if self._compress_output_buf is None or self._compress_output_buf.size(0) < cap:
            self._compress_output_buf = torch.zeros(
                cap, dtype=torch.uint8, device=self._device)
            self._compress_ac_state = torch.zeros(
                4, dtype=torch.uint32, device=self._device)
        # Reset in place — zero only the head we'll touch (faster than full).
        self._compress_output_buf[:cap].zero_()
        self._compress_ac_state.zero_()
        self._compress_ac_state[1] = 0xFFFFFFFF
        output_buf = self._compress_output_buf
        ac_state = self._compress_ac_state
        symbols = torch.as_tensor(tokens, dtype=torch.int32, device=self._device).contiguous()

        # Adaptive head (KRUNCH_ADAPTIVE_HEAD=1) intercepts post-softmax,
        # pre-CDF. Encoder + decoder both maintain a per-chunk bias state
        # and apply identical adjust+update at every token. Bytes diverge
        # from the baseline codec — a different MODEL_ID is written to
        # the blob header.
        adaptive_on = os.environ.get("KRUNCH_ADAPTIVE_HEAD") == "1"
        adaptive_head = None
        if adaptive_on:
            from krunch.codec.adaptive_head_gpu import AdaptiveHeadGPU
            from krunch.codec.gpu_encode import probs_to_cdf_gpu_fp64
            n_vocab = int(self._tokenizer.get_vocab_size())
            adaptive_head = AdaptiveHeadGPU(
                vocab_size=n_vocab, batch_size=1, device=self._device,
            )

        with torch.no_grad():
            for off in range(0, T, SEQ_BATCH):
                n_w = min(SEQ_BATCH, T - off)
                logits_w = cpp_path.forward_packed_window(
                    weights, full_input_t, state, off, n_w)
                sym_w = symbols[off:off + n_w].contiguous()
                if adaptive_on:
                    # Probs in fp64 for symmetric encoder/decoder rounding.
                    probs_w = torch.softmax(
                        logits_w.to(torch.float64), dim=-1
                    )
                    # Sequential per-token: bias[t+1] depends on
                    # adjusted[t] and tokens[t]. CUDA-graph-able later;
                    # for v1 just live with the loop.
                    for t in range(n_w):
                        adj = adaptive_head.adjust(probs_w[t])  # [V]
                        cdf_one = probs_to_cdf_gpu_fp64(
                            adj.unsqueeze(0)
                        )  # [1, V+1] int32
                        krunch_ac_cuda.encode_step(
                            cdf_one, sym_w[t:t+1].contiguous(),
                            output_buf, ac_state,
                        )
                        adaptive_head.update(sym_w[t:t+1].long(), adj)
                else:
                    cdfs_w = cpp_path.softmax_cdfs_per_row(logits_w)
                    krunch_ac_cuda.encode_step(
                        cdfs_w, sym_w, output_buf, ac_state)
        if prof:
            torch.cuda.synchronize(); t3 = t4 = _time.time()
        krunch_ac_cuda.encode_finalize(output_buf, ac_state)
        torch.cuda.synchronize()
        if prof:
            t5 = _time.time()

        bit_offset = int(ac_state[3].item())
        n_bytes = (bit_offset + 7) // 8
        ac_bytes = bytes(output_buf[:n_bytes].cpu().numpy())
        if prof:
            t6 = _time.time()
            logger.info(
                "cpp_compress T=%d: weights=%.1fms state_init=%.1fms "
                "forward=%.1fms cdf=%.1fms ac=%.1fms copy=%.1fms total=%.1fms",
                T, (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000,
                (t4-t3)*1000, (t5-t4)*1000, (t6-t5)*1000, (t6-t0)*1000)
        # No empty_cache() here — buffers are reused across calls
        # (cached on engine), so the allocator doesn't accumulate
        # per-chunk allocations to flush.
        mini_header = struct.pack(">II", len(data), len(tokens))
        return mini_header + ac_bytes

    def _text_to_bytes_for(self, tokenizer_id: int):
        """Pick the reverse text→bytes function matching the blob's
        tokenizer_id. Image bundles converters for every shipped
        tokenizer_id so old blobs continue to decode."""
        if tokenizer_id == TOKENIZER_ID:           # 2 — byte-safe PUA
            return _text_to_bytes
        if tokenizer_id == TOKENIZER_ID_LEGACY:    # 1 — direct UTF-8 (lossy on non-UTF-8)
            return _text_to_bytes_legacy
        raise ValueError(
            f"unsupported tokenizer_id {tokenizer_id} "
            f"(supported: {SUPPORTED_TOKENIZER_IDS})")

    def _decompress_chunks_batched_cpp(self, encoded_chunks: list[bytes],
                                        tokenizer_id: int = TOKENIZER_ID) -> list[bytes]:
        """Bit-exact cross-chunk batched decompress.

        Decodes up to B_MAX chunks in parallel per batched stepped
        forward call (B_MAX picked per-GPU by cpp_path.compute_decompress_batch).
        If the input exceeds B_MAX, splits into B_MAX-sized groups and
        processes them sequentially. Each group's per-timestep launch
        overhead is fixed; the GPU is saturated within each group.

        Same numerics as `_decompress_chunk_cpp` per-chunk, just processed
        in lockstep — verified bit-exact in tests/unit/gpu/test_batched_stepped.py
        (3-chunk batched == 3-chunk sequential, all state diffs = 0).
        """
        import torch
        import krunch_ac_cuda
        from krunch import cpp_path

        # Pick per-GPU batch size from runtime SM count + memory; clamped
        # by n_chunks. Replaces the prior static per-GPU table — see
        # cpp_path.compute_decompress_batch.
        B_MAX = cpp_path.compute_decompress_batch(n_chunks=len(encoded_chunks))
        if len(encoded_chunks) > B_MAX:
            out: list[bytes] = []
            for i in range(0, len(encoded_chunks), B_MAX):
                out.extend(self._decompress_chunks_batched_cpp(
                    encoded_chunks[i:i + B_MAX], tokenizer_id=tokenizer_id))
            return out

        B = len(encoded_chunks)
        # Parse mini-headers
        orig_lens: list[int] = []
        n_tokens_per: list[int] = []
        bitstreams: list[bytes] = []
        for enc in encoded_chunks:
            ol, nt = struct.unpack(">II", enc[:8])
            orig_lens.append(ol)
            n_tokens_per.append(nt)
            bitstreams.append(enc[8:])
        max_T = max(n_tokens_per)

        # Concatenate bitstreams with per-stream byte offsets + 64-byte
        # tail padding per stream.
        TAIL_PAD = 64
        base_offsets: list[int] = []
        pos = 0
        for bs in bitstreams:
            base_offsets.append(pos)
            pos += len(bs) + TAIL_PAD
        cat = bytearray(pos)
        for off, bs in zip(base_offsets, bitstreams):
            cat[off:off + len(bs)] = bs

        device = self._device
        input_buf = torch.frombuffer(bytes(cat), dtype=torch.uint8).clone().to(device)
        base_byte_offsets = torch.tensor(base_offsets, dtype=torch.int32, device=device)
        ac_states = torch.zeros(B * 4, dtype=torch.uint32, device=device)
        krunch_ac_cuda.decode_init_batched(input_buf, base_byte_offsets, ac_states)

        weights = cpp_path.init_weights(self._model, self._device)
        # W8A8 mode: decoder must use same path as encoder for bit-exact
        # AC roundtrip. Force eager (graph captures aren't yet threaded
        # through the W8A8 layer step).
        w8a8_active = ("w8a8_int8" in weights)
        # CUDA-graph dispatch: three modes via KRUNCH_DECOMPRESS_GRAPH.
        #   "full"      (default if KRUNCH_OWN_WKV=1): emb → 12 layers →
        #               ln_out → head → softmax+CDF → AC decode all in
        #               one captured graph. 1 replay() per step.
        #   "per_layer": 12 graphs, one per layer; softmax+CDF + decode
        #                outside the graph. Effectively neutral.
        #   "eager"     (KRUNCH_OWN_WKV=0): no graph; for debugging.
        own_wkv = os.environ.get("KRUNCH_OWN_WKV") == "1"
        graph_mode = os.environ.get(
            "KRUNCH_DECOMPRESS_GRAPH", "full" if own_wkv else "eager")
        # W8A8 layer step has no graph capture support yet — force eager.
        if w8a8_active:
            graph_mode = "eager"
        # Adaptive head mutates per-token bias state between cdf/decode →
        # graph capture would freeze a stale bias. Force eager.
        adaptive_on = os.environ.get("KRUNCH_ADAPTIVE_HEAD") == "1"
        if adaptive_on:
            graph_mode = "eager"
        use_graph = graph_mode in ("full", "per_layer")
        # Graphs are pointer-bound; reuse same state tensors across calls
        # (in-place reset). Eager path uses fresh allocations as before.
        if use_graph:
            state = cpp_path.fresh_state_batched_cached(weights, B)
        else:
            state = cpp_path.fresh_state_batched(weights, B)
        decoded_tokens = torch.zeros((B, max_T), dtype=torch.int32, device=device)

        with torch.no_grad():
            if graph_mode == "full":
                # Whole-step graph: emb → ... → AC decode in one g.replay().
                bufs = cpp_path._get_full_step_bufs(weights, B)
                bufs["last_input"].fill_(BOS_TOKEN)
                for t in range(max_T):
                    out_syms_buf = cpp_path.forward_step_full_graphed_v3(
                        weights, ac_states, input_buf,
                        base_byte_offsets, state, B)
                    decoded_tokens[:, t] = out_syms_buf
                    # int32 → int64 cast in place; bufs['last_input'] is
                    # the graph's input buffer — must stay at same address.
                    bufs["last_input"].copy_(out_syms_buf)
            else:
                # per_layer or eager
                last_input = torch.full(
                    (B,), BOS_TOKEN, dtype=torch.long, device=device)
                out_syms = torch.empty(B, dtype=torch.int32, device=device)
                fwd = (cpp_path.forward_stepped_batched_graphed_v2
                       if graph_mode == "per_layer"
                       else cpp_path.forward_stepped_batched)
                if adaptive_on:
                    from krunch.codec.adaptive_head_gpu import AdaptiveHeadGPU
                    from krunch.codec.gpu_encode import probs_to_cdf_gpu_fp64
                    n_vocab = int(self._tokenizer.get_vocab_size())
                    head = AdaptiveHeadGPU(
                        vocab_size=n_vocab, batch_size=B, device=device,
                    )
                    for t in range(max_T):
                        logits = fwd(weights, last_input, state)
                        probs = torch.softmax(
                            logits.to(torch.float64), dim=-1
                        )  # [B, V]
                        adj = head.adjust(probs)  # [B, V] fp64
                        cdfs = probs_to_cdf_gpu_fp64(adj)  # [B, V+1] int32
                        krunch_ac_cuda.decode_step_batched(
                            cdfs, input_buf, base_byte_offsets,
                            ac_states, out_syms,
                        )
                        decoded_tokens[:, t] = out_syms
                        head.update(out_syms.long(), adj)
                        last_input = out_syms.long()
                else:
                    for t in range(max_T):
                        logits = fwd(weights, last_input, state)
                        cdfs = cpp_path.softmax_cdfs_per_row(logits)
                        krunch_ac_cuda.decode_step_batched(
                            cdfs, input_buf, base_byte_offsets,
                            ac_states, out_syms)
                        decoded_tokens[:, t] = out_syms
                        last_input = out_syms.long()

        # Single sync at the end.
        decoded_cpu = decoded_tokens.cpu().numpy()
        to_bytes = self._text_to_bytes_for(tokenizer_id)
        out: list[bytes] = []
        for i in range(B):
            toks = decoded_cpu[i, :n_tokens_per[i]].tolist()
            text = self._tokenizer.decode(toks)
            out.append(to_bytes(text)[:orig_lens[i]])
        return out

    def _decompress_chunk_cpp(self, encoded: bytes, orig_len: int,
                                n_tokens: int, bitstream: bytes,
                                tokenizer_id: int = TOKENIZER_ID) -> bytes:
        """Bit-exact C++ orchestration path. Stepped forward per token,
        per-row softmax+CDF, GPU AC decode."""
        import torch
        import krunch_ac_cuda
        from krunch import cpp_path

        weights = cpp_path.init_weights(self._model, self._device)
        state = cpp_path.fresh_state(weights)

        bs_padded = bitstream + b"\x00" * 64
        input_buf = torch.frombuffer(bytearray(bs_padded), dtype=torch.uint8).to(self._device)
        ac_state = torch.zeros(4, dtype=torch.uint32, device=self._device)
        out_sym = torch.empty(1, dtype=torch.int32, device=self._device)
        krunch_ac_cuda.decode_init(input_buf, ac_state)

        tokens: list[int] = []
        last = BOS_TOKEN
        # KRUNCH_CPP_GRAPH=1 enables CUDA-graph-captured per-layer
        # forward. First call per layer captures, subsequent calls
        # replay one graph (saves ~12× ATen launch overhead).
        use_graph = os.environ.get("KRUNCH_CPP_GRAPH", "0") == "1"
        adaptive_on = os.environ.get("KRUNCH_ADAPTIVE_HEAD") == "1"
        if adaptive_on:
            # Graph capture would freeze a stale bias state.
            use_graph = False
        # v2 = snapshot/restore-around-capture variant; v1 is broken.
        fwd = (cpp_path.forward_stepped_graphed_v2 if use_graph
               else cpp_path.forward_stepped)
        head = None
        if adaptive_on:
            from krunch.codec.adaptive_head_gpu import AdaptiveHeadGPU
            from krunch.codec.gpu_encode import probs_to_cdf_gpu_fp64
            n_vocab = int(self._tokenizer.get_vocab_size())
            head = AdaptiveHeadGPU(
                vocab_size=n_vocab, batch_size=1, device=self._device,
            )
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = fwd(weights, last, state)
                if adaptive_on:
                    probs = torch.softmax(
                        logits.view(-1).to(torch.float64), dim=-1
                    )
                    adj = head.adjust(probs)  # [V] fp64
                    cdf_row = probs_to_cdf_gpu_fp64(
                        adj.unsqueeze(0)
                    ).squeeze(0)  # [V+1] int32
                else:
                    cdf_row = cpp_path.softmax_cdf_one_row(logits)
                krunch_ac_cuda.decode_step(cdf_row, input_buf, ac_state, out_sym)
                tok = int(out_sym.item())
                tokens.append(tok)
                if adaptive_on:
                    head.update(tok, adj)
                last = tok

        text = self._tokenizer.decode(tokens)
        return self._text_to_bytes_for(tokenizer_id)(text)[:orig_len]

    def decompress_chunk(self, encoded: bytes,
                          tokenizer_id: int = TOKENIZER_ID) -> bytes:
        """Decompress a single AC-encoded chunk produced by compress_chunk.

        GPU decode path: state (low/high/value/bit_offset) lives in a
        4-uint32 GPU tensor across calls; per-step CDF stays on GPU; only
        the decoded symbol (one int) crosses to CPU each token.

        ``tokenizer_id`` selects the reverse text→bytes function — read
        from the blob header by the caller (cli.cmd_decompress). Defaults
        to the current TOKENIZER_ID so direct programmatic use without a
        blob still works.
        """
        orig_len, n_tokens = struct.unpack(">II", encoded[:8])
        bitstream = encoded[8:]
        return self._decompress_chunk_cpp(encoded, orig_len, n_tokens, bitstream,
                                          tokenizer_id=tokenizer_id)

    def compress_chunks_batched(self, chunks: list[bytes]) -> list[bytes]:
        """Compress N chunks in lockstep, B=N forward + B=N AC encode per
        timestep. Symmetric to `decompress_chunks_batched`: both sides
        call cpp_path.forward_stepped_batched (B=N, T=1) so logits are
        bit-identical → AC roundtrip holds, including under bf16 / int8
        codecs that swap kernels in cpp_path.
        Returns list of N compressed-chunk byte strings (mini-header + AC).
        """
        import torch
        import krunch_ac_cuda
        from krunch import cpp_path

        B = len(chunks)
        if B == 0:
            return []
        if B == 1:
            return [self.compress_chunk(chunks[0])]

        # Tokenize each chunk separately. Pad to common length T_max with a
        # benign pad token (BOS); past each chunk's true length we ignore
        # the encoded bits (chunk's mini-header records its true token count).
        per_chunk_tokens: list[list[int]] = []
        orig_lens: list[int] = []
        for c in chunks:
            orig_lens.append(len(c))
            text = _bytes_to_text(c)
            toks = self._tokenizer.encode(text).ids
            if len(toks) < 1:
                raise ValueError("chunk has no tokens after tokenization")
            per_chunk_tokens.append(toks)
        n_tokens_per = [len(t) for t in per_chunk_tokens]
        T_max = max(n_tokens_per)

        device = self._device
        # Per-chunk token tensor [B, T_max], padded with BOS in the unused
        # tail (the AC encode just ignores symbols at t >= n_tokens[i]).
        tokens_padded = torch.full((B, T_max), BOS_TOKEN,
                                    dtype=torch.long, device=device)
        for i, toks in enumerate(per_chunk_tokens):
            tokens_padded[i, :len(toks)] = torch.tensor(
                toks, dtype=torch.long, device=device)
        # Inputs: [BOS] + tokens[:-1]; outputs: tokens. Build by shifting.
        inputs_padded = torch.full((B, T_max), BOS_TOKEN,
                                    dtype=torch.long, device=device)
        inputs_padded[:, 1:] = tokens_padded[:, :-1]

        # Per-chunk output buffer: worst-case size 2× input bytes + slack.
        per_cap = max(max(orig_lens) * 2, 64 << 10)
        TAIL_PAD = 64
        per_stride = per_cap + TAIL_PAD
        # Concatenated output buffer + base offsets (matches decompress's
        # bitstream concat layout, in reverse).
        base_offsets = [i * per_stride for i in range(B)]
        output_buf = torch.zeros(B * per_stride, dtype=torch.uint8, device=device)
        base_byte_offsets = torch.tensor(base_offsets, dtype=torch.int32, device=device)
        # Encoder states: [B, 4] uint32, low=0 high=0xFFFFFFFF pending=0 bit_offset=0
        ac_states = torch.zeros(B * 4, dtype=torch.uint32, device=device)
        ac_states.view(B, 4)[:, 1] = 0xFFFFFFFF

        # Use cpp_path's stepped-batched forward — same code path as the
        # decompress side. Symmetric forward = identical logits = roundtrip
        # holds (this matters when bf16 / int8 codecs swap kernels).
        weights = cpp_path.init_weights(self._model, device)
        rwkv_state = cpp_path.fresh_state_batched(weights, B)

        for t in range(T_max):
            cur_in = inputs_padded[:, t].contiguous()
            logits = cpp_path.forward_stepped_batched(weights, cur_in, rwkv_state)
            # Use the same softmax+CDF kernel the decoder uses
            # (cpp_path.softmax_cdfs_per_row → det_softmax_cdf + cumsum).
            # Otherwise encoder builds CDFs via probs_to_cdf_gpu and
            # decoder via softmax_cdfs_per_row — even with bit-identical
            # logits, the two CDF kernels can differ by 1 LSB → AC
            # bitstream un-recoverable. Pinned by docs/Bugs.md #2.
            cdfs = cpp_path.softmax_cdfs_per_row(logits).contiguous()
            sym_t = tokens_padded[:, t].to(torch.int32).contiguous()
            krunch_ac_cuda.encode_step_batched(
                cdfs, sym_t, output_buf, base_byte_offsets, ac_states)

        krunch_ac_cuda.encode_finalize_batched(
            output_buf, base_byte_offsets, ac_states)
        torch.cuda.synchronize()

        # Pull bit_offsets per stream, slice output, build per-chunk results.
        bit_offsets_cpu = ac_states.view(B, 4)[:, 3].cpu().numpy()
        output_cpu = output_buf.cpu().numpy()
        out: list[bytes] = []
        for i in range(B):
            n_bytes = int((bit_offsets_cpu[i] + 7) // 8)
            ac_bytes = bytes(output_cpu[base_offsets[i]:base_offsets[i] + n_bytes])
            mini_header = struct.pack(">II", orig_lens[i], n_tokens_per[i])
            out.append(mini_header + ac_bytes)
        return out

    def decompress_chunks_batched(self, encoded_chunks: list[bytes],
                                    tokenizer_id: int = TOKENIZER_ID) -> list[bytes]:
        """Decompress B independent chunks in lockstep via the bit-exact
        C++ orchestration path (matches compress_chunk so the bitstream
        roundtrips byte-for-byte).

        ``tokenizer_id`` selects the reverse text→bytes function; read
        from the blob header by the caller (cli.cmd_decompress).
        """
        B = len(encoded_chunks)
        if B == 0:
            return []
        if B == 1:
            return [self.decompress_chunk(encoded_chunks[0],
                                           tokenizer_id=tokenizer_id)]
        return self._decompress_chunks_batched_cpp(encoded_chunks,
                                                    tokenizer_id=tokenizer_id)


def _gpu_softmax_to_numpy(logits) -> np.ndarray:
    """Softmax on the same device as `logits` (typically GPU), in fp32 for
    numerical stability, then transfer to CPU as float32 numpy. The CPU
    side never sees raw logits — only normalized probabilities — saving
    ~500 ms per (1024, 50K) batch versus doing softmax in numpy on CPU."""
    import torch
    with torch.no_grad():
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        probs = torch.softmax(logits.float(), dim=-1)
    return probs.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_numpy(t) -> np.ndarray:
    """Tensor → numpy fp32, regardless of device. CUDA tensors must be moved
    to host memory first; the rwkv pkg returns CUDA tensors when the model
    is loaded with `cuda fp16` strategy."""
    if hasattr(t, "detach"):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


# Module-level singleton — imported by main.py
engine = InferenceEngine()
