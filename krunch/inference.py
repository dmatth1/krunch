"""
RWKV-4-Pile-169M inference core.

Uses BlinkDL's `rwkv` pip package (NOT HF transformers): the WKV CUDA
kernel only engages from BlinkDL's path; HF's RWKV implementation
silently falls back to a ~1000× slower Python loop in eval mode.
"""

import os
import gc
import time
import struct
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

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
# Model + tokenizer IDs for RWKV-4-Pile-169M + GPT-NeoX BPE
MODEL_ID = 1
TOKENIZER_ID = 1

# Header: magic(4) + blob_version(1) + model_id(4) + tokenizer_id(4) +
#         adapter_id(4) + adapter_version(2) + flags(2) +
#         original_len(8) + n_chunks(4) + crc32(4) = 42 bytes
HEADER_FMT = ">4sBIIIHHQII"  # big-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def encode_header(original_len: int, n_chunks: int, crc32: int,
                  adapter_id: int = 0, adapter_version: int = 0,
                  flags: int = 0) -> bytes:
    return struct.pack(
        HEADER_FMT,
        BLOB_MAGIC, BLOB_VERSION,
        MODEL_ID, TOKENIZER_ID,
        adapter_id, adapter_version, flags,
        original_len, n_chunks, crc32
    )


def decode_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"blob too short: {len(data)} < {HEADER_SIZE}")
    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic, bv, mid, tid, aid, av, flags, orig_len, n_chunks, crc = fields
    if magic != BLOB_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    return {
        "blob_version": bv, "model_id": mid, "tokenizer_id": tid,
        "adapter_id": aid, "adapter_version": av, "flags": flags,
        "original_len": orig_len, "n_chunks": n_chunks, "crc32": crc,
    }


# ---------------------------------------------------------------------------
# Arithmetic coder (constriction)
# ---------------------------------------------------------------------------

import constriction  # noqa: E402


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


BOS_TOKEN = 0  # initial seed for both encode and decode — must match


def _categorical_from_logits(logits_row: np.ndarray):
    """Softmax → clip → renormalize → Categorical. Same numerical recipe
    on encode and decode sides so the bitstream is byte-exact."""
    probs = _softmax_np(logits_row.astype(np.float64))
    probs = np.clip(probs, 1e-9, 1.0)
    probs /= probs.sum()
    return constriction.stream.model.Categorical(probs, perfect=False)


def ac_encode(tokens: list[int], logits_seq: np.ndarray) -> bytes:
    """
    Range-encode tokens[0..N-1] using logits_seq[0..N-1] as distributions.
    logits_seq[i] must be the model's prediction for tokens[i] given the prefix
    [BOS, tokens[0], ..., tokens[i-1]]. Range coding is symmetric (encode and
    decode both run in forward order), which lets compress_chunk stream:
    forward batch i → range-encode immediately → free logits → next batch.
    """
    assert len(tokens) == logits_seq.shape[0], \
        f"tokens ({len(tokens)}) vs logits ({logits_seq.shape[0]}) length mismatch"
    enc = constriction.stream.queue.RangeEncoder()
    for t in range(len(tokens)):
        enc.encode(np.array([tokens[t]], dtype=np.int32),
                   _categorical_from_logits(logits_seq[t]))
    return np.array(enc.get_compressed(), dtype=np.uint32).tobytes()


def ac_decode(bitstream: bytes, n_tokens: int, logits_fn) -> list[int]:
    """
    Range-decode n_tokens. logits_fn(state, last_input) -> (logits, new_state)
    is called n_tokens times: first with last_input=BOS_TOKEN, state=None;
    subsequent calls feed the just-decoded token.
    """
    compressed = np.frombuffer(bitstream, dtype=np.uint32)
    dec = constriction.stream.queue.RangeDecoder(compressed)
    tokens = []
    state = None
    last_input = BOS_TOKEN
    for _ in range(n_tokens):
        logits, state = logits_fn(state, last_input)
        tok = int(dec.decode(_categorical_from_logits(logits)))
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

    def load(self):
        """Load model + tokenizer. Blocks until ready."""
        import torch
        self._load_start = time.time()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
        self._tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

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

    def compress_chunk(self, data: bytes) -> bytes:
        """
        Compress a single chunk to a range-coded bitstream + mini-header.

        Streaming: each forward batch's logits are range-encoded immediately
        and freed before the next forward call. Peak memory is O(SEQ_BATCH ×
        vocab) ≈ 200 MB regardless of chunk size — the per-chunk memory
        bound that was capping chunk size on small instances is gone.
        """
        text = data.decode("utf-8", errors="replace")
        tokens = self._tokenizer.encode(text).ids
        if len(tokens) < 1:
            raise ValueError("chunk has no tokens after tokenization")

        SEQ_BATCH = int(os.environ.get("KRUNCH_FORWARD_BATCH", 1024))
        full_input = [BOS_TOKEN] + tokens[:-1]  # length N

        enc = constriction.stream.queue.RangeEncoder()
        state = None
        pos = 0  # next token index to encode
        for i in range(0, len(full_input), SEQ_BATCH):
            batch = full_input[i:i + SEQ_BATCH]
            logits, state = self._model.forward(batch, state, full_output=True)
            np_logits = _to_numpy(logits)  # (B, vocab)
            for row in np_logits:
                enc.encode(np.array([tokens[pos]], dtype=np.int32),
                           _categorical_from_logits(row))
                pos += 1
            del np_logits, logits  # release before the next forward call

        ac_bytes = np.array(enc.get_compressed(), dtype=np.uint32).tobytes()

        # Force the torch CUDA allocator to return cached memory across chunks
        # (otherwise the cache drifts upward over a long compress run).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Mini-header: original byte length (4) + token count (4)
        mini_header = struct.pack(">II", len(data), len(tokens))
        return mini_header + ac_bytes

    def decompress_chunk(self, encoded: bytes) -> bytes:
        """Decompress a single AC-encoded chunk produced by compress_chunk."""
        orig_len, n_tokens = struct.unpack(">II", encoded[:8])
        bitstream = encoded[8:]

        def logits_fn(state, token):
            logits, new_state = self._model.forward([token], state)
            return _to_numpy(logits), new_state

        tokens = ac_decode(bitstream, n_tokens, logits_fn)
        text = self._tokenizer.decode(tokens)
        result = text.encode("utf-8")
        return result[:orig_len]


def _to_numpy(t) -> np.ndarray:
    """Tensor → numpy fp32, regardless of device. CUDA tensors must be moved
    to host memory first; the rwkv pkg returns CUDA tensors when the model
    is loaded with `cuda fp16` strategy."""
    if hasattr(t, "detach"):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


# Module-level singleton — imported by main.py
engine = InferenceEngine()
