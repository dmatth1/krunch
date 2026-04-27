"""
RWKV-4-Pile-169M inference server core.

Uses BlinkDL/RWKV-LM directly (NOT HF transformers — see CLAUDE.md).
The WKV CUDA kernel only engages when the model is in training mode;
HF's eval() path silently falls back to a ~1000x slower Python loop.
"""

import os
import gc
import time
import struct
import logging
from pathlib import Path
from typing import Optional

import torch
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

_rwkv_lm_path = Path(os.environ.get("RWKV_LM_PATH", "/opt/rwkv-lm/RWKV-v4/src"))


def _load_rwkv_lm():
    import sys
    sys.path.insert(0, str(_rwkv_lm_path))
    import model_run  # noqa: F401 — side-effect: registers RWKV ops
    from model_run import RWKV_RNN
    return RWKV_RNN


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


def ac_encode(tokens: list[int], logits_seq: np.ndarray) -> bytes:
    """
    Arithmetic-encode tokens[1:] using logits_seq[:-1] as distributions.
    logits_seq shape: (T, vocab)
    Returns raw AC bitstream bytes.
    """
    codec = constriction.stream.stack.AnsCoder()
    vocab = logits_seq.shape[1]
    # Encode in reverse order for stack-based ANS
    for t in range(len(tokens) - 1, 0, -1):
        probs = _softmax_np(logits_seq[t - 1].astype(np.float64))
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        codec.encode_symbol(tokens[t], model)
    return bytes(codec.get_compressed())


def ac_decode(bitstream: bytes, n_tokens: int,
              logits_fn, initial_token: int) -> list[int]:
    """
    Arithmetic-decode n_tokens using logits_fn(state, token) -> (logits, state).
    Returns decoded token list (length n_tokens, including initial_token).
    """
    codec = constriction.stream.stack.AnsCoder(
        np.frombuffer(bitstream, dtype=np.uint32)
    )
    tokens = [initial_token]
    state = None
    for _ in range(n_tokens - 1):
        logits, state = logits_fn(state, tokens[-1])
        probs = _softmax_np(logits.astype(np.float64))
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        tok = codec.decode_symbol(model)
        tokens.append(int(tok))
    return tokens


# ---------------------------------------------------------------------------
# InferenceEngine: model + tokenizer, loaded once per process
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
        self._model = None
        self._tokenizer: Optional[Tokenizer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ready = False
        self._load_start: Optional[float] = None

    def load(self):
        """Load model + tokenizer. Blocks until ready."""
        self._load_start = time.time()
        logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
        self._tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

        logger.info("Loading RWKV-4-Pile-169M from %s (device=%s)",
                    MODEL_PATH, self._device)
        RWKV_RNN = _load_rwkv_lm()

        # BlinkDL's RWKV_RNN constructor: (model_path, strategy)
        # strategy controls dtype + device. fp16 on CUDA, fp32 on CPU.
        strategy = (
            f"cuda fp16" if self._device == "cuda"
            else "cpu fp32"
        )
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1" if self._device == "cuda" else "0"

        self._model = RWKV_RNN(str(MODEL_PATH), strategy)
        self._ready = True
        elapsed = time.time() - self._load_start
        logger.info("Model loaded in %.1fs", elapsed)

    @property
    def ready(self) -> bool:
        return self._ready

    def compress_chunk(self, data: bytes) -> bytes:
        """
        Compress a single chunk to AC bitstream + mini-header.
        Returns raw encoded bytes (no blob header — chunking.py wraps these).
        """
        text = data.decode("utf-8", errors="replace")
        enc = self._tokenizer.encode(text)
        tokens = enc.ids
        if len(tokens) < 2:
            raise ValueError("chunk too short to compress (< 2 tokens)")

        # Full-sequence forward pass: (T, vocab) logits in one shot
        logits_seq = self._forward_sequence(tokens)

        ac_bytes = ac_encode(tokens, logits_seq)

        # Mini-header: original byte length (4) + token count (4)
        mini_header = struct.pack(">II", len(data), len(tokens))
        return mini_header + ac_bytes

    def decompress_chunk(self, encoded: bytes) -> bytes:
        """Decompress a single AC-encoded chunk produced by compress_chunk."""
        orig_len, n_tokens = struct.unpack(">II", encoded[:8])
        bitstream = encoded[8:]

        def logits_fn(state, token):
            logits, new_state = self._model.forward([token], state)
            return np.array(logits, dtype=np.float32), new_state

        # BOS token (0) as initial context; decode n_tokens
        tokens = ac_decode(bitstream, n_tokens, logits_fn, initial_token=0)
        text = self._tokenizer.decode(tokens[1:])  # drop BOS
        result = text.encode("utf-8")
        # Trim/pad to original length for byte-exactness on UTF-8 boundaries
        return result[:orig_len]

    def _forward_sequence(self, tokens: list[int]) -> np.ndarray:
        """
        Run a full sequence forward pass.
        Returns logits array of shape (T, vocab).
        """
        # Prepend BOS (token 0) so each position t has a valid context
        full_tokens = [0] + tokens
        logits_list = []
        state = None
        for tok in full_tokens[:-1]:
            logits, state = self._model.forward([tok], state)
            logits_list.append(np.array(logits, dtype=np.float32))
        return np.stack(logits_list, axis=0)  # (T, vocab)


# Module-level singleton — imported by main.py
engine = InferenceEngine()
