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

        Pipelined: a background thread does GPU forward + softmax + transfer;
        the main thread does CPU clip+renormalize + range-encode. With a
        bounded queue between them, GPU and CPU work overlap — per-batch
        wall-clock becomes max(GPU_time, CPU_time) instead of sum. On A10G
        with SEQ_BATCH=1024, this collapses ~160ms sequential per-batch to
        ~60-80ms (whichever side is slower), so ~3× compress speedup over
        the unpipelined GPU-softmax version.

        Peak memory: O(SEQ_BATCH × vocab) ≈ 200 MB × queue_depth (= ~600 MB
        with default depth=2).
        """
        import threading
        import queue
        text = data.decode("utf-8", errors="replace")
        tokens = self._tokenizer.encode(text).ids
        if len(tokens) < 1:
            raise ValueError("chunk has no tokens after tokenization")

        SEQ_BATCH = int(os.environ.get("KRUNCH_FORWARD_BATCH", 1024))
        full_input = [BOS_TOKEN] + tokens[:-1]  # length N

        enc = constriction.stream.queue.RangeEncoder()
        model_family = constriction.stream.model.Categorical(perfect=False)
        tokens_arr = np.asarray(tokens, dtype=np.int32)

        # Bounded producer-consumer queue. Producer (background thread) runs
        # forward+softmax+transfer on GPU; consumer (this thread) does CPU
        # encode. Depth=2 = 1 batch in flight on GPU + 1 ready for encode.
        SENTINEL = object()
        probs_q: "queue.Queue" = queue.Queue(maxsize=2)
        producer_exc: list = []

        def gpu_producer():
            try:
                state = None
                for i in range(0, len(full_input), SEQ_BATCH):
                    batch = full_input[i:i + SEQ_BATCH]
                    logits, state = self._model.forward(
                        batch, state, full_output=True)
                    probs = _gpu_softmax_to_numpy(logits)  # (B, V) float32 on CPU
                    del logits
                    probs_q.put(probs)
            except Exception as e:
                producer_exc.append(e)
            finally:
                probs_q.put(SENTINEL)

        producer = threading.Thread(target=gpu_producer, daemon=True)
        producer.start()

        pos = 0
        while True:
            probs = probs_q.get()
            if probs is SENTINEL:
                break
            np.clip(probs, 1e-9, 1.0, out=probs)
            probs /= probs.sum(axis=1, keepdims=True)
            B = probs.shape[0]
            enc.encode(tokens_arr[pos:pos + B], model_family, probs)
            pos += B
            del probs

        producer.join()
        if producer_exc:
            raise producer_exc[0]

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
        """Decompress a single AC-encoded chunk produced by compress_chunk.

        Uses the same GPU-softmax recipe as compress_chunk so the per-step
        probabilities match bit-for-bit. CPU softmax on encode + CPU softmax
        on decode would also work but introduces a precision trap (float32
        vs float64 paths can disagree in the last bits, breaking the AC
        decoder). Always softmaxing on GPU keeps the recipe identical.
        """
        import torch
        orig_len, n_tokens = struct.unpack(">II", encoded[:8])
        bitstream = encoded[8:]

        compressed = np.frombuffer(bitstream, dtype=np.uint32)
        dec = constriction.stream.queue.RangeDecoder(compressed)
        model_family = constriction.stream.model.Categorical(perfect=False)

        tokens: list[int] = []
        state = None
        last_input = BOS_TOKEN
        for _ in range(n_tokens):
            logits, state = self._model.forward([last_input], state)
            probs = _gpu_softmax_to_numpy(logits).reshape(1, -1)  # (1, V)
            del logits
            np.clip(probs, 1e-9, 1.0, out=probs)
            probs /= probs.sum(axis=1, keepdims=True)
            tok = int(dec.decode(model_family, probs)[0])
            tokens.append(tok)
            last_input = tok
            del probs

        text = self._tokenizer.decode(tokens)
        return text.encode("utf-8")[:orig_len]


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
