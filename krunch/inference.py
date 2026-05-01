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

        v1.1 GPU path: probs stay on GPU, CDF is computed via
        torch.compile-d probs_to_cdf_gpu, then our custom CUDA range
        coder kernel encodes batch-by-batch. AC state persists across
        batches in a (4,) uint32 GPU tensor. No prob transfer to CPU
        per batch (was the v1 bottleneck — 200 MB/batch × ~25K batches
        = 5 TB cross-PCIe).

        With KRUNCH_CPP_PATH=1 (and KRUNCH_DETERMINISTIC_MATMUL=1):
        uses the bit-exact C++ orchestration path so the bitstream is
        byte-identical to what `decompress_chunk` reproduces stepped.

        Peak memory: O(SEQ_BATCH × vocab) for probs + ~200 MB for CDF.
        """
        import torch
        import krunch_ac_cuda
        from krunch_ac.gpu_encode import probs_to_cdf_gpu
        from krunch import cpp_path

        text = data.decode("utf-8", errors="replace")
        tokens = self._tokenizer.encode(text).ids
        if len(tokens) < 1:
            raise ValueError("chunk has no tokens after tokenization")

        if cpp_path.cpp_path_enabled():
            return self._compress_chunk_cpp(data, tokens)

        SEQ_BATCH = int(os.environ.get("KRUNCH_FORWARD_BATCH", 1024))
        full_input = [BOS_TOKEN] + tokens[:-1]
        tokens_arr_gpu = torch.as_tensor(tokens, dtype=torch.int32, device=self._device)

        # Compile-and-cache probs_to_cdf_gpu once per process. Default mode
        # (not reduce-overhead) — last batch in a chunk has variable shape,
        # which is incompatible with reduce-overhead's CUDA graphs. Default
        # mode still gets ~2× over eager. reduce-overhead is a v1.2 win
        # if we add a static-shape padded path for the tail.
        if not hasattr(self, "_cdf_compiled"):
            self._cdf_compiled = torch.compile(probs_to_cdf_gpu)

        # Output buffer + state on GPU. Worst case AC output ~= input
        # (zstd-equivalent uniform encoding); allocate len(data) + slack.
        cap = max(len(data) * 2, 64 << 10)
        output_buf = torch.zeros(cap, dtype=torch.uint8, device=self._device)
        state = torch.zeros(4, dtype=torch.uint32, device=self._device)
        state[1] = 0xFFFFFFFF

        # Synchronous default-stream pipeline. Per-batch breakdown on
        # A10G: forward ~10.8 ms, softmax 3.6 ms, cdf 4.4 ms, encode 0.7 ms
        # = ~19.5 ms/batch ≈ 52K tok/s ≈ 200 KB/s steady-state. The forward
        # pass is now the wall (56% of per-batch time). Side-stream AC
        # pipelining was tried and gave ~7% gain on this hardware/torch
        # version — not worth the complexity. The real next win is
        # cross-chunk batched forward (v1.1 backlog "true batched RWKV
        # decode"), which collapses 16 sequential per-token forwards
        # into one launch and unblocks the 300+ KB/s tier.
        rwkv_state = None
        pos = 0
        for i in range(0, len(full_input), SEQ_BATCH):
            batch = full_input[i:i + SEQ_BATCH]
            logits, rwkv_state = self._model.forward(
                batch, rwkv_state, full_output=True)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, device=self._device)
            B = logits.size(0)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                cdf = self._cdf_compiled(probs).contiguous()
            sym_batch = tokens_arr_gpu[pos:pos + B].contiguous()
            krunch_ac_cuda.encode_step(cdf, sym_batch, output_buf, state)
            pos += B

        krunch_ac_cuda.encode_finalize(output_buf, state)
        torch.cuda.synchronize()

        bit_offset = int(state[3].item())
        n_bytes = (bit_offset + 7) // 8
        ac_bytes = bytes(output_buf[:n_bytes].cpu().numpy())

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Mini-header: original byte length (4) + token count (4)
        mini_header = struct.pack(">II", len(data), len(tokens))
        return mini_header + ac_bytes

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
        output_buf = torch.zeros(cap, dtype=torch.uint8, device=self._device)
        ac_state = torch.zeros(4, dtype=torch.uint32, device=self._device)
        ac_state[1] = 0xFFFFFFFF
        symbols = torch.as_tensor(tokens, dtype=torch.int32, device=self._device).contiguous()

        with torch.no_grad():
            for off in range(0, T, SEQ_BATCH):
                n_w = min(SEQ_BATCH, T - off)
                logits_w = cpp_path.forward_packed_window(
                    weights, full_input_t, state, off, n_w)
                cdfs_w = cpp_path.softmax_cdfs_per_row(logits_w)
                sym_w = symbols[off:off + n_w].contiguous()
                krunch_ac_cuda.encode_step(cdfs_w, sym_w, output_buf, ac_state)
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
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        mini_header = struct.pack(">II", len(data), len(tokens))
        return mini_header + ac_bytes

    def _decompress_chunks_batched_cpp(self, encoded_chunks: list[bytes]) -> list[bytes]:
        """Bit-exact cross-chunk batched decompress.

        Decodes up to B_MAX chunks in parallel per batched stepped
        forward call (B_MAX picked per-GPU by cpp_path.pick_decompress_batch).
        If the input exceeds B_MAX, splits into B_MAX-sized groups and
        processes them sequentially. Each group's per-timestep launch
        overhead is fixed; the GPU is saturated within each group.

        Same numerics as `_decompress_chunk_cpp` per-chunk, just processed
        in lockstep — verified bit-exact in scripts/test_batched_stepped.py
        (3-chunk batched == 3-chunk sequential, all state diffs = 0).
        """
        import torch
        import krunch_ac_cuda
        from krunch import cpp_path

        # Auto-pick per-GPU batch size; split input into B_MAX-sized groups.
        B_MAX = cpp_path.pick_decompress_batch()
        if len(encoded_chunks) > B_MAX:
            out: list[bytes] = []
            for i in range(0, len(encoded_chunks), B_MAX):
                out.extend(self._decompress_chunks_batched_cpp(
                    encoded_chunks[i:i + B_MAX]))
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
        state = cpp_path.fresh_state_batched(weights, B)
        last_input = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=device)
        out_syms = torch.empty(B, dtype=torch.int32, device=device)
        decoded_tokens = torch.zeros((B, max_T), dtype=torch.int32, device=device)

        with torch.no_grad():
            for t in range(max_T):
                logits = cpp_path.forward_stepped_batched(weights, last_input, state)
                cdfs = cpp_path.softmax_cdfs_per_row(logits)
                krunch_ac_cuda.decode_step_batched(
                    cdfs, input_buf, base_byte_offsets, ac_states, out_syms)
                decoded_tokens[:, t] = out_syms
                last_input = out_syms.long()

        # Single sync at the end.
        decoded_cpu = decoded_tokens.cpu().numpy()
        out: list[bytes] = []
        for i in range(B):
            toks = decoded_cpu[i, :n_tokens_per[i]].tolist()
            text = self._tokenizer.decode(toks)
            out.append(text.encode("utf-8")[:orig_lens[i]])
        return out

    def _decompress_chunk_cpp(self, encoded: bytes, orig_len: int,
                                n_tokens: int, bitstream: bytes) -> bytes:
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
        # v2 = snapshot/restore-around-capture variant; v1 is broken.
        fwd = (cpp_path.forward_stepped_graphed_v2 if use_graph
               else cpp_path.forward_stepped)
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = fwd(weights, last, state)
                cdf_row = cpp_path.softmax_cdf_one_row(logits)
                krunch_ac_cuda.decode_step(cdf_row, input_buf, ac_state, out_sym)
                tok = int(out_sym.item())
                tokens.append(tok)
                last = tok

        text = self._tokenizer.decode(tokens)
        return text.encode("utf-8")[:orig_len]

    def decompress_chunk(self, encoded: bytes) -> bytes:
        """Decompress a single AC-encoded chunk produced by compress_chunk.

        GPU decode path: state (low/high/value/bit_offset) lives in a
        4-uint32 GPU tensor across calls; per-step CDF stays on GPU; only
        the decoded symbol (one int) crosses to CPU each token (required
        because rwkv's `m.forward([last_input])` takes a Python int).
        Roughly 2-3× faster than the pure-Python reference on real LM data —
        the floor is the autoregressive forward+sync latency, not the AC
        path.
        """
        import torch
        import krunch_ac_cuda
        from krunch_ac.gpu_encode import probs_to_cdf_gpu
        from krunch import cpp_path

        orig_len, n_tokens = struct.unpack(">II", encoded[:8])
        bitstream = encoded[8:]

        if cpp_path.cpp_path_enabled():
            return self._decompress_chunk_cpp(encoded, orig_len, n_tokens, bitstream)

        # Pad the bitstream so over-reads at the last-token renorm don't
        # walk off the end. PRECISION extra bits is safe.
        bs_padded = bitstream + b"\x00" * 64
        input_buf = torch.frombuffer(bytearray(bs_padded), dtype=torch.uint8).to(self._device)
        ac_state = torch.zeros(4, dtype=torch.uint32, device=self._device)
        out_sym = torch.empty(1, dtype=torch.int32, device=self._device)
        krunch_ac_cuda.decode_init(input_buf, ac_state)

        # Optional: torch.compile the per-step forward (idea I in V1_PLAN).
        # Goal: pay launch overhead once at compile, then replay the captured
        # graph per step. Toggle via KRUNCH_DECOMPRESS_COMPILE=1. Falls back
        # silently to eager if rwkv's forward graph-breaks. mode=
        # "reduce-overhead" enables CUDA graphs trees automatically.
        compile_fwd = os.environ.get("KRUNCH_DECOMPRESS_COMPILE") == "1"
        if compile_fwd and not hasattr(self, "_compiled_forward"):
            try:
                fwd = self._model.forward
                self._compiled_forward = torch.compile(
                    fwd, mode="reduce-overhead", fullgraph=False, dynamic=False)
                logger.info("decompress: torch.compile enabled on per-step forward")
            except Exception as e:
                logger.warning("decompress: torch.compile setup failed: %s", e)
                self._compiled_forward = self._model.forward

        # Optional: record top-K(logits) vs actual-decoded-token to measure
        # self-speculation acceptance rate (idea G in V1_PLAN). Off by
        # default — KRUNCH_DECOMPRESS_INSTRUMENT=1 to enable. This adds one
        # GPU sync per step; measurement run only.
        instrument = os.environ.get("KRUNCH_DECOMPRESS_INSTRUMENT") == "1"
        if instrument:
            top1_matches = 0
            top2_matches = 0
            top4_matches = 0
            top8_matches = 0

        tokens: list[int] = []
        rwkv_state = None
        last_input = BOS_TOKEN
        forward_fn = (self._compiled_forward if compile_fwd
                      else self._model.forward)
        for _ in range(n_tokens):
            logits, rwkv_state = forward_fn([last_input], rwkv_state)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, device=self._device)
            with torch.no_grad():
                probs = torch.softmax(logits.float().reshape(1, -1), dim=-1)
                cdf_row = probs_to_cdf_gpu(probs)[0].contiguous()
                if instrument:
                    top8 = torch.topk(probs, 8, dim=-1).indices[0].cpu().numpy()
            krunch_ac_cuda.decode_step(cdf_row, input_buf, ac_state, out_sym)
            tok = int(out_sym.item())  # forces sync; required to feed next forward
            if instrument:
                if tok == int(top8[0]): top1_matches += 1
                if tok in top8[:2]: top2_matches += 1
                if tok in top8[:4]: top4_matches += 1
                if tok in top8[:8]: top8_matches += 1
            tokens.append(tok)
            last_input = tok

        if instrument:
            n = max(1, n_tokens)
            logger.info(
                "instrument: chunk %d tokens — top1=%.1f%% top2=%.1f%% top4=%.1f%% top8=%.1f%%",
                n_tokens,
                100 * top1_matches / n, 100 * top2_matches / n,
                100 * top4_matches / n, 100 * top8_matches / n,
            )

        text = self._tokenizer.decode(tokens)
        return text.encode("utf-8")[:orig_len]

    def compress_chunks_batched(self, chunks: list[bytes]) -> list[bytes]:
        """Compress N chunks in lockstep, B=N forward + B=N AC encode per
        timestep. Symmetric to `decompress_chunks_batched`: identical
        forward shape (`forward_batched`, B=N, T=1) on both sides
        guarantees bit-equivalent logits → AC roundtrip.
        Returns list of N compressed-chunk byte strings (mini-header + AC).
        """
        import torch
        import krunch_ac_cuda
        from krunch_ac.gpu_encode import probs_to_cdf_gpu
        from krunch.batched_rwkv4 import init_state_batched, forward_batched

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
            text = c.decode("utf-8", errors="replace")
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

        rwkv_state = init_state_batched(self._model, B, device=device)

        for t in range(T_max):
            cur_in = inputs_padded[:, t]
            logits, rwkv_state = forward_batched(
                self._model, cur_in.unsqueeze(1), rwkv_state, full_output=False)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                cdfs = probs_to_cdf_gpu(probs).contiguous()  # [B, V+1]
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

    def decompress_chunks_batched(self, encoded_chunks: list[bytes]) -> list[bytes]:
        """Decompress B independent chunks in lockstep, one batched forward
        + one batched AC decode launch per timestep, with decoded symbols
        living on the GPU between iterations. The Python loop runs `max_T`
        times instead of `max_T × B` (cf. per-chunk path). One CPU sync at
        the end, not per-token.
        """
        import torch
        import krunch_ac_cuda
        from krunch_ac.gpu_encode import probs_to_cdf_gpu
        from krunch import cpp_path

        B = len(encoded_chunks)
        if B == 0:
            return []
        if B == 1:
            return [self.decompress_chunk(encoded_chunks[0])]

        # Bit-exact C++ orchestration path (matches compress_chunk's
        # cpp_path so the bitstream roundtrips byte-for-byte).
        if cpp_path.cpp_path_enabled():
            return self._decompress_chunks_batched_cpp(encoded_chunks)

        from krunch.batched_rwkv4 import init_state_batched, forward_batched

        # Per-chunk mini-headers + bitstreams.
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
        # tail padding per stream so post-final renormalization reads
        # don't walk into the next stream's bytes.
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

        rwkv_state = init_state_batched(self._model, B, device=device)
        last_input = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=device)
        out_syms = torch.empty(B, dtype=torch.int32, device=device)
        decoded_tokens = torch.zeros((B, max_T), dtype=torch.int32, device=device)

        for t in range(max_T):
            logits, rwkv_state = forward_batched(
                self._model, last_input.unsqueeze(1), rwkv_state, full_output=False)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                cdfs = probs_to_cdf_gpu(probs).contiguous()  # [B, V+1]
            krunch_ac_cuda.decode_step_batched(
                cdfs, input_buf, base_byte_offsets, ac_states, out_syms)
            decoded_tokens[:, t] = out_syms
            last_input = out_syms.long()

        # Single sync at the end.
        decoded_cpu = decoded_tokens.cpu().numpy()
        out: list[bytes] = []
        for i in range(B):
            toks = decoded_cpu[i, :n_tokens_per[i]].tolist()
            text = self._tokenizer.decode(toks)
            out.append(text.encode("utf-8")[:orig_lens[i]])
        return out


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
