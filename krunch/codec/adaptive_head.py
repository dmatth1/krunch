"""
Adaptive bias head — per-document online correction in log-prob space.

Direct port of the algorithm in
[Nacrith-GPU/adaptive_head.py](https://github.com/robtacconelli/Nacrith-GPU)
(Apache-2.0). Sits AFTER the LLM softmax and BEFORE the AC encoder:

    state:    bias[V]  — initialized to zero
    per token:
        adjusted = softmax(log(probs) + bias)
        ENCODE / DECODE the actual_token using adjusted
        grad = adjusted; grad[actual_token] -= 1
        bias -= lr * grad

Both encoder and decoder run the same adjust + update sequence, so
they produce identical CDFs at every step → AC roundtrip holds.

This is the CPU reference. The bit-exact GPU implementation lives
in `adaptive_head_gpu.py`. Tests pin the two against each other.

Bitstream impact: enabling adaptive head changes the emitted CDF
per token, which changes the AC bytes. Requires a MODEL_ID bump.
"""

import numpy as np


# Defaults — match Nacrith's adaptive_head.py.
DEFAULT_LR = 0.001


class AdaptiveHead:
    """Per-document adaptive bias in log-probability space.

    State is a single float64 vector of length vocab_size, initialized
    to zero. After each observed token, the bias is updated by one
    step of gradient descent on cross-entropy loss using the adjusted
    probabilities returned for that token.

    Float64 throughout so the bias trajectory is bit-exact between
    encoder and decoder, regardless of platform.
    """

    __slots__ = ("vocab_size", "lr", "bias", "_log_buf", "_grad_buf")

    def __init__(self, vocab_size: int, lr: float = DEFAULT_LR):
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not (0.0 < lr < 1.0):
            raise ValueError(f"lr must be in (0, 1), got {lr}")
        self.vocab_size = vocab_size
        self.lr = float(lr)
        self.bias = np.zeros(vocab_size, dtype=np.float64)
        # Pre-allocated scratch (Nacrith's pattern; keeps the inner loop
        # allocation-free).
        self._log_buf = np.zeros(vocab_size, dtype=np.float64)
        self._grad_buf = np.zeros(vocab_size, dtype=np.float64)

    def reset(self) -> None:
        """Re-initialize bias to zero. Call when starting a new sequence
        (compress a fresh chunk, decompress a fresh blob). Encoder + decoder
        MUST call reset at the same logical position."""
        self.bias[:] = 0.0

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        """Apply adaptive bias to LLM probabilities.

        Args:
            probs: 1-D float (any precision) shape (vocab_size,), summing
                   to ~1. Cast internally to float64.

        Returns:
            Adjusted probabilities as a fresh float64 array of shape
            (vocab_size,), summing to 1 (within fp64 rounding).
        """
        if probs.shape != (self.vocab_size,):
            raise ValueError(
                f"probs must be shape ({self.vocab_size},), got {probs.shape}"
            )
        log_buf = self._log_buf
        # log(probs + eps) — tiny eps so log(0) doesn't blow up; same eps
        # as Nacrith for bit-compat.
        np.log(probs.astype(np.float64, copy=False) + 1e-10, out=log_buf)
        log_buf += self.bias
        # Numerically stable softmax.
        log_buf -= log_buf.max()
        np.exp(log_buf, out=log_buf)
        log_buf /= log_buf.sum()
        # Return a fresh copy — the caller may keep it across the next
        # call (Nacrith returns the internal buffer; we copy for safety
        # since the AC path may hold the array briefly).
        return log_buf.copy()

    def update(self, actual_token: int, adjusted_probs: np.ndarray) -> None:
        """Update bias after observing actual_token.

        Gradient of cross-entropy loss w.r.t. bias is
        ``adjusted - one_hot(actual_token)``.

        Args:
            actual_token: the token in the input/output stream at this step.
            adjusted_probs: the array returned from this step's adjust(),
                            float64, length vocab_size.
        """
        if not (0 <= actual_token < self.vocab_size):
            raise ValueError(
                f"actual_token {actual_token} out of range [0, {self.vocab_size})"
            )
        grad = self._grad_buf
        np.copyto(grad, adjusted_probs)
        grad[actual_token] -= 1.0
        self.bias -= self.lr * grad


def encode_with_adaptive_head(
    probs_seq: np.ndarray, tokens: np.ndarray, lr: float = DEFAULT_LR
) -> tuple[np.ndarray, np.ndarray]:
    """Roundtrip-friendly encoder helper for tests.

    Given a [T, V] array of LLM probabilities and a [T] array of actual
    tokens, returns the [T, V] adjusted probabilities + the final bias
    vector. Mirrors the encoder side of an actual compress run.

    Designed to be called identically on the decoder side
    (`decode_with_adaptive_head`) so the two trajectories match
    bit-for-bit.
    """
    if probs_seq.ndim != 2:
        raise ValueError(f"probs_seq must be 2-D [T, V], got {probs_seq.shape}")
    T, V = probs_seq.shape
    if tokens.shape != (T,):
        raise ValueError(f"tokens shape {tokens.shape} != ({T},)")
    head = AdaptiveHead(V, lr=lr)
    out = np.empty((T, V), dtype=np.float64)
    for t in range(T):
        out[t] = head.adjust(probs_seq[t])
        head.update(int(tokens[t]), out[t])
    return out, head.bias.copy()
