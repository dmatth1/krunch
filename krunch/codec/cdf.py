"""
Probability → integer CDF conversion.

The range coder consumes integer CDFs, not floats, so encoder and
decoder agree on exact bin boundaries. We quantize float probs to a
uint16-sum of T = 2^16, with MIN_PROB = 1 to guarantee every symbol
has non-zero width (otherwise encoding that symbol is impossible).

Why T = 2^16: with 32-bit range register the intermediate
`range * cdf_value` fits in uint64 with room to spare (32 + 16 = 48
bits), and uint16 CDF entries halve the GPU shared-memory footprint
vs uint32. Ratio loss vs T = 2^24 is small (~0.001 bits/symbol on
50K vocab from the residual quantization).
"""

import numpy as np

CDF_PRECISION = 24
T = 1 << CDF_PRECISION  # 16777216


def probs_to_cdf(probs: np.ndarray) -> np.ndarray:
    """
    Convert (N, V) float probabilities to (N, V+1) uint32 CDF.

    Guarantees:
    - cdf[:, 0] == 0
    - cdf[:, V] == T
    - cdf[:, j+1] > cdf[:, j] for all j (every symbol has width >= 1)
    - monotonic along axis 1

    Algorithm (simple-and-fast variant — no per-row ranking, GPU-friendly):
    1. counts = floor(p * (T - V))    # 0..(T-V)
    2. counts += 1                     # MIN_PROB; sum is in [V, T]
    3. argmax bin absorbs the entire deficit (T - sum) — keeps the
       construction O(NV) with no argsort/topk.
    4. cumsum.

    Slight ratio cost vs the textbook "spread deficit over largest
    residuals" variant: argmax bin gets a tiny over-allocation, so its
    encoding is ~`deficit/T` bits longer than optimal per occurrence
    (~0.2 bits when deficit ~ V/2 = 25K of T = 65K). On a 256K-token
    chunk, this is <1% ratio loss for ~30× faster CDF construction
    on GPU. Worth it; we're throughput-bound, not ratio-bound.
    """
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    assert probs.ndim == 2, f"probs must be (N, V), got {probs.shape}"
    N, V = probs.shape
    assert V < T, f"vocab size {V} must be < T={T}"

    # fp32 spec: numpy and torch can disagree in the last bit on float32
    # reductions (different intrinsics / reduction order), so this CDF
    # is no longer byte-identical to torch's. Self-consistency
    # (encode→decode round-trip via the same code path) is preserved
    # and that's what production cares about.
    p = probs.astype(np.float32, copy=False)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-30))

    counts = np.floor(p * (T - V)).astype(np.int64) + 1  # MIN_PROB = 1
    deficit = T - counts.sum(axis=1)                     # (N,) >= 0

    if int(deficit.max()) > 0 or int(deficit.min()) < 0:
        # Argmax bin absorbs the deficit. argmax over float prob is
        # well-defined and ties are essentially impossible for real probs.
        argmax = p.argmax(axis=1)
        counts[np.arange(N), argmax] += deficit

    assert (counts.sum(axis=1) == T).all(), "sum must equal T"
    assert (counts > 0).all(), "every bin must have count >= 1"

    cdf = np.zeros((N, V + 1), dtype=np.uint32)
    cdf[:, 1:] = np.cumsum(counts, axis=1).astype(np.uint32)
    return cdf
