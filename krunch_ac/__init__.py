"""
krunch_ac: GPU-native arithmetic coder for krunch.

Two implementations behind one interface:

- cpu_reference: pure-Python integer range coder. Bit-exact spec
  for the CUDA kernel — ground truth for correctness tests. Slow
  but readable.
- cuda_kernel: nvcc-compiled CUDA kernel via pybind11. Same
  bitstream output as cpu_reference (byte-identical). Built on
  the GPU instance at install time.

Both consume:
- cdf: int32 tensor of shape (N, V+1), cdf[i, 0] = 0, cdf[i, V] = T,
  monotonically non-decreasing along axis 1.
- symbols: int32 tensor of shape (N,), values in [0, V).

Output: bytes (the encoded bitstream).
"""

from krunch_ac.cdf import probs_to_cdf, CDF_PRECISION
from krunch_ac.cpu_reference import (
    encode as encode_cpu,
    decode as decode_cpu,
)

__all__ = [
    "probs_to_cdf",
    "CDF_PRECISION",
    "encode_cpu",
    "decode_cpu",
]
