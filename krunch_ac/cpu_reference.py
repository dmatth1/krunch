"""
CPU reference range coder. Bit-exact spec for the CUDA kernel.

Standard 32-bit precision range coder with E1/E2/E3 renormalization.
Encoder consumes (cdf, symbol) pairs; decoder reverses given the
same CDF stream. Output bytes from encode() must round-trip through
decode() byte-identically.

Naming/convention:
- PRECISION = 32 (state bits)
- TOP = 2**32, HALF = 2**31, QTR = 2**30
- T = 2**CDF_PRECISION (sum of every CDF row), default 2**16
- range coder state: low (uint32), high (uint32), pending E3 bits

The arithmetic is integer-exact. We use Python ints (arbitrary
precision) to dodge overflow questions; the CUDA kernel uses
uint64 for the range*cdf_value intermediate (32 + 16 = 48 bits,
fits trivially).

Bitstream serialization: bits are packed MSB-first into bytes.
Final flush writes one tail bit + pending bits, then pads the
last partial byte with zeros. Decoder reads MSB-first, padding
the input with zero bits past EOF (standard).
"""

import numpy as np

PRECISION = 32
TOP = 1 << PRECISION                # 2^32
TOP_MASK = TOP - 1                  # 0xFFFFFFFF
HALF = 1 << (PRECISION - 1)         # 2^31
QTR = 1 << (PRECISION - 2)          # 2^30
THREE_QTR = HALF + QTR              # 3 * 2^30


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

class _BitWriter:
    __slots__ = ("buf", "byte", "n")

    def __init__(self):
        self.buf = bytearray()
        self.byte = 0
        self.n = 0

    def write(self, bit: int):
        self.byte = (self.byte << 1) | (bit & 1)
        self.n += 1
        if self.n == 8:
            self.buf.append(self.byte)
            self.byte = 0
            self.n = 0

    def finish(self) -> bytes:
        if self.n > 0:
            self.buf.append(self.byte << (8 - self.n))
            self.byte = 0
            self.n = 0
        return bytes(self.buf)


class _BitReader:
    __slots__ = ("data", "pos", "byte", "n")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.byte = 0
        self.n = 0

    def read(self) -> int:
        if self.n == 0:
            if self.pos < len(self.data):
                self.byte = self.data[self.pos]
                self.pos += 1
            else:
                self.byte = 0  # past EOF: zero padding
            self.n = 8
        bit = (self.byte >> 7) & 1
        self.byte = (self.byte << 1) & 0xFF
        self.n -= 1
        return bit


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode(cdf: np.ndarray, symbols: np.ndarray) -> bytes:
    """
    cdf: uint32 (N, V+1), cdf[:, 0]=0, cdf[:, -1]=T (constant).
    symbols: int32/int64 (N,), values in [0, V).

    Returns: encoded bitstream as bytes.
    """
    assert cdf.ndim == 2, f"cdf must be 2D, got {cdf.shape}"
    N, Vp1 = cdf.shape
    assert symbols.shape == (N,), f"symbols shape {symbols.shape} != ({N},)"
    T = int(cdf[0, -1])
    assert T > 0 and (T & (T - 1)) == 0, f"T={T} must be power of 2"

    low = 0
    high = TOP_MASK
    pending = 0
    out = _BitWriter()

    for i in range(N):
        sym = int(symbols[i])
        sym_lo = int(cdf[i, sym])
        sym_hi = int(cdf[i, sym + 1])
        rng = high - low + 1

        # Narrow to the symbol's sub-interval.
        high = low + (rng * sym_hi) // T - 1
        low = low + (rng * sym_lo) // T

        # Renormalize.
        while True:
            if high < HALF:
                _emit(out, 0, pending)
                pending = 0
                low <<= 1
                high = (high << 1) | 1
            elif low >= HALF:
                _emit(out, 1, pending)
                pending = 0
                low = (low - HALF) << 1
                high = ((high - HALF) << 1) | 1
            elif low >= QTR and high < THREE_QTR:
                # E3 condition — straddling HALF, defer the bit.
                pending += 1
                low = (low - QTR) << 1
                high = ((high - QTR) << 1) | 1
            else:
                break
            low &= TOP_MASK
            high &= TOP_MASK

    # Final flush: emit one more bit to disambiguate.
    pending += 1
    if low < QTR:
        _emit(out, 0, pending)
    else:
        _emit(out, 1, pending)
    return out.finish()


def _emit(out: _BitWriter, bit: int, pending: int):
    out.write(bit)
    inv = 1 - bit
    for _ in range(pending):
        out.write(inv)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

class RangeDecoder:
    """
    Stateful CPU range decoder. Use for autoregressive decoding where
    each step's CDF depends on the previous decoded symbol.

    Usage:
        dec = RangeDecoder(bitstream)
        for step in range(N):
            cdf_row = compute_cdf(prev_symbol)   # (V+1,) uint32
            sym = dec.decode_symbol(cdf_row)
            prev_symbol = sym
    """

    def __init__(self, bitstream: bytes):
        self._rd = _BitReader(bitstream)
        self.low = 0
        self.high = TOP_MASK
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._rd.read()

    def decode_symbol(self, cdf_row) -> int:
        """cdf_row: (V+1,) array-like, cdf_row[0]=0, cdf_row[V]=T."""
        V = len(cdf_row) - 1
        T = int(cdf_row[-1])
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * T - 1) // rng
        sym = _bsearch(cdf_row, target, V)

        sym_lo = int(cdf_row[sym])
        sym_hi = int(cdf_row[sym + 1])
        self.high = self.low + (rng * sym_hi) // T - 1
        self.low = self.low + (rng * sym_lo) // T

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QTR and self.high < THREE_QTR:
                self.low -= QTR
                self.high -= QTR
                self.value -= QTR
            else:
                break
            self.low = (self.low << 1) & TOP_MASK
            self.high = ((self.high << 1) | 1) & TOP_MASK
            self.value = ((self.value << 1) | self._rd.read()) & TOP_MASK
        return sym


def decode(bitstream: bytes, cdfs: np.ndarray) -> np.ndarray:
    """One-shot decode of N symbols given a (N, V+1) CDF tensor."""
    assert cdfs.ndim == 2
    N = cdfs.shape[0]
    dec = RangeDecoder(bitstream)
    out = np.empty(N, dtype=np.int32)
    for i in range(N):
        out[i] = dec.decode_symbol(cdfs[i])
    return out


def _bsearch(cdf_row: np.ndarray, target: int, V: int) -> int:
    """Return largest sym in [0, V) with cdf_row[sym] <= target."""
    lo, hi = 0, V
    while lo < hi:
        mid = (lo + hi) >> 1
        if int(cdf_row[mid + 1]) <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
