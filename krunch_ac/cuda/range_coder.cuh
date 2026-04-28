#pragma once
// Constants shared between encode + decode kernels and the host glue.
// MUST match krunch_ac/cpu_reference.py exactly — that file is the spec.

#include <cstdint>

constexpr int PRECISION = 32;
constexpr uint64_t TOP = 1ULL << PRECISION;       // 2^32
constexpr uint32_t TOP_MASK = 0xFFFFFFFFu;        // TOP - 1
constexpr uint32_t HALF = 1u << (PRECISION - 1);  // 2^31
constexpr uint32_t QTR = 1u << (PRECISION - 2);   // 2^30
constexpr uint32_t THREE_QTR = HALF + QTR;        // 3 * 2^30

// CDF precision must match krunch_ac/cdf.py.
constexpr int CDF_PRECISION = 16;
constexpr uint32_t CDF_T = 1u << CDF_PRECISION;   // 65536

// Range coder state passed across kernel launches (one chunk encodes
// many forward-batches; AC state must persist).
struct RangeState {
    uint32_t low;
    uint32_t high;
    uint32_t pending;     // # of E3 underflow bits pending
    uint32_t bit_offset;  // current write position in output_buf, in BITS
};
