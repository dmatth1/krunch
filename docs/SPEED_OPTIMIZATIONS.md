# Speed Optimizations

Identified via `l3tc profile` on the 6L × 96H × vocab 32K model.
Ranked by expected impact.

## Current bottleneck breakdown

From `l3tc profile` on fiction.txt (152 KB, single-thread):

| phase | µs/step | % of time |
|-------|---------|-----------|
| forward pass (RWKV + head matvec) | 541 | 84.5% |
| cum_freqs (softmax + prefix sum) | 99 | 15.4% |
| AC encode | 0.26 | 0.04% |

Decompress is ~2× slower than compress on the 6L model due
to a linear search in the AC decoder.

For comparison, the 2L × 16K model:

| phase | µs/step | % of time |
|-------|---------|-----------|
| forward pass | 234 | 82.2% |
| cum_freqs | 50 | 17.7% |
| AC encode | 0.17 | 0.06% |

---

## Compress path

### 1. Head matvec: pre-widen INT8 columns

**File:** `l3tc-rust/src/tensor.rs:656`
**Impact:** ~30% reduction in head matvec time

The i8→f32 widening (`sxtl` + `scvtf`) happens inside the
inner 32K-iteration loop, limiting NEON autovectorization.
Pre-widening one column of i8 to f32 in a temp buffer before
the AXPY multiply would let the inner loop be a pure `fmla`
broadcast.

```
Current:  for j in 0..96 { for i in 0..32768 { out[i] += xs * (col[i] as f32) } }
Proposed: for j in 0..96 { widen(col, tmp); for i in 0..32768 { out[i] += xs * tmp[i] } }
```

### 2. Top-K cum_freqs truncation

**File:** `l3tc-rust/src/codec.rs:1388`
**Impact:** cum_freqs from 99 µs → ~10 µs (15% → 2% of total)

Most probability mass is in <1000 tokens. Zero out the tail,
compute cum_freqs only over the top-K. Cuts the 32K loop to
~1K per token. Requires matching truncation on the decode side
(set all non-top-K freqs to the minimum floor value).

### 3. Vectorize cum_freqs prefix sum

**File:** `l3tc-rust/src/codec.rs:1450`
**Impact:** moderate (diminishing if top-K done first)

The scalar serial loop over 32K elements has a carried data
dependency (`cum[i+1] = cum[i] + freqs[i]`). A SIMD parallel
prefix scan (tree reduction) could help but is complex. Top-K
(#2) is simpler and more impactful — do that first.

---

## Decompress path

### 4. AC decode: binary search over cum_freqs

**File:** `l3tc-rust/src/arithmetic.rs:228`
**Impact:** decompress speed roughly doubles

Currently a linear O(V) scan per decoded token with a comment:
"Linear search is fine for small vocabularies; we can switch
to binary search later if the vocab gets large."

At 32K vocab: ~16K comparisons/token → ~15 with binary search.
The cum_freqs array is already sorted (it's a prefix sum), so
`partition_point` or manual binary search is a drop-in fix.

This is the single biggest decompress optimization.

---

## Both paths

### 5. Larger segment size

**Current:** 4096 bytes per segment.
**Impact:** better ratio + less per-segment overhead

Larger segments (8K–16K) mean:
- Fewer segments → less rayon dispatch overhead
- Fewer session resets → less wasted context
- Longer context window → better predictions → better ratio
- Tradeoff: less rayon parallelism on small files

Should benchmark 4K vs 8K vs 16K on enwik6 to quantify the
ratio improvement vs parallelism tradeoff.

---

## Implementation order

1. **Binary search** (#4) — trivial fix, big decompress win
2. **Top-K cum_freqs** (#2) — moderate effort, cleans up 15%
3. **Head pre-widen** (#1) — small change, biggest compress win
4. **Larger segments** (#5) — config change + benchmarking
5. **Prefix sum SIMD** (#3) — complex, only if still needed
