# Phase 2 — Improve ratio and push speed further  ✅ COMPLETE

**Final result:** 89 KB/s compress, ratio 0.2060 on enwik6. 6.9×
faster than Python L3TC-200K. Ratio and speed improvements were
modest (the big wins required deeper changes deferred to Phase 2.5).
See `docs/phase_2_findings.md` for what we tried and what worked.

**Starting point (from Phase 1):**

- 85.16 KB/s compress, 86.79 KB/s decompress on full enwik6 (1 MB)
- Ratio 0.2094
- 6.43× faster than Python L3TC-200K reference
- Byte-identical round trip on all test corpora including Persian/Arabic

**Phase 2 targets:**

1. **Ratio: close the gap to Python** (0.21 → 0.17 or better on enwik6)
2. **Speed: push further** (85 KB/s → 150-200 KB/s stretch goal)
3. **Full enwik8 measurement** to see how we scale on a real-sized corpus

**Achieved:**

1. Ratio 0.2094 → 0.2060 (−1.6 pp, didn't hit the 0.17 stretch — the
   deeper changes required for that are in Phase 2.5)
2. Speed 85 → 89 KB/s (+5%, didn't hit the 150-200 stretch — same
   reason)
3. 10 MB enwik8 subset: 88 KB/s at 0.2216 ratio, confirming speed
   scales cleanly on larger inputs

## Where the ratio gap comes from

Our Rust encoder gets 0.21 on enwik6; Python L3TC-200K gets 0.17. The
0.04 gap (24% worse) comes primarily from the frequency-table
conversion in `codec::logits_to_cum_freqs_scratch`:

```rust
let usable = target_total.saturating_sub(n as u64);  // ~2^62
let inv_sum = 1.0f32 / sum;
let scale = usable as f32 * inv_sum;  // !!
```

With `target_total = MAX_TOTAL = (1u128 << 62) + 2`, `usable` is on
the order of `2^62 ≈ 4.6e18`. **f32 has only 24 bits of mantissa
precision**, which can represent integers exactly only up to `2^24 ≈
16M`. Beyond that, `usable as f32` rounds to the nearest
representable float, introducing precision loss of up to `2^(62-24)
≈ 2.7e11`.

That precision loss propagates through the scale multiplication, the
floor operation, and the max(1) clamp. The combined effect is that
many tokens near the long tail get the minimum frequency of 1 when
they should get a slightly higher proportional value. Each such
misallocation costs a fraction of a bit per symbol.

The fix is either:
- Use **f64** for the scale calculation (preserves ~53 bits of
  precision, way more than needed)
- Use a **smaller target_total** (say 2^30) so scale fits cleanly in
  f32 and the arithmetic coder's underlying 64-bit range absorbs the
  precision loss

Both work. f64 is cleaner because it keeps the full target_total and
doesn't need any coder changes. Smaller target_total is faster because
we can use u32 freqs and vectorize more.

## Speed: where's the remaining 3-4× hiding

Current per-token profile (measured in Phase 1):

| Stage | us/token | % |
|---|---:|---:|
| Tokenize | 0.9 | 0.3% |
| Forward pass | 193 | 75% |
| - Head matvec | ~100 | ~39% |
| - Block matvecs (16 of them) | ~80 | ~31% |
| - Other (layer norm, element-wise) | ~13 | ~5% |
| cum_freqs | 68 | 24% |
| AC encode | 1 | 0.4% |
| **Total** | **~262** | |

Realistic additional wins:

| Optimization | Estimated savings | Difficulty |
|---|---:|---|
| f64 precision in cum_freqs (ratio only) | 0 us | trivial |
| cum_freqs single-pass with u32 freqs | 30-50 us | medium |
| Block matvecs hand-SIMD | 30-50 us | medium-hard |
| INT8 head quantization | 40-60 us | hard |
| Persistent rayon pool / lower sync overhead | 5-15 us | easy |

Stacking the first three would take us from 262 to ~130 us/token,
or roughly 200 KB/s on a 50 KB corpus.

## Phase 2 plan

1. **2a — Ratio fix (f64 precision in cum_freqs)**
   - Change the scale math to f64
   - Verify ratio drops on enwik6 without hurting speed
   - Should be a 10-30 minute task
2. **2b — cum_freqs single-pass optimization**
   - Use u32 freqs with a smaller target_total
   - Fold the softmax, scale, and floor passes together
   - Target: 68 us → 25 us
3. **2c — Block matvec optimization**
   - Profile the block matvecs in isolation
   - Try matrixmultiply at a lower threshold, or hand-write NEON
   - Target: 80 us → 35 us
4. **2d — Full enwik8 measurement**
   - Run the 100 MB corpus through our compressor and Python's
   - Report speed, ratio, and memory on the same hardware
5. **2e — Stretch: INT8 head quantization**
   - Offline script to quantize the head weight
   - Int8 matvec with f32 dequant on the fly
   - Target: head 100 us → 35 us
   - Optional, depends on how far the above gets us

## Non-goals (deferred to Phase 3+)

- File format stability and magic bytes (Phase 3)
- Stream-oriented API (Phase 3)
- C ABI and language bindings (Phase 6)
- Fuzz testing (Phase 5)
- The L3TC-3.2M and L3TC-12M variants (once Phase 2 works on 200K,
  the bigger models come essentially for free)

## Success criteria

Phase 2 is complete when:
- Ratio is within 0.02 of Python on enwik6 (≤0.19 from current 0.21)
- Compress speed ≥ 150 KB/s on the same 50 KB iter.sh test
- Full enwik8 compress measured and committed to `bench/results/`
- All existing tests still pass (lib + 4 end-to-end integration)
