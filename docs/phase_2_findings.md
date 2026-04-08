# Phase 2 Findings

**Starting from Phase 1:** 85 KB/s compress, ratio 0.2094 on enwik6.

**Ending at Phase 2:** 89 KB/s compress, ratio 0.2060 on enwik6.

## Headline numbers

### Apples-to-apples on enwik6 (1 MB), full round-trip verified

| Compressor | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| **l3tc-rust (phase 2)** | **0.2060** | **0.090** | **0.090** |
| bzip2-9 | 0.2813 | 16.67 | 35.09 |
| xz-9e | 0.2907 | 3.77 | 52.39 |
| zstd-22 | 0.3001 | 4.34 | 125.17 |
| gzip-9 | 0.3558 | 23.04 | 151.49 |
| Python L3TC-200K (reference) | 0.1665 | 0.013 | — |

l3tc-rust has the **best compression ratio** in the table: 27% better
than bzip2-9, 41% better than zstd-22. It's also the slowest by a
wide margin; the closest classical compressor on speed (xz-9e) is
~42× faster.

### vs Python L3TC-200K

- **Speed:** 0.090 MB/s ÷ 0.013 MB/s = **6.9× faster** (Phase 1: 6.43×)
- **Ratio:** 0.2060 vs 0.1665 = **23.7% worse**

### Scaling on bigger inputs

| Corpus | Size | Ratio | Compress KB/s |
|---|---:|---:|---:|
| e6_50k | 50 KB | 0.1815 | 71.6 |
| enwik6 | 1 MB | 0.2060 | 89.8 |
| enwik8[:10M] | 10 MB | 0.2216 | 88.4 |

Speed is stable at ~88-90 KB/s on files ≥ 1 MB. Smaller files suffer
because segment-level parallelism can't fill all 8 cores.

Ratio gets slightly worse on bigger files because they contain more
non-ASCII content (interlanguage links, symbols), which triggers the
raw-fallback path and stores those fragments unmodified.

## What moved the numbers

### Ratio improvements

1. **Default segment size 2048 → 4096** (Phase 2a). A segment sweep
   showed ratio improvement plateaus around 4096-8192 on Wikipedia
   text. 4096 is the sweet spot: ~2% better ratio than 2048 with
   essentially the same throughput.
2. **f64 precision in cum_freqs scale** (Phase 2a). Didn't actually
   move the ratio (f32 precision was already enough), but kept for
   safety because `usable ≈ 2^62` is right at the edge of f32's
   integer range.

### Speed improvements

3. **Serial head matvec instead of parallel** (Phase 2a). Removed
   the `matvec_col_major_par` call in `Session::forward`. Rayon's
   thread-pool dispatch overhead exceeded the multi-threading
   savings when segment-level parallelism was already using all 8
   cores. +5-7% throughput.
4. **4-accumulator unrolled scalar matvec** (Phase 2a, not helpful).
   Changed the scalar matvec inner loop to use 4 independent f32
   accumulators to break the reduction dependency chain. LLVM was
   already doing this, so it had no measurable effect. Kept for
   code clarity.
5. **Thread-local Session pool** (Phase 2, reverted). Tried to
   amortize session allocation across segments via thread-local
   storage with an unsafe lifetime transmute. Per-segment
   allocation cost turned out to be much smaller than the compute
   cost, so the benefit was noise-level and not worth the unsafe.
   Reverted.

## What didn't work

Several things I expected to help didn't:

- **Parallel head matvec via rayon**: thread-pool dispatch overhead
  eats the gains at small workloads (~100 us per head matvec).
- **3-pass cum_freqs with separate freqs buffer**: the added
  memory pass dominated the theoretical vectorization gain. The
  single-pass version is measurably faster.
- **u32 freqs with CUM_TOTAL = 2^30**: ratio regressed badly
  because low-probability tokens couldn't get meaningful
  proportional freq values (everything near the tail rounded to 1).
- **MIN_RAW_FALLBACK_BYTES = 16 (vs 64)**: smaller raw-fallback
  segments created more segment headers, which outweighed the
  savings on raw bytes stored.
- **Larger segments (8192, 16384, 32768)**: diminishing ratio
  gains, small throughput losses from less segment-level
  parallelism. 4096 is the sweet spot.

## Where the remaining speed is hiding

Per-token profile (on a single segment in isolation, no parallelism):

| Stage | us/token |
|---|---:|
| Tokenize | ~1 |
| Forward pass | ~190 |
|   — Head matvec (serial col-major AXPY) | ~100 |
|   — 16 block matvecs (scalar with 4-acc unroll) | ~50 |
|   — Layer norms, time mix, element-wise ops | ~40 |
| cum_freqs (softmax + freq table) | ~80 |
| AC encode (arithmetic coder) | ~4 |
| **Total** | **~275** |

To hit a 10× improvement over Python (150 KB/s), we need to cut
per-core per-token time from ~275 us to ~130 us — still ~145 us to
shave.

The biggest levers remaining:

1. **INT8 quantization of the head weight.** Head is ~100 us, the
   single largest block. Quantized inference would bring it to
   ~35-50 us via faster memory bandwidth. Requires offline
   quantization script + INT8 matvec in Rust. Estimated savings:
   50-60 us/token.

2. **Hand-tuned NEON intrinsics for the 96x96 block matvecs.** At
   20 GFLOPS of f32 NEON throughput, 96x96 matvec should take
   0.5 us. Current scalar takes ~3 us each × 16 = ~50 us total.
   Hand-tuned could get this to ~8-10 us. Estimated savings:
   40 us/token.

3. **cum_freqs vectorization**. The main blocker is `f64 → u64`
   casting which doesn't vectorize on ARM NEON. A dual-precision
   scheme (compute in f32 → u32, then widen) could work, but
   requires careful handling of the dynamic range. Estimated
   savings: 40-60 us/token.

Stacking any two of these would likely hit the 5× goal.

## Test status

- Unit tests: 35/35 passing
- Integration tests: 4/4 passing incl. full enwik6 (1 MB)
- New: 10 MB enwik8 subset round-trip verified

## Known Phase 3+ work

- **File format stabilization**: magic bytes + version + content
  checksum. Current format is versioned but not checksummed.
- **INT8 head quantization**: biggest remaining speed win.
- **NEON intrinsics for block matvecs**: second biggest.
- **Adaptive segment size**: optimal segment size depends on
  file size (small files want 2048 for parallelism, large files
  want 4096-8192 for ratio). Could auto-tune.
- **Benchmark on Silesia corpus**: diverse file types (JPEG, XML,
  binary, text). Would stress-test the raw-fallback path.
- **Full enwik8 and enwik9 runs**: 100 MB and 1 GB respectively.
  Deferred because of the test time; estimated at ~19 minutes
  (enwik8) and ~3 hours (enwik9) based on the 10 MB measurement.
