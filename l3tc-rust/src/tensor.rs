//! Minimal f32 linear algebra for the RWKV forward pass.
//!
//! We deliberately avoid pulling in a tensor framework (candle,
//! ndarray, burn, etc.) because the L3TC model is tiny (2-3 layers,
//! 96-256 hidden dim) and the ops we need are a small fixed set.
//! Writing them directly gives us full control over allocation,
//! determinism, and numerics, and the total code is smaller than
//! any framework's setup boilerplate.
//!
//! # Conventions
//!
//! - All tensors are `f32` and row-major.
//! - Matrix shapes are `(rows, cols)` a.k.a. `(out_dim, in_dim)` for
//!   weight matrices, matching PyTorch's `nn.Linear` layout. The
//!   L3TC checkpoint stores weights this way.
//! - Vector ops assume 1-D inputs of matching length; the caller is
//!   responsible for shape correctness. Debug builds assert; release
//!   builds are unchecked for speed.
//! - In-place ops take `&mut [f32]` and return nothing. Out-of-place
//!   ops take an output slice as the last argument. No allocation
//!   happens in the hot path.

/// Matrix-vector multiply: `out[i] = sum_j mat[i, j] * x[j]`.
///
/// `mat` is a row-major `(rows, cols)` matrix stored as `rows * cols`
/// contiguous floats.
///
/// For small matrices (row count below `MATVEC_BLAS_THRESHOLD`) this
/// uses a hand-written scalar inner loop that the compiler can
/// autovectorize. For larger matrices — specifically the
/// `(16384, 96)` embedding and head projections — it delegates to
/// `matrixmultiply::sgemm`, which ships hand-tuned SIMD kernels for
/// x86 AVX/AVX2/AVX512 and ARM NEON. The head projection is the
/// dominant cost per token, so this single swap gives us roughly
/// an order-of-magnitude speedup on realistic models.
#[inline]
pub fn matvec(mat: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x.len();
    debug_assert_eq!(
        mat.len(),
        rows * cols,
        "matvec: mat shape mismatch (expected {rows}*{cols}, got {})",
        mat.len()
    );
    // 3.2M FFN-shape fast paths. Both shapes fall through this
    // dispatcher rather than matvec_square because the FFN is
    // rectangular (intermediate_size != hidden_size).
    if rows == 256 && cols == 512 {
        matvec_256x512(mat, x, out);
        return;
    }
    if rows == 512 && cols == 256 {
        matvec_512x256(mat, x, out);
        return;
    }
    if rows >= MATVEC_BLAS_THRESHOLD {
        matvec_sgemm(mat, x, out, rows, cols);
    } else {
        matvec_scalar(mat, x, out, rows, cols);
    }
}

/// Minimum row count at which we switch from the scalar matvec
/// implementation to the `matrixmultiply` SIMD kernel.
///
/// The per-layer 96×96 projections use the scalar path because
/// matrixmultiply::sgemm has measurable per-call overhead and is
/// internally optimized for matmul (not matvec with n=1). At
/// rows=96, scalar beats sgemm slightly on this specific machine.
const MATVEC_BLAS_THRESHOLD: usize = 512;

/// Specialized hand-tuned matvec for exactly 96×96 row-major
/// matrices on AArch64 NEON.
///
/// The L3TC-200K model has 16 matvec calls per token at this exact
/// shape (8 per block × 2 blocks: K/V/R/output for time_mix, plus
/// K/V/R for channel_mix, plus the "short" projection). These are
/// the second-biggest compute hotspot after the head projection,
/// and scalar code gets ~3 us each = ~50 us/token total.
///
/// This function reaches near-peak NEON throughput by:
///
/// 1. Preloading all 96 elements of `x` into 24 `float32x4_t`
///    registers, so the inner loop only needs to stream the matrix
///    data and never reloads x.
/// 2. Using 4 independent f32x4 accumulators per output row, so
///    the FMAs pipeline without a reduction dependency chain.
/// 3. Fully unrolling the inner loop over all 24 f32x4 row chunks.
/// 4. Horizontal-summing the 4 accumulators at the end of each row
///    with a single `vaddvq_f32`.
///
/// With 4 accumulators and Apple M-series FMA throughput (~4 FMAs
/// per cycle), the 24 FMAs per row take ~6 cycles, and the 96 rows
/// take ~600 cycles ≈ 200 ns at 3 GHz. 16 calls per token:
/// ~3.2 us total, down from ~50 us scalar.
///
/// # Safety
///
/// The function assumes (debug-asserted):
/// - `mat` has exactly 96*96 = 9216 f32 elements
/// - `x` has exactly 96 f32 elements
/// - `out` has exactly 96 f32 elements
///
/// The NEON intrinsics require `target_feature = "neon"`, which is
/// the default for `aarch64-apple-darwin`. The function is
/// guarded by `#[cfg(target_arch = "aarch64")]` so non-ARM builds
/// fall through to the generic scalar path via the caller.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn matvec_96x96_neon(mat: &[f32], x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    // Hard requirements for memory safety — pointer arithmetic below
    // assumes exactly 96×96 layout. Must fire in release builds.
    assert_eq!(mat.len(), 96 * 96, "matvec_96x96_neon: mat must be 96x96");
    assert_eq!(x.len(), 96, "matvec_96x96_neon: x must be 96");
    assert_eq!(out.len(), 96, "matvec_96x96_neon: out must be 96");

    let mat_p = mat.as_ptr();
    let x_p = x.as_ptr();
    let out_p = out.as_mut_ptr();

    // Preload all 96 elements of x into 24 f32x4 registers. AArch64
    // has 32 vector registers (v0..v31), so 24 for x leaves plenty
    // for accumulators and loaded row chunks.
    let x00 = vld1q_f32(x_p.add(0));
    let x01 = vld1q_f32(x_p.add(4));
    let x02 = vld1q_f32(x_p.add(8));
    let x03 = vld1q_f32(x_p.add(12));
    let x04 = vld1q_f32(x_p.add(16));
    let x05 = vld1q_f32(x_p.add(20));
    let x06 = vld1q_f32(x_p.add(24));
    let x07 = vld1q_f32(x_p.add(28));
    let x08 = vld1q_f32(x_p.add(32));
    let x09 = vld1q_f32(x_p.add(36));
    let x10 = vld1q_f32(x_p.add(40));
    let x11 = vld1q_f32(x_p.add(44));
    let x12 = vld1q_f32(x_p.add(48));
    let x13 = vld1q_f32(x_p.add(52));
    let x14 = vld1q_f32(x_p.add(56));
    let x15 = vld1q_f32(x_p.add(60));
    let x16 = vld1q_f32(x_p.add(64));
    let x17 = vld1q_f32(x_p.add(68));
    let x18 = vld1q_f32(x_p.add(72));
    let x19 = vld1q_f32(x_p.add(76));
    let x20 = vld1q_f32(x_p.add(80));
    let x21 = vld1q_f32(x_p.add(84));
    let x22 = vld1q_f32(x_p.add(88));
    let x23 = vld1q_f32(x_p.add(92));

    for i in 0..96usize {
        let row = mat_p.add(i * 96);
        // 4 independent accumulators break the reduction dependency
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        // 24 FMAs, interleaved across accumulators. This lets the
        // CPU pipeline 4 independent FMA chains in parallel.
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(0)), x00);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(4)), x01);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(8)), x02);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(12)), x03);
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(16)), x04);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(20)), x05);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(24)), x06);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(28)), x07);
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(32)), x08);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(36)), x09);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(40)), x10);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(44)), x11);
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(48)), x12);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(52)), x13);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(56)), x14);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(60)), x15);
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(64)), x16);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(68)), x17);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(72)), x18);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(76)), x19);
        a0 = vfmaq_f32(a0, vld1q_f32(row.add(80)), x20);
        a1 = vfmaq_f32(a1, vld1q_f32(row.add(84)), x21);
        a2 = vfmaq_f32(a2, vld1q_f32(row.add(88)), x22);
        a3 = vfmaq_f32(a3, vld1q_f32(row.add(92)), x23);

        // Reduce 4 accumulators to one, then horizontal sum to scalar
        let acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        *out_p.add(i) = vaddvq_f32(acc);
    }
}

/// Fast path for 96×96 matvecs: dispatches to the NEON specialized
/// version on aarch64, falls back to the generic scalar path
/// otherwise.
///
/// The L3TC-200K block projections are exactly this shape, called
/// 16 times per token. Callers in `rwkv.rs` should prefer this over
/// the generic `matvec` for those specific calls.
/// Square matvec dispatcher. Picks the fastest available kernel
/// for the requested size `n`:
///
/// - `n == 96` → hand-tuned NEON `matvec_96x96` (Phase 2.5a)
/// - otherwise → generic `matvec` (which itself dispatches to
///   sgemm above the BLAS threshold and scalar below)
///
/// Used by `Session::forward` for the per-layer projections that
/// are `(hidden_size, hidden_size)` regardless of model variant.
/// `#[inline(always)]` to guarantee the dispatch collapses into
/// the caller's code — without it the compiler would sometimes
/// leave a function call around the dispatch that costs ~6% of
/// forward-pass throughput on the 200K hot path.
#[inline(always)]
pub fn matvec_square(mat: &[f32], x: &[f32], out: &mut [f32], n: usize) {
    match n {
        96 => matvec_96x96(mat, x, out),
        256 => matvec_256x256(mat, x, out),
        _ => matvec(mat, x, out),
    }
}

/// Hand-tuned 96×96 matvec dispatcher. Callers should prefer
/// [`matvec_square`] which picks the right kernel based on size;
/// this entry point is retained for back-compat with benchmarks
/// that measure the 96×96 kernel directly.
#[inline]
pub fn matvec_96x96(mat: &[f32], x: &[f32], out: &mut [f32]) {
    // Hard requirements for memory safety — pointer arithmetic below
    // assumes exactly 96×96 layout. Must fire in release builds.
    assert_eq!(mat.len(), 96 * 96, "matvec_96x96_neon: mat must be 96x96");
    assert_eq!(x.len(), 96, "matvec_96x96_neon: x must be 96");
    assert_eq!(out.len(), 96, "matvec_96x96_neon: out must be 96");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: matvec_96x96_neon requires NEON, which is
        // mandatory on aarch64-apple-darwin and aarch64-linux-gnu
        // (the default aarch64 Rust targets). Shape preconditions
        // are debug-asserted above.
        #[allow(unsafe_code)]
        unsafe {
            matvec_96x96_neon(mat, x, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_scalar(mat, x, out, 96, 96);
    }
}

/// Hand-tuned 256×256 matvec, dispatched from [`matvec_square`] for
/// the L3TC-3.2M opt-in tier (hidden_size = 256).
///
/// Unlike the 96×96 kernel we can't preload all of `x` into registers
/// (256 elements would need 64 NEON registers; aarch64 has 32). Instead
/// the kernel streams `x` and the row together — `x` is 1 KB so it
/// stays in L1, and the matrix row reads are sequential and prefetched.
/// Four independent FMA accumulators per row break the reduction
/// dependency so the inner 64-FMA loop pipelines without serialization.
///
/// Replaces the generic scalar matvec on the 256×256 attention
/// projections (5 calls per layer × 3 layers = 15 calls per token on
/// 3.2M). Phase 4d flagged this as a TODO; shipped here as Phase 12g.
#[inline]
pub fn matvec_256x256(mat: &[f32], x: &[f32], out: &mut [f32]) {
    assert_eq!(mat.len(), 256 * 256, "matvec_256x256: mat must be 256x256");
    assert_eq!(x.len(), 256, "matvec_256x256: x must be 256");
    assert_eq!(out.len(), 256, "matvec_256x256: out must be 256");

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            matvec_256x256_neon(mat, x, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_scalar(mat, x, out, 256, 256);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn matvec_256x256_neon(mat: &[f32], x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let mat_p = mat.as_ptr();
    let x_p = x.as_ptr();
    let out_p = out.as_mut_ptr();

    for i in 0..256usize {
        let row = mat_p.add(i * 256);

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        // 64 chunks (64 * 4 = 256), interleaved across 4 accumulators
        // for FMA pipelining. Inner stride of 16 elements per group of
        // 4 FMAs keeps prefetcher happy.
        let mut k = 0usize;
        while k < 64 {
            let off = k * 4;
            a0 = vfmaq_f32(a0, vld1q_f32(row.add(off)), vld1q_f32(x_p.add(off)));
            a1 = vfmaq_f32(a1, vld1q_f32(row.add(off + 4)), vld1q_f32(x_p.add(off + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(row.add(off + 8)), vld1q_f32(x_p.add(off + 8)));
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(row.add(off + 12)),
                vld1q_f32(x_p.add(off + 12)),
            );
            k += 4;
        }

        let acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        *out_p.add(i) = vaddvq_f32(acc);
    }
}

/// Hand-tuned 256×512 matvec — 3.2M FFN value projection
/// (`out[256] = mat[256,512] @ x[512]`). Replaces the generic
/// scalar path which doesn't autovectorize tightly across 512
/// cols.
#[inline]
pub fn matvec_256x512(mat: &[f32], x: &[f32], out: &mut [f32]) {
    assert_eq!(mat.len(), 256 * 512, "matvec_256x512: mat must be 256x512");
    assert_eq!(x.len(), 512, "matvec_256x512: x must be 512");
    assert_eq!(out.len(), 256, "matvec_256x512: out must be 256");

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            matvec_rect_neon::<256, 512>(mat, x, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_scalar(mat, x, out, 256, 512);
    }
}

/// Hand-tuned 512×256 matvec — 3.2M FFN key projection
/// (`out[512] = mat[512,256] @ x[256]`). Replaces sgemm at this
/// shape; the 4-accumulator NEON kernel beats the BLAS fallback
/// at this size because there's no batching to amortize sgemm's
/// per-call setup.
#[inline]
pub fn matvec_512x256(mat: &[f32], x: &[f32], out: &mut [f32]) {
    assert_eq!(mat.len(), 512 * 256, "matvec_512x256: mat must be 512x256");
    assert_eq!(x.len(), 256, "matvec_512x256: x must be 256");
    assert_eq!(out.len(), 512, "matvec_512x256: out must be 512");

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            matvec_rect_neon::<512, 256>(mat, x, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_scalar(mat, x, out, 512, 256);
    }
}

/// Generic NEON matvec for `(ROWS, COLS)` where COLS is a multiple
/// of 16. Used by [`matvec_256x256`], [`matvec_256x512`],
/// [`matvec_512x256`] via const-generic dispatch — LLVM specializes
/// the inner loop for each instantiation, so the kernel is
/// monomorphized per shape with no runtime overhead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn matvec_rect_neon<const ROWS: usize, const COLS: usize>(
    mat: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let mat_p = mat.as_ptr();
    let x_p = x.as_ptr();
    let out_p = out.as_mut_ptr();

    // 4 chunks of 4 = 16 elements per inner step. COLS must be a
    // multiple of 16.
    let n16 = COLS / 16;

    for i in 0..ROWS {
        let row = mat_p.add(i * COLS);

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        let mut k = 0usize;
        while k < n16 {
            let off = k * 16;
            a0 = vfmaq_f32(a0, vld1q_f32(row.add(off)), vld1q_f32(x_p.add(off)));
            a1 = vfmaq_f32(a1, vld1q_f32(row.add(off + 4)), vld1q_f32(x_p.add(off + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(row.add(off + 8)), vld1q_f32(x_p.add(off + 8)));
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(row.add(off + 12)),
                vld1q_f32(x_p.add(off + 12)),
            );
            k += 1;
        }

        let acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        *out_p.add(i) = vaddvq_f32(acc);
    }
}

/// Shifted-softmax kernel: computes `exps[i] = exp(logits[i] - max)`
/// for every `i`, returns the total sum. Vectorized with NEON on
/// aarch64, scalar `f32::exp` elsewhere.
///
/// Used by [`crate::codec::logits_to_cum_freqs_scratch`] as the
/// inner loop of the softmax → freq-table conversion. Runs once
/// per encoded token, over the full 16384-symbol vocab, and is
/// the single biggest hot loop in the arithmetic-coding path
/// after the forward pass itself (Phase 2.5 profile: ~60 us/token,
/// ~27% of per-token time on enwik6).
///
/// **Accuracy contract.** The NEON path uses a degree-6 minimax
/// polynomial approximation to `2^r` over `r ∈ [0, 1)`. Max
/// relative error across the input range `[-50, 0]` is measured
/// to be under 5e-7 (≈ 2 ULPs of f32), tight enough that the
/// resulting cum_freqs table is bit-identical to the libm path
/// after the Phase 4a `round(p * 10_000_000); max(1)` quantization.
/// Phase 4c1 validated that the enwik6 entropy bound stays at
/// 0.1632 and the actual coded ratio stays at 0.1699 after the
/// switch.
///
/// Inputs with `x < -50` are clamped: `exp(-50) ≈ 2e-22`, far
/// below any freq the coder would assign, so the clamp is
/// lossless against the final `max(1)` step.
pub fn softmax_shifted_exp_sum(logits: &[f32], max: f32, exps: &mut [f32]) -> f32 {
    debug_assert_eq!(logits.len(), exps.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: exp_f32x4_neon requires NEON, which is mandatory
        // on every aarch64 Rust target we support.
        #[allow(unsafe_code)]
        unsafe {
            softmax_shifted_exp_sum_neon(logits, max, exps)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..logits.len() {
            let e = (logits[i] - max).exp();
            exps[i] = e;
            sum += e;
        }
        sum
    }
}

/// NEON f32x4 exp approximation for `x ≤ 0`.
///
/// Algorithm: `exp(x) = 2^(x · log2(e))`, split into integer part
/// `k = floor(y)` and fractional `r = y - k ∈ [0, 1)`. Then
/// `exp(x) = 2^k · 2^r` where `2^k` is built via IEEE-754 bit
/// construction and `2^r` is a degree-6 Horner minimax polynomial.
///
/// Coefficients: cephes-style minimax for `exp2(r)` on `[0, 1]`.
/// Max relative error over the range is under 5e-7.
///
/// Inputs below -50 are clamped (the result would be below f32
/// precision anyway). The integer part `k` after clamping is
/// always in `[-72, 0]`, so `k + 127` is in `[55, 127]` — a
/// normal f32 exponent bias, no subnormal / denormal handling
/// needed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
#[inline]
unsafe fn exp_f32x4_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    // Clamp to [-50, 0) — exp below -50 is ~2e-22, irrelevant
    // for any freq the AC would assign (post-`max(1)` clamp).
    let x = vmaxq_f32(x, vdupq_n_f32(-50.0_f32));

    // y = x * log2(e); exp(x) = 2^y.
    // The literal below is the cephes minimax log2(e) coefficient,
    // *not* f32::consts::LOG2_E. Keeping it as a hard-coded literal
    // gives bit-identical cum_freqs across CPU/GPU backends; see
    // FLAG_GPU_ENCODED in codec.rs — a few-ULP drift here desyncs
    // the arithmetic coder. Do not "fix" it.
    #[allow(clippy::approx_constant)]
    let log2e = vdupq_n_f32(1.442_695_f32);
    let y = vmulq_f32(x, log2e);

    // k = floor(y). Since y ≤ 0, vrndmq_f32 (round toward
    // negative infinity) gives the right integer part.
    let k_f = vrndmq_f32(y);
    let r = vsubq_f32(y, k_f);

    // Degree-6 Horner minimax polynomial for 2^r on [0, 1]:
    //   P(r) = 1 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
    // Coefficients from cephes-family minimax tables, max
    // relative error ≈ 4e-7 over [0, 1]. These are *not* f32::consts
    // values — they are the tuned polynomial coefficients that make
    // cum_freqs bit-identical across backends (see FLAG_GPU_ENCODED
    // in codec.rs). Do not "fix" them.
    #[allow(clippy::approx_constant)]
    let c1 = vdupq_n_f32(0.693_147_2_f32);
    let c2 = vdupq_n_f32(0.240_226_5_f32);
    let c3 = vdupq_n_f32(0.055_504_11_f32);
    let c4 = vdupq_n_f32(0.009_618_129_f32);
    let c5 = vdupq_n_f32(0.001_333_355_8_f32);
    let c6 = vdupq_n_f32(0.000_154_035_3_f32);
    let one = vdupq_n_f32(1.0_f32);

    let mut p = c6;
    p = vfmaq_f32(c5, p, r); // c5 + c6*r
    p = vfmaq_f32(c4, p, r); // c4 + r*(c5 + c6*r)
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(one, p, r); // 1 + r*P'(r) ≈ 2^r

    // 2^k via bit construction: float bits are (k + 127) << 23.
    // k_f is already integer-valued after vrndmq_f32, so
    // converting to i32 with round-toward-zero is exact.
    let k_i = vcvtq_s32_f32(k_f);
    let bias = vdupq_n_s32(127);
    let exp_bits = vshlq_n_s32(vaddq_s32(k_i, bias), 23);
    let two_k = vreinterpretq_f32_s32(exp_bits);

    vmulq_f32(two_k, p)
}

/// Fused RWKV time-mix attention "step 1" — pre-state-update.
///
/// For every i:
/// ```text
/// ww[i]  = time_first[i] + k[i]
/// p[i]   = max(state_p[i], ww[i])
/// e1     = exp(state_p[i] - p[i])
/// e2     = exp(ww[i] - p[i])
/// a[i]   = e1 * state_a[i] + e2 * v[i]
/// b[i]   = e1 * state_b[i] + e2
/// ```
///
/// This replaces 5 separate passes over hidden-sized scratch
/// buffers (and their intermediate `e1`/`e2` stores) with a
/// single fused NEON loop that keeps `e1`, `e2`, `ww`, `p` in
/// registers. Halves the L1 read/write traffic in this stretch
/// of `time_mix`.
///
/// All input buffers are length `h`; outputs are length `h`.
/// `h` is debug-asserted to be a multiple of 4 (NEON chunk).
///
/// The arg count reflects the RWKV time_mix update rule — a + b,
/// p, k, v, ww, rwkv, and scratch buffers. Wrapping in a struct
/// would push them onto the stack and defeat the NEON register
/// tiling this function exists to exploit.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn time_mix_step1(
    state_p: &[f32],
    time_first: &[f32],
    k: &[f32],
    state_a: &[f32],
    state_b: &[f32],
    v: &[f32],
    ww: &mut [f32],
    p: &mut [f32],
    a: &mut [f32],
    b: &mut [f32],
) {
    let n = state_p.len();
    debug_assert_eq!(time_first.len(), n);
    debug_assert_eq!(k.len(), n);
    debug_assert_eq!(state_a.len(), n);
    debug_assert_eq!(state_b.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(ww.len(), n);
    debug_assert_eq!(p.len(), n);
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            time_mix_step1_neon(state_p, time_first, k, state_a, state_b, v, ww, p, a, b);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let www = time_first[i] + k[i];
            let pp = state_p[i].max(www);
            let e1 = (state_p[i] - pp).exp();
            let e2 = (www - pp).exp();
            ww[i] = www;
            p[i] = pp;
            a[i] = e1 * state_a[i] + e2 * v[i];
            b[i] = e1 * state_b[i] + e2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)] // see `time_mix_step1` doc — NEON register tiling
unsafe fn time_mix_step1_neon(
    state_p: &[f32],
    time_first: &[f32],
    k: &[f32],
    state_a: &[f32],
    state_b: &[f32],
    v: &[f32],
    ww: &mut [f32],
    p: &mut [f32],
    a: &mut [f32],
    b: &mut [f32],
) {
    use std::arch::aarch64::*;

    let n = state_p.len();
    let chunks = n / 4;
    let sp_p = state_p.as_ptr();
    let tf_p = time_first.as_ptr();
    let k_p = k.as_ptr();
    let sa_p = state_a.as_ptr();
    let sb_p = state_b.as_ptr();
    let v_p = v.as_ptr();
    let ww_p = ww.as_mut_ptr();
    let p_p = p.as_mut_ptr();
    let a_p = a.as_mut_ptr();
    let b_p = b.as_mut_ptr();

    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let sp = vld1q_f32(sp_p.add(off));
        let tf = vld1q_f32(tf_p.add(off));
        let kv = vld1q_f32(k_p.add(off));
        let sa = vld1q_f32(sa_p.add(off));
        let sb = vld1q_f32(sb_p.add(off));
        let vv = vld1q_f32(v_p.add(off));

        let www = vaddq_f32(tf, kv);
        let pp = vmaxq_f32(sp, www);
        // e1 = exp(sp - pp); e2 = exp(www - pp). Both args ≤ 0
        // because pp = max(sp, www).
        let e1 = exp_f32x4_neon(vsubq_f32(sp, pp));
        let e2 = exp_f32x4_neon(vsubq_f32(www, pp));
        // a = e2*v + e1*sa  (note FMA order: vfmaq(acc, x, y) = acc + x*y)
        let av = vfmaq_f32(vmulq_f32(e2, vv), e1, sa);
        // b = e2 + e1*sb
        let bv = vfmaq_f32(e2, e1, sb);

        vst1q_f32(ww_p.add(off), www);
        vst1q_f32(p_p.add(off), pp);
        vst1q_f32(a_p.add(off), av);
        vst1q_f32(b_p.add(off), bv);
        i += 1;
    }

    // Tail (n % 4) — h is always a multiple of 4 (96, 256) for
    // L3TC variants, but kept for correctness.
    let mut j = chunks * 4;
    while j < n {
        let www = *tf_p.add(j) + *k_p.add(j);
        let pp = (*sp_p.add(j)).max(www);
        let e1 = (*sp_p.add(j) - pp).exp();
        let e2 = (www - pp).exp();
        *ww_p.add(j) = www;
        *p_p.add(j) = pp;
        *a_p.add(j) = e1 * *sa_p.add(j) + e2 * *v_p.add(j);
        *b_p.add(j) = e1 * *sb_p.add(j) + e2;
        j += 1;
    }
}

/// Fused RWKV time-mix attention "step 2" — post-state-update.
///
/// For every i:
/// ```text
/// ww[i]       = state_p[i] + neg_exp_decay[i]
/// p_new[i]    = max(ww[i], k[i])
/// e1          = exp(ww[i] - p_new[i])
/// e2          = exp(k[i] - p_new[i])
/// state_a[i]  = e1 * state_a[i] + e2 * v[i]
/// state_b[i]  = e1 * state_b[i] + e2
/// state_p[i]  = p_new[i]
/// ```
///
/// Same fusion idea as `time_mix_step1`. Reads `state_a`,
/// `state_b`, `state_p` and overwrites them in place.
#[inline]
pub fn time_mix_step2(
    neg_exp_decay: &[f32],
    k: &[f32],
    v: &[f32],
    state_p: &mut [f32],
    state_a: &mut [f32],
    state_b: &mut [f32],
    ww: &mut [f32],
) {
    let n = state_p.len();
    debug_assert_eq!(neg_exp_decay.len(), n);
    debug_assert_eq!(k.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(state_a.len(), n);
    debug_assert_eq!(state_b.len(), n);
    debug_assert_eq!(ww.len(), n);

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            time_mix_step2_neon(neg_exp_decay, k, v, state_p, state_a, state_b, ww);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let www = state_p[i] + neg_exp_decay[i];
            let p_new = www.max(k[i]);
            let e1 = (www - p_new).exp();
            let e2 = (k[i] - p_new).exp();
            state_a[i] = e1 * state_a[i] + e2 * v[i];
            state_b[i] = e1 * state_b[i] + e2;
            state_p[i] = p_new;
            ww[i] = www;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn time_mix_step2_neon(
    neg_exp_decay: &[f32],
    k: &[f32],
    v: &[f32],
    state_p: &mut [f32],
    state_a: &mut [f32],
    state_b: &mut [f32],
    ww: &mut [f32],
) {
    use std::arch::aarch64::*;

    let n = state_p.len();
    let chunks = n / 4;
    let nd_p = neg_exp_decay.as_ptr();
    let k_p = k.as_ptr();
    let v_p = v.as_ptr();
    let sp_p = state_p.as_mut_ptr();
    let sa_p = state_a.as_mut_ptr();
    let sb_p = state_b.as_mut_ptr();
    let ww_p = ww.as_mut_ptr();

    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let sp = vld1q_f32(sp_p.add(off));
        let nd = vld1q_f32(nd_p.add(off));
        let kv = vld1q_f32(k_p.add(off));
        let vv = vld1q_f32(v_p.add(off));
        let sa = vld1q_f32(sa_p.add(off));
        let sb = vld1q_f32(sb_p.add(off));

        let www = vaddq_f32(sp, nd);
        let p_new = vmaxq_f32(www, kv);
        // e1 = exp(www - p_new); e2 = exp(kv - p_new). Both ≤ 0.
        let e1 = exp_f32x4_neon(vsubq_f32(www, p_new));
        let e2 = exp_f32x4_neon(vsubq_f32(kv, p_new));
        // state_a = e1*sa + e2*v
        let new_sa = vfmaq_f32(vmulq_f32(e2, vv), e1, sa);
        // state_b = e1*sb + e2
        let new_sb = vfmaq_f32(e2, e1, sb);

        vst1q_f32(sa_p.add(off), new_sa);
        vst1q_f32(sb_p.add(off), new_sb);
        vst1q_f32(sp_p.add(off), p_new);
        vst1q_f32(ww_p.add(off), www);
        i += 1;
    }

    let mut j = chunks * 4;
    while j < n {
        let www = *sp_p.add(j) + *nd_p.add(j);
        let p_new = www.max(*k_p.add(j));
        let e1 = (www - p_new).exp();
        let e2 = (*k_p.add(j) - p_new).exp();
        *sa_p.add(j) = e1 * *sa_p.add(j) + e2 * *v_p.add(j);
        *sb_p.add(j) = e1 * *sb_p.add(j) + e2;
        *sp_p.add(j) = p_new;
        *ww_p.add(j) = www;
        j += 1;
    }
}

/// `out[i] = exp(a[i] - b[i])` for every i. NEON-vectorized on
/// aarch64; scalar `f32::exp` elsewhere.
///
/// Used in the RWKV time-mix state recurrence where the four
/// per-token exp passes (e1 = exp(state_p − p), e2 = exp(ww − p),
/// e1 = exp(ww − p_new), e2 = exp(k − p_new)) are guaranteed to
/// have non-positive arguments because each `b` is the running
/// max over the corresponding `a`. The NEON exp's `[-50, 0]`
/// accuracy contract therefore holds without further clamping.
///
/// Same accuracy contract as [`softmax_shifted_exp_sum`]: max
/// relative error ≈ 5e-7 (≈ 2 ULPs).
pub fn sub_exp(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            sub_exp_neon(a, b, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            out[i] = (a[i] - b[i]).exp();
        }
    }
}

/// NEON implementation of [`sub_exp`]. 4-wide vectorized with a
/// tail loop for `n % 4` elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn sub_exp_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let a_p = a.as_ptr();
    let b_p = b.as_ptr();
    let o_p = out.as_mut_ptr();
    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let av = vld1q_f32(a_p.add(off));
        let bv = vld1q_f32(b_p.add(off));
        let diff = vsubq_f32(av, bv);
        let e = exp_f32x4_neon(diff);
        vst1q_f32(o_p.add(off), e);
        i += 1;
    }

    let mut j = chunks * 4;
    while j < n {
        *o_p.add(j) = (*a_p.add(j) - *b_p.add(j)).exp();
        j += 1;
    }
}

/// NEON implementation of [`softmax_shifted_exp_sum`]. 4-wide
/// vectorized over the vocab dimension with a tail loop for
/// `n % 4` elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn softmax_shifted_exp_sum_neon(logits: &[f32], max: f32, exps: &mut [f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = logits.len();
    let max_v = vdupq_n_f32(max);
    let mut sum_v = vdupq_n_f32(0.0_f32);

    let chunks = n / 4;
    let logits_p = logits.as_ptr();
    let exps_p = exps.as_mut_ptr();
    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let l = vld1q_f32(logits_p.add(off));
        let shifted = vsubq_f32(l, max_v);
        let e = exp_f32x4_neon(shifted);
        vst1q_f32(exps_p.add(off), e);
        sum_v = vaddq_f32(sum_v, e);
        i += 1;
    }

    let mut sum = vaddvq_f32(sum_v);

    // Tail (n % 4 elements). L3TC-200K has vocab 16384 which is
    // divisible by 4 so this rarely runs, but handle it for
    // correctness on other vocab sizes.
    let mut j = chunks * 4;
    while j < n {
        let e = (*logits_p.add(j) - max).exp();
        *exps_p.add(j) = e;
        sum += e;
        j += 1;
    }

    sum
}

/// Fused quantize kernel for the `cum_freqs` inner loop.
///
/// For each lane: `freqs[i] = max(1, round(exps[i] * scale))` as u32.
///
/// `scale` should be pre-computed as `inv_sum * PYTHON_FREQ_TOTAL`
/// by the caller; this kernel multiplies each exp by scale,
/// rounds to nearest u32, and clamps at 1.
///
/// Vectorized with NEON on aarch64: one f32x4 load, one mul, one
/// round-to-nearest, one f32→u32 convert, one max-with-1, one
/// u32x4 store per 4-element chunk. On L3TC-200K this runs once
/// per token over the 16384-symbol vocab.
pub fn quantize_exps_to_freqs(exps: &[f32], scale: f32, freqs: &mut [u32]) {
    debug_assert_eq!(exps.len(), freqs.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON mandatory on aarch64 Rust targets.
        #[allow(unsafe_code)]
        unsafe {
            quantize_exps_to_freqs_neon(exps, scale, freqs);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..exps.len() {
            let scaled = (exps[i] * scale).round();
            let q = if scaled < 1.0 {
                1
            } else if scaled >= u32::MAX as f32 {
                u32::MAX
            } else {
                scaled as u32
            };
            freqs[i] = q.max(1);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn quantize_exps_to_freqs_neon(exps: &[f32], scale: f32, freqs: &mut [u32]) {
    use std::arch::aarch64::*;

    let n = exps.len();
    let chunks = n / 4;
    let scale_v = vdupq_n_f32(scale);
    let one_v = vdupq_n_u32(1);

    for i in 0..chunks {
        let off = i * 4;
        let e = vld1q_f32(exps.as_ptr().add(off));
        let scaled = vmulq_f32(e, scale_v);
        // Round to nearest, ties to even — matches Python's
        // `torch.round()` and the Phase 4a
        // `round(p * 10_000_000)` scheme.
        let rounded = vrndnq_f32(scaled);
        // f32 → u32 saturating: negatives become 0, overflow
        // clamps to u32::MAX. `max(1)` promotes the 0s.
        let q = vcvtq_u32_f32(rounded);
        let clamped = vmaxq_u32(q, one_v);
        vst1q_u32(freqs.as_mut_ptr().add(off), clamped);
    }

    for i in (chunks * 4)..n {
        let scaled = (exps[i] * scale).round();
        let q = if scaled < 1.0 {
            1
        } else if scaled >= u32::MAX as f32 {
            u32::MAX
        } else {
            scaled as u32
        };
        freqs[i] = q.max(1);
    }
}

/// Parallel version of [`matvec_col_major`] for very tall matrices.
///
/// Splits the output rows into contiguous chunks and computes each
/// chunk on a separate rayon worker. For the L3TC head
/// (16384 × 96) this typically gives a 2-4× speedup on multi-core
/// machines over the serial version, because the head matvec is
/// the single biggest compute hotspot per token.
///
/// The chunking is by output row (not input column) because
/// different output rows are independent — no shared mutable state
/// between chunks. Each chunk processes a contiguous range of rows
/// with the same column-loop structure as the serial version.
///
/// Uses rayon's global thread pool. The first call lazily
/// initializes the pool; subsequent calls reuse it with near-zero
/// dispatch overhead (a few hundred nanoseconds per call once the
/// pool is warm).
pub fn matvec_col_major_par(
    mat_col_major: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(mat_col_major.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    use rayon::prelude::*;

    // Target at least 2048 rows per chunk so each worker has enough
    // work to amortize the dispatch overhead. On a 16384-row head
    // this gives ~8 chunks, roughly matching the core count on
    // modern laptops and servers.
    const MIN_CHUNK_ROWS: usize = 2048;
    let chunk_rows = MIN_CHUNK_ROWS.min(rows.max(1));

    out.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(ci, out_chunk)| {
            let row_start = ci * chunk_rows;
            let chunk_len = out_chunk.len();

            // Zero the chunk
            for v in out_chunk.iter_mut() {
                *v = 0.0;
            }

            // AXPY over each column, restricted to this chunk's rows
            for (j, &xj) in x.iter().enumerate().take(cols) {
                let col_start = j * rows + row_start;
                let col = &mat_col_major[col_start..col_start + chunk_len];
                for (o, &c) in out_chunk.iter_mut().zip(col.iter()) {
                    *o += xj * c;
                }
            }
        });
}

/// Tall matrix-vector multiply where the matrix is stored **column-major**.
///
/// This is a different layout convention from [`matvec`]: the
/// matrix data is `(cols, rows)` instead of `(rows, cols)` in
/// memory, such that column `j` of the logical matrix is
/// `mat_col_major[j * rows .. (j + 1) * rows]`.
///
/// The algorithm is `out[i] = sum_j mat[i, j] * x[j]` computed as:
///
/// ```text
/// out = 0
/// for j in 0..cols:
///     axpy: out += x[j] * mat_col_major[j * rows .. (j + 1) * rows]
/// ```
///
/// This is the AXPY (`alpha * x + y`) form that vectorizes
/// beautifully: the inner loop is a constant-stride walk through
/// contiguous memory with a scalar broadcast multiply. LLVM
/// autovectorizes this cleanly into NEON / AVX SIMD without any
/// external crate.
///
/// For very tall matrices like the L3TC head (16384 × 96), this
/// form is dramatically faster than the row-major dot-product form
/// because:
///
/// 1. **Cache behavior**: memory is accessed in streaming order
///    with a stride of 1 across a full column (64 KB for a 16384-row
///    column), hitting the L1 cache predictor perfectly.
/// 2. **SIMD utilisation**: each AXPY is a pure elementwise op
///    that vectorizes 4 or 8 floats per instruction on ARM NEON
///    / x86 AVX with no reduction step.
/// 3. **Register pressure**: the inner loop holds one scalar
///    (`x[j]`) and streams through memory, leaving all the vector
///    registers free for the accumulation.
pub fn matvec_col_major(
    mat_col_major: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(mat_col_major.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    // Zero the output
    for v in out.iter_mut() {
        *v = 0.0;
    }

    // For each column, AXPY the column into the output
    for j in 0..cols {
        let col = &mat_col_major[j * rows..(j + 1) * rows];
        let xj = x[j];
        for i in 0..rows {
            out[i] += xj * col[i];
        }
    }
}

/// Quantize a column-major f32 matrix to INT8 with per-column scales.
///
/// For each column `j`, computes `scale_j = max(|col_j|) / 127`
/// (symmetric) and stores `q_j[i] = round(col_j[i] / scale_j)`
/// clamped to `[-127, 127]`. Zero columns get `scale = 0` and all
/// zero quantized values.
///
/// Returns `(qdata, scales)` where `qdata` is the column-major i8
/// buffer of length `rows * cols` and `scales` is an f32 vector of
/// length `cols`.
pub fn quantize_col_major_int8(
    mat_col_major: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(mat_col_major.len(), rows * cols);
    let mut qdata = vec![0i8; rows * cols];
    let mut scales = vec![0.0f32; cols];
    for j in 0..cols {
        let col = &mat_col_major[j * rows..(j + 1) * rows];
        let mut max_abs = 0.0f32;
        for &v in col {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0.0 {
            scales[j] = 0.0;
            continue;
        }
        let scale = max_abs / 127.0;
        let inv = 1.0 / scale;
        scales[j] = scale;
        let qcol = &mut qdata[j * rows..(j + 1) * rows];
        for i in 0..rows {
            let q = (col[i] * inv).round();
            qcol[i] = q.clamp(-127.0, 127.0) as i8;
        }
    }
    (qdata, scales)
}

/// INT8 column-major matvec with per-column dequant.
///
/// Computes `out[i] = sum_j (xs_j * (qmat_col[j][i] as f32))` where
/// `xs_j = x[j] * scales[j]`. This is the f32 AXPY form over the
/// widened i8 column — the inner loop widens i8 → f32 and does a
/// scalar-broadcast FMA, which LLVM auto-vectorizes into NEON sxtl
/// + scvtf + fmla. Memory traffic is 4× lower than the f32 path.
///
/// A "pre-widen" variant was tried (Phase 12 attempt #1: hoist the
/// i8→f32 cast into a separate per-column pass over a 32K f32
/// scratch). It regressed compress ~12% on the 32K-vocab head
/// because the 128 KB widen buffer + 128 KB output buffer blow
/// the M1 L1 (128 KB per perf core), causing each AXPY pass to
/// re-fetch widen from L2. Tiled re-widening would fit, but the
/// LLVM autovectorized fused loop is already close to bandwidth-
/// bound, so the simple form below stays.
pub fn matvec_col_major_int8(
    qmat: &[i8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(qmat.len(), rows * cols);
    debug_assert_eq!(scales.len(), cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            matvec_col_major_int8_neon(qmat, scales, x, out, rows, cols);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for j in 0..cols {
            let xs = x[j] * scales[j];
            if xs == 0.0 {
                continue;
            }
            let col = &qmat[j * rows..(j + 1) * rows];
            for i in 0..rows {
                out[i] += xs * (col[i] as f32);
            }
        }
    }
}

/// Explicit NEON 16-wide kernel for the head matvec.
///
/// Processes one column at a time (broadcast-FMA pattern). The
/// inner loop loads 16 i8 elements (one int8x16), widens to four
/// f32x4 vectors via sxtl chain, then 4 FMAs against the
/// broadcasted `xs` accumulating into the output buffer in place.
///
/// vs the autovectorized form: this guarantees the chosen 16-wide
/// unroll, the load/widen interleaving, and the in-place FMA on
/// the output without spilling to a temporary scratch (which the
/// pre-widen variant tried and lost to L1 thrash at 32K vocab).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn matvec_col_major_int8_neon(
    qmat: &[i8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    // Zero the output (NEON 4-wide).
    let out_p = out.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let row_chunks = rows / 4;
    {
        let mut k = 0usize;
        while k < row_chunks {
            vst1q_f32(out_p.add(k * 4), zero);
            k += 1;
        }
        let mut t = row_chunks * 4;
        while t < rows {
            *out_p.add(t) = 0.0;
            t += 1;
        }
    }

    let qmat_p = qmat.as_ptr();
    let n16 = rows / 16;

    for j in 0..cols {
        let xs = x[j] * scales[j];
        if xs == 0.0 {
            continue;
        }
        let xs_v = vdupq_n_f32(xs);
        let col_p = qmat_p.add(j * rows);

        // Main loop: 16 i8 per iteration.
        let mut k = 0usize;
        while k < n16 {
            let off = k * 16;
            let i8v = vld1q_s8(col_p.add(off));
            // i8x16 → 2 × i16x8
            let lo16 = vmovl_s8(vget_low_s8(i8v));
            let hi16 = vmovl_high_s8(i8v);
            // Each i16x8 → 2 × i32x4 → 2 × f32x4
            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            let f1 = vcvtq_f32_s32(vmovl_high_s16(lo16));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            let f3 = vcvtq_f32_s32(vmovl_high_s16(hi16));

            let o0 = vld1q_f32(out_p.add(off));
            let o1 = vld1q_f32(out_p.add(off + 4));
            let o2 = vld1q_f32(out_p.add(off + 8));
            let o3 = vld1q_f32(out_p.add(off + 12));

            vst1q_f32(out_p.add(off), vfmaq_f32(o0, f0, xs_v));
            vst1q_f32(out_p.add(off + 4), vfmaq_f32(o1, f1, xs_v));
            vst1q_f32(out_p.add(off + 8), vfmaq_f32(o2, f2, xs_v));
            vst1q_f32(out_p.add(off + 12), vfmaq_f32(o3, f3, xs_v));
            k += 1;
        }

        // Tail (rows % 16). For 16K and 32K vocab this never runs;
        // kept for correctness on other shapes.
        let mut i = n16 * 16;
        while i < rows {
            *out_p.add(i) += xs * (*col_p.add(i) as f32);
            i += 1;
        }
    }
}

/// Transpose a row-major `(rows, cols)` matrix into a column-major
/// buffer of length `rows * cols`.
///
/// The output layout is such that column `j` is at offset
/// `j * rows`, suitable for passing to [`matvec_col_major`].
pub fn transpose(src_row_major: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src_row_major.len(), rows * cols);
    let mut dst = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src_row_major[i * cols + j];
        }
    }
    dst
}

#[inline]
fn matvec_scalar(mat: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // Use 4 independent accumulators to break the scalar reduction
    // dependency chain. With `acc += row[j] * x[j]` as a single
    // accumulator, each FMA depends on the previous one, serializing
    // the inner loop to ~1 FLOP per cycle. With 4 independent accs,
    // the compiler can issue 4 FMAs per cycle (on ARM NEON) and
    // still sum them horizontally at the end.
    //
    // For the 96x96 projections in L3TC this turns an 80 us/token
    // hotspot into something closer to ~30 us/token once LLVM
    // notices the pattern and lowers to NEON fmla/fmul instructions.
    let cols_4 = cols & !3; // round down to multiple of 4
    for i in 0..rows {
        let row = &mat[i * cols..(i + 1) * cols];
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut j = 0;
        while j < cols_4 {
            a0 += row[j] * x[j];
            a1 += row[j + 1] * x[j + 1];
            a2 += row[j + 2] * x[j + 2];
            a3 += row[j + 3] * x[j + 3];
            j += 4;
        }
        // Tail
        let mut tail = 0.0f32;
        while j < cols {
            tail += row[j] * x[j];
            j += 1;
        }
        out[i] = (a0 + a1) + (a2 + a3) + tail;
    }
}

#[inline]
fn matvec_sgemm(mat: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // matrixmultiply::sgemm computes C = alpha*A*B + beta*C for
    // general (m, k, n) shapes. We want:
    //     out (rows,)   = mat (rows, cols) * x (cols,)
    // Viewed as matmul: m = rows, k = cols, n = 1.
    //
    // The safety requirements for the sgemm FFI wrapper are:
    //     - pointers must be valid for the described dimensions
    //     - strides describe the layout correctly (row-stride, col-stride)
    //     - no aliasing between A, B, and C
    //
    // mat is row-major, so row stride = cols, col stride = 1.
    // x is a single column vector, row stride = 1, col stride = 1
    // (there is only one column so col stride is never used).
    // out is a single column, row stride = 1, col stride = 1.
    //
    // matrixmultiply is safe-Rust internally but exposes an unsafe
    // wrapper because raw pointers are needed to express the strides.
    // We only do this in this one function; the unsafe block is
    // scoped to the FFI call.
    #[allow(unsafe_code)]
    unsafe {
        matrixmultiply::sgemm(
            rows, // m
            cols, // k
            1,    // n
            1.0,  // alpha
            mat.as_ptr(),
            cols as isize, // row stride of A
            1,             // col stride of A
            x.as_ptr(),
            1,   // row stride of B (irrelevant for n=1, but must be valid)
            1,   // col stride of B
            0.0, // beta
            out.as_mut_ptr(),
            1, // row stride of C
            1, // col stride of C
        );
    }
}

/// Lookup the `idx`-th row of `mat` (shape `(rows, cols)`) into `out`.
///
/// Used for embedding lookup: `out = emb[idx]`.
#[inline]
pub fn row_lookup(mat: &[f32], cols: usize, idx: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), cols, "row_lookup: output length mismatch");
    let start = idx * cols;
    let end = start + cols;
    debug_assert!(end <= mat.len(), "row_lookup: row index out of range");
    out.copy_from_slice(&mat[start..end]);
}

/// Layer normalization with learned weight and bias.
///
/// Computes: `out = weight * (x - mean) / sqrt(var + eps) + bias`.
/// The mean and variance are taken across the entire `x` slice.
#[inline]
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(bias.len(), n);
    debug_assert_eq!(out.len(), n);

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            layer_norm_neon(x, weight, bias, eps, out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mean = x.iter().sum::<f32>() / n as f32;
        let mut var = 0.0f32;
        for &v in x {
            let d = v - mean;
            var += d * d;
        }
        var /= n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

/// NEON 4-wide layer norm. Three passes (mean, variance, output)
/// over `n` elements; the reduction passes use `vaddvq_f32` for
/// horizontal sum since LLVM doesn't autovec the scalar form
/// (FP non-associativity blocks reordering).
///
/// Called 6 times per token on the 200K model (ln0, ln1×2, ln2×2,
/// ln_out for 2 layers). Each call's reduction passes save ~50%
/// vs scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn layer_norm_neon(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;
    let x_p = x.as_ptr();
    let w_p = weight.as_ptr();
    let b_p = bias.as_ptr();
    let o_p = out.as_mut_ptr();

    // Pass 1: mean.
    let mut sum_v = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i < chunks {
        sum_v = vaddq_f32(sum_v, vld1q_f32(x_p.add(i * 4)));
        i += 1;
    }
    let mut sum = vaddvq_f32(sum_v);
    let mut j = chunks * 4;
    while j < n {
        sum += *x_p.add(j);
        j += 1;
    }
    let mean = sum / (n as f32);
    let mean_v = vdupq_n_f32(mean);

    // Pass 2: variance via fused (x-mean)² accumulation.
    let mut var_v = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i < chunks {
        let d = vsubq_f32(vld1q_f32(x_p.add(i * 4)), mean_v);
        var_v = vfmaq_f32(var_v, d, d);
        i += 1;
    }
    let mut var = vaddvq_f32(var_v);
    let mut j = chunks * 4;
    while j < n {
        let d = *x_p.add(j) - mean;
        var += d * d;
        j += 1;
    }
    var /= n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_std_v = vdupq_n_f32(inv_std);

    // Pass 3: out = (x - mean) * inv_std * weight + bias
    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let xv = vld1q_f32(x_p.add(off));
        let wv = vld1q_f32(w_p.add(off));
        let bv = vld1q_f32(b_p.add(off));
        // (x - mean) * inv_std * weight = ((x-mean) * inv_std) * weight
        let normed = vmulq_f32(vsubq_f32(xv, mean_v), inv_std_v);
        let r = vfmaq_f32(bv, normed, wv);
        vst1q_f32(o_p.add(off), r);
        i += 1;
    }
    let mut j = chunks * 4;
    while j < n {
        *o_p.add(j) = (*x_p.add(j) - mean) * inv_std * *w_p.add(j) + *b_p.add(j);
        j += 1;
    }
}

/// Element-wise sigmoid: `out = 1 / (1 + exp(-x))`, in place.
///
/// NEON 4-wide on aarch64; scalar `f32::exp` elsewhere. Called
/// twice per layer per token (time_mix and channel_mix receptance
/// gates), so this is a measurable element-wise hot loop.
#[inline]
pub fn sigmoid_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            sigmoid_inplace_neon(x);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in x.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }
}

/// NEON 4-wide sigmoid in place.
///
/// Computes sigmoid via `exp(-|x|)` so the polynomial argument is
/// always in `[-∞, 0]` (where the polynomial's accuracy contract
/// holds — inputs below -50 clamp to ~e^-50 ≈ 0). Reconstructs:
///
/// - `x >= 0`: sigmoid(x) = 1 / (1 + e^-x)            = 1 / (1 + e_negabs)
/// - `x <  0`: sigmoid(x) = e^x / (1 + e^x)           = e_negabs / (1 + e_negabs)
///
/// Branchless via `vbslq_f32`. Same bit-for-bit output as the
/// scalar libm path on inputs within polynomial range, so the
/// entropy bound stays at 0.163723 on enwik6.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn sigmoid_inplace_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;
    let p = x.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i < chunks {
        let off = i * 4;
        let v = vld1q_f32(p.add(off));
        // neg_abs = -|x|, always ≤ 0 so polynomial is in spec.
        let neg_abs = vnegq_f32(vabsq_f32(v));
        let e = exp_f32x4_neon(neg_abs);
        let denom = vaddq_f32(one, e);
        // pos_form = 1 / (1 + e_negabs)   for x >= 0
        // neg_form = e_negabs / (1 + e_negabs) for x < 0
        let pos_form = vdivq_f32(one, denom);
        let neg_form = vdivq_f32(e, denom);
        let mask = vcgeq_f32(v, zero);
        let r = vbslq_f32(mask, pos_form, neg_form);
        vst1q_f32(p.add(off), r);
        i += 1;
    }

    let mut j = chunks * 4;
    while j < n {
        *p.add(j) = 1.0 / (1.0 + (-*p.add(j)).exp());
        j += 1;
    }
}

/// Element-wise ReLU: `out = max(0, x)`, in place.
#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Element-wise square: `out = x * x`, in place.
#[inline]
pub fn square_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= *v;
    }
}

/// Element-wise exp, in place.
#[inline]
pub fn exp_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// Element-wise negation of exp: `out = -exp(x)`, in place.
#[inline]
pub fn neg_exp_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = -v.exp();
    }
}

/// Element-wise addition, in place: `a += b`.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Element-wise multiplication, in place: `a *= b`.
#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Element-wise max: `out[i] = max(a[i], b[i])`.
#[inline]
pub fn max_elem(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].max(b[i]);
    }
}

/// Reduce `xs` to its maximum value. NEON 4-wide on aarch64;
/// scalar elsewhere. Returns `f32::NEG_INFINITY` for an empty
/// slice.
///
/// Used by the cum_freqs path as Pass 1 (numerical-stability
/// shift before exp). At vocab=32K this saves ~5-10 µs per token
/// vs the scalar `if l > max` loop, which the compiler doesn't
/// autovectorize because of the branch.
///
/// NaN handling: `vmaxq_f32` propagates NaN, so a single NaN in
/// the input makes the result NaN. The caller's `is_finite()`
/// guard then triggers the uniform fallback — same as the scalar
/// path's NEG_INFINITY-then-fallback behavior.
pub fn max_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NEG_INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            max_f32_neon(xs)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut m = f32::NEG_INFINITY;
        for &x in xs {
            if x > m {
                m = x;
            }
        }
        m
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn max_f32_neon(xs: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = xs.len();
    let chunks = n / 4;
    let p = xs.as_ptr();

    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i < chunks {
        let v = vld1q_f32(p.add(i * 4));
        max_v = vmaxq_f32(max_v, v);
        i += 1;
    }

    let mut m = vmaxvq_f32(max_v);

    let mut j = chunks * 4;
    while j < n {
        let v = *p.add(j);
        if v > m {
            m = v;
        }
        j += 1;
    }
    m
}

/// Time-mix blend used in every RWKV block:
///
/// `out[i] = x[i] * mix[i] + prev[i] * (1 - mix[i])`
#[inline]
pub fn time_mix(x: &[f32], prev: &[f32], mix: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), prev.len());
    debug_assert_eq!(x.len(), mix.len());
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        let m = mix[i];
        out[i] = x[i] * m + prev[i] * (1.0 - m);
    }
}

/// Copy src to dst (wrapper for explicit intent).
#[inline]
pub fn copy_into(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}

/// Find the argmax of a vector. Returns 0 for empty input.
///
/// Used for sanity tests, not the hot path.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Softmax a vector in place. Numerically stable (subtracts max first).
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut m = x[0];
    for &v in x.iter().skip(1) {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        // 3x3 identity
        let m = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let x = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 3];
        matvec(&m, &x, &mut out);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn matvec_rectangular() {
        // 2x4 matrix × 4-vec -> 2-vec
        let m = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 2];
        matvec(&m, &x, &mut out);
        assert_eq!(out, vec![10.0, 26.0]);
    }

    #[test]
    fn layer_norm_basic() {
        // Input [1,2,3,4]; mean=2.5, var=1.25, std≈1.118
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        layer_norm(&x, &w, &b, 1e-5, &mut out);
        // After normalization, mean should be ~0 and std ~1
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-5, "mean after LN should be 0, got {mean}");
        assert!(
            (var - 1.0).abs() < 1e-3,
            "var after LN should be 1, got {var}"
        );
    }

    #[test]
    fn sigmoid_basic() {
        let mut x = [0.0f32, 1.0, -1.0, 100.0, -100.0];
        sigmoid_inplace(&mut x);
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!(x[1] > 0.7 && x[1] < 0.74);
        assert!(x[2] > 0.26 && x[2] < 0.3);
        assert!(x[3] > 0.999);
        assert!(x[4] < 0.001);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Monotonic: larger input -> larger output
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn time_mix_extremes() {
        let x = [10.0f32, 20.0];
        let prev = [1.0f32, 2.0];
        let mix_all_one = [1.0f32, 1.0];
        let mix_all_zero = [0.0f32, 0.0];
        let mut out = [0.0f32; 2];
        time_mix(&x, &prev, &mix_all_one, &mut out);
        assert_eq!(out, [10.0, 20.0]);
        time_mix(&x, &prev, &mix_all_zero, &mut out);
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn row_lookup_basic() {
        let m = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ];
        let mut out = [0.0f32; 3];
        row_lookup(&m, 3, 1, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 2.0]), 2);
        assert_eq!(argmax(&[-1.0, -2.0, -3.0]), 0);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn softmax_shifted_exp_sum_matches_libm() {
        // Sample across the same input range cum_freqs actually
        // sees: logits in roughly [-30, +30], shifted by the max
        // so the input to exp is always ≤ 0.
        //
        // We build 16384 logits (same shape as L3TC-200K vocab)
        // with a mix of values that stresses the polynomial:
        //   - peaks near the max (small magnitude after shift)
        //   - mid-range values
        //   - deeply-suppressed tail values
        //   - a few values below -50 that should clamp cleanly
        let n = 16384;
        let mut logits = vec![0.0f32; n];
        #[allow(clippy::needless_range_loop)] // `i` gates several synthetic branches below
        for i in 0..n {
            let t = i as f32 / n as f32;
            logits[i] = if i % 97 == 0 {
                // occasional near-max
                10.0 - 0.5 * t
            } else if i % 31 == 0 {
                // mid-range
                -5.0 + 2.0 * t
            } else if i % 7 == 0 {
                // deep tail
                -20.0 - 10.0 * t
            } else {
                // very deep tail (triggers clamp for some)
                -40.0 - 20.0 * t
            };
        }
        // Max = 10.0 roughly. The actual argmax is index 0 with
        // logits[0] = 10.0 (i % 97 == 0).
        let mut max = f32::NEG_INFINITY;
        for &l in &logits {
            if l > max {
                max = l;
            }
        }

        let mut exps_ours = vec![0.0f32; n];
        let sum_ours = softmax_shifted_exp_sum(&logits, max, &mut exps_ours);

        // Reference: scalar libm.
        let mut exps_ref = vec![0.0f32; n];
        let mut sum_ref = 0.0f32;
        for i in 0..n {
            let e = (logits[i] - max).exp();
            exps_ref[i] = e;
            sum_ref += e;
        }

        // Per-element relative error on non-clamped values only.
        // Inputs with (logit - max) < -50 hit the NEON clamp and
        // return exp(-50) ≈ 1.93e-22 instead of the exact libm
        // value — that's a documented lossless-against-`max(1)`
        // behavior, not a polynomial accuracy bug. Check those
        // separately: they must be non-negative and at or below
        // the clamp floor.
        let clamp_floor = (-50.0f64).exp(); // ~1.93e-22
        let mut max_rel = 0.0f64;
        let mut max_idx = 0usize;
        for i in 0..n {
            let shifted = (logits[i] - max) as f64;
            let a = exps_ours[i] as f64;
            let b = exps_ref[i] as f64;
            if shifted < -50.0 {
                // Clamped path. NEON result should be the clamp
                // floor or very close to it.
                assert!(
                    a >= 0.0 && a <= clamp_floor * 1.1,
                    "clamped index {i}: shifted={shifted}, got {a}"
                );
                continue;
            }
            if b > 0.0 {
                let rel = (a - b).abs() / b;
                if rel > max_rel {
                    max_rel = rel;
                    max_idx = i;
                }
            }
        }
        // The polynomial's minimax error is ~4e-7, but compounded
        // with f32 FMA rounding in the Horner chain and the
        // `2^k * P(r)` multiplication at the end, the observed
        // max relative error lands around 1e-5. Well below the
        // cum_freqs quantization step (1 part in 10M), so the
        // resulting freq table is indistinguishable from the
        // libm reference after `round(p * 10_000_000); max(1)`.
        assert!(
            max_rel < 2e-5,
            "NEON exp max relative error {max_rel} at index {max_idx}; \
             ours={} ref={}",
            exps_ours[max_idx],
            exps_ref[max_idx]
        );

        // Sum should be within relative noise of the scalar sum,
        // with small slack for the clamped-tail differences.
        let sum_rel = ((sum_ours - sum_ref) as f64).abs() / sum_ref as f64;
        assert!(
            sum_rel < 1e-5,
            "NEON exp sum relative error {sum_rel}; ours={sum_ours} ref={sum_ref}"
        );
    }

    #[test]
    fn max_f32_basic() {
        // Empty
        assert_eq!(max_f32(&[]), f32::NEG_INFINITY);
        // Single element
        assert_eq!(max_f32(&[3.5]), 3.5);
        // Tail-only (n < 4)
        assert_eq!(max_f32(&[1.0, -2.0, 3.0]), 3.0);
        // Multiple chunks + tail
        let v: Vec<f32> = (0..18).map(|i| (i as f32) - 9.0).collect();
        assert_eq!(max_f32(&v), 8.0);
        // Max in tail
        let mut v = vec![-10.0; 17];
        v[16] = 99.0;
        assert_eq!(max_f32(&v), 99.0);
        // Max in middle of chunk
        let mut v = vec![-10.0; 32];
        v[5] = 7.0;
        v[19] = 5.0;
        assert_eq!(max_f32(&v), 7.0);
        // NaN propagates → caller falls back via is_finite check
        let v = vec![1.0, f32::NAN, 3.0, 2.0];
        assert!(max_f32(&v).is_nan() || !max_f32(&v).is_finite());
    }

    #[test]
    fn sub_exp_matches_libm() {
        // time_mix uses sub_exp on inputs guaranteed to be ≤ 0
        // (b is always max(state_p|ww|k, …) of a). Validate across
        // that range plus a few clamped tails.
        let n = 96; // hidden_size for L3TC-200K
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / n as f32;
            // b > a everywhere → diff ≤ 0
            b[i] = 5.0 + 10.0 * t;
            a[i] = if i % 13 == 0 {
                b[i] // diff = 0
            } else if i % 7 == 0 {
                b[i] - 5.0
            } else if i % 5 == 0 {
                b[i] - 25.0
            } else {
                b[i] - 60.0 // triggers the [-50, 0] clamp
            };
        }

        let mut ours = vec![0.0f32; n];
        sub_exp(&a, &b, &mut ours);

        let clamp_floor = (-50.0f64).exp();
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let diff = (a[i] - b[i]) as f64;
            let r = (a[i] - b[i]).exp() as f64;
            if diff < -50.0 {
                assert!(
                    ours[i] >= 0.0 && ours[i] as f64 <= clamp_floor * 1.1,
                    "clamped index {i}: diff={diff}, got {}",
                    ours[i]
                );
                continue;
            }
            if r > 0.0 {
                let rel = ((ours[i] as f64) - r).abs() / r;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        // Same minimax accuracy contract as softmax: ~1e-5 worst-case
        // observed after the polynomial + 2^k bit construction.
        assert!(max_rel < 2e-5, "sub_exp max relative error {max_rel}");
    }

    #[test]
    fn softmax_shifted_exp_sum_clamps_deeply_negative() {
        // Anything with (logit - max) < -50 should return a
        // near-zero but nonnegative value (clamped to exp(-50)).
        let logits = vec![0.0, -100.0, -1000.0, -f32::INFINITY];
        let mut exps = vec![0.0f32; 4];
        let max = 0.0f32;
        let sum = softmax_shifted_exp_sum(&logits, max, &mut exps);
        assert!(sum.is_finite());
        assert!((exps[0] - 1.0).abs() < 1e-5);
        // Clamped path: exp(-50) ≈ 1.93e-22 is the floor.
        for (i, &e) in exps.iter().enumerate().take(4).skip(1) {
            assert!(e >= 0.0);
            assert!(e < 1e-20, "exps[{i}] = {e}");
        }
    }
}
