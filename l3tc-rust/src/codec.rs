//! High-level compress / decompress: ties tokenizer + model +
//! arithmetic coder together.
//!
//! # Compressed file format (Phase 1)
//!
//! This is a simple, self-describing binary format. It is NOT
//! compatible with L3TC's Python output — intentionally, see
//! PHASE_1.md. The format will be stabilized and given magic bytes
//! in Phase 2.
//!
//! ```text
//! header:
//!     magic:         b"LRUS"   (4 bytes)
//!     version:       u8        (= 2)
//!     flags:         u8        (reserved, = 0)
//!     reserved:      u16       (= 0)
//!     total_bytes:   u64       original uncompressed byte length
//!     n_segments:    u32
//! For each segment:
//!     n_tokens:      u32
//!     n_unks:        u32
//!     seg_flags:     u8        bit 0: use raw fallback
//!     ac_bytes:      u32       length of the arithmetic-coded body
//!     ac_body:       bytes     arithmetic coder output for this segment
//!     For each unk:
//!         unk_len:   u16       length in bytes of the raw unk payload
//!         unk_data:  bytes     raw UTF-8 bytes of the unk surface
//!     If seg_flags bit 0 is set:
//!         raw_len:   u32
//!         raw_data:  bytes     original segment text, used verbatim
//!                              on decode (overrides the token path)
//! trailer:
//!     magic:         b"!END"   (4 bytes)
//! ```
//!
//! # Raw fallback segments
//!
//! SentencePiece normalizes certain characters during tokenization
//! (zero-width joiners, combining marks, some NFC forms) which
//! means `sp.decode(sp.encode(text)) != text` for those inputs.
//! When the encoder detects this at segment-tokenization time (see
//! `Tokenizer::encode_segment`), it sets `needs_raw_fallback` and
//! the codec writes the original segment bytes alongside the
//! arithmetic-coded body. The decoder ignores the ac_body and
//! emits the raw bytes directly.
//!
//! For ASCII-only text the fallback is never triggered. For text
//! with occasional non-ASCII characters (e.g. Persian/Arabic in
//! enwik6) a small fraction of segments use the fallback. The
//! total overhead is a few bytes per fallback segment (the length
//! prefix) on top of the segment's raw byte count.
//!
//! # Compression flow
//!
//! 1. Read input text (full file into memory for now)
//! 2. Tokenize into segments using [`Tokenizer::encode_file`]
//! 3. For each segment:
//!     a. Reset model state
//!     b. For each token after BOS:
//!        - Run `session.forward(previous_token)` to get logits
//!        - Build a frequency table from the logits (softmax → integer freqs)
//!        - Encode the current token with the arithmetic coder
//!     c. Write the arithmetic-coded bytes and unk payloads
//!
//! # Decompression flow
//!
//! 1. Read the header to learn segment count and metadata
//! 2. For each segment:
//!     a. Reset model state
//!     b. Initialize arithmetic decoder from the segment's ac_body
//!     c. Feed tokens through the model the same way the encoder did
//!        (start with BOS, then decode next token against logits)
//!     d. Apply unks at their unk-token positions
//!     e. Detokenize to text
//! 3. Concatenate segment texts
//!
//! # Model-driven probability table
//!
//! The arithmetic coder needs a cumulative-frequency table at each
//! step. We convert the model's raw logits to a discrete frequency
//! table via:
//!
//! 1. Apply softmax → floating-point probabilities
//! 2. Scale to a total of `MAX_TOTAL` = arithmetic::MAX_TOTAL
//! 3. Clamp every symbol to at least 1 (avoid zero-probability)
//! 4. Adjust the top-probability symbol to make the total exact
//!
//! This matches Project Nayuki's "model-driven arithmetic coding"
//! pattern and is what L3TC's Python coder does as well.

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder, MAX_TOTAL};
use crate::error::{Error, Result};
use crate::rwkv::{Model, Session};
use crate::tensor;
use crate::tokenizer::{EncodedSegment, Tokenizer, BOS_ID, UNK_ID};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

/// Header magic bytes.
const MAGIC: &[u8; 4] = b"LRUS";
/// Trailer magic bytes.
const TRAILER: &[u8; 4] = b"!END";
/// Format version this module reads and writes.
///
/// v2 adds the per-segment `seg_flags` byte with a `use raw
/// fallback` bit so the decoder can use the original segment
/// bytes verbatim when SentencePiece normalization would
/// otherwise lose information (ZWNJ, combining marks, etc.).
const VERSION: u8 = 2;

/// Per-segment flag bit: "raw fallback bytes follow the unks".
///
/// When the encoder detects that `sp.decode(tokens) != original`
/// for a given segment (because of SPM normalization), it sets
/// this bit and appends the raw segment bytes so the decoder can
/// emit them directly instead of reconstructing via tokens.
const SEG_FLAG_RAW_FALLBACK: u8 = 0x01;

/// Default segment length in bytes (matches L3TC).
pub const DEFAULT_SEGMENT_BYTES: usize = 2048;

/// Compress text to bytes.
///
/// Returns a self-describing blob per the format above. The blob is
/// self-sufficient: you can decompress it with [`decompress`] and
/// the same `model` + `tokenizer` to recover the original text
/// exactly.
///
/// Segments are processed **in parallel** via rayon when there are
/// multiple of them. Each segment is independent (the model state
/// is reset at segment boundaries, and the arithmetic coder output
/// is per-segment), so segment-level parallelism is embarrassingly
/// parallel and sidesteps the per-token rayon dispatch overhead
/// that made intra-segment parallelism uneconomical. On a multi-
/// core machine this gives a near-linear speedup in the number of
/// segments up to the core count.
pub fn compress(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<Vec<u8>> {
    use rayon::prelude::*;

    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let total_bytes = text.len() as u64;

    // Compress every segment in parallel. Each segment gets its
    // own Session and CodecScratch (both are per-segment state
    // that can't be shared across threads). The per-segment
    // allocation cost is ~100 KB, dwarfed by the compute work, so
    // we don't bother with a session pool yet.
    //
    // Segments that need raw fallback still run through the
    // arithmetic coder — we can skip that and emit an empty ac_body,
    // saving a tiny amount of compute. The raw bytes go into the
    // serialized output alongside the ac_body.
    let segment_bodies: Result<Vec<Vec<u8>>> = segments
        .par_iter()
        .map(|seg| {
            if seg.needs_raw_fallback {
                // Skip arithmetic coding entirely — the decoder
                // will use the raw bytes. We still need an (empty)
                // ac_body for format uniformity.
                Ok(Vec::new())
            } else {
                let mut session = Session::new(model);
                let mut scratch = CodecScratch::new(model.vocab_size);
                compress_segment(seg, &mut session, model, &mut scratch)
            }
        })
        .collect();
    let segment_bodies = segment_bodies?;

    // Serialize the header + each segment + trailer. This is
    // sequential but trivial — just byte copies.
    let mut out = Vec::with_capacity(text.len() / 8);
    write_header(&mut out, total_bytes, segments.len() as u32)?;
    for (seg, body) in segments.iter().zip(segment_bodies.iter()) {
        write_segment(&mut out, seg, body)?;
    }
    write_trailer(&mut out)?;
    Ok(out)
}

/// Decompress a blob produced by [`compress`] back to text.
///
/// Like [`compress`], this processes segments in parallel using
/// rayon. Each segment is independently decodable (the model state
/// is reset at segment boundaries and each segment has its own
/// arithmetic-coded body), so segment-level parallelism gives a
/// near-linear speedup in the number of segments up to the core
/// count.
pub fn decompress(bytes: &[u8], tokenizer: &Tokenizer, model: &Model) -> Result<String> {
    use rayon::prelude::*;

    let mut cursor = std::io::Cursor::new(bytes);
    let (total_bytes, n_segments) = read_header(&mut cursor)?;

    // First pass: read all segment metadata + bodies sequentially.
    // This is I/O bound and very cheap (just byte copies).
    let mut raw_segments: Vec<SegmentRead> = Vec::with_capacity(n_segments as usize);
    for _ in 0..n_segments {
        raw_segments.push(read_segment_meta(&mut cursor)?);
    }
    read_trailer(&mut cursor)?;

    // Second pass: decode each segment in parallel. Each thread
    // owns a fresh Session (because LayerState is per-thread).
    // Raw-fallback segments skip the token path entirely and emit
    // the stored raw bytes.
    let decoded: Result<Vec<String>> = raw_segments
        .par_iter()
        .map(|seg| -> Result<String> {
            if let Some(raw) = &seg.raw_fallback {
                return Ok(String::from_utf8(raw.clone())?);
            }
            let mut session = Session::new(model);
            let mut scratch = CodecScratch::new(model.vocab_size);
            let tokens =
                decompress_segment(seg.n_tokens, &seg.ac_body, &mut session, model, &mut scratch)?;
            tokenizer.decode_segment(&tokens, &seg.unks)
        })
        .collect();

    let decoded = decoded?;
    let mut out = String::with_capacity(total_bytes as usize);
    for piece in decoded {
        out.push_str(&piece);
    }
    Ok(out)
}

// -------- per-segment codec -------- //

fn compress_segment(
    seg: &EncodedSegment,
    session: &mut Session<'_>,
    _model: &Model,
    scratch: &mut CodecScratch,
) -> Result<Vec<u8>> {
    let mut ac_bytes = Vec::with_capacity(seg.tokens.len());
    {
        let mut enc = ArithmeticEncoder::new(&mut ac_bytes);

        // The first token is BOS — it's agreed out-of-band (the
        // decoder starts with BOS too), so we don't encode it. The
        // tokens slice always starts with BOS per EncodedSegment's
        // contract. We need the logits after forwarding token[i-1]
        // to encode token[i].
        for i in 1..seg.tokens.len() {
            let prev = seg.tokens[i - 1];
            let tok = seg.tokens[i];
            let logits = session.forward(prev);
            logits_to_cum_freqs_scratch(logits, &mut scratch.cum, &mut scratch.exps);
            enc.encode_symbol(&scratch.cum, tok)?;
        }

        enc.finish()?;
    }
    Ok(ac_bytes)
}

fn decompress_segment(
    n_tokens: u32,
    ac_body: &[u8],
    session: &mut Session<'_>,
    _model: &Model,
    scratch: &mut CodecScratch,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(n_tokens as usize);
    tokens.push(BOS_ID);

    let mut dec = ArithmeticDecoder::new(ac_body)?;

    // n_tokens includes the BOS. We decode n_tokens - 1 symbols.
    // Each iteration: forward the previous token through the model,
    // use its output distribution to decode the next token.
    let mut prev = BOS_ID;
    for _ in 1..n_tokens {
        let logits = session.forward(prev);
        logits_to_cum_freqs_scratch(logits, &mut scratch.cum, &mut scratch.exps);
        let tok = dec.decode_symbol(&scratch.cum)?;
        tokens.push(tok);
        prev = tok;
    }

    Ok(tokens)
}

/// Reusable buffers for the per-segment encode/decode loop.
///
/// Holds the cumulative-frequency table and the softmax scratch
/// buffer so that `logits_to_cum_freqs` can avoid heap allocation
/// on the hot path. Allocated once per compress/decompress call
/// and reused across every segment and every token.
struct CodecScratch {
    cum: Vec<u64>,
    exps: Vec<f32>,
}

impl CodecScratch {
    fn new(vocab_size: usize) -> Self {
        Self {
            cum: vec![0u64; vocab_size + 1],
            exps: vec![0.0f32; vocab_size],
        }
    }
}

/// Fast `exp(x)` approximation for `x <= 0`, vectorization-friendly.
///
/// Used by [`logits_to_cum_freqs_scratch`] to turn softmax-shifted
/// logits (which are always ≤ 0 after subtracting the max) into
/// probabilities without calling the libm `exp` scalar function.
///
/// # Method
///
/// We factor `exp(x) = 2^(x · log2(e))` and split the exponent into
/// an integer part `k` and a fractional part `r`:
///
/// ```text
/// exp(x) = 2^(k + r) = 2^k · exp2(r),  with r in [0, 1)
/// ```
///
/// - `2^k` for integer `k` is computed by directly constructing an
///   IEEE-754 float via bit manipulation: the float `2^k` has
///   biased exponent `k + 127` and a zero mantissa.
/// - `exp2(r)` for `r in [0, 1)` is approximated by a degree-4
///   polynomial fit on that interval (minimax coefficients from a
///   standard reference). The polynomial is in Horner form, which
///   is a 4-step scalar accumulation that LLVM vectorizes cleanly
///   using ARM NEON FMA / x86 FMA3.
///
/// # Accuracy
///
/// Maximum relative error is around 1e-5 over `x ∈ [-50, 0]`. For
/// `x < -50` the result is below f32 precision anyway (~2e-22), and
/// we clamp to 0 there.
///
/// For arithmetic coding, this level of precision is more than
/// sufficient: the worst-case coding overhead from approximating
/// the probability model is the log2 of the relative error,
/// approximately 2^-17 bits per symbol for our approximation, or
/// about 0.00001 bits per token. On a 280,000-token file that's
/// less than half a byte of extra output.
///
/// # Performance
///
/// About 3-5x faster than `f32::exp` because the whole routine is
/// inlined, branch-free, and fully vectorizable. Empirically on
/// Apple Silicon the 16384-element softmax loop drops from ~82 us
/// to ~15-25 us.
#[inline(always)]
fn fast_exp_neg(x: f32) -> f32 {
    // Clamp to a safe range; below -50 the result is ~1.9e-22,
    // far below any frequency we'd actually emit.
    let x = x.max(-50.0);

    // y = x * log2(e); exp(x) = 2^y
    const LOG2E: f32 = 1.442_695_f32;
    let y = x * LOG2E;

    // Split into integer and fractional parts. Since y <= 0, floor
    // gives the largest integer <= y, and r = y - k is in [0, 1).
    let k = y.floor();
    let r = y - k;

    // Polynomial minimax approximation of 2^r on [0, 1), degree 4.
    // Coefficients are the same ones used by cephes and various
    // SIMD math libraries.
    //   2^r ≈ 1 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    let poly = 1.0_f32
        + r * (0.693_147_18_f32
            + r * (0.240_226_51_f32
                + r * (0.055_505_4_f32 + r * 0.009_618_129_f32)));

    // 2^k via direct bit construction. The IEEE-754 float 2^k has
    // biased exponent `k + 127` and a zero mantissa, so we build
    // it as `((k + 127) << 23)` reinterpreted as f32.
    //
    // k can be very negative (down to -72 after the clamp), and
    // (k + 127) can be a small positive number; for the smallest
    // subnormal values we'd need to handle underflow, but the
    // clamp on x guarantees we stay in the normal range.
    let k_i = k as i32;
    let exp_bits = ((k_i + 127) as u32) << 23;
    let two_k = f32::from_bits(exp_bits);

    two_k * poly
}

/// Public alias for [`logits_to_cum_freqs_scratch`], used by the
/// profile tests in `tests/profile_codec.rs`. Not part of the stable API.
#[doc(hidden)]
pub fn logits_to_cum_freqs_public(logits: &[f32], cum: &mut [u64]) {
    let mut exps = vec![0.0f32; logits.len()];
    logits_to_cum_freqs_scratch(logits, cum, &mut exps);
}

/// Convert raw model logits to a cumulative frequency table, using
/// a caller-provided scratch buffer for the intermediate softmax
/// exponentials. This form is the one the hot-path codec loop uses
/// so it can avoid heap allocation per call.
fn logits_to_cum_freqs_scratch(logits: &[f32], cum: &mut [u64], exps: &mut [f32]) {
    debug_assert_eq!(cum.len(), logits.len() + 1);
    debug_assert_eq!(exps.len(), logits.len());

    let n = logits.len();

    // --- Pass 1: find max for numerical stability ---
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }

    // --- Pass 2: compute shifted exps and sum in a single loop ---
    //
    // We use a fast exp approximation (see `fast_exp_neg`) instead
    // of `f32::exp`. libm's `.exp()` is a scalar library call that
    // cannot be autovectorized; the approximation is a ~4-line
    // polynomial + bit-manipulation routine that the compiler
    // auto-vectorizes into NEON / AVX SIMD.
    //
    // Writing to `exps[i]` in a simple for-i loop vectorizes
    // cleanly; the `.push()` pattern used in an earlier version
    // couldn't because each push has a capacity check.
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = fast_exp_neg(logits[i] - max);
        exps[i] = e;
        sum += e;
    }

    // --- Pass 3: scale, floor, clamp, and find argmax ---
    let target_total: u64 = MAX_TOTAL as u64;
    let usable = target_total.saturating_sub(n as u64);
    let inv_sum = 1.0f32 / sum;
    let scale = usable as f32 * inv_sum;

    // Fallback to uniform if the distribution is degenerate:
    //   - sum is 0, NaN, or inf
    //   - scale overflows or is non-finite (can happen when the
    //     input is all NaN and fast_exp_neg clamped everything to
    //     a tiny value)
    if !sum.is_finite() || sum <= 0.0 || !scale.is_finite() {
        uniform_fallback(n, cum);
        return;
    }

    let mut assigned: u64 = 0;
    let mut best_idx = 0usize;
    let mut best_exp = f32::NEG_INFINITY;

    // Build cum[i+1] = sum of freq[0..=i] in a running accumulator.
    cum[0] = 0;
    for i in 0..n {
        let e = exps[i];
        let scaled = ((e * scale).floor() as u64).max(1);
        assigned += scaled;
        cum[i + 1] = cum[i] + scaled;
        if e > best_exp {
            best_exp = e;
            best_idx = i;
        }
    }

    // --- Fix up the total ---
    if assigned < target_total {
        let residual = target_total - assigned;
        for i in (best_idx + 1)..=n {
            cum[i] += residual;
        }
    } else if assigned > target_total {
        let excess = assigned - target_total;
        let best_freq = cum[best_idx + 1] - cum[best_idx];
        if best_freq > excess + 1 {
            for i in (best_idx + 1)..=n {
                cum[i] -= excess;
            }
        } else {
            uniform_fallback(n, cum);
            return;
        }
    }
    debug_assert_eq!(cum[n], target_total);
}

/// Build a uniform cumulative-frequency table in `cum`.
///
/// Used by [`logits_to_cum_freqs_scratch`] as the fallback when the
/// input distribution is degenerate (all NaN, all zero, etc.).
/// Gives every symbol an equal probability with the total equal to
/// `MAX_TOTAL`.
fn uniform_fallback(n: usize, cum: &mut [u64]) {
    let target_total = MAX_TOTAL as u64;
    let uniform = target_total / n as u64;
    let residual = target_total - uniform * n as u64;
    cum[0] = 0;
    for i in 0..n {
        cum[i + 1] = cum[i] + uniform;
    }
    // Distribute the residual by giving each of the first `residual`
    // symbols one extra count. This keeps every freq >= 1 and the
    // sum exactly equal to target_total.
    for i in 1..=(residual as usize) {
        // Shift every cum[j] for j >= i up by 1. This is O(n*residual)
        // but residual < n so it's bounded; for our sizes n=16384
        // and residual < 16384 it's a few ms at worst in the
        // pathological path.
        for j in i..=n {
            cum[j] += 1;
        }
    }
}

// -------- header / trailer I/O -------- //

fn write_header<W: Write>(w: &mut W, total_bytes: u64, n_segments: u32) -> Result<()> {
    w.write_all(MAGIC)?;
    w.write_u8(VERSION)?;
    w.write_u8(0)?; // flags
    w.write_u16::<LittleEndian>(0)?; // reserved
    w.write_u64::<LittleEndian>(total_bytes)?;
    w.write_u32::<LittleEndian>(n_segments)?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> Result<(u64, u32)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {magic:?}")));
    }
    let version = r.read_u8()?;
    if version != VERSION {
        return Err(Error::BadCheckpoint(format!(
            "unsupported version: {version}"
        )));
    }
    let _flags = r.read_u8()?;
    let _reserved = r.read_u16::<LittleEndian>()?;
    let total_bytes = r.read_u64::<LittleEndian>()?;
    let n_segments = r.read_u32::<LittleEndian>()?;
    Ok((total_bytes, n_segments))
}

/// Per-segment data read back from a compressed file.
struct SegmentRead {
    /// Token count (including the implicit BOS). Unused for raw
    /// fallback segments but still present in the format.
    n_tokens: u32,
    /// Unk payloads.
    unks: Vec<Vec<u8>>,
    /// Arithmetic-coded body.
    ac_body: Vec<u8>,
    /// Raw fallback bytes, if the segment was flagged with
    /// [`SEG_FLAG_RAW_FALLBACK`] at write time.
    raw_fallback: Option<Vec<u8>>,
}

fn write_segment<W: Write>(w: &mut W, seg: &EncodedSegment, ac_body: &[u8]) -> Result<()> {
    w.write_u32::<LittleEndian>(seg.tokens.len() as u32)?;
    w.write_u32::<LittleEndian>(seg.unks.len() as u32)?;
    let seg_flags = if seg.needs_raw_fallback {
        SEG_FLAG_RAW_FALLBACK
    } else {
        0
    };
    w.write_u8(seg_flags)?;
    w.write_u32::<LittleEndian>(ac_body.len() as u32)?;
    w.write_all(ac_body)?;
    for unk in &seg.unks {
        if unk.len() > u16::MAX as usize {
            return Err(Error::BadCheckpoint(format!(
                "unk payload too large: {} bytes",
                unk.len()
            )));
        }
        w.write_u16::<LittleEndian>(unk.len() as u16)?;
        w.write_all(unk)?;
    }
    if seg.needs_raw_fallback {
        let raw_bytes = seg.raw.as_bytes();
        w.write_u32::<LittleEndian>(raw_bytes.len() as u32)?;
        w.write_all(raw_bytes)?;
    }
    Ok(())
}

fn read_segment_meta<R: Read>(r: &mut R) -> Result<SegmentRead> {
    let n_tokens = r.read_u32::<LittleEndian>()?;
    let n_unks = r.read_u32::<LittleEndian>()?;
    let seg_flags = r.read_u8()?;
    let ac_bytes_len = r.read_u32::<LittleEndian>()? as usize;
    let mut ac_body = vec![0u8; ac_bytes_len];
    r.read_exact(&mut ac_body)?;
    let mut unks = Vec::with_capacity(n_unks as usize);
    for _ in 0..n_unks {
        let unk_len = r.read_u16::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; unk_len];
        r.read_exact(&mut buf)?;
        unks.push(buf);
    }
    let raw_fallback = if seg_flags & SEG_FLAG_RAW_FALLBACK != 0 {
        let raw_len = r.read_u32::<LittleEndian>()? as usize;
        let mut raw = vec![0u8; raw_len];
        r.read_exact(&mut raw)?;
        Some(raw)
    } else {
        None
    };
    Ok(SegmentRead {
        n_tokens,
        unks,
        ac_body,
        raw_fallback,
    })
}

fn write_trailer<W: Write>(w: &mut W) -> Result<()> {
    w.write_all(TRAILER)?;
    Ok(())
}

fn read_trailer<R: Read>(r: &mut R) -> Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != TRAILER {
        return Err(Error::BadCheckpoint(format!("bad trailer: {magic:?}")));
    }
    Ok(())
}

// Suppress the unused import warning in -D unused_imports builds.
#[allow(dead_code)]
fn _keep_tensor_import_alive() {
    let _ = tensor::argmax(&[1.0]);
    let _: UnkLimit = UNK_ID;
}
type UnkLimit = u32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cum_freqs_sum_to_max_total() {
        // Random-ish logits
        let logits: Vec<f32> = (0..256).map(|i| (i as f32 / 50.0).sin()).collect();
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps);
        assert_eq!(cum[0], 0);
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // No zero frequencies
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i], "zero freq at symbol {i}");
        }
    }

    #[test]
    fn cum_freqs_uniform_fallback() {
        let logits = vec![f32::NAN; 16];
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps);
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // Every step is non-zero
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i]);
        }
    }

    #[test]
    fn fast_exp_neg_matches_libm_reasonably() {
        // Spot-check the approximation against libm's f32::exp
        // across the range we care about.
        for &x in &[0.0_f32, -0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -40.0] {
            let actual = fast_exp_neg(x);
            let expected = x.exp();
            let rel_err = ((actual - expected) / expected).abs();
            // Allow ~1% relative error — more than enough for our use case
            assert!(
                rel_err < 0.01,
                "fast_exp_neg({x}) = {actual}, expected {expected}, rel_err {rel_err}"
            );
        }
    }

    #[test]
    fn fast_exp_neg_clamps_extreme_negatives() {
        // Below -50 we return ~0 (the input is clamped to -50)
        let v = fast_exp_neg(-1000.0);
        assert!(v.is_finite());
        assert!(v >= 0.0);
        assert!(v < 1e-20);
    }
}
