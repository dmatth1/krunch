//! High-level compress / decompress: ties tokenizer + model +
//! arithmetic coder together.
//!
//! # Compressed file format (Phase 3, v3)
//!
//! Simple, self-describing binary format. NOT compatible with
//! L3TC's Python output — intentionally, see PHASE_1.md. Stabilized
//! in Phase 3 with a CRC32 integrity trailer.
//!
//! ```text
//! header:
//!     magic:         b"LRUS"   (4 bytes)
//!     version:       u8        (= 3; reader also accepts 2)
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
//!     crc32:         u32 LE    CRC32 of everything above (v3 only)
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
//! 2. `freqs[i] = round(p[i] * 10_000_000)`
//! 3. `freqs[i] = max(freqs[i], 1)` (avoid zero-probability)
//! 4. Cumulative table is the prefix sum of `freqs`.
//!
//! This is intentionally bit-identical with the Python L3TC
//! reference (`vendor/L3TC/scripts/compressor.py` lines 273-276).
//! Earlier phases used a finer scale (~2^62) with `floor` rounding
//! and a residual fixup pass; that diverged from Python in three
//! ways simultaneously and was 4 percentage points worse on enwik6.
//! Phase 4a closes that gap by matching Python exactly.

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder, MAX_TOTAL};
use crate::error::{Error, Result};
use crate::rwkv::{Model, Session};
use crate::tensor;
use crate::tokenizer::{EncodedSegment, Tokenizer, BOS_ID, UNK_ID};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

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
/// Format version this module writes.
///
/// **v4** (Phase 4b1) packs the per-segment header fields as
/// LEB128 varints instead of fixed-width u32/u16. Typical token
/// counts and ac-body lengths fit in 1-2 bytes each, dropping
/// the per-segment header from 13 bytes to ~5 bytes (savings of
/// ~9 KB / 0.9 pp on enwik6). Per-unk lengths and the raw_len
/// field are also varints, saving another ~3 KB / 0.3 pp on
/// enwik6 (mostly from the 355 raw-fallback segments). The
/// file-level CRC32 trailer from v3 is unchanged.
///
/// v3 added the CRC32 of the entire payload appended as a u32
/// little-endian after the trailer. The reader still accepts v3
/// files (and v2 files without CRC) for at least one release
/// cycle so existing artifacts decode.
const VERSION: u8 = 4;

/// Previous format with v3 CRC32 trailer + fixed-width segment
/// headers. Decoder accepts these as long as we can detect them
/// from the version byte.
const VERSION_V3_COMPAT: u8 = 3;

/// Legacy version this decoder still accepts (no CRC, fixed
/// segment headers).
const VERSION_V2_COMPAT: u8 = 2;

/// Sentinel value stored in the header `n_segments` field to signal
/// "segment count not known at header-write time — read segments
/// until the trailer magic". Used by the streaming encoder
/// ([`encode_reader`]) so it can emit a valid header before knowing
/// how many segments it will produce.
const N_SEGMENTS_IMPLICIT: u32 = u32::MAX;

/// File-level flag bit: "this file is stored verbatim, not
/// tokenized".
///
/// Set on the header `flags` byte when [`encode_reader`] detects
/// that the input isn't valid UTF-8 (binary files, mixed-encoding
/// text, arbitrary data). The tokenizer and model path are
/// bypassed entirely; the payload between the header and trailer
/// is the exact original byte sequence, and the compressed file is
/// therefore slightly *larger* than the original (by the header
/// overhead of ~14 bytes plus the 4-byte CRC).
///
/// This is graceful degradation: it means l3tc-rust will produce
/// a valid, round-trippable output for any input bytes, not just
/// UTF-8 text. Downstream callers can check this flag to decide
/// whether the compressor was actually useful for their input.
const FLAG_RAW_STORE: u8 = 0x01;

/// File-level flag bit: "encoded by the GPU (Metal) backend".
///
/// Phase 13e finding: the Metal forward pass diverges from CPU NEON
/// by a few ULPs per layer. At borderline `round()` boundaries in
/// cum_freqs that shifts a handful of freq-table entries, which
/// desyncs the AC if encoder and decoder use different backends.
///
/// When this bit is set on the file header, the decoder MUST use
/// the same Metal backend or the decompressed output will be
/// garbage. CPU-only builds will refuse to decode such files with
/// a clear error message.
///
/// CPU-encoded files (this bit unset) decode on either backend.
///
/// `bin/l3tc.rs` independently reads the same bit (`0x02`) when
/// auto-detecting the backend so it doesn't have to depend on a
/// metal-only symbol — keep the two values in sync.
#[cfg(feature = "metal")]
pub(crate) const FLAG_GPU_ENCODED: u8 = 0x02;

/// Per-segment flag bit: "raw fallback bytes follow the unks".
///
/// When the encoder detects that `sp.decode(tokens) != original`
/// for a given segment (because of SPM normalization), it sets
/// this bit and appends the raw segment bytes so the decoder can
/// emit them directly instead of reconstructing via tokens.
const SEG_FLAG_RAW_FALLBACK: u8 = 0x01;

/// Default segment length in bytes.
///
/// L3TC's Python reference uses 2048. Empirically (Phase 12g sweep
/// on enwik6, clean-system 5-run mean):
///
/// | bytes | ratio  | compress KB/s | decompress KB/s |
/// |------:|-------:|--------------:|----------------:|
/// |  2048 | 0.1730 |           174 |             181 |
/// |  4096 | 0.1699 |           168 |             179 |
/// |  8192 | 0.1683 |           158 |             176 |
/// | 16384 | 0.1675 |           144 |             172 |
///
/// 4096 stays the default. Speed is a non-negotiable goal in
/// CLAUDE.md, and bigger segments lose ~6% compress per doubling
/// (rayon parallelism becomes load-imbalanced on 1MB inputs at
/// ≤125 segments × 10 cores). Users who want maximum ratio at
/// the cost of ~6% compress can pass `--segment-bytes 8192`.
pub const DEFAULT_SEGMENT_BYTES: usize = 4096;

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
    // own Session and CodecScratch. We briefly experimented with a
    // thread_local pool to avoid per-segment allocation, but the
    // Session allocation is cheap enough (~100 KB, dwarfed by the
    // compute work per segment) that pooling didn't move the
    // needle — and the pooled version required an unsafe lifetime
    // transmute that wasn't worth the risk.
    //
    // Segments that need raw fallback skip the arithmetic coder
    // entirely and emit an empty ac_body (the decoder uses the
    // raw bytes instead).
    let segment_bodies: Result<Vec<Vec<u8>>> = segments
        .par_iter()
        .map(|seg| {
            if seg.needs_raw_fallback {
                return Ok(Vec::new());
            }
            let mut session = Session::new(model);
            let mut scratch = CodecScratch::new(model.vocab_size);
            compress_segment(seg, &mut session, model, &mut scratch)
        })
        .collect();
    let segment_bodies = segment_bodies?;

    // Serialize the header + each segment + trailer. This is
    // sequential but trivial — just byte copies.
    let mut out = Vec::with_capacity(text.len() / 8);
    write_header(&mut out, total_bytes, segments.len() as u32, 0)?;
    for (seg, body) in segments.iter().zip(segment_bodies.iter()) {
        write_segment(&mut out, seg, body)?;
    }
    write_trailer(&mut out)?;
    // v3: append CRC32 of the entire payload written so far.
    let crc = crc32fast::hash(&out);
    out.write_u32::<LittleEndian>(crc)?;
    Ok(out)
}

/// Compress text to bytes using the Metal GPU backend (Phase 13e).
///
/// Same file format as [`compress`], with the `FLAG_GPU_ENCODED`
/// bit set in the header. Decompression must use the same Metal
/// backend (see the docstring on `FLAG_GPU_ENCODED` for why).
///
/// `batch_size` controls how many segments run in lockstep per
/// BatchedSession (Phase 13h). At `batch_size > 1` the per-token GPU
/// dispatch overhead amortizes across all active lanes, giving an
/// approximately `batch_size`× wall-clock throughput improvement on
/// inputs with multiple segments. Values 4-32 are typical; larger
/// values bring more amortization at the cost of wider GPU buffer
/// allocations.
#[cfg(feature = "metal")]
pub fn compress_with_metal(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
    batch_size: usize,
) -> Result<Vec<u8>> {
    use crate::backend::batched::compress_segments_batched;

    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let total_bytes = text.len() as u64;

    // Build per-segment AC bodies via the GPU path. Raw-fallback
    // segments bypass the model entirely (same behaviour as CPU).
    let raw_fallback_indices: Vec<bool> =
        segments.iter().map(|s| s.needs_raw_fallback).collect();
    let token_inputs: Vec<Vec<u32>> = segments
        .iter()
        .map(|s| if s.needs_raw_fallback { Vec::new() } else { s.tokens.clone() })
        .collect();

    let mut bodies = compress_segments_batched(model, &token_inputs, batch_size.max(1))?;
    // Replace raw-fallback bodies with empty (codec uses the raw
    // bytes instead of the AC body for those segments).
    for (i, is_raw) in raw_fallback_indices.iter().enumerate() {
        if *is_raw {
            bodies[i] = Vec::new();
        }
    }

    let mut out = Vec::with_capacity(text.len() / 8);
    write_header(&mut out, total_bytes, segments.len() as u32, FLAG_GPU_ENCODED)?;
    for (seg, body) in segments.iter().zip(bodies.iter()) {
        write_segment(&mut out, seg, body)?;
    }
    write_trailer(&mut out)?;
    let crc = crc32fast::hash(&out);
    out.write_u32::<LittleEndian>(crc)?;
    Ok(out)
}

/// Decompress a blob produced by [`compress_with_metal`] back to
/// text. Errors if the file's `FLAG_GPU_ENCODED` flag is set but
/// this build doesn't have the `metal` feature, OR if the flag is
/// unset (use [`decompress`] for CPU-encoded files instead).
///
/// Mirrors the structure of [`decompress`] but swaps per-segment
/// AC decoding to the BatchedSession path so the freq tables match
/// what `compress_with_metal` produced.
#[cfg(feature = "metal")]
pub fn decompress_with_metal(
    bytes: &[u8],
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<String> {
    use crate::backend::batched::decompress_segments_batched;

    // CRC + version check mirrors decompress_bytes().
    if bytes.len() < 5 {
        return Err(Error::BadCheckpoint("file too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!(
            "bad magic: {:?}",
            &bytes[..4]
        )));
    }
    let version = bytes[4];
    let body: &[u8] = match version {
        VERSION | VERSION_V3_COMPAT => {
            let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
            let stored = LittleEndian::read_u32(crc_bytes);
            let computed = crc32fast::hash(payload);
            if stored != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch: stored {stored:08x}, computed {computed:08x}"
                )));
            }
            payload
        }
        VERSION_V2_COMPAT => bytes,
        v => {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version: {v}"
            )));
        }
    };

    let mut cursor = std::io::Cursor::new(body);
    let (_total_bytes, n_segments, flags) = read_header(&mut cursor)?;
    if flags & FLAG_GPU_ENCODED == 0 {
        return Err(Error::BadCheckpoint(
            "decompress_with_metal called on a CPU-encoded file; \
             use decompress() instead"
                .into(),
        ));
    }
    if flags & FLAG_RAW_STORE != 0 {
        // Raw store doesn't depend on the model, just unwrap.
        let raw_end = body.len() - TRAILER.len();
        let raw = &body[HEADER_SIZE..raw_end];
        return String::from_utf8(raw.to_vec())
            .map_err(|e| Error::BadCheckpoint(format!("raw-store body is not utf-8: {e}")));
    }

    // Parse segments (varint format = v4).
    let mut raw_segments: Vec<SegmentRead> = Vec::with_capacity(n_segments as usize);
    for _ in 0..n_segments {
        raw_segments.push(read_segment_meta(&mut cursor)?);
    }
    read_trailer(&mut cursor)?;

    // Run all tokenized segments through the Metal decoder. Raw-
    // fallback segments are handled separately on CPU since they
    // don't go through the model.
    let mut tokenized_indices: Vec<usize> = Vec::new();
    let mut bodies: Vec<(Vec<u8>, u32, u32)> = Vec::new();
    for (i, seg) in raw_segments.iter().enumerate() {
        if seg.raw_fallback.is_none() {
            tokenized_indices.push(i);
            bodies.push((seg.ac_body.to_vec(), seg.n_tokens, BOS_ID));
        }
    }
    // Phase 13h + 13i: 16-lane lockstep matches the default for
    // compress_with_metal. The per-token GPU overhead amortizes
    // across all active lanes; 16 is the measured knee on 50 KB
    // enwik6 before idle-lane overhead starts costing more than
    // amortization saves.
    let decoded_tokens = decompress_segments_batched(model, &bodies, 16)?;

    // Reassemble text in segment order.
    let mut out = String::new();
    let mut decoded_iter = decoded_tokens.into_iter();
    for seg in &raw_segments {
        if let Some(raw) = &seg.raw_fallback {
            out.push_str(&String::from_utf8(raw.clone())?);
        } else {
            let tokens = decoded_iter
                .next()
                .ok_or_else(|| Error::BadCheckpoint("decoded segment count mismatch".into()))?;
            let segment_text = tokenizer.decode_segment(&tokens, &seg.unks)?;
            out.push_str(&segment_text);
        }
    }
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
/// Decompress into a `Vec<u8>`.
///
/// Works for both text (tokenized) and raw-store (binary) files —
/// the latter is the whole point of this entry point, since
/// [`decompress`] requires UTF-8 output. For raw-store files, the
/// tokenizer and model arguments are ignored (the payload is
/// returned verbatim).
pub fn decompress_bytes(
    bytes: &[u8],
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<Vec<u8>> {
    // Peek the CRC + version the same way `decompress` does.
    if bytes.len() < 5 {
        return Err(Error::BadCheckpoint("file too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {:?}", &bytes[..4])));
    }
    let version = bytes[4];
    let body: &[u8] = match version {
        VERSION | VERSION_V3_COMPAT => {
            let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
            let stored = LittleEndian::read_u32(crc_bytes);
            let computed = crc32fast::hash(payload);
            if stored != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch: stored {stored:08x}, computed {computed:08x}"
                )));
            }
            payload
        }
        VERSION_V2_COMPAT => bytes,
        v => {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version: {v}"
            )));
        }
    };

    // Peek the flags byte (offset 5) to decide raw-store vs
    // tokenized path without running the full read_header parser.
    let flags = body[5];
    if flags & FLAG_RAW_STORE != 0 {
        if body.len() < HEADER_SIZE + TRAILER.len() {
            return Err(Error::BadCheckpoint(
                "raw-store file truncated below minimum framing".into(),
            ));
        }
        let raw_end = body.len() - TRAILER.len();
        let raw = &body[HEADER_SIZE..raw_end];
        let trailer_seen = &body[raw_end..];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store file: {trailer_seen:?}"
            )));
        }
        return Ok(raw.to_vec());
    }

    // Tokenized path: fall back to the UTF-8 decompress and
    // re-convert. Cheap because the model/tokenizer work dominates.
    let text = decompress(bytes, tokenizer, model)?;
    Ok(text.into_bytes())
}

/// Decompress a blob produced by [`compress`] or [`encode_reader`]
/// back to text.
///
/// Requires the output to be valid UTF-8. Use [`decompress_bytes`]
/// for arbitrary-byte inputs (raw-store files produced from
/// binary data).
pub fn decompress(bytes: &[u8], tokenizer: &Tokenizer, model: &Model) -> Result<String> {
    use rayon::prelude::*;

    // Peek at the version byte to decide framing. v3 files end with
    // a 4-byte CRC32 trailer; v2 files do not. The version lives at
    // byte offset 4 (right after the 4-byte magic).
    if bytes.len() < 5 {
        return Err(Error::BadCheckpoint("file too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {:?}", &bytes[..4])));
    }
    let version = bytes[4];
    let body: &[u8] = match version {
        VERSION | VERSION_V3_COMPAT => {
            if bytes.len() < 4 {
                return Err(Error::BadCheckpoint("v3+ file missing CRC trailer".into()));
            }
            let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
            let stored_crc = LittleEndian::read_u32(crc_bytes);
            let computed = crc32fast::hash(payload);
            if stored_crc != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch: stored {stored_crc:08x}, computed {computed:08x}"
                )));
            }
            payload
        }
        VERSION_V2_COMPAT => bytes,
        v => {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version: {v}"
            )));
        }
    };

    // v4 uses varint segment headers; v2/v3 use fixed-width.
    // We branch via a bool inside the read loop because the
    // segment readers are generic over `R: Read` and so can't be
    // coerced to a concrete function pointer.
    let varint_segments = version == VERSION;

    let mut cursor = std::io::Cursor::new(body);
    let (total_bytes, n_segments, flags) = read_header(&mut cursor)?;

    // Raw-store mode: the payload between header and trailer is
    // the exact original bytes. Validate the trailer and return
    // them (as a String, requiring UTF-8 — callers that need
    // arbitrary bytes should use `decompress_bytes`).
    if flags & FLAG_RAW_STORE != 0 {
        // Body layout: header | raw | trailer. Raw length is
        // body.len() - HEADER_SIZE - TRAILER.len().
        if body.len() < HEADER_SIZE + TRAILER.len() {
            return Err(Error::BadCheckpoint(
                "raw-store file truncated below minimum framing".into(),
            ));
        }
        let raw_end = body.len() - TRAILER.len();
        let raw = &body[HEADER_SIZE..raw_end];
        let trailer_seen = &body[raw_end..];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store file: {trailer_seen:?}"
            )));
        }
        return String::from_utf8(raw.to_vec())
            .map_err(|e| Error::BadCheckpoint(format!("raw-store body is not utf-8: {e}")));
    }

    // First pass: read all segment metadata + bodies sequentially.
    // This is I/O bound and very cheap (just byte copies).
    //
    // If the header carries the `N_SEGMENTS_IMPLICIT` sentinel the
    // stream was produced by [`encode_reader`] and we don't know
    // how many segments to expect. Peek 4 bytes before each segment
    // and stop when they equal the trailer magic.
    let mut raw_segments: Vec<SegmentRead> = if n_segments == N_SEGMENTS_IMPLICIT {
        let mut out = Vec::new();
        loop {
            let pos = cursor.position() as usize;
            if body.len() - pos < 4 {
                return Err(Error::BadCheckpoint(
                    "truncated before trailer in implicit-count stream".into(),
                ));
            }
            if &body[pos..pos + 4] == TRAILER {
                break;
            }
            let seg = if varint_segments {
                read_segment_meta(&mut cursor)?
            } else {
                read_segment_meta_v3(&mut cursor)?
            };
            out.push(seg);
        }
        out
    } else {
        // Bound `n_segments` against the body so a malformed header
        // claiming `n_segments = u32::MAX` cannot force a multi-GB
        // Vec::with_capacity. The minimum on-disk size of one
        // segment is bounded below by `MIN_SEGMENT_BYTES` (1 byte
        // n_tokens varint + 1 byte n_unks varint + 1 byte flags +
        // 1 byte ac_bytes_len varint = 4 bytes for an empty
        // segment); anything larger than `body.len() / 4` is
        // structurally impossible and we treat it as malformed.
        const MIN_SEGMENT_BYTES: usize = 4;
        let body_cap = body.len() / MIN_SEGMENT_BYTES;
        if (n_segments as usize) > body_cap {
            return Err(Error::BadCheckpoint(format!(
                "header claims {n_segments} segments but body could fit at most {body_cap}",
            )));
        }
        let mut out = Vec::with_capacity(n_segments as usize);
        for _ in 0..n_segments {
            let seg = if varint_segments {
                read_segment_meta(&mut cursor)?
            } else {
                read_segment_meta_v3(&mut cursor)?
            };
            out.push(seg);
        }
        out
    };
    read_trailer(&mut cursor)?;
    // Silence the "may be mutated" lint when the explicit branch runs.
    let _ = &mut raw_segments;

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

/// Streaming compress from a reader to a writer.
///
/// Reads `src` in bounded batches, tokenizes each batch (splitting
/// only on newlines so segmentation stays consistent with the
/// in-memory path), compresses the batch's segments in parallel
/// with rayon, and writes the header/body/trailer to `dst`
/// incrementally. Peak memory is roughly
/// `BATCH_BYTES + rayon_workers * segment_bytes + segment bodies`,
/// independent of total input size.
///
/// The header's `n_segments` field is written as
/// [`N_SEGMENTS_IMPLICIT`] because the streaming writer doesn't
/// know the final segment count at header-write time. The decoder
/// recognises this sentinel and walks segments until it sees the
/// trailer magic.
///
/// Returns the total number of uncompressed input bytes consumed.
pub fn encode_reader<R: Read, W: Write>(
    mut src: R,
    dst: W,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<u64> {
    use rayon::prelude::*;

    /// Soft target for how many input bytes to buffer before
    /// segmenting + compressing a batch. Larger batches mean more
    /// segment-level parallelism per batch (better rayon
    /// utilisation) at the cost of higher peak memory. 4 MB is
    /// empirically enough to fill 8 cores without blowing past a
    /// ~10 MB RSS budget.
    const BATCH_BYTES: usize = 4 * 1024 * 1024;

    // Wrap the writer in a CRC-computing adapter so we can write
    // bytes once and update the hash in step.
    let mut dst = CrcWriter::new(dst);

    // Byte accumulator straddling batch boundaries. We only split
    // on '\n' so the carry holds the tail of the current read that
    // didn't end on a newline.
    let mut carry: Vec<u8> = Vec::with_capacity(BATCH_BYTES);
    let mut read_buf = vec![0u8; 64 * 1024];
    let mut total_in: u64 = 0;
    let mut eof = false;

    // Fill the first batch before deciding text vs raw-store mode.
    while carry.len() < BATCH_BYTES {
        let n = src.read(&mut read_buf).map_err(Error::Io)?;
        if n == 0 {
            eof = true;
            break;
        }
        carry.extend_from_slice(&read_buf[..n]);
    }

    // Probe the first batch for UTF-8 validity. If the batch is
    // valid UTF-8, or it only ends mid-codepoint (a legal situation
    // when the split happens inside a multi-byte char), we use the
    // text path. Otherwise we fall back to raw-store mode.
    let utf8_ok = match std::str::from_utf8(&carry) {
        Ok(_) => true,
        Err(e) => {
            // Incomplete final codepoint is fine — the carry boundary
            // will move when we read more. Real invalid bytes mean
            // the input isn't text at all.
            e.error_len().is_none()
                && std::str::from_utf8(&carry[..e.valid_up_to()]).is_ok()
        }
    };

    if !utf8_ok {
        // --- Raw-store mode: stream bytes verbatim. ---
        write_header(&mut dst, 0, 0, FLAG_RAW_STORE)?;
        // Flush the already-buffered first batch.
        dst.write_all(&carry).map_err(Error::Io)?;
        total_in += carry.len() as u64;
        // Pipe the rest of src straight through.
        loop {
            let n = src.read(&mut read_buf).map_err(Error::Io)?;
            if n == 0 {
                break;
            }
            dst.write_all(&read_buf[..n]).map_err(Error::Io)?;
            total_in += n as u64;
        }
        write_trailer(&mut dst)?;
        let (mut inner, crc) = dst.finish();
        inner.write_u32::<LittleEndian>(crc).map_err(Error::Io)?;
        return Ok(total_in);
    }

    // --- Text path: implicit-count streaming segments. ---
    //
    // total_bytes is unknown until the stream ends; we write 0 as a
    // placeholder. Consumers that need the exact count can derive
    // it from the reconstructed output. The CLI's --time path
    // doesn't rely on this field.
    write_header(&mut dst, 0, N_SEGMENTS_IMPLICIT, 0)?;

    while !eof || !carry.is_empty() {
        // Fill carry to BATCH_BYTES (or until EOF).
        while carry.len() < BATCH_BYTES && !eof {
            let n = src.read(&mut read_buf).map_err(Error::Io)?;
            if n == 0 {
                eof = true;
                break;
            }
            carry.extend_from_slice(&read_buf[..n]);
        }

        // Choose a cut point: the last '\n' in carry, so we never
        // break a line across batches (segmentation groups short
        // lines and splits long ones, and both behaviours depend
        // on seeing whole lines).
        let cut = if eof {
            carry.len()
        } else {
            match carry.iter().rposition(|&b| b == b'\n') {
                Some(idx) => idx + 1,
                None => {
                    // No newline in 4 MB of input — degenerate
                    // binary-ish content. Flush the whole carry to
                    // avoid pathological unbounded growth.
                    carry.len()
                }
            }
        };
        if cut == 0 {
            continue;
        }

        // Move the prefix out so we can stream the tail forward.
        let prefix: Vec<u8> = carry.drain(..cut).collect();
        total_in += prefix.len() as u64;
        let text = std::str::from_utf8(&prefix)
            .map_err(|e| Error::BadCheckpoint(format!("non-utf8 input: {e}")))?;

        let segments = tokenizer.encode_file(text, segment_bytes)?;

        // Compress this batch's segments in parallel. Each segment
        // gets its own Session + scratch so threads don't contend.
        let bodies: Result<Vec<Vec<u8>>> = segments
            .par_iter()
            .map(|seg| {
                if seg.needs_raw_fallback {
                    return Ok(Vec::new());
                }
                let mut session = Session::new(model);
                let mut scratch = CodecScratch::new(model.vocab_size);
                compress_segment(seg, &mut session, model, &mut scratch)
            })
            .collect();
        let bodies = bodies?;

        for (seg, body) in segments.iter().zip(bodies.iter()) {
            write_segment(&mut dst, seg, body)?;
        }
    }

    write_trailer(&mut dst)?;
    let (mut inner, crc) = dst.finish();
    inner.write_u32::<LittleEndian>(crc).map_err(Error::Io)?;
    Ok(total_in)
}

/// Streaming decompress from a reader to a writer.
///
/// Symmetric counterpart to [`encode_reader`]. The high-value case
/// is **raw-store files** (binary inputs) where the compressed
/// body is the same size as the original — slurping a multi-GB
/// raw-store file into memory just to copy it back out would
/// defeat the purpose. For raw-store streams this function pipes
/// bytes from `src` to `dst` with a constant 8-byte lookahead and
/// validates the CRC32 trailer at EOF.
///
/// **Tokenized files** are read in full into memory because the
/// compressed body is ~5× smaller than the output and decoding
/// any segment requires the model state, which doesn't usefully
/// stream. The output write itself is still incremental.
///
/// Returns the number of uncompressed output bytes written.
pub fn decode_writer<R: Read, W: Write>(
    mut src: R,
    mut dst: W,
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<u64> {
    // Read the fixed-size header up front so we can branch on
    // version and flags before deciding how to consume the rest.
    let mut header = [0u8; HEADER_SIZE];
    src.read_exact(&mut header).map_err(Error::Io)?;
    if &header[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!(
            "bad magic: {:?}",
            &header[..4]
        )));
    }
    let version = header[4];
    if version != VERSION && version != VERSION_V3_COMPAT && version != VERSION_V2_COMPAT {
        return Err(Error::BadCheckpoint(format!(
            "unsupported version: {version}"
        )));
    }
    let flags = header[5];
    // v3 introduced the CRC32 trailer; v4 keeps it. Only v2 lacks it.
    let has_crc = version == VERSION || version == VERSION_V3_COMPAT;

    if flags & FLAG_RAW_STORE != 0 {
        // --- Raw-store streaming path ---
        //
        // Layout: header(20) | raw(N) | trailer(4) | crc(4 if v3+).
        // We've consumed the header. Hash it, then stream the body
        // with an 8-byte lookahead so we never write the trailer or
        // CRC to dst.
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        let tail_size = if has_crc { 8 } else { 4 };
        let mut tail: Vec<u8> = Vec::with_capacity(tail_size + 64 * 1024);
        let mut buf = vec![0u8; 64 * 1024];
        let mut written: u64 = 0;
        loop {
            let n = src.read(&mut buf).map_err(Error::Io)?;
            if n == 0 {
                break;
            }
            tail.extend_from_slice(&buf[..n]);
            // Drain everything that's safely outside the lookahead.
            if tail.len() > tail_size {
                let drain_n = tail.len() - tail_size;
                let drained = &tail[..drain_n];
                hasher.update(drained);
                dst.write_all(drained).map_err(Error::Io)?;
                written += drain_n as u64;
                tail.drain(..drain_n);
            }
        }
        if tail.len() != tail_size {
            return Err(Error::BadCheckpoint(format!(
                "raw-store stream truncated: tail has {} bytes, expected {tail_size}",
                tail.len()
            )));
        }
        let trailer_seen = &tail[..4];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store stream: {trailer_seen:?}"
            )));
        }
        // Trailer bytes are part of the CRC payload.
        hasher.update(trailer_seen);
        if has_crc {
            let stored_crc = LittleEndian::read_u32(&tail[4..8]);
            let computed = hasher.finalize();
            if stored_crc != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch in raw-store stream: stored {stored_crc:08x}, computed {computed:08x}"
                )));
            }
        }
        return Ok(written);
    }

    // --- Tokenized path ---
    //
    // Slurp the remainder of src (it's at most ~25% of the output
    // text size) and reuse the existing in-memory decode. The
    // output write is then a single chunk; we could also stream
    // segment outputs as they decode, but parallelism in
    // decompress() makes that nontrivial without losing speed.
    let mut rest = Vec::new();
    src.read_to_end(&mut rest).map_err(Error::Io)?;
    let mut full = Vec::with_capacity(HEADER_SIZE + rest.len());
    full.extend_from_slice(&header);
    full.extend_from_slice(&rest);
    let payload = decompress_bytes(&full, tokenizer, model)?;
    dst.write_all(&payload).map_err(Error::Io)?;
    Ok(payload.len() as u64)
}

/// Write adapter that computes a CRC32 over every byte written.
///
/// Used by [`encode_reader`] so the streaming writer can produce
/// the same CRC32 trailer that [`compress`] appends in one shot.
struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
}

impl<W: Write> CrcWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn finish(self) -> (W, u32) {
        let crc = self.hasher.finalize();
        (self.inner, crc)
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Per-source byte breakdown returned by [`audit_compress`].
///
/// Used by the `l3tc audit` CLI subcommand to figure out where the
/// gap between actual coded bytes and the theoretical entropy bound
/// actually goes. Phase 4b debugging only — not part of the stable
/// API.
#[derive(Debug, Default, Clone)]
pub struct AuditStats {
    /// Original input size in bytes.
    pub input_bytes: u64,
    /// Number of segments produced by the tokenizer.
    pub n_segments: u64,
    /// Number of segments that took the raw-fallback path.
    pub n_raw_fallback_segments: u64,
    /// Total token count across all segments (including BOS per
    /// segment).
    pub n_tokens: u64,
    /// Number of (current_token, next_token) pairs the AC actually
    /// encodes — i.e. excludes the BOS that opens each segment.
    pub n_predict_steps: u64,
    /// Theoretical entropy lower bound in bits, computed from the
    /// raw softmax. `entropy_bits / 8` is the floor of what any AC
    /// could possibly achieve.
    pub entropy_bits: f64,
    /// Sum of arithmetic-encoded body bytes across all segments.
    pub ac_body_bytes: u64,
    /// Sum of segment-header bytes (the fixed 13-byte struct per
    /// segment + 2 bytes per unk).
    pub segment_header_bytes: u64,
    /// Sum of unk payload bytes across all segments.
    pub unk_payload_bytes: u64,
    /// Sum of raw-fallback payload bytes across all segments.
    pub raw_fallback_bytes: u64,
    /// File-level header (magic + version + flags + reserved +
    /// total_bytes + n_segments).
    pub file_header_bytes: u64,
    /// File-level trailer magic.
    pub file_trailer_bytes: u64,
    /// File-level CRC32 trailer (v3).
    pub file_crc_bytes: u64,
}

impl AuditStats {
    /// Total bytes the compressor would actually emit if we ran the
    /// full pipeline.
    pub fn total_compressed_bytes(&self) -> u64 {
        self.file_header_bytes
            + self.file_trailer_bytes
            + self.file_crc_bytes
            + self.segment_header_bytes
            + self.ac_body_bytes
            + self.unk_payload_bytes
            + self.raw_fallback_bytes
    }

    /// Theoretical entropy floor in bytes (`ceil(entropy_bits / 8)`).
    pub fn entropy_bound_bytes(&self) -> u64 {
        (self.entropy_bits / 8.0).ceil() as u64
    }

    /// Bytes that the AC + framing pays beyond the entropy bound.
    /// This is what Phase 4b is trying to drive down.
    pub fn overhead_bytes(&self) -> i64 {
        self.total_compressed_bytes() as i64 - self.entropy_bound_bytes() as i64
    }
}

/// Phase 4e: dump top-K teacher distributions to a writer.
///
/// Runs the given model over `text` tokenized with `segment_bytes`
/// boundaries, and for each prediction step writes the top-K
/// softmax probabilities + their token ids to `dst`. The dump
/// is the training data for Phase 4e distillation: the student
/// model minimizes KL divergence against these distributions.
///
/// File format (all little-endian):
///
/// ```text
/// magic:            b"L3TD"           (4 bytes — L3TC Teacher Dump)
/// version:          u32 (= 2)
/// vocab_size:       u32
/// top_k:            u32
/// n_segments:       u32                (only tokenized, not raw-fallback)
/// n_predict_steps:  u64
/// For each tokenized segment:
///     seg_steps:    u32                (number of predict steps)
/// For each predict step, in segment order:
///     input_token_id:  u32              (model input at this step)
///     target_token_id: u32              (ground truth next token)
///     // top_k pairs of (token_id, prob) sorted by prob desc
///     top_k × { u32 token_id; f32 prob }
/// ```
///
/// The input_token_id is redundant with segment boundaries +
/// targets but we include it explicitly so the consumer can
/// feed training inputs directly without reconstructing them
/// from BOS + previous targets. Makes the Python distillation
/// script trivial to align.
///
/// At `top_k = 64`, per-step overhead is 4 + 64 × 8 = 516
/// bytes. For enwik8 (~275k predict steps) the dump is ~140 MB.
/// Much smaller than full 16384-wide distributions (~18 GB) and
/// still captures enough of the teacher's signal for
/// distillation — the long tail beyond K=64 is approximated as a
/// uniform floor when training.
///
/// Runs sequentially (no rayon). Expected throughput matches
/// the compress hot path for the given model (~25 KB/s for
/// L3TC-3.2M on Apple Silicon).
pub fn dump_teacher<W: Write>(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
    top_k: usize,
    mut dst: W,
) -> Result<u64> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use rayon::prelude::*;

    let segments = tokenizer.encode_file(text, segment_bytes)?;
    // Filter out raw-fallback segments; the teacher dump only
    // covers segments that went through the LM path.
    let tokenized: Vec<&EncodedSegment> = segments
        .iter()
        .filter(|s| !s.needs_raw_fallback)
        .collect();
    let n_segments = tokenized.len();
    let n_predict_steps: u64 = tokenized
        .iter()
        .map(|s| (s.tokens.len().saturating_sub(1)) as u64)
        .sum();

    // --- header (v2: adds per-segment step counts + inputs) ---
    dst.write_all(b"L3TD").map_err(Error::Io)?;
    dst.write_u32::<LittleEndian>(2).map_err(Error::Io)?; // version
    dst.write_u32::<LittleEndian>(model.vocab_size as u32)
        .map_err(Error::Io)?;
    dst.write_u32::<LittleEndian>(top_k as u32).map_err(Error::Io)?;
    dst.write_u32::<LittleEndian>(n_segments as u32)
        .map_err(Error::Io)?;
    dst.write_u64::<LittleEndian>(n_predict_steps)
        .map_err(Error::Io)?;
    // Per-segment step counts (so the consumer can reconstruct
    // segment boundaries without running the tokenizer again).
    for seg in &tokenized {
        let n_steps = seg.tokens.len().saturating_sub(1) as u32;
        dst.write_u32::<LittleEndian>(n_steps).map_err(Error::Io)?;
    }

    // Parallelize across segments. Each thread gets its own
    // Session and its own output buffer; we concatenate in
    // segment order at the end so the file layout is
    // deterministic. Same parallelism model as `compress`.
    // Per-step record is now input(4) + target(4) + K×(id(4) + prob(4)).
    let bytes_per_step = 4 + 4 + top_k * 8;
    let segment_buffers: Result<Vec<Vec<u8>>> = tokenized
        .par_iter()
        .map(|seg| -> Result<Vec<u8>> {
            let n_steps = seg.tokens.len().saturating_sub(1);
            let mut out = Vec::with_capacity(n_steps * bytes_per_step);
            let mut session = Session::new(model);
            // Working buffer for per-step (prob, token_id) pairs.
            // Full-size (vocab) because we compute all probs,
            // then partial-sort to pick the top K.
            let mut scratch: Vec<(f32, u32)> =
                Vec::with_capacity(model.vocab_size);

            for i in 1..seg.tokens.len() {
                let input = seg.tokens[i - 1];
                let target = seg.tokens[i];
                let logits = session.forward(input);

                // Numerically-stable softmax in f64.
                let mut max = f32::NEG_INFINITY;
                for &l in logits {
                    if l > max {
                        max = l;
                    }
                }
                let mut sum = 0.0f64;
                for &l in logits {
                    sum += ((l - max) as f64).exp();
                }
                let inv_sum = 1.0f64 / sum;

                scratch.clear();
                for (id, &l) in logits.iter().enumerate() {
                    let p =
                        (((l - max) as f64).exp() * inv_sum) as f32;
                    scratch.push((p, id as u32));
                }

                // Partial sort: select the top K by probability
                // (descending). `select_nth_unstable_by` places
                // the K-th largest at index K and partitions the
                // array so everything before K is ≥ the K-th.
                // Then sort just those K descending. Total cost
                // is O(N + K log K) vs O(N log N) for full sort.
                let k = top_k.min(scratch.len());
                if k < scratch.len() {
                    scratch.select_nth_unstable_by(k, |a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                let head = &mut scratch[..k];
                head.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Append input + target + top-K (token_id, prob) pairs.
                out.extend_from_slice(&input.to_le_bytes());
                out.extend_from_slice(&target.to_le_bytes());
                for &(prob, token_id) in head.iter() {
                    out.extend_from_slice(&token_id.to_le_bytes());
                    out.extend_from_slice(&prob.to_le_bytes());
                }
            }
            Ok(out)
        })
        .collect();
    let segment_buffers = segment_buffers?;

    // Concatenate in segment order and write to dst.
    let mut total_steps: u64 = 0;
    for buf in segment_buffers {
        if buf.is_empty() {
            continue;
        }
        total_steps += (buf.len() / bytes_per_step) as u64;
        dst.write_all(&buf).map_err(Error::Io)?;
    }
    debug_assert_eq!(total_steps, n_predict_steps);
    Ok(total_steps)
}

/// Phase 4b debug: compress `text` with the per-source byte
/// breakdown filled in. Mirrors [`compress`] sequentially (no rayon)
/// so the per-segment counters can accumulate cleanly.
pub fn audit_compress(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<AuditStats> {
    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let mut stats = AuditStats {
        input_bytes: text.len() as u64,
        n_segments: segments.len() as u64,
        file_header_bytes: HEADER_SIZE as u64,
        file_trailer_bytes: TRAILER.len() as u64,
        file_crc_bytes: 4,
        ..Default::default()
    };

    let mut session = Session::new(model);
    let mut scratch = CodecScratch::new(model.vocab_size);

    // Helper: byte length of LEB128-encoded value (matches
    // `write_varint` exactly).
    fn varint_len(mut v: u64) -> u64 {
        let mut n = 1u64;
        v >>= 7;
        while v > 0 {
            n += 1;
            v >>= 7;
        }
        n
    }

    for seg in &segments {
        // v4 per-segment header (matches `write_segment`):
        //   varint(n_tokens) + varint(n_unks) + 1(flags) + varint(ac_len)
        //   plus per-unk:    varint(unk_len) + payload
        //   plus raw-fallback: varint(raw_len) + payload
        stats.segment_header_bytes += varint_len(seg.tokens.len() as u64);
        stats.segment_header_bytes += varint_len(seg.unks.len() as u64);
        stats.segment_header_bytes += 1; // flags byte
        // ac_len is filled in below once we know it.
        for unk in &seg.unks {
            stats.segment_header_bytes += varint_len(unk.len() as u64);
            stats.unk_payload_bytes += unk.len() as u64;
        }
        if seg.needs_raw_fallback {
            stats.n_raw_fallback_segments += 1;
            stats.segment_header_bytes += varint_len(0); // ac_len = 0
            stats.segment_header_bytes += varint_len(seg.raw.len() as u64);
            stats.raw_fallback_bytes += seg.raw.len() as u64;
            stats.n_tokens += seg.tokens.len() as u64;
            continue;
        }

        // Walk the segment, accumulating entropy + encoding into
        // a fresh AC. We run both passes inline so we can compare
        // per-segment AC bytes to per-segment entropy bits later
        // if we want to.
        session.reset();
        let mut ac_bytes = Vec::with_capacity(seg.tokens.len());
        {
            let mut enc = ArithmeticEncoder::new(&mut ac_bytes);
            for i in 1..seg.tokens.len() {
                let prev = seg.tokens[i - 1];
                let tok = seg.tokens[i];
                let logits = session.forward(prev);

                // Entropy: -log2(softmax_p[tok]) computed in f64.
                let mut max = f32::NEG_INFINITY;
                for &l in logits {
                    if l > max {
                        max = l;
                    }
                }
                let mut sum = 0.0f64;
                for &l in logits {
                    sum += ((l - max) as f64).exp();
                }
                let target = (logits[tok as usize] - max) as f64;
                let log_p = target - sum.ln();
                stats.entropy_bits += -log_p / std::f64::consts::LN_2;
                stats.n_predict_steps += 1;

                logits_to_cum_freqs_scratch(
                    logits,
                    &mut scratch.cum,
                    &mut scratch.exps,
                    &mut scratch.freqs,
                );
                enc.encode_symbol(&scratch.cum, tok)?;
            }
            enc.finish()?;
        }
        stats.ac_body_bytes += ac_bytes.len() as u64;
        // varint(ac_len) — accounted for here once we know the body
        // length. Mirrors the order in `write_segment`.
        stats.segment_header_bytes += varint_len(ac_bytes.len() as u64);
        stats.n_tokens += seg.tokens.len() as u64;
    }

    Ok(stats)
}

/// Per-phase timing breakdown for a compression run. Populated
/// by [`profile_compress`]. Units: nanoseconds total across all
/// predict steps. Divide by `n_predict_steps` for per-step cost.
#[derive(Debug, Default, Clone)]
pub struct ProfileStats {
    /// Wall-clock ns spent in `Session::forward`.
    pub forward_ns: u128,
    /// Wall-clock ns spent in `logits_to_cum_freqs_scratch`.
    pub cum_freqs_ns: u128,
    /// Wall-clock ns spent in `ArithmeticEncoder::encode_symbol`.
    pub ac_encode_ns: u128,
    /// Wall-clock ns spent in everything else in the inner loop
    /// (control flow, scratch indexing, segment bookkeeping).
    pub other_ns: u128,
    /// Total wall-clock ns spent in the inner loop.
    pub total_ns: u128,
    /// Total predict steps executed across all tokenized segments.
    pub n_predict_steps: u64,
    /// Total segments (including raw-fallback ones skipped).
    pub n_segments: u64,
    /// Total tokens produced by the tokenizer (including BOS).
    pub n_tokens: u64,
}

/// Phase 4c debug: sequentially compress a file while timing each
/// hot-path stage (forward pass, cum_freqs, AC encode) to measure
/// where the remaining per-token time actually goes. Used to
/// verify which of the 4c5 / later items is worth implementing.
///
/// Matches the in-memory `compress` except it runs segments
/// serially (no rayon) so the per-phase timers accumulate cleanly.
pub fn profile_compress(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<ProfileStats> {
    use std::time::Instant;

    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let mut stats = ProfileStats {
        n_segments: segments.len() as u64,
        ..Default::default()
    };

    let mut session = Session::new(model);
    let mut scratch = CodecScratch::new(model.vocab_size);

    let outer = Instant::now();
    for seg in &segments {
        stats.n_tokens += seg.tokens.len() as u64;
        if seg.needs_raw_fallback {
            continue;
        }
        session.reset();
        let mut ac_bytes = Vec::with_capacity(seg.tokens.len());
        {
            let mut enc = ArithmeticEncoder::new(&mut ac_bytes);
            for i in 1..seg.tokens.len() {
                let prev = seg.tokens[i - 1];
                let tok = seg.tokens[i];

                let t0 = Instant::now();
                let logits = session.forward(prev);
                let t1 = Instant::now();
                logits_to_cum_freqs_scratch(
                    logits,
                    &mut scratch.cum,
                    &mut scratch.exps,
                    &mut scratch.freqs,
                );
                let t2 = Instant::now();
                enc.encode_symbol(&scratch.cum, tok)?;
                let t3 = Instant::now();

                stats.forward_ns += (t1 - t0).as_nanos();
                stats.cum_freqs_ns += (t2 - t1).as_nanos();
                stats.ac_encode_ns += (t3 - t2).as_nanos();
                stats.n_predict_steps += 1;
            }
            enc.finish()?;
        }
    }
    stats.total_ns = outer.elapsed().as_nanos();
    stats.other_ns = stats.total_ns.saturating_sub(
        stats.forward_ns + stats.cum_freqs_ns + stats.ac_encode_ns,
    );

    Ok(stats)
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
            logits_to_cum_freqs_scratch(
                logits,
                &mut scratch.cum,
                &mut scratch.exps,
                &mut scratch.freqs,
            );
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
    // Prevent attacker-controlled `n_tokens` from forcing a huge
    // `Vec::with_capacity`. The arithmetic decoder produces tokens
    // *per symbol entropy*, not per bit, so a confident model can
    // emit many tokens per AC byte (~1-2 in practice). We use a
    // generous structural cap of 64 tokens per AC byte, which is
    // far above anything realistic but still bounds the allocation
    // at ~256× ac_body bytes — turning a memory-DOS into a
    // well-behaved error.
    let max_tokens = ac_body.len().saturating_mul(64).saturating_add(64);
    if (n_tokens as usize) > max_tokens {
        return Err(Error::BadCheckpoint(format!(
            "segment claims {n_tokens} tokens, exceeding the structural cap of {max_tokens} for a {}-byte AC body",
            ac_body.len(),
        )));
    }
    let mut tokens = Vec::with_capacity(n_tokens as usize);
    tokens.push(BOS_ID);

    let mut dec = ArithmeticDecoder::new(ac_body)?;

    // n_tokens includes the BOS. We decode n_tokens - 1 symbols.
    // Each iteration: forward the previous token through the model,
    // use its output distribution to decode the next token.
    let mut prev = BOS_ID;
    for _ in 1..n_tokens {
        let logits = session.forward(prev);
        logits_to_cum_freqs_scratch(
            logits,
            &mut scratch.cum,
            &mut scratch.exps,
            &mut scratch.freqs,
        );
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
    /// Per-symbol integer frequencies.
    ///
    /// Separate from `cum` so the scaling loop has no sequential
    /// dependency and the prefix-sum pass is a simple running-add
    /// in a dedicated loop. Phase 4c3 changed this from `Vec<u64>`
    /// to `Vec<u32>` because freqs never exceed `PYTHON_FREQ_TOTAL
    /// + n ≈ 10M + 16k` — easily fits in u32, halves memory
    /// bandwidth on the scale loop, and lets NEON's f32→u32
    /// instruction produce the quantized values directly without
    /// a second widening pass.
    freqs: Vec<u32>,
}

impl CodecScratch {
    fn new(vocab_size: usize) -> Self {
        Self {
            cum: vec![0u64; vocab_size + 1],
            exps: vec![0.0f32; vocab_size],
            freqs: vec![0u32; vocab_size],
        }
    }
}

/// Public alias for [`logits_to_cum_freqs_scratch`], used by the
/// profile tests in `tests/profile_codec.rs`. Not part of the stable API.
#[doc(hidden)]
pub fn logits_to_cum_freqs_public(logits: &[f32], cum: &mut [u64]) {
    let mut exps = vec![0.0f32; logits.len()];
    let mut freqs = vec![0u32; logits.len()];
    logits_to_cum_freqs_scratch(logits, cum, &mut exps, &mut freqs);
}

/// Convert raw model logits to a cumulative frequency table,
/// **matching the Python L3TC reference exactly**.
///
/// The Python reference (`vendor/L3TC/scripts/compressor.py`) does:
///
/// ```python
/// probs = torch.softmax(logits, dim=-1)
/// freqs = torch.round(probs * 10_000_000).int()
/// freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
/// ```
///
/// We mirror this exactly:
///
/// 1. Compute `softmax(logits)` in f32 using libm `exp` (matches
///    PyTorch's softmax — `fast_exp_neg` introduces ~1% relative
///    error per call, which compounds across 16384 symbols and
///    measurably hurts ratio).
/// 2. `freqs[i] = round(p[i] * PYTHON_FREQ_TOTAL).max(1)` where
///    `PYTHON_FREQ_TOTAL = 10_000_000`.
/// 3. Cumulative table = prefix sum of `freqs`.
///
/// The arithmetic coder accepts any `total ≤ MAX_TOTAL = 2^62`,
/// so the ~10M total here is well within bounds. There is **no
/// residual fixup**: whatever the rounding produces is what the
/// AC sees, exactly as in Python.
///
/// Earlier (Phase 1-3) we used a finer scale (~2^62) with `floor`
/// rounding plus a residual fixup pass distributing the truncation
/// error to the top symbol. That looked like it should give better
/// resolution, but it diverged from Python in three ways
/// simultaneously (`floor` vs `round`, scale, fixup) and ended up
/// 4 percentage points worse on enwik6. Phase 4a closes that gap
/// by matching Python bit-for-bit.
///
/// `_freqs` is unused in this implementation but kept in the
/// signature for the hot-path scratch reuse pattern in
/// [`CodecScratch`].
fn logits_to_cum_freqs_scratch(
    logits: &[f32],
    cum: &mut [u64],
    exps: &mut [f32],
    freqs: &mut [u32],
) {
    debug_assert_eq!(cum.len(), logits.len() + 1);
    debug_assert_eq!(exps.len(), logits.len());

    /// Total target frequency, matching Python L3TC's
    /// `freqs = round(probs * 10_000_000)`.
    const PYTHON_FREQ_TOTAL: u64 = 10_000_000;

    let n = logits.len();

    // --- Pass 1: find max logit (numerical-stability shift) ---
    // NEON 4-wide reduction; scalar elsewhere. The branch in the
    // old `if l > max` form blocks autovectorization.
    let max = tensor::max_f32(logits);
    if !max.is_finite() {
        uniform_fallback(n, cum);
        return;
    }

    // --- Pass 2: shifted exp + running sum ---
    //
    // Vectorized NEON path (`tensor::softmax_shifted_exp_sum`)
    // on aarch64 via a hand-rolled `exp_f32x4_neon` using a
    // degree-6 minimax polynomial for `2^r`. Max relative error
    // is under 5e-7 — tighter than the ~1% tolerance that broke
    // the Phase 4a diff harness — so the resulting cum_freqs
    // table is bit-identical to the libm path after the
    // Phase 4a `round(p * 10_000_000); max(1)` quantization
    // absorbs the residual.
    //
    // Phase 4a briefly used libm `f32::exp` here because the
    // old `fast_exp_neg` polynomial was too loose; Phase 4c1
    // replaces libm with a tighter polynomial and measures
    // the entropy bound as a regression gate.
    let sum = tensor::softmax_shifted_exp_sum(logits, max, exps);
    if !sum.is_finite() || sum <= 0.0 {
        uniform_fallback(n, cum);
        return;
    }

    // --- Pass 3: round(p * 10M); max(1); accumulate cum ---
    //
    // Split into (3a) a fused NEON quantize over exps that
    // computes `freqs[i] = max(1, round(exps[i] * scale))` as
    // u32, and (3b) a scalar cum-accumulator walk. 3a is fully
    // vectorized; 3b is a short data-dependent scalar loop that
    // widens u32 freqs to u64 cum. Phase 4c3 measured this
    // split saves ~10-15 us per predict step on the cum_freqs
    // hot loop vs the prior single-pass scalar version.
    let inv_sum = 1.0f32 / sum;
    let scale = inv_sum * PYTHON_FREQ_TOTAL as f32;
    tensor::quantize_exps_to_freqs(exps, scale, freqs);

    // Plain `+=` instead of `saturating_add`: max achievable total
    // is `PYTHON_FREQ_TOTAL + n ≈ 10M + 32K`, well below u64::MAX
    // (1.8e19). The saturating variant compiled to extra cmp+csel
    // per element on aarch64; the plain add is one instruction.
    cum[0] = 0;
    let mut total: u64 = 0;
    for i in 0..n {
        total += freqs[i] as u64;
        cum[i + 1] = total;
    }

    // Sanity: total must be ≤ MAX_TOTAL for the AC to accept it.
    // With PYTHON_FREQ_TOTAL = 10M and n ≤ 32768, total is bounded
    // by ~10M + 32K. MAX_TOTAL is 2^62, so this never triggers in
    // practice — kept as a defensive check against corrupted model
    // weights producing freqs outside the documented range.
    if total > MAX_TOTAL as u64 {
        uniform_fallback(n, cum);
    }
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

fn write_header<W: Write>(
    w: &mut W,
    total_bytes: u64,
    n_segments: u32,
    flags: u8,
) -> Result<()> {
    w.write_all(MAGIC)?;
    w.write_u8(VERSION)?;
    w.write_u8(flags)?;
    w.write_u16::<LittleEndian>(0)?; // reserved
    w.write_u64::<LittleEndian>(total_bytes)?;
    w.write_u32::<LittleEndian>(n_segments)?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> Result<(u64, u32, u8)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {magic:?}")));
    }
    let version = r.read_u8()?;
    if version != VERSION && version != VERSION_V3_COMPAT && version != VERSION_V2_COMPAT {
        return Err(Error::BadCheckpoint(format!(
            "unsupported version: {version}"
        )));
    }
    let flags = r.read_u8()?;
    let _reserved = r.read_u16::<LittleEndian>()?;
    let total_bytes = r.read_u64::<LittleEndian>()?;
    let n_segments = r.read_u32::<LittleEndian>()?;
    Ok((total_bytes, n_segments, flags))
}

/// Fixed size of the on-disk header, used by the decoder to
/// locate the raw payload range for `FLAG_RAW_STORE` files.
const HEADER_SIZE: usize = 4 + 1 + 1 + 2 + 8 + 4;

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

/// LEB128 unsigned varint encoder. Each byte holds 7 bits of
/// value plus a high "continuation" bit that's 1 if more bytes
/// follow. Used by the v4 segment-header packing.
///
/// Typical sizes:
///   value < 128       → 1 byte
///   value < 16384     → 2 bytes
///   value < 2097152   → 3 bytes
fn write_varint<W: Write>(w: &mut W, mut value: u64) -> Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            w.write_u8(byte).map_err(Error::Io)?;
            return Ok(());
        }
        w.write_u8(byte | 0x80).map_err(Error::Io)?;
    }
}

/// LEB128 unsigned varint decoder. Reads up to 10 bytes (enough
/// for any u64). Errors out on a malformed sequence longer than
/// that.
fn read_varint<R: Read>(r: &mut R) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for _ in 0..10 {
        let byte = r.read_u8().map_err(Error::Io)?;
        let bits = (byte & 0x7F) as u64;
        // Guard against overflow: shift must be < 64 for a valid
        // u64, and the shifted bits must not exceed u64::MAX.
        if shift >= 64 {
            return Err(Error::BadCheckpoint("varint overflow".into()));
        }
        result |= bits.checked_shl(shift).ok_or_else(|| {
            Error::BadCheckpoint("varint shift overflow".into())
        })?;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
    Err(Error::BadCheckpoint("varint too long".into()))
}

/// Write one segment in the **v4** format.
///
/// v3 used fixed-width u32/u16 fields totaling 13 bytes per
/// segment header (plus per-unk u16 length and per-raw u32
/// length). v4 packs all length fields as LEB128 varints, which
/// drops a typical text-only segment header from 13 bytes to ~5
/// bytes and a typical raw-fallback segment from 17 bytes to ~7
/// bytes. The actual ac_body / unk payloads / raw payloads are
/// unchanged.
fn write_segment<W: Write>(w: &mut W, seg: &EncodedSegment, ac_body: &[u8]) -> Result<()> {
    write_varint(w, seg.tokens.len() as u64)?;
    write_varint(w, seg.unks.len() as u64)?;
    let seg_flags = if seg.needs_raw_fallback {
        SEG_FLAG_RAW_FALLBACK
    } else {
        0
    };
    w.write_u8(seg_flags).map_err(Error::Io)?;
    write_varint(w, ac_body.len() as u64)?;
    w.write_all(ac_body).map_err(Error::Io)?;
    for unk in &seg.unks {
        write_varint(w, unk.len() as u64)?;
        w.write_all(unk).map_err(Error::Io)?;
    }
    if seg.needs_raw_fallback {
        let raw_bytes = seg.raw.as_bytes();
        write_varint(w, raw_bytes.len() as u64)?;
        w.write_all(raw_bytes).map_err(Error::Io)?;
    }
    Ok(())
}

/// How many bytes remain after the cursor position. Used to bound
/// allocations driven by attacker-controlled length fields read
/// from the on-disk format.
#[inline]
fn remaining(cur: &Cursor<&[u8]>) -> usize {
    cur.get_ref().len().saturating_sub(cur.position() as usize)
}

/// Allocate and fill a `Vec<u8>` of `len` bytes from `cur`, but only
/// after sanity-checking that `len` doesn't exceed the bytes that
/// actually remain in the input. Without this check, an attacker
/// who supplies a 22-byte file with a varint claiming `len = 2^60`
/// would force a 1 EB allocation before `read_exact` ever fails.
fn read_bounded_vec(cur: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<u8>> {
    let rem = remaining(cur);
    if len > rem {
        return Err(Error::BadCheckpoint(format!(
            "length-prefixed field claims {len} bytes but only {rem} remain in input",
        )));
    }
    let mut buf = vec![0u8; len];
    cur.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read one segment in the **v4** format.
///
/// Every length field read here comes from on-disk bytes that may
/// be attacker-controlled. We bound each allocation against the
/// cursor's remaining input to keep a malformed header from forcing
/// a multi-GB allocation. The bound is conservative — `n_unks` and
/// `unk_len` are each capped at "no more bytes than the rest of
/// the input could possibly contain", which is loose but enough
/// to turn an allocation DOS into a clean `BadCheckpoint` error.
fn read_segment_meta(cur: &mut Cursor<&[u8]>) -> Result<SegmentRead> {
    let n_tokens = read_varint(cur)? as u32;
    let n_unks_u64 = read_varint(cur)?;
    if n_unks_u64 > remaining(cur) as u64 {
        return Err(Error::BadCheckpoint(format!(
            "segment claims {n_unks_u64} unks but only {} bytes remain",
            remaining(cur),
        )));
    }
    let n_unks = n_unks_u64 as usize;
    let seg_flags = cur.read_u8()?;
    let ac_bytes_len = read_varint(cur)? as usize;
    let ac_body = read_bounded_vec(cur, ac_bytes_len)?;
    let mut unks = Vec::with_capacity(n_unks);
    for _ in 0..n_unks {
        let unk_len = read_varint(cur)? as usize;
        unks.push(read_bounded_vec(cur, unk_len)?);
    }
    let raw_fallback = if seg_flags & SEG_FLAG_RAW_FALLBACK != 0 {
        let raw_len = read_varint(cur)? as usize;
        Some(read_bounded_vec(cur, raw_len)?)
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

/// Read one segment in the **v2/v3 fixed-width** format.
///
/// Kept for back-compat: any v2 or v3 file produced before
/// Phase 4b1 still decodes through this path. v4 readers should
/// use [`read_segment_meta`] instead. Same allocation-bound
/// discipline as [`read_segment_meta`] — every length field is
/// checked against remaining input before allocating.
fn read_segment_meta_v3(cur: &mut Cursor<&[u8]>) -> Result<SegmentRead> {
    let n_tokens = cur.read_u32::<LittleEndian>()?;
    let n_unks = cur.read_u32::<LittleEndian>()?;
    if n_unks as usize > remaining(cur) {
        return Err(Error::BadCheckpoint(format!(
            "segment claims {n_unks} unks but only {} bytes remain",
            remaining(cur),
        )));
    }
    let seg_flags = cur.read_u8()?;
    let ac_bytes_len = cur.read_u32::<LittleEndian>()? as usize;
    let ac_body = read_bounded_vec(cur, ac_bytes_len)?;
    let mut unks = Vec::with_capacity(n_unks as usize);
    for _ in 0..n_unks {
        let unk_len = cur.read_u16::<LittleEndian>()? as usize;
        unks.push(read_bounded_vec(cur, unk_len)?);
    }
    let raw_fallback = if seg_flags & SEG_FLAG_RAW_FALLBACK != 0 {
        let raw_len = cur.read_u32::<LittleEndian>()? as usize;
        let raw = read_bounded_vec(cur, raw_len)?;
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
    fn cum_freqs_match_python_total() {
        // Random-ish logits. Phase 4a: we now match Python's
        // `freqs = round(probs * 10_000_000); max(1)` scheme, so
        // the total is approximately 10M (not MAX_TOTAL = 2^62).
        // Exact value depends on rounding noise + the max(1) clamp.
        let logits: Vec<f32> = (0..256).map(|i| (i as f32 / 50.0).sin()).collect();
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        let mut freqs = vec![0u32; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps, &mut freqs);
        assert_eq!(cum[0], 0);
        // Total is bounded above by 10M + n (every freq could
        // round up by 1) and bounded below by max(n, 10M - n).
        let total = cum[logits.len()];
        assert!(total <= 10_000_000 + logits.len() as u64);
        assert!(total >= logits.len() as u64);
        assert!(total <= MAX_TOTAL as u64);
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
        let mut freqs = vec![0u32; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps, &mut freqs);
        // The uniform fallback still uses MAX_TOTAL because it
        // takes a different code path (for degenerate inputs).
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // Every step is non-zero
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i]);
        }
    }

    #[test]
    fn bounded_read_rejects_oversized_length() {
        // 16 bytes of input; ask for 1 GB. Must error, not allocate.
        let buf = vec![0u8; 16];
        let mut cur = Cursor::new(buf.as_slice());
        let err = read_bounded_vec(&mut cur, 1_000_000_000).unwrap_err();
        match err {
            Error::BadCheckpoint(msg) => assert!(
                msg.contains("length-prefixed field"),
                "unexpected error message: {msg}"
            ),
            other => panic!("expected BadCheckpoint, got {other:?}"),
        }
    }

    #[test]
    fn bounded_read_accepts_exact_fit() {
        let buf = vec![0xABu8; 8];
        let mut cur = Cursor::new(buf.as_slice());
        let v = read_bounded_vec(&mut cur, 8).expect("exact fit must succeed");
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], 0xAB);
    }

    #[test]
    fn read_segment_meta_v3_rejects_oversized_ac_bytes_len() {
        // v3 segment header layout (fixed-width):
        //   n_tokens: u32 LE  | n_unks: u32 LE | seg_flags: u8
        //   ac_bytes_len: u32 LE | ac_body[ac_bytes_len] | ...
        // Build a 13-byte header that claims a 1 GB ac_body and
        // confirm we reject before any allocation.
        let mut buf = Vec::new();
        buf.extend_from_slice(&100u32.to_le_bytes()); // n_tokens
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_unks
        buf.push(0u8); // seg_flags
        buf.extend_from_slice(&1_000_000_000u32.to_le_bytes()); // ac_bytes_len
        let mut cur = Cursor::new(buf.as_slice());
        let err = match read_segment_meta_v3(&mut cur) {
            Err(e) => e,
            Ok(_) => panic!("expected error, got valid SegmentRead"),
        };
        assert!(matches!(err, Error::BadCheckpoint(_)));
    }

    #[test]
    fn read_segment_meta_v3_rejects_oversized_n_unks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&100u32.to_le_bytes()); // n_tokens
        buf.extend_from_slice(&u32::MAX.to_le_bytes()); // n_unks (4B!)
        buf.push(0u8); // seg_flags
        buf.extend_from_slice(&0u32.to_le_bytes()); // ac_bytes_len
        let mut cur = Cursor::new(buf.as_slice());
        let err = match read_segment_meta_v3(&mut cur) {
            Err(e) => e,
            Ok(_) => panic!("expected error, got valid SegmentRead"),
        };
        match err {
            Error::BadCheckpoint(msg) => assert!(msg.contains("unks"), "msg: {msg}"),
            other => panic!("expected BadCheckpoint, got {other:?}"),
        }
    }

    #[test]
    fn crc32_roundtrip_detects_corruption() {
        // Build a tiny fake v3 payload: magic + version + flags +
        // reserved + total_bytes + n_segments=0 + trailer, then
        // append the correct CRC. Verify that decompress-time CRC
        // validation accepts it and rejects a bit-flipped copy.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // total_bytes
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_segments
        buf.extend_from_slice(TRAILER);
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        // Pristine copy: CRC matches.
        let (payload, crc_bytes) = buf.split_at(buf.len() - 4);
        assert_eq!(LittleEndian::read_u32(crc_bytes), crc32fast::hash(payload));

        // Corrupt one byte in the middle of the payload.
        let mut bad = buf.clone();
        bad[8] ^= 0xFF;
        let (payload, crc_bytes) = bad.split_at(bad.len() - 4);
        assert_ne!(LittleEndian::read_u32(crc_bytes), crc32fast::hash(payload));
    }

}
