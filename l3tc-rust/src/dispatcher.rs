//! Hybrid-codec dispatcher — Tier 1 implementation.
//!
//! Splits the input into fixed-size chunks, runs every enabled codec
//! on each chunk, picks the shortest output (with a zstd-shadow
//! safety net), and writes a versioned blob tagged per chunk.
//!
//! Design reference: `HYBRID_CODEC_DESIGN.md` at the repo root.
//!
//! Blob layout:
//!
//! ```text
//! [magic:     4 bytes "L3H\0"]
//! [version:   2 bytes u16 (= BLOB_VERSION)]
//! [chunk_sz:  4 bytes u32 (uncompressed bytes per full chunk)]
//! [raw_len:   8 bytes u64 (total uncompressed input length)]
//! [chunk_ct:  4 bytes u32 (number of chunks in stream)]
//! --- per-chunk ---
//!   [codec:   1 byte  CodecTag]
//!   [enc_len: 4 bytes u32 (length of encoded bytes that follow)]
//!   [bytes:   enc_len bytes]
//! ```
//!
//! Per-chunk overhead: 5 bytes (1 tag + 4 length). On a 64 KB chunk
//! that's 0.008%.

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor, Read, Write};
use std::time::Instant;

use crate::error::{Error, Result};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Magic bytes at the head of every hybrid blob. Chosen so a v1 l3tc
/// blob (which starts with different bytes) can't be confused for a
/// hybrid blob and vice versa.
pub const BLOB_MAGIC: &[u8; 4] = b"L3H\0";

/// Current hybrid blob format version. Bump on any layout change.
pub const BLOB_VERSION: u16 = 1;

/// Default chunk size: 64 KB. Per-dataset tunable once we wire the
/// training pipeline to emit a suggested value — larger for
/// templated-log datasets so zstd's window has room to work, smaller
/// for text-heavy datasets where per-chunk neural works better.
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Stage 3 safety net: if the dispatcher's picked codec is worse than
/// zstd by more than this factor, substitute zstd instead. 1.01 means
/// "tolerate up to 1% worse than zstd before falling back."
pub const SAFETY_NET_THRESHOLD: f64 = 1.01;

// -----------------------------------------------------------------------------
// Codec identity
// -----------------------------------------------------------------------------

/// Codec tag for a single chunk. Matches the assignments in
/// `HYBRID_CODEC_DESIGN.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CodecTag {
    /// Store raw bytes (magic-byte-prescreen fallback for binary).
    Passthrough = 0,
    /// lz4 frame codec (latency-friendly; tiny/incompressible chunks).
    Lz4 = 1,
    /// zstd -22 --long=27, plain (no dictionary).
    Zstd22 = 2,
    /// zstd --dict with a per-dataset trained dictionary.
    ZstdDict = 3,
    /// bzip3 at default settings.
    Bzip3 = 4,
    /// Brotli with an RFC 9842 shared dictionary (Tier 1+ feature).
    BrotliDict = 5,
    /// CLP IR + per-column zstd (Tier 1+ feature).
    Clp = 6,
    /// Our neural model (RWKV + SPM + arithmetic coding).
    Neural = 7,
}

impl CodecTag {
    /// Short human-readable name. Used in metrics + logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::Lz4 => "lz4",
            Self::Zstd22 => "zstd",
            Self::ZstdDict => "zstd_dict",
            Self::Bzip3 => "bzip3",
            Self::BrotliDict => "brotli_dict",
            Self::Clp => "clp",
            Self::Neural => "neural",
        }
    }

    /// Convert from the on-disk u8 tag. Fails with `UnknownCodecTag`
    /// on values we don't recognise (future forward-compat boundary).
    pub fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::Passthrough,
            1 => Self::Lz4,
            2 => Self::Zstd22,
            3 => Self::ZstdDict,
            4 => Self::Bzip3,
            5 => Self::BrotliDict,
            6 => Self::Clp,
            7 => Self::Neural,
            other => return Err(Error::UnknownCodecTag(other)),
        })
    }
}

// -----------------------------------------------------------------------------
// Per-codec encode/decode
// -----------------------------------------------------------------------------

/// Trait a codec implements to participate in the dispatcher.
///
/// Keep the surface small: `encode` and `decode`, both fallible,
/// both pure (given the codec's own config which is held in the
/// implementing struct). Dispatcher handles chunk slicing, tag
/// writing, and safety-net logic — individual codecs only need to
/// know how to turn `&[u8]` into `Vec<u8>` and back.
pub trait Codec: Send + Sync {
    /// Tag this codec writes for its chunks.
    fn tag(&self) -> CodecTag;
    /// Compress `input` to bytes.
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>>;
    /// Decompress bytes produced by a prior `encode` call.
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>>;
}

// --- Passthrough -------------------------------------------------------------

/// Trivial codec that stores input bytes verbatim. Used for chunks
/// the detector flagged as already-compressed (via magic-byte
/// prescreen) or for the dispatcher's "don't make it larger" edge
/// case on incompressible data.
pub struct PassthroughCodec;

impl Codec for PassthroughCodec {
    fn tag(&self) -> CodecTag {
        CodecTag::Passthrough
    }
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>> {
        Ok(input.to_vec())
    }
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        Ok(input.to_vec())
    }
}

// --- zstd --------------------------------------------------------------------

/// Plain zstd at level 22 with `--long=27`-equivalent window. No
/// dictionary.
pub struct Zstd22Codec;

impl Codec for Zstd22Codec {
    fn tag(&self) -> CodecTag {
        CodecTag::Zstd22
    }
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>> {
        // `zstd_safe::max_c_level` == 22 at time of writing. We also
        // want the long-mode (27-bit window log) that matches our
        // published zstd-22 --long=27 baseline.
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 22).map_err(|e| {
            Error::ClassicalCodec { codec: "zstd", message: format!("encoder init: {e}") }
        })?;
        encoder
            .set_pledged_src_size(Some(input.len() as u64))
            .map_err(|e| Error::ClassicalCodec { codec: "zstd", message: format!("pledge: {e}") })?;
        encoder
            .long_distance_matching(true)
            .map_err(|e| Error::ClassicalCodec { codec: "zstd", message: format!("long: {e}") })?;
        encoder
            .window_log(27)
            .map_err(|e| Error::ClassicalCodec { codec: "zstd", message: format!("wlog: {e}") })?;
        encoder
            .write_all(input)
            .map_err(|e| Error::ClassicalCodec { codec: "zstd", message: format!("write: {e}") })?;
        encoder
            .finish()
            .map_err(|e| Error::ClassicalCodec { codec: "zstd", message: format!("finish: {e}") })
    }
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        zstd::stream::copy_decode(input, &mut out).map_err(|e| {
            Error::ClassicalCodec { codec: "zstd", message: format!("decode: {e}") }
        })?;
        Ok(out)
    }
}

/// zstd with a trained dictionary. The dictionary is passed by
/// reference (not cloned) to keep encoder init cheap when compressing
/// many chunks.
pub struct ZstdDictCodec {
    dict: Vec<u8>,
}

impl ZstdDictCodec {
    /// Create a dict-backed zstd codec from a pre-trained dictionary
    /// blob (output of `zstd --train` or `ZSTD_trainFromBuffer`).
    pub fn new(dict: Vec<u8>) -> Self {
        Self { dict }
    }
}

impl Codec for ZstdDictCodec {
    fn tag(&self) -> CodecTag {
        CodecTag::ZstdDict
    }
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut encoder =
            zstd::stream::Encoder::with_dictionary(Vec::new(), 22, &self.dict)
                .map_err(|e| Error::ClassicalCodec {
                    codec: "zstd_dict",
                    message: format!("encoder init: {e}"),
                })?;
        encoder.write_all(input).map_err(|e| Error::ClassicalCodec {
            codec: "zstd_dict",
            message: format!("write: {e}"),
        })?;
        encoder.finish().map_err(|e| Error::ClassicalCodec {
            codec: "zstd_dict",
            message: format!("finish: {e}"),
        })
    }
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = zstd::stream::Decoder::with_dictionary(input, &self.dict)
            .map_err(|e| Error::ClassicalCodec {
                codec: "zstd_dict",
                message: format!("decoder init: {e}"),
            })?;
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(|e| Error::ClassicalCodec {
            codec: "zstd_dict",
            message: format!("decode: {e}"),
        })?;
        Ok(out)
    }
}

// --- bzip3 -------------------------------------------------------------------

/// bzip3 at default settings. Small ratio win on prose where neural
/// is unavailable.
pub struct Bzip3Codec;

/// Block size passed to bzip3 — smaller than the library's default
/// (16 MB) so individual 64 KB chunks don't trigger costly overhead
/// on every chunk. 1 MB is the minimum permitted, plenty for our
/// single-chunk use.
const BZIP3_BLOCK_SIZE: usize = 1024 * 1024;

impl Codec for Bzip3Codec {
    fn tag(&self) -> CodecTag {
        CodecTag::Bzip3
    }
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>> {
        // bzip3 0.12 API: stream writer owns the block state.
        let mut out = Vec::with_capacity(input.len() / 4);
        let mut enc = bzip3::write::Bz3Encoder::new(&mut out, BZIP3_BLOCK_SIZE).map_err(|e| {
            Error::ClassicalCodec { codec: "bzip3", message: format!("enc init: {e}") }
        })?;
        enc.write_all(input).map_err(|e| Error::ClassicalCodec {
            codec: "bzip3",
            message: format!("write: {e}"),
        })?;
        drop(enc); // flushes.
        Ok(out)
    }
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut dec = bzip3::read::Bz3Decoder::new(input).map_err(|e| {
            Error::ClassicalCodec { codec: "bzip3", message: format!("dec init: {e}") }
        })?;
        let mut out = Vec::new();
        dec.read_to_end(&mut out).map_err(|e| Error::ClassicalCodec {
            codec: "bzip3",
            message: format!("read: {e}"),
        })?;
        Ok(out)
    }
}

// --- lz4 ---------------------------------------------------------------------

/// lz4 frame codec. Gigabyte-class throughput; ratio-neutral. Used as
/// the small/incompressible-chunk escape and the latency-friendly
/// default for pre-compressed content.
pub struct Lz4Codec;

impl Codec for Lz4Codec {
    fn tag(&self) -> CodecTag {
        CodecTag::Lz4
    }
    fn encode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
        enc.write_all(input).map_err(|e: io::Error| Error::ClassicalCodec {
            codec: "lz4",
            message: format!("write: {e}"),
        })?;
        enc.finish().map_err(|e| Error::ClassicalCodec {
            codec: "lz4",
            message: format!("finish: {e}"),
        })
    }
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = lz4_flex::frame::FrameDecoder::new(input);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(|e: io::Error| Error::ClassicalCodec {
            codec: "lz4",
            message: e.to_string(),
        })?;
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Magic-byte prescreen (Stage 1 of the detector)
// -----------------------------------------------------------------------------

/// Known binary-format magic byte prefixes. Chunks that start with
/// one of these go straight to `Passthrough` — neural and classical
/// text codecs can't compress them meaningfully and we'd just waste
/// the probe cost.
const MAGIC_PREFIXES: &[&[u8]] = &[
    b"\xff\xd8\xff",            // JPEG
    b"\x89PNG\r\n\x1a\n",       // PNG
    b"GIF87a",                  // GIF
    b"GIF89a",                  // GIF
    b"BM",                      // BMP (also matches "BM" prefix inside text — acceptable false-positive for v1)
    b"RIFF",                    // WAV / AVI
    b"\x7fELF",                 // ELF
    b"MZ",                      // PE
    b"%PDF",                    // PDF
    b"PK\x03\x04",              // ZIP / JAR / DOCX
    b"\x1f\x8b",                // gzip
    b"\x28\xb5\x2f\xfd",        // zstd
    b"\xfd7zXZ\x00",            // xz
    b"BZh",                     // bzip2
    b"7z\xbc\xaf\x27\x1c",      // 7-Zip
];

/// Returns true if the buffer starts with one of our known-binary
/// magic bytes. Runs in O(len_of_longest_magic) — effectively a
/// handful of byte compares.
pub fn looks_binary(buf: &[u8]) -> bool {
    MAGIC_PREFIXES.iter().any(|p| buf.starts_with(p))
}

// -----------------------------------------------------------------------------
// Per-chunk probe + pick
// -----------------------------------------------------------------------------

/// Result of running a single chunk through the dispatcher.
#[derive(Debug, Clone)]
pub struct ChunkOutcome {
    /// Winning codec's tag.
    pub tag: CodecTag,
    /// Winning codec's output bytes.
    pub bytes: Vec<u8>,
    /// What zstd-22 alone produced for this chunk. Useful for
    /// metrics (the `zstd_shadow_bytes` number) and cheap to keep
    /// since we run zstd per chunk anyway.
    pub zstd_shadow_bytes: usize,
    /// True if the Stage 3 safety net substituted zstd because the
    /// otherwise-picked codec's output was more than
    /// `SAFETY_NET_THRESHOLD` × zstd.
    pub safety_net_fired: bool,
}

/// Pick the best codec for a single chunk.
///
/// - If the chunk matches a known binary-format magic → Passthrough.
/// - Otherwise, run every provided codec, pick shortest, apply the
///   safety net vs. zstd's own output.
///
/// `codecs` must contain at least one `Zstd22` codec (or the safety
/// net has no reference point). The dispatcher passes both zstd and
/// every other candidate here.
pub fn dispatch_chunk(chunk: &[u8], codecs: &[Box<dyn Codec>]) -> Result<ChunkOutcome> {
    // Stage 1: magic-byte prescreen.
    if looks_binary(chunk) {
        let pt = PassthroughCodec;
        return Ok(ChunkOutcome {
            tag: CodecTag::Passthrough,
            bytes: pt.encode(chunk)?,
            zstd_shadow_bytes: chunk.len(), // no zstd run; use raw length as shadow (worst case)
            safety_net_fired: false,
        });
    }

    // Stage 2: run every codec. Collect (tag, output_bytes) pairs.
    let mut results: Vec<(CodecTag, Vec<u8>)> = Vec::with_capacity(codecs.len());
    let mut zstd_bytes: Option<Vec<u8>> = None;
    for c in codecs {
        let out = c.encode(chunk)?;
        if c.tag() == CodecTag::Zstd22 {
            zstd_bytes = Some(out.clone());
        }
        results.push((c.tag(), out));
    }

    // Pick shortest.
    let (picked_tag, picked_bytes) = results
        .into_iter()
        .min_by_key(|(_, b)| b.len())
        .expect("dispatcher called with empty codec list");

    // Stage 3: safety net vs zstd if picked wasn't zstd already.
    let zstd_shadow = zstd_bytes.as_ref().map(|b| b.len()).unwrap_or(chunk.len());
    let (final_tag, final_bytes, safety_fired) = if picked_tag != CodecTag::Zstd22
        && (picked_bytes.len() as f64) > (zstd_shadow as f64) * SAFETY_NET_THRESHOLD
    {
        // Picked codec is worse than zstd by more than threshold;
        // substitute zstd's output.
        (
            CodecTag::Zstd22,
            zstd_bytes.expect("codec list must include Zstd22 for safety net"),
            true,
        )
    } else {
        (picked_tag, picked_bytes, false)
    };

    Ok(ChunkOutcome {
        tag: final_tag,
        bytes: final_bytes,
        zstd_shadow_bytes: zstd_shadow,
        safety_net_fired: safety_fired,
    })
}

// -----------------------------------------------------------------------------
// Whole-blob encode / decode
// -----------------------------------------------------------------------------

/// Per-run metrics aggregate. Populated as the dispatcher walks the
/// input; caller emits these to CloudWatch + DDB per
/// `HYBRID_CODEC_DESIGN.md`'s metrics spec.
#[derive(Debug, Default, Clone)]
pub struct DispatchStats {
    /// Total raw bytes in.
    pub bytes_in: u64,
    /// Total compressed bytes out (including the blob header + per-chunk overhead).
    pub bytes_out: u64,
    /// Number of chunks processed.
    pub chunks_total: u32,
    /// Sum of `zstd_shadow_bytes` across all chunks — what the
    /// output would have been under zstd-22 alone.
    pub zstd_shadow_bytes: u64,
    /// Per-codec bytes emitted, keyed by tag name.
    pub per_codec_bytes: std::collections::HashMap<&'static str, u64>,
    /// Per-codec chunk counts.
    pub per_codec_chunks: std::collections::HashMap<&'static str, u64>,
    /// Number of chunks where the Stage 3 safety net substituted zstd.
    pub safety_net_substitutions: u32,
    /// Wall-clock encode seconds.
    pub encode_seconds: f64,
}

impl DispatchStats {
    /// Dispatcher's achieved ratio (bytes_out / bytes_in).
    pub fn ratio(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            self.bytes_out as f64 / self.bytes_in as f64
        }
    }
    /// zstd-only shadow ratio.
    pub fn zstd_shadow_ratio(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            self.zstd_shadow_bytes as f64 / self.bytes_in as f64
        }
    }
    /// Percent savings vs zstd alone — positive = we beat zstd.
    pub fn savings_vs_zstd_pct(&self) -> f64 {
        if self.zstd_shadow_bytes == 0 {
            0.0
        } else {
            100.0
                * (self.zstd_shadow_bytes as f64 - self.bytes_out as f64)
                / self.zstd_shadow_bytes as f64
        }
    }
    /// Compression throughput in MB/s.
    pub fn throughput_mb_per_sec(&self) -> f64 {
        if self.encode_seconds <= 0.0 {
            0.0
        } else {
            (self.bytes_in as f64 / 1_048_576.0) / self.encode_seconds
        }
    }
}

/// Encode `input` into a hybrid blob. Returns the blob plus per-run
/// stats for metrics emission.
///
/// `codecs` is the codec menu to probe per chunk; must include at
/// least `Zstd22` (the safety-net reference). `chunk_size` controls
/// chunk boundaries; default is `DEFAULT_CHUNK_SIZE`.
pub fn encode_blob(
    input: &[u8],
    codecs: &[Box<dyn Codec>],
    chunk_size: usize,
) -> Result<(Vec<u8>, DispatchStats)> {
    assert!(
        codecs.iter().any(|c| c.tag() == CodecTag::Zstd22),
        "encode_blob requires Zstd22 in the codec list (safety-net reference)"
    );
    assert!(chunk_size > 0 && chunk_size <= u32::MAX as usize);

    let start = Instant::now();
    let mut out = Vec::with_capacity(input.len() / 4);
    let mut stats = DispatchStats::default();

    // Blob header (placeholder for chunk count; we rewrite at end).
    out.write_all(BLOB_MAGIC)?;
    out.write_u16::<BigEndian>(BLOB_VERSION)?;
    out.write_u32::<BigEndian>(chunk_size as u32)?;
    out.write_u64::<BigEndian>(input.len() as u64)?;
    let chunk_count_pos = out.len();
    out.write_u32::<BigEndian>(0)?; // placeholder

    // Walk chunks.
    let mut chunk_ct: u32 = 0;
    for chunk in input.chunks(chunk_size) {
        let outcome = dispatch_chunk(chunk, codecs)?;

        out.write_u8(outcome.tag as u8)?;
        out.write_u32::<BigEndian>(outcome.bytes.len() as u32)?;
        out.write_all(&outcome.bytes)?;

        stats.bytes_in += chunk.len() as u64;
        stats.zstd_shadow_bytes += outcome.zstd_shadow_bytes as u64;
        *stats
            .per_codec_bytes
            .entry(outcome.tag.name())
            .or_insert(0) += outcome.bytes.len() as u64;
        *stats
            .per_codec_chunks
            .entry(outcome.tag.name())
            .or_insert(0) += 1;
        if outcome.safety_net_fired {
            stats.safety_net_substitutions += 1;
        }
        chunk_ct += 1;
    }

    // Patch chunk count in header.
    out[chunk_count_pos..chunk_count_pos + 4].copy_from_slice(&chunk_ct.to_be_bytes());

    stats.bytes_out = out.len() as u64;
    stats.chunks_total = chunk_ct;
    stats.encode_seconds = start.elapsed().as_secs_f64();

    Ok((out, stats))
}

/// Decode a hybrid blob back to bytes.
pub fn decode_blob(blob: &[u8], codecs: &[Box<dyn Codec>]) -> Result<Vec<u8>> {
    let mut cur = Cursor::new(blob);

    // Header.
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != BLOB_MAGIC {
        return Err(Error::BadHybridHeader("magic bytes mismatch"));
    }
    let version = cur.read_u16::<BigEndian>()?;
    if version != BLOB_VERSION {
        return Err(Error::BadHybridHeader("unsupported blob version"));
    }
    let _chunk_size = cur.read_u32::<BigEndian>()?;
    let raw_len = cur.read_u64::<BigEndian>()?;
    let chunk_ct = cur.read_u32::<BigEndian>()?;

    // Build a tag → codec lookup from the supplied codec list so
    // decoding is just "read the tag, dispatch to the right codec."
    // Codecs present in the blob but absent from `codecs` yield a
    // clean error instead of silent corruption.
    let mut lookup: std::collections::HashMap<CodecTag, &Box<dyn Codec>> =
        std::collections::HashMap::with_capacity(codecs.len());
    for c in codecs {
        lookup.insert(c.tag(), c);
    }

    let mut out = Vec::with_capacity(raw_len as usize);
    for _ in 0..chunk_ct {
        let tag_u8 = cur.read_u8()?;
        let tag = CodecTag::from_u8(tag_u8)?;
        let enc_len = cur.read_u32::<BigEndian>()? as usize;
        let pos = cur.position() as usize;
        let slice = blob
            .get(pos..pos + enc_len)
            .ok_or(Error::UnexpectedEof { pos })?;
        let codec = lookup
            .get(&tag)
            .ok_or_else(|| Error::BadHybridHeader("chunk uses codec not in decode registry"))?;
        let decoded = codec.decode(slice)?;
        out.extend_from_slice(&decoded);
        cur.set_position((pos + enc_len) as u64);
    }

    if out.len() as u64 != raw_len {
        return Err(Error::BadHybridHeader("decoded length mismatch"));
    }
    Ok(out)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Codec list the tests use. Neural is out of scope for the pure
    /// classical dispatcher tests — its round-trip is covered in
    /// `codec.rs` already.
    fn classical_codecs() -> Vec<Box<dyn Codec>> {
        vec![
            Box::new(PassthroughCodec),
            Box::new(Lz4Codec),
            Box::new(Zstd22Codec),
            Box::new(Bzip3Codec),
        ]
    }

    #[test]
    fn round_trip_empty() {
        let codecs = classical_codecs();
        let (blob, _stats) = encode_blob(b"", &codecs, DEFAULT_CHUNK_SIZE).unwrap();
        let out = decode_blob(&blob, &codecs).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn round_trip_single_chunk_text() {
        let input = "The quick brown fox jumps over the lazy dog. ".repeat(100);
        let codecs = classical_codecs();
        let (blob, stats) = encode_blob(input.as_bytes(), &codecs, DEFAULT_CHUNK_SIZE).unwrap();
        let out = decode_blob(&blob, &codecs).unwrap();
        assert_eq!(out, input.as_bytes());
        assert_eq!(stats.chunks_total, 1);
        assert!(stats.bytes_out < stats.bytes_in); // something compressed
    }

    #[test]
    fn round_trip_multi_chunk_binary() {
        // Ensure a multi-chunk blob round-trips across codec-tag
        // switches. Use a mix of text-like and binary-looking content.
        let mut input = Vec::new();
        input.extend(b"lorem ipsum dolor sit amet ".repeat(3000).iter());
        input.extend((0u8..=255).cycle().take(200_000)); // semi-random binary-ish
        let codecs = classical_codecs();
        let (blob, stats) = encode_blob(&input, &codecs, 32 * 1024).unwrap();
        let out = decode_blob(&blob, &codecs).unwrap();
        assert_eq!(out, input);
        assert!(stats.chunks_total > 1);
    }

    #[test]
    fn magic_byte_prescreen() {
        // JPEG magic bytes → Passthrough (not compressed further).
        let mut fake_jpeg = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10];
        fake_jpeg.extend(vec![0u8; 32_000]); // pretend JPEG body
        let codecs = classical_codecs();
        let (blob, stats) = encode_blob(&fake_jpeg, &codecs, DEFAULT_CHUNK_SIZE).unwrap();
        let out = decode_blob(&blob, &codecs).unwrap();
        assert_eq!(out, fake_jpeg);
        let &pt_chunks = stats
            .per_codec_chunks
            .get(CodecTag::Passthrough.name())
            .unwrap_or(&0);
        assert_eq!(pt_chunks, 1);
    }

    #[test]
    fn safety_net_never_produces_worse_than_zstd() {
        // Some data bzip3 wins, some zstd wins. Verify that in every
        // chunk, the final emitted bytes are <= zstd alone × 1.01.
        let input = b"abcdefghij".repeat(20_000);
        let codecs = classical_codecs();
        let (_blob, stats) = encode_blob(&input, &codecs, 64 * 1024).unwrap();
        // The zstd-shadow should be an upper bound (within threshold)
        // on what we emitted.
        let tolerance = (stats.zstd_shadow_bytes as f64 * SAFETY_NET_THRESHOLD) as u64;
        // bytes_out includes blob-framing overhead; subtract a
        // generous allowance for header (22 B + per-chunk 5 B).
        let framing = 22 + (stats.chunks_total as u64) * 5;
        assert!(
            stats.bytes_out.saturating_sub(framing) <= tolerance,
            "dispatcher bytes_out ({}) exceeded zstd shadow ({}) × {} + framing ({})",
            stats.bytes_out,
            stats.zstd_shadow_bytes,
            SAFETY_NET_THRESHOLD,
            framing,
        );
    }

    #[test]
    fn codec_tag_roundtrip() {
        for &t in &[
            CodecTag::Passthrough,
            CodecTag::Lz4,
            CodecTag::Zstd22,
            CodecTag::ZstdDict,
            CodecTag::Bzip3,
            CodecTag::BrotliDict,
            CodecTag::Clp,
            CodecTag::Neural,
        ] {
            assert_eq!(CodecTag::from_u8(t as u8).unwrap(), t);
        }
        assert!(CodecTag::from_u8(200).is_err());
    }
}
