//! Error type for the l3tc library.
//!
//! The library uses `thiserror` to produce typed errors. Every failure
//! mode is a named variant, which means callers can match on it. The
//! CLI binary converts these to `anyhow::Error` for ergonomic reporting.

use std::io;
use thiserror::Error;

/// The library's unified error type.
///
/// Every public function in this crate returns `Result<T, Error>`
/// (aliased as [`Result<T>`]). Callers should match on the variants
/// if they need programmatic handling; for display, `{}` gives a
/// human-readable message.
#[derive(Debug, Error)]
pub enum Error {
    /// I/O error from the file system, stdin, stdout, etc.
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),

    /// The input data was not valid UTF-8 where UTF-8 was expected.
    #[error("invalid utf-8 in input")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// The compressed stream ended unexpectedly.
    #[error("unexpected end of compressed stream at byte {pos}")]
    UnexpectedEof {
        /// Byte offset within the compressed stream where EOF was hit.
        pos: usize,
    },

    /// A checkpoint file could not be parsed.
    #[error("bad checkpoint: {0}")]
    BadCheckpoint(String),

    /// A tensor shape mismatch during model loading or inference.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Shape we expected, as a string for error reporting.
        expected: String,
        /// Shape we actually found.
        actual: String,
    },

    /// The arithmetic coder received a symbol with zero probability,
    /// which would correspond to an infinite code length.
    #[error("arithmetic coder: zero probability symbol {symbol}")]
    ZeroProbability {
        /// The symbol index that had zero probability.
        symbol: u32,
    },

    /// The tokenizer model file could not be loaded.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// A feature or code path has not yet been implemented.
    ///
    /// Distinct variant (rather than a `todo!()` panic) so partial
    /// builds during Phase 1 can fail gracefully with clear messages.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),

    /// A hybrid-dispatcher blob had an unrecognised codec tag. Bumped
    /// when a newer dispatcher version writes a blob an older binary
    /// then tries to decode.
    #[error("hybrid dispatcher: unknown codec tag {0}")]
    UnknownCodecTag(u8),

    /// A classical codec (zstd / bzip3 / lz4) returned an error through
    /// its library API. The string wraps the underlying error's display.
    #[error("classical codec {codec}: {message}")]
    ClassicalCodec {
        /// Which codec produced the error, e.g. `"zstd"` or `"bzip3"`.
        codec: &'static str,
        /// Underlying error message.
        message: String,
    },

    /// The hybrid blob header's magic bytes, version, or length fields
    /// were malformed. This catches accidental feeding of a non-hybrid
    /// blob to the hybrid decoder.
    #[error("hybrid dispatcher: bad blob header ({0})")]
    BadHybridHeader(&'static str),
}

/// Convenience `Result` alias bound to [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
