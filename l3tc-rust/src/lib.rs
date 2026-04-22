//! l3tc — learned lossless text compression.
//!
//! This crate is the Rust port of L3TC (Alipay, AAAI 2025): a neural
//! text compressor that uses a small RWKV language model plus
//! arithmetic coding to compress English text to about half the size
//! of classical compressors at the cost of 10-100x the compute.
//!
//! See the project README and PHASE_1.md for design and scope.
//!
//! # Architecture
//!
//! ```text
//!           +-----------+    +-----------+    +-----------+
//!   text -> | tokenizer | -> |   rwkv    | -> |arithmetic | -> bytes
//!           +-----------+    +-----------+    +-----------+
//!                              ^
//!                              |
//!                        +-----------+
//!                        |checkpoint |
//!                        +-----------+
//! ```
//!
//! - `tokenizer`: SentencePiece BPE with outlier bypass
//! - `rwkv`: RWKV-v4 forward pass (HiRA weights pre-merged)
//! - `arithmetic`: Nayuki-style arithmetic coder (integer-only, deterministic)
//! - `checkpoint`: reads a Rust-friendly binary format (produced by
//!   `scripts/convert_checkpoint.py`, not the original PyTorch `.pth`)
//! - `codec`: ties them together into compress/decompress functions
//!
//! # Status
//!
//! Phase 1 in progress. Modules are implemented in dependency order:
//! `bitio` and `arithmetic` first (pure logic, no external data),
//! then `checkpoint` and `tensor`, then `tokenizer`, then `rwkv`,
//! then `codec`.

// Unsafe is denied globally except in a single function in tensor.rs
// that wraps matrixmultiply::sgemm. See the comment there for the
// safety argument.
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod arithmetic;
pub mod backend;
pub mod bitio;
pub mod checkpoint;
pub mod codec;
pub mod dispatcher;
pub mod error;
pub mod rwkv;
pub mod tensor;
pub mod tokenizer;

pub use dispatcher::{
    decode_blob as hybrid_decode, encode_blob as hybrid_encode, Bzip3Codec, ClpStub, Codec,
    CodecTag, DispatchStats, Lz4Codec, NeuralCodec, PassthroughCodec, Zstd22Codec, ZstdDictCodec,
    DEFAULT_CHUNK_SIZE,
};

pub use backend::{Backend, GPU_AUTO_THRESHOLD_BYTES};
pub use checkpoint::{Checkpoint, Tensor};
pub use codec::{
    audit_compress, compress, decode_writer, decompress, decompress_bytes, dump_teacher,
    encode_reader, peek_header, profile_compress, AuditStats, HeaderPeek, ProfileStats,
    DEFAULT_SEGMENT_BYTES,
};
pub use error::{Error, Result};
pub use rwkv::{Model, Session};
pub use tokenizer::{EncodedSegment, Tokenizer};

/// Highest specialist-model-bundle format version this binary
/// understands. Bumped when the on-disk layout or the manifest
/// schema changes in a way a newer bundle could exercise.
///
/// The `install-models` subcommand refuses to install a bundle whose
/// `bundle_version` exceeds this number, so an old binary won't
/// silently extract a tarball it can't actually use. See
/// `l3tc-rust/src/bin/l3tc/install_models.rs` for the manifest
/// schema.
pub const MODEL_BUNDLE_VERSION: u32 = 1;
