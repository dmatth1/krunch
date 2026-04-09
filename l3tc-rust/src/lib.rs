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
pub mod bitio;
pub mod checkpoint;
pub mod codec;
pub mod error;
pub mod rwkv;
pub mod tensor;
pub mod tokenizer;

pub use checkpoint::{Checkpoint, Tensor};
pub use codec::{
    audit_compress, compress, decode_writer, decompress, decompress_bytes, encode_reader,
    AuditStats, DEFAULT_SEGMENT_BYTES,
};
pub use error::{Error, Result};
pub use rwkv::{Model, Session};
pub use tokenizer::{EncodedSegment, Tokenizer};
