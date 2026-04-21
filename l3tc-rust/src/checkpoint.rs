//! Reader for the Rust-friendly checkpoint format produced by
//! `scripts/convert_checkpoint.py`.
//!
//! This module intentionally does NOT parse PyTorch pickle files.
//! The Python converter does all the hard work (HiRA merging,
//! renaming, shape normalization) and emits a single flat binary
//! that we can read with trivial byte-level code.
//!
//! # Format
//!
//! ```text
//! magic:          "L3TC"          (4 bytes)
//! version:        u32 LE
//! n_tensors:      u32 LE
//! For each tensor:
//!     name_len:   u32 LE
//!     name:       utf-8 bytes (name_len of them)
//!     ndim:       u32 LE
//!     dims:       u32 LE * ndim
//!     dtype:      u32 LE (0 = f32)
//!     data_len:   u64 LE (bytes)
//!     data:       f32 LE * (data_len / 4)
//! trailer:        "END!"          (4 bytes)
//! ```

use crate::error::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use std::collections::HashMap;
use std::path::Path;

/// The file header we expect at the start of every checkpoint.
const MAGIC: &[u8; 4] = b"L3TC";

/// The trailer we expect at the end.
const TRAILER: &[u8; 4] = b"END!";

/// The format version this reader understands.
const VERSION: u32 = 1;

/// Dtype tag for f32.
const DTYPE_F32: u32 = 0;

/// A single tensor loaded from the checkpoint.
///
/// Shape is stored as a `Vec<usize>`; data is a plain `Vec<f32>` in
/// row-major layout. The tensor owns its data (no aliasing back into
/// the mmap) so the backing file can be closed after loading.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shape of the tensor, e.g. `[16384, 96]` for an embedding.
    pub shape: Vec<usize>,
    /// Flat row-major f32 data.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Total number of elements in the tensor (`product of dims`).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True if the tensor has rank `n`.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return `(rows, cols)` if the tensor is a 2-D matrix, else an error.
    pub fn as_matrix(&self) -> Result<(usize, usize)> {
        if self.shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: "rank-2".into(),
                actual: format!("{:?}", self.shape),
            });
        }
        Ok((self.shape[0], self.shape[1]))
    }

    /// Return the length if the tensor is a 1-D vector, else an error.
    pub fn as_vector(&self) -> Result<usize> {
        if self.shape.len() != 1 {
            return Err(Error::ShapeMismatch {
                expected: "rank-1".into(),
                actual: format!("{:?}", self.shape),
            });
        }
        Ok(self.shape[0])
    }
}

/// A loaded checkpoint: a map from tensor name to tensor data.
///
/// Construct with [`Checkpoint::load`]. Look up tensors with
/// [`Checkpoint::get`] or [`Checkpoint::take`].
#[derive(Debug)]
pub struct Checkpoint {
    tensors: HashMap<String, Tensor>,
}

impl Checkpoint {
    /// Load a checkpoint from a file on disk.
    ///
    /// The entire file is read into memory. For the L3TC-200K model
    /// this is ~13 MB; for L3TC-3.2M it's ~15 MB. Both fit easily in
    /// any modern machine's memory budget.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a checkpoint from an in-memory buffer. Primarily used for
    /// testing, but also useful if the caller wants to load from
    /// somewhere other than a file path.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut p = Parser::new(bytes);
        p.expect_magic(MAGIC, "header magic")?;
        let version = p.read_u32()?;
        if version != VERSION {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version {version}, expected {VERSION}"
            )));
        }
        let n_tensors = p.read_u32()? as usize;
        // Each tensor has at least: name_len(4) + 1 byte name + ndim(4)
        // + dtype(4) + data_len(8) = 21 bytes minimum. Cap allocation
        // against remaining input to prevent OOM on malformed headers.
        let max_possible = p.remaining() / 21;
        if n_tensors > max_possible {
            return Err(Error::BadCheckpoint(format!(
                "n_tensors={n_tensors} exceeds maximum possible for {} remaining bytes",
                p.remaining()
            )));
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let (name, tensor) = p.read_tensor().map_err(|e| {
                if let Error::BadCheckpoint(msg) = e {
                    Error::BadCheckpoint(format!("tensor {i}: {msg}"))
                } else {
                    e
                }
            })?;
            tensors.insert(name, tensor);
        }

        p.expect_magic(TRAILER, "trailer magic")?;
        if !p.is_at_end() {
            return Err(Error::BadCheckpoint(format!(
                "{} trailing bytes after trailer",
                p.remaining()
            )));
        }

        Ok(Self { tensors })
    }

    /// Number of tensors in the checkpoint.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Borrow a tensor by name.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::BadCheckpoint(format!("missing tensor: {name}")))
    }

    /// Take ownership of a tensor, removing it from the checkpoint.
    ///
    /// Useful when assembling the model from raw tensors: each
    /// tensor should be moved once into its final home, and `take`
    /// makes the ownership transfer explicit.
    pub fn take(&mut self, name: &str) -> Result<Tensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| Error::BadCheckpoint(format!("missing tensor: {name}")))
    }

    /// Iterate over all tensor names in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Take a tensor of any rank by name, asserting it matches the expected shape.
    pub fn take_shape(&mut self, name: &str, expected: &[usize]) -> Result<Tensor> {
        let tensor = self.take(name)?;
        if tensor.shape != expected {
            return Err(Error::ShapeMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{:?}", tensor.shape),
            });
        }
        Ok(tensor)
    }
}

/// Byte-level parser over the checkpoint buffer.
///
/// Private helper; the public interface is [`Checkpoint::load`].
struct Parser<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn is_at_end(&self) -> bool {
        self.pos == self.buf.len()
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(Error::BadCheckpoint(format!(
                "truncated at offset {} (need {n} bytes, have {})",
                self.pos,
                self.remaining()
            )));
        }
        Ok(())
    }

    fn expect_magic(&mut self, expected: &[u8; 4], what: &str) -> Result<()> {
        self.ensure(4)?;
        let seen = &self.buf[self.pos..self.pos + 4];
        if seen != expected {
            return Err(Error::BadCheckpoint(format!(
                "bad {what}: expected {:?}, got {:?}",
                std::str::from_utf8(expected).unwrap_or("?"),
                std::str::from_utf8(seen).unwrap_or("?")
            )));
        }
        self.pos += 4;
        Ok(())
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = LittleEndian::read_u32(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = LittleEndian::read_u64(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(v)
    }

    fn read_name(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        self.ensure(len)?;
        let slice = &self.buf[self.pos..self.pos + len];
        let name = std::str::from_utf8(slice)
            .map_err(|e| Error::BadCheckpoint(format!("bad tensor name utf-8: {e}")))?
            .to_owned();
        self.pos += len;
        Ok(name)
    }

    fn read_tensor(&mut self) -> Result<(String, Tensor)> {
        let name = self.read_name()?;
        let ndim = self.read_u32()? as usize;
        // Sanity: ndim × 4 bytes (u32 per dim) must fit in remaining
        // input. Prevents OOM from a malformed ndim value.
        if ndim > self.remaining() / 4 {
            return Err(Error::BadCheckpoint(format!(
                "{name}: ndim={ndim} exceeds remaining bytes"
            )));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(self.read_u32()? as usize);
        }
        let dtype = self.read_u32()?;
        if dtype != DTYPE_F32 {
            return Err(Error::BadCheckpoint(format!(
                "unsupported dtype {dtype} for {name}"
            )));
        }
        let data_len = self.read_u64()? as usize;
        self.ensure(data_len)?;

        let expected_numel: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                Error::BadCheckpoint(format!("{name}: shape {shape:?} overflows usize"))
            })?;
        let expected_bytes = expected_numel.checked_mul(4).ok_or_else(|| {
            Error::BadCheckpoint(format!("{name}: numel {expected_numel} * 4 overflows"))
        })?;
        if data_len != expected_bytes {
            return Err(Error::BadCheckpoint(format!(
                "{name}: data length {data_len} does not match shape {shape:?} * 4 = {expected_bytes}"
            )));
        }

        let mut data = Vec::with_capacity(expected_numel);
        for i in 0..expected_numel {
            let off = self.pos + i * 4;
            let v = LittleEndian::read_f32(&self.buf[off..off + 4]);
            data.push(v);
        }
        self.pos += data_len;

        Ok((name, Tensor { shape, data }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tensor(buf: &mut Vec<u8>, name: &str, shape: &[usize], data: &[f32]) {
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype f32
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        buf.extend_from_slice(&(data_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&data_bytes);
    }

    fn build_checkpoint(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, shape, data) in tensors {
            write_tensor(&mut buf, name, shape, data);
        }
        buf.extend_from_slice(TRAILER);
        buf
    }

    #[test]
    fn roundtrip_single_tensor() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let buf = build_checkpoint(&[("foo", vec![3, 4], data.clone())]);
        let ckpt = Checkpoint::from_bytes(&buf).unwrap();
        assert_eq!(ckpt.len(), 1);
        let t = ckpt.get("foo").unwrap();
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.data, data);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.as_matrix().unwrap(), (3, 4));
    }

    #[test]
    fn roundtrip_multiple_tensors() {
        let buf = build_checkpoint(&[
            ("weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("bias", vec![2], vec![0.5, 0.25]),
        ]);
        let mut ckpt = Checkpoint::from_bytes(&buf).unwrap();
        assert_eq!(ckpt.len(), 2);

        let b = ckpt.take_shape("bias", &[2]).unwrap();
        assert_eq!(b.data, vec![0.5, 0.25]);

        let w = ckpt.take_shape("weight", &[2, 2]).unwrap();
        assert_eq!(w.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = build_checkpoint(&[("foo", vec![1], vec![1.0])]);
        buf[0] = b'X';
        let err = Checkpoint::from_bytes(&buf).unwrap_err();
        assert!(matches!(err, Error::BadCheckpoint(_)));
    }

    #[test]
    fn bad_version_rejected() {
        let mut buf = build_checkpoint(&[("foo", vec![1], vec![1.0])]);
        buf[4] = 99; // overwrite the low byte of version with 99
        let err = Checkpoint::from_bytes(&buf).unwrap_err();
        assert!(matches!(err, Error::BadCheckpoint(_)));
    }

    #[test]
    fn truncated_rejected() {
        let buf = build_checkpoint(&[("foo", vec![1], vec![1.0])]);
        let err = Checkpoint::from_bytes(&buf[..buf.len() - 4]).unwrap_err();
        assert!(matches!(err, Error::BadCheckpoint(_)));
    }

    #[test]
    fn shape_mismatch_on_take_shape() {
        let buf = build_checkpoint(&[("foo", vec![3], vec![1.0, 2.0, 3.0])]);
        let mut ckpt = Checkpoint::from_bytes(&buf).unwrap();
        let err = ckpt.take_shape("foo", &[4]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }
}
