//! RWKV-4-Pile-169M pretrained inference for Tier-2 compression.
//!
//! Forward-pass implementation of RWKV-v4 architecture, specialized
//! for the published BlinkDL `RWKV-4-Pile-169M-20220807-8023` weights
//! (12 layers × 768 hidden × 50277 vocab, GPT-NeoX BPE tokenizer).
//!
//! # Status
//!
//! **Phase 1 complete**: weight loading + model struct + forward API.
//! **Phase 2 pending**: CUDA implementations of the tensor ops
//! (matmul via cuBLAS, layer norm, elementwise sigmoid/relu²/mix,
//! WKV kernel integration). The `forward()` method currently returns
//! `Error::NotImplemented` — Phase 2 fills in the CUDA path.
//!
//! # Design
//!
//! The forward pass mirrors BlinkDL's `RWKV_GPT` in
//! `RWKV-LM/RWKV-v4/src/model_run.py`, step by step:
//!
//! 1. Embed input tokens `(B, T) → (B, T, C)` via emb lookup
//! 2. Apply `blocks.0.ln0` layer norm (only on block 0)
//! 3. For each block:
//!    - Time-mix (attention):
//!      - LN1
//!      - Shift by one, blend with time_mix_{k,v,r} → xk, xv, xr
//!      - k = key @ xk, v = value @ xv, r = sigmoid(receptance @ xr)
//!      - wkv = WKV_CUDA(w=time_decay, u=time_first, k, v)  [fused kernel]
//!      - attn_out = output @ (r * wkv)
//!      - residual: x = x + attn_out
//!    - Channel-mix (ffn):
//!      - LN2
//!      - Shift by one, blend with ffn.time_mix_{k,r} → xk, xr
//!      - k = relu(ffn.key @ xk) ** 2
//!      - r = sigmoid(ffn.receptance @ xr)
//!      - ffn_out = r * (ffn.value @ k)
//!      - residual: x = x + ffn_out
//! 4. Final ln_out, head projection → `(B, T, vocab)` logits
//!
//! The WKV CUDA kernel lives in `src/cuda/wkv.cu` (compiled by
//! `build.rs`). All other ops are generic tensor math that Phase 2
//! will implement via cudarc + cuBLAS.

#![cfg(feature = "rwkv-v4-pile")]

use std::path::Path;
use std::sync::Arc;

use crate::backend::cuda::CudaContextHandle;
use crate::error::{Error, Result};
use crate::weights_rwkv::{RwkvConfig, RwkvV4PileWeights};

/// Per-block running state for sequence-forward inference.
///
/// All three buffers are `(B, C)` fp32 — the WKV recurrence runs in
/// fp32 for numerical stability even when activations are fp16.
/// State persists across chunks of a single compression stream so a
/// long document can be compressed without re-processing the prefix.
pub struct BlockState {
    /// Numerator of the softmax-weighted rolling sum (B, C).
    pub aa: Vec<f32>,
    /// Denominator of the softmax-weighted rolling sum (B, C).
    pub bb: Vec<f32>,
    /// Log-max for numerical stability (B, C).
    pub pp: Vec<f32>,
    /// Last token's residual input for time-mix shift (B, C).
    pub xx: Vec<f32>,
    /// Last token's residual input for channel-mix shift (B, C).
    pub xx_ffn: Vec<f32>,
}

impl BlockState {
    /// Fresh state zeroed for a new stream of batch size B.
    pub fn new(batch: usize, n_embd: usize) -> Self {
        let sz = batch * n_embd;
        Self {
            aa: vec![0.0; sz],
            bb: vec![0.0; sz],
            // pp starts at -1e38 (log-space -infinity)
            pp: vec![-1e38; sz],
            xx: vec![0.0; sz],
            xx_ffn: vec![0.0; sz],
        }
    }
}

/// RWKV-4-Pile-169M model, weights + backend handles.
///
/// Instantiated once per GPU worker; thread-safe to share across
/// requests via `Arc`. Each inference session creates fresh
/// `BlockState` vectors and holds a reference to this model.
pub struct RwkvV4Pile169m {
    pub config: RwkvConfig,
    pub weights: RwkvV4PileWeights,
    pub cuda: Arc<CudaContextHandle>,
}

impl RwkvV4Pile169m {
    /// Load pretrained weights + init the CUDA backend.
    ///
    /// Call once per process. Weight upload to GPU happens lazily on
    /// first `forward()` call (Phase 2) or at construction once that
    /// path is wired.
    pub fn from_safetensors(path: &Path, config: RwkvConfig) -> Result<Self> {
        let weights = RwkvV4PileWeights::from_safetensors(path, config)?;
        let cuda = Arc::new(CudaContextHandle::new()?);
        Ok(Self { config, weights, cuda })
    }

    /// Initial per-block state for a new compression stream.
    pub fn new_state(&self, batch: usize) -> Vec<BlockState> {
        (0..self.config.n_layer)
            .map(|_| BlockState::new(batch, self.config.n_embd))
            .collect()
    }

    /// Forward pass over a sequence of tokens.
    ///
    /// # Inputs
    /// - `tokens`: `(B, T)` flat row-major, `batch × seq_len` u32 ids
    /// - `state`: per-layer state updated in-place; caller may
    ///   persist state across chunks for long-document compression
    ///
    /// # Output
    /// `(B, T, vocab)` fp32 logits, row-major, ready to feed into
    /// `log_softmax` + the arithmetic coder.
    ///
    /// # Phase 2 work
    /// The body of this function needs the following CUDA ops:
    /// - Embedding lookup (single launch — trivial)
    /// - Layer norm (cuDNN or custom)
    /// - GEMM for all the `{key, value, receptance, output}.weight`
    ///   matmuls (cuBLAS `Sgemm` or `HGemm` for fp16)
    /// - Sigmoid + element-wise multiply (custom small kernels)
    /// - WKV kernel (DONE — see `cuda::launch_wkv_forward`)
    /// - ReLU squared for FFN (custom kernel or two generic ops)
    pub fn forward(
        &self,
        tokens: &[u32],
        batch: usize,
        seq_len: usize,
        _state: &mut [BlockState],
    ) -> Result<Vec<f32>> {
        if tokens.len() != batch * seq_len {
            return Err(Error::ShapeMismatch {
                expected: format!("batch * seq_len = {}", batch * seq_len),
                actual: format!("tokens.len() = {}", tokens.len()),
            });
        }
        if _state.len() != self.config.n_layer {
            return Err(Error::ShapeMismatch {
                expected: format!("n_layer = {}", self.config.n_layer),
                actual: format!("state.len() = {}", _state.len()),
            });
        }

        // Phase 2 fills this in. Structure:
        //
        // 1. Upload tokens to GPU (small, one memcpy)
        // 2. Allocate device buffers for intermediate activations if
        //    this is the first call (or reuse from a scratch cache)
        // 3. emb_gpu[tokens] → x  (lookup kernel)
        // 4. If block 0: x = layer_norm(x, ln0_w, ln0_b)
        // 5. for block in &self.weights.blocks:
        //        x = time_mix(x, block, &mut state[i])
        //        x = channel_mix(x, block, &mut state[i])
        // 6. x = layer_norm(x, ln_out_w, ln_out_b)
        // 7. logits = head_w @ x
        // 8. Download (B, T, vocab) logits to host; return as Vec<f32>
        Err(Error::NotImplemented(
            "RwkvV4Pile169m::forward — Phase 2 will implement the CUDA \
             forward path (embed, LN, GEMM, time-mix, channel-mix). The \
             WKV kernel itself is already ported to src/cuda/wkv.cu and \
             loaded by backend::cuda::CudaContextHandle.",
        ))
    }
}
