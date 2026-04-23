//! Weight loader for RWKV-4-Pile safetensors files.
//!
//! Input: safetensors file produced by
//! `scripts/convert_rwkv_pth_to_safetensors.py`.
//!
//! Output: a `RwkvV4PileWeights` struct holding device-resident
//! tensors, ready to be borrowed by `RwkvV4Pile169m::forward()`.
//!
//! Layout (matches BlinkDL's published checkpoint; 222 tensors for
//! the 169M-param 12-layer variant):
//!
//! - `emb.weight`            (vocab × n_embd)
//! - `blocks.0.ln0.{weight,bias}`   pre-embed LayerNorm (only on block 0)
//! - per-block (×N layers):
//!     `blocks.i.ln1.{weight,bias}`       (n_embd,)
//!     `blocks.i.ln2.{weight,bias}`       (n_embd,)
//!     `blocks.i.att.time_decay`          (n_embd,)
//!     `blocks.i.att.time_first`          (n_embd,)
//!     `blocks.i.att.time_mix_{k,v,r}`    (1, 1, n_embd)
//!     `blocks.i.att.{key,value,receptance,output}.weight`   (n_embd × n_embd)
//!     `blocks.i.ffn.time_mix_{k,r}`      (1, 1, n_embd)
//!     `blocks.i.ffn.key.weight`          (4×n_embd × n_embd)
//!     `blocks.i.ffn.receptance.weight`   (n_embd × n_embd)
//!     `blocks.i.ffn.value.weight`        (n_embd × 4×n_embd)
//! - `ln_out.{weight,bias}`            (n_embd,)
//! - `head.weight`                     (vocab × n_embd)

#![cfg(feature = "rwkv-v4-pile")]

use std::path::Path;

use crate::error::{Error, Result};

/// RWKV-4-Pile config for the 169M variant. Other variants (430M,
/// 1.5B) would need different `n_layer`, `n_embd`.
#[derive(Debug, Clone, Copy)]
pub struct RwkvConfig {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub ctx_len: usize,
}

impl RwkvConfig {
    /// The published RWKV-4-Pile-169M config.
    pub const RWKV_4_PILE_169M: Self = Self {
        vocab: 50277,
        n_embd: 768,
        n_layer: 12,
        ctx_len: 1024,
    };
}

/// Host-resident f32 tensor. Converted to device tensors by the
/// backend when the model is moved to GPU.
#[derive(Debug)]
pub struct HostTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Weights for one RWKV-v4 block. All tensors are (n_embd,)
/// unless noted.
#[derive(Debug)]
pub struct BlockWeights {
    /// Pre-embed LN, only populated on block 0 (None on others).
    pub ln0: Option<(HostTensor, HostTensor)>, // (weight, bias)

    pub ln1_w: HostTensor,
    pub ln1_b: HostTensor,
    pub ln2_w: HostTensor,
    pub ln2_b: HostTensor,

    // Time-mix (attention) params
    pub att_time_decay: HostTensor,  // (n_embd,) — `w` for WKV
    pub att_time_first: HostTensor,  // (n_embd,) — `u` for WKV
    pub att_time_mix_k: HostTensor,  // (1, 1, n_embd), flattened
    pub att_time_mix_v: HostTensor,
    pub att_time_mix_r: HostTensor,
    pub att_key_w: HostTensor,        // (n_embd × n_embd)
    pub att_value_w: HostTensor,
    pub att_receptance_w: HostTensor,
    pub att_output_w: HostTensor,

    // Channel-mix (ffn) params
    pub ffn_time_mix_k: HostTensor,
    pub ffn_time_mix_r: HostTensor,
    pub ffn_key_w: HostTensor,        // (4×n_embd × n_embd)
    pub ffn_receptance_w: HostTensor,
    pub ffn_value_w: HostTensor,      // (n_embd × 4×n_embd)
}

/// All weights for the RWKV-4-Pile model, host-resident.
///
/// Call `upload_to_cuda` to move to GPU memory for inference.
#[derive(Debug)]
pub struct RwkvV4PileWeights {
    pub config: RwkvConfig,
    pub emb: HostTensor,       // (vocab × n_embd)
    pub blocks: Vec<BlockWeights>,
    pub ln_out_w: HostTensor,
    pub ln_out_b: HostTensor,
    pub head_w: HostTensor,    // (vocab × n_embd)
}

impl RwkvV4PileWeights {
    /// Load weights from a safetensors file. Does NOT move to GPU —
    /// call `upload_to_cuda` for that.
    ///
    /// Validates shapes against `config`; returns `Error` on mismatch.
    pub fn from_safetensors(path: &Path, config: RwkvConfig) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::Io(e))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Model { message: format!("safetensors parse: {e}") })?;

        let read = |name: &str, expected_shape: &[usize]| -> Result<HostTensor> {
            let view = st.tensor(name).map_err(|e| Error::Model {
                message: format!("missing tensor {name}: {e}"),
            })?;
            let shape: Vec<usize> = view.shape().to_vec();
            if shape != expected_shape {
                return Err(Error::Model {
                    message: format!(
                        "{name}: expected shape {:?}, got {:?}",
                        expected_shape, shape
                    ),
                });
            }
            // Convert to f32. safetensors supports f32 + f16 + bf16.
            let data = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let raw = view.data();
                    let mut out = vec![0f32; raw.len() / 4];
                    for (i, chunk) in raw.chunks_exact(4).enumerate() {
                        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    out
                }
                dt => {
                    return Err(Error::Model {
                        message: format!(
                            "{name}: dtype {:?} not yet supported (need f32 — \
                             run converter with --dtype fp32)",
                            dt
                        ),
                    });
                }
            };
            Ok(HostTensor { data, shape })
        };

        // Embedding + head
        let emb = read("emb.weight", &[config.vocab, config.n_embd])?;
        let head_w = read("head.weight", &[config.vocab, config.n_embd])?;
        let ln_out_w = read("ln_out.weight", &[config.n_embd])?;
        let ln_out_b = read("ln_out.bias", &[config.n_embd])?;

        // Blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let prefix = format!("blocks.{i}");

            // ln0 only on block 0
            let ln0 = if i == 0 {
                let w = read(&format!("{prefix}.ln0.weight"), &[config.n_embd])?;
                let b = read(&format!("{prefix}.ln0.bias"), &[config.n_embd])?;
                Some((w, b))
            } else {
                None
            };

            let b = BlockWeights {
                ln0,
                ln1_w: read(&format!("{prefix}.ln1.weight"), &[config.n_embd])?,
                ln1_b: read(&format!("{prefix}.ln1.bias"), &[config.n_embd])?,
                ln2_w: read(&format!("{prefix}.ln2.weight"), &[config.n_embd])?,
                ln2_b: read(&format!("{prefix}.ln2.bias"), &[config.n_embd])?,

                att_time_decay: read(&format!("{prefix}.att.time_decay"), &[config.n_embd])?,
                att_time_first: read(&format!("{prefix}.att.time_first"), &[config.n_embd])?,
                att_time_mix_k: read(&format!("{prefix}.att.time_mix_k"), &[1, 1, config.n_embd])?,
                att_time_mix_v: read(&format!("{prefix}.att.time_mix_v"), &[1, 1, config.n_embd])?,
                att_time_mix_r: read(&format!("{prefix}.att.time_mix_r"), &[1, 1, config.n_embd])?,
                att_key_w: read(&format!("{prefix}.att.key.weight"), &[config.n_embd, config.n_embd])?,
                att_value_w: read(&format!("{prefix}.att.value.weight"), &[config.n_embd, config.n_embd])?,
                att_receptance_w: read(&format!("{prefix}.att.receptance.weight"), &[config.n_embd, config.n_embd])?,
                att_output_w: read(&format!("{prefix}.att.output.weight"), &[config.n_embd, config.n_embd])?,

                ffn_time_mix_k: read(&format!("{prefix}.ffn.time_mix_k"), &[1, 1, config.n_embd])?,
                ffn_time_mix_r: read(&format!("{prefix}.ffn.time_mix_r"), &[1, 1, config.n_embd])?,
                ffn_key_w: read(&format!("{prefix}.ffn.key.weight"), &[4 * config.n_embd, config.n_embd])?,
                ffn_receptance_w: read(&format!("{prefix}.ffn.receptance.weight"), &[config.n_embd, config.n_embd])?,
                ffn_value_w: read(&format!("{prefix}.ffn.value.weight"), &[config.n_embd, 4 * config.n_embd])?,
            };
            blocks.push(b);
        }

        Ok(Self {
            config,
            emb,
            blocks,
            ln_out_w,
            ln_out_b,
            head_w,
        })
    }
}
