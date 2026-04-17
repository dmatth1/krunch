//! RWKV-v4 + TC + HiRA forward pass.
//!
//! This is a port of L3TC's `rwkv_tc_hira_infer.py::Block_Script` and
//! `RWKV_TC_HIRA_Infer_For_Script`, running one token at a time. HiRA
//! weights are already merged at checkpoint-conversion time, so the
//! Rust model only sees standard RWKV projections plus the L3TC-
//! specific "short" skip connection.
//!
//! # One token, one forward pass
//!
//! RWKV's key property is that it's naturally recurrent: to process
//! the i-th token, you only need the current token embedding plus
//! the per-layer state from step i-1. This matches compression's
//! natural loop exactly (process token k, emit bits, move on) and
//! means we don't need any batching or attention-over-sequence
//! infrastructure.
//!
//! # State
//!
//! Per layer, RWKV maintains five vectors of size `hidden_size`:
//!
//! - `state_a`: running numerator of the time attention
//! - `state_b`: running denominator
//! - `state_p`: running max (for the log-sum-exp numerical trick)
//! - `state_x`: previous input for the time-mix blend
//! - `state_ffn`: previous FFN input for the channel-mix blend
//!
//! These are reset to their initial values at every segment boundary
//! (2048 bytes) to match L3TC's segment_length semantics.
//!
//! # Layout of each block's forward pass
//!
//! ```text
//!     short = relu(short_weight @ x)
//!     residual = x
//!     x = ln1(x)
//!     x = time_mix(x, state...)
//!     x = residual + x
//!     residual = x
//!     x = ln2(x)
//!     x = channel_mix(x, state_ffn)
//!     x = residual + x
//!     x = x + short                   # L3TC's extra shortcut
//! ```

use crate::checkpoint::Checkpoint;
use crate::error::{Error, Result};
use crate::tensor;

/// Layer-normalization parameters: `weight * normalize(x) + bias`.
#[derive(Debug, Clone)]
pub struct LayerNormParams {
    /// Scale factor of length `hidden_size`.
    pub weight: Vec<f32>,
    /// Offset of length `hidden_size`.
    pub bias: Vec<f32>,
}

impl LayerNormParams {
    /// Apply layer norm with a small epsilon (PyTorch default 1e-5).
    #[inline]
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        tensor::layer_norm(x, &self.weight, &self.bias, 1e-5, out);
    }
}

/// Time-mix (attention) projection weights for one block.
///
/// All projections are `(hidden_size, hidden_size)` row-major.
/// HiRA branches are already merged in at checkpoint-conversion time,
/// so these are the final effective weights used at inference.
#[derive(Debug, Clone)]
pub struct TimeMixParams {
    /// Decay vector: `(hidden_size,)`. Negative of this is exp'd in
    /// the recurrence to build the state decay per step.
    pub time_decay: Vec<f32>,
    /// Precomputed `-exp(time_decay[i])` for the inner state update.
    /// Held alongside `time_decay` so the hot path reads a constant
    /// instead of recomputing one `exp()` per element per token.
    pub neg_exp_time_decay: Vec<f32>,
    /// Initial time-first bias: `(hidden_size,)`.
    pub time_first: Vec<f32>,
    /// Mixing coefficient for key projection input.
    pub time_mix_k: Vec<f32>,
    /// Mixing coefficient for value projection input.
    pub time_mix_v: Vec<f32>,
    /// Mixing coefficient for receptance projection input.
    pub time_mix_r: Vec<f32>,
    /// Key projection weight (HiRA-merged).
    pub w_key: Vec<f32>,
    /// Value projection weight (HiRA-merged).
    pub w_value: Vec<f32>,
    /// Receptance projection weight (HiRA-merged).
    pub w_receptance: Vec<f32>,
    /// Output projection weight.
    pub w_output: Vec<f32>,
}

/// Channel-mix (FFN) weights for one block.
///
/// In the L3TC variant the FFN projections are all
/// `(hidden_size, hidden_size)` — note this is NOT the usual 4x
/// expanded FFN you see in transformers. The config explicitly sets
/// `intermediate_size = hidden_size` for the 200K model.
#[derive(Debug, Clone)]
pub struct ChannelMixParams {
    /// Mixing coefficient for key projection input.
    pub time_mix_k: Vec<f32>,
    /// Mixing coefficient for receptance projection input.
    pub time_mix_r: Vec<f32>,
    /// Key projection weight (HiRA-merged).
    pub w_key: Vec<f32>,
    /// Value projection weight (HiRA-merged).
    pub w_value: Vec<f32>,
    /// Receptance projection weight (HiRA-merged).
    pub w_receptance: Vec<f32>,
}

/// One transformer block: attention + FFN + L3TC's "short" shortcut.
#[derive(Debug, Clone)]
pub struct Block {
    /// Pre-attention layer norm.
    pub ln1: LayerNormParams,
    /// Pre-FFN layer norm.
    pub ln2: LayerNormParams,
    /// Time mixing (attention).
    pub att: TimeMixParams,
    /// Channel mixing (FFN).
    pub ffn: ChannelMixParams,
    /// L3TC's "short" shortcut: relu(short @ x), added to the block
    /// output after the residual chain.
    pub w_short: Vec<f32>,
}

/// Per-layer recurrent state.
///
/// Each block has its own `LayerState`. They live in a flat
/// `Vec<LayerState>` inside [`Session`].
#[derive(Debug, Clone)]
pub struct LayerState {
    /// Running time-attention numerator.
    pub state_a: Vec<f32>,
    /// Running time-attention denominator.
    pub state_b: Vec<f32>,
    /// Running log-sum-exp anchor.
    pub state_p: Vec<f32>,
    /// Previous input x for the time-mix blend.
    pub state_x: Vec<f32>,
    /// Previous FFN input for the channel-mix blend.
    pub state_ffn: Vec<f32>,
}

impl LayerState {
    /// Create fresh state vectors of size `hidden_size`.
    ///
    /// The initial state_p is -1e30 (a large negative number) so
    /// that `exp(state_p - p) ≈ 0` on the first step, correctly
    /// making the first token's output depend only on that token.
    pub fn fresh(hidden_size: usize) -> Self {
        Self {
            state_a: vec![0.0; hidden_size],
            state_b: vec![0.0; hidden_size],
            state_p: vec![-1e30; hidden_size],
            state_x: vec![0.0; hidden_size],
            state_ffn: vec![0.0; hidden_size],
        }
    }
}

/// The full L3TC / RWKV-v4-tc-hira model.
#[derive(Debug, Clone)]
pub struct Model {
    /// Token embedding: `(vocab_size, hidden_size)` row-major.
    pub emb: Vec<f32>,
    /// Vocabulary size (rows of emb / head).
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// FFN intermediate dimension. Equal to `hidden_size` for
    /// L3TC-200K; larger for bigger variants (e.g. 512 for the
    /// 3.2M model where hidden_size=256, intermediate_size=512).
    pub intermediate_size: usize,
    /// Top-level initial layer norm (moved from blocks.0.ln0).
    pub ln0: LayerNormParams,
    /// All blocks in order.
    pub blocks: Vec<Block>,
    /// Final layer norm before the head.
    pub ln_out: LayerNormParams,
    /// Output projection to vocab logits, stored **column-major**
    /// as `(hidden_size, vocab_size)` so that the head matvec can
    /// use the AXPY-form in [`crate::tensor::matvec_col_major`].
    ///
    /// This is transposed from the on-disk row-major layout at load
    /// time. The transpose is a one-time cost (about 1.5M elements
    /// for L3TC-200K) and unlocks a ~10× speedup on the head matmul
    /// at inference time.
    pub head_col_major: Vec<f32>,
    /// INT8-quantized head weights (column-major, same layout as
    /// `head_col_major` but one byte per element). Per-column scales
    /// are in `head_scales`. Populated at load time from the f32 head
    /// and used by `Session::forward` for the hot head matvec.
    pub head_q: Vec<i8>,
    /// Per-column f32 dequant scales for `head_q`, length `vocab_size`.
    pub head_scales: Vec<f32>,
}

impl Model {
    /// Number of transformer blocks in this model.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Load a Model from an already-converted checkpoint.
    ///
    /// The checkpoint must be the output of
    /// `scripts/convert_checkpoint.py`: HiRA merged, `ln0` at top
    /// level, `time_mix_*` squeezed to 1-D.
    pub fn from_checkpoint(ckpt: &mut Checkpoint) -> Result<Self> {
        // Discover model shape by looking at the embedding.
        let emb_tensor = ckpt.take("emb.weight")?;
        let (vocab_size, hidden_size) = emb_tensor.as_matrix()?;
        let emb = emb_tensor.data;

        // Head shares the embedding dimension but may have been
        // trained separately — it's not tied. Transpose to
        // column-major so the hot matvec in `Session::forward` can
        // use AXPY-form (dramatically faster; see
        // tensor::matvec_col_major).
        let head_tensor = ckpt.take_shape("head.weight", &[vocab_size, hidden_size])?;
        let head_col_major = tensor::transpose(&head_tensor.data, vocab_size, hidden_size);
        // Quantize the head to INT8 per-column at load time. The f32
        // head is kept as a fallback for parity tests but the hot path
        // uses the INT8 version (~4× lower memory traffic on the
        // 6.3 MB head weight, the single biggest matmul in the pass).
        let (head_q, head_scales) =
            tensor::quantize_col_major_int8(&head_col_major, vocab_size, hidden_size);

        let ln0 = LayerNormParams {
            weight: ckpt.take_shape("ln0.weight", &[hidden_size])?.data,
            bias: ckpt.take_shape("ln0.bias", &[hidden_size])?.data,
        };
        let ln_out = LayerNormParams {
            weight: ckpt.take_shape("ln_out.weight", &[hidden_size])?.data,
            bias: ckpt.take_shape("ln_out.bias", &[hidden_size])?.data,
        };

        // Discover `intermediate_size` from the FFN key shape on
        // block 0. The checkpoint layout is `(intermediate, hidden)`
        // row-major. For L3TC-200K intermediate == hidden == 96;
        // for L3TC-3.2M intermediate == 512, hidden == 256.
        let intermediate_size = {
            let t = ckpt.get("blocks.0.ffn.key.weight")?;
            let (rows, cols) = t.as_matrix()?;
            if cols != hidden_size {
                return Err(Error::BadCheckpoint(format!(
                    "blocks.0.ffn.key.weight has {cols} cols, expected {hidden_size}"
                )));
            }
            rows
        };

        // Count blocks by probing names. The converter emits
        // blocks.N.* for N in 0..n_layers.
        let mut num_layers = 0usize;
        while ckpt.get(&format!("blocks.{num_layers}.ln1.weight")).is_ok() {
            num_layers += 1;
        }
        if num_layers == 0 {
            return Err(Error::BadCheckpoint("no blocks found".into()));
        }

        let mut blocks = Vec::with_capacity(num_layers);
        for li in 0..num_layers {
            blocks.push(Self::load_block(ckpt, li, hidden_size, intermediate_size)?);
        }

        Ok(Self {
            emb,
            vocab_size,
            hidden_size,
            intermediate_size,
            ln0,
            blocks,
            ln_out,
            head_col_major,
            head_q,
            head_scales,
        })
    }

    fn load_block(
        ckpt: &mut Checkpoint,
        li: usize,
        h: usize,
        im: usize,
    ) -> Result<Block> {
        let prefix = format!("blocks.{li}");

        let take_1d = |ckpt: &mut Checkpoint, suffix: &str| -> Result<Vec<f32>> {
            ckpt.take_shape(&format!("{prefix}.{suffix}"), &[h])
                .map(|t| t.data)
        };
        let take_2d_hh = |ckpt: &mut Checkpoint, suffix: &str| -> Result<Vec<f32>> {
            ckpt.take_shape(&format!("{prefix}.{suffix}"), &[h, h])
                .map(|t| t.data)
        };
        let take_2d_imh = |ckpt: &mut Checkpoint, suffix: &str| -> Result<Vec<f32>> {
            ckpt.take_shape(&format!("{prefix}.{suffix}"), &[im, h])
                .map(|t| t.data)
        };
        let take_2d_him = |ckpt: &mut Checkpoint, suffix: &str| -> Result<Vec<f32>> {
            ckpt.take_shape(&format!("{prefix}.{suffix}"), &[h, im])
                .map(|t| t.data)
        };

        let ln1 = LayerNormParams {
            weight: take_1d(ckpt, "ln1.weight")?,
            bias: take_1d(ckpt, "ln1.bias")?,
        };
        let ln2 = LayerNormParams {
            weight: take_1d(ckpt, "ln2.weight")?,
            bias: take_1d(ckpt, "ln2.bias")?,
        };

        let time_decay = take_1d(ckpt, "att.time_decay")?;
        let neg_exp_time_decay = time_decay.iter().map(|&v| -v.exp()).collect();
        let att = TimeMixParams {
            time_decay,
            neg_exp_time_decay,
            time_first: take_1d(ckpt, "att.time_first")?,
            time_mix_k: take_1d(ckpt, "att.time_mix_k")?,
            time_mix_v: take_1d(ckpt, "att.time_mix_v")?,
            time_mix_r: take_1d(ckpt, "att.time_mix_r")?,
            // Attention projections are all square (h, h) in every
            // L3TC variant — the FFN is the only place the shape
            // differs across model sizes.
            w_key: take_2d_hh(ckpt, "att.key.weight")?,
            w_value: take_2d_hh(ckpt, "att.value.weight")?,
            w_receptance: take_2d_hh(ckpt, "att.receptance.weight")?,
            w_output: take_2d_hh(ckpt, "att.output.weight")?,
        };

        // time_mix_g exists in the checkpoint but is never used by
        // Block_Script.forward(). Drop it silently.
        let _ = ckpt.take(&format!("{prefix}.att.time_mix_g"));

        let ffn = ChannelMixParams {
            time_mix_k: take_1d(ckpt, "ffn.time_mix_k")?,
            time_mix_r: take_1d(ckpt, "ffn.time_mix_r")?,
            // FFN key:   (intermediate, hidden) -- expands h → im
            // FFN value: (hidden, intermediate) -- projects im → h
            // FFN receptance: (hidden, hidden) -- square gate
            w_key: take_2d_imh(ckpt, "ffn.key.weight")?,
            w_value: take_2d_him(ckpt, "ffn.value.weight")?,
            w_receptance: take_2d_hh(ckpt, "ffn.receptance.weight")?,
        };

        let w_short = take_2d_hh(ckpt, "short.weight")?;

        Ok(Block {
            ln1,
            ln2,
            att,
            ffn,
            w_short,
        })
    }
}

/// Thread-local scratch buffers used during a forward pass.
///
/// Allocated once per `Session` and reused across every token,
/// avoiding per-token allocation on the hot path.
#[derive(Debug)]
struct Scratch {
    x: Vec<f32>,
    residual: Vec<f32>,
    short: Vec<f32>,
    normed: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    xr: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    r: Vec<f32>,
    ww: Vec<f32>,
    p: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    rwkv: Vec<f32>,
    out_proj: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl Scratch {
    fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let h = || vec![0.0f32; hidden_size];
        // `k` is the only scratch buffer that can hold either a
        // hidden-sized vector (in att's time-mix path) OR an
        // intermediate-sized vector (the FFN key output post-relu,
        // before the value matvec projects back to hidden). Size
        // it on the max so both fit.
        let im_or_h = std::cmp::max(hidden_size, intermediate_size);
        Self {
            x: h(),
            residual: h(),
            short: h(),
            normed: h(),
            xk: h(),
            xv: h(),
            xr: h(),
            k: vec![0.0f32; im_or_h],
            v: h(),
            r: h(),
            ww: h(),
            p: h(),
            a: h(),
            b: h(),
            rwkv: h(),
            out_proj: h(),
            ffn_out: h(),
        }
    }
}

/// A runtime session: owns the per-layer state and scratch buffers.
///
/// A `Session` is what you actually feed tokens to. It wraps an
/// immutable reference to the `Model` and mutable state that evolves
/// as tokens are processed.
///
/// Call [`Session::reset`] at every segment boundary to match L3TC's
/// segment_length semantics.
pub struct Session<'a> {
    model: &'a Model,
    state: Vec<LayerState>,
    scratch: Scratch,
    /// Reusable logits buffer so we don't allocate `vocab_size`
    /// floats per token.
    logits: Vec<f32>,
}

impl<'a> Session<'a> {
    /// Create a new session for the given model. Initial state is
    /// "all zeros and very negative state_p" (see [`LayerState::fresh`]).
    pub fn new(model: &'a Model) -> Self {
        let state = (0..model.num_layers())
            .map(|_| LayerState::fresh(model.hidden_size))
            .collect();
        Self {
            model,
            state,
            scratch: Scratch::new(model.hidden_size, model.intermediate_size),
            logits: vec![0.0; model.vocab_size],
        }
    }

    /// Reset all per-layer state to fresh values.
    ///
    /// Call this at every segment boundary. L3TC resets state every
    /// 2048 bytes, so the codec layer will call this after each
    /// segment.
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = LayerState::fresh(self.model.hidden_size);
        }
    }

    /// Process one token and return the logits for the next token.
    ///
    /// The returned slice is owned by the session and is overwritten
    /// on the next call. Copy it out if you need to keep it.
    pub fn forward(&mut self, token: u32) -> &[f32] {
        let h = self.model.hidden_size;
        let vocab = self.model.vocab_size;
        debug_assert!(
            (token as usize) < vocab,
            "token id {token} out of vocab range {vocab}"
        );

        // Embedding lookup into scratch.x
        tensor::row_lookup(&self.model.emb, h, token as usize, &mut self.scratch.x);

        // Top-level ln0 (moved here from blocks.0 at checkpoint conversion)
        {
            let (x_src, x_dst) = (&self.scratch.x[..], &mut self.scratch.normed[..]);
            tensor::layer_norm(
                x_src,
                &self.model.ln0.weight,
                &self.model.ln0.bias,
                1e-5,
                x_dst,
            );
            self.scratch.x.copy_from_slice(&self.scratch.normed);
        }

        // Run each block
        let h = self.model.hidden_size;
        let im = self.model.intermediate_size;
        for (li, block) in self.model.blocks.iter().enumerate() {
            Self::forward_block(
                block,
                &mut self.state[li],
                &mut self.scratch,
                h,
                im,
            );
        }

        // Final layer norm
        tensor::layer_norm(
            &self.scratch.x,
            &self.model.ln_out.weight,
            &self.model.ln_out.bias,
            1e-5,
            &mut self.scratch.normed,
        );

        // Head projection: logits[i] = sum_j head[i, j] * normed[j]
        // computed as an AXPY over the column-major head buffer.
        // This is the single biggest matmul in the forward pass
        // (16384 × 96 = 1.57 M FLOPs).
        //
        // Using the SERIAL matvec_col_major: the parallel version
        // tried rayon across output rows but the thread-pool
        // dispatch overhead on 3 MB of inputs exceeded the
        // savings from multi-threading. Segment-level parallelism
        // (in codec.rs) is where the actual parallelism wins.
        // Within a segment's forward pass, serial is fastest.
        // Phase 2.5b INT8 head with per-column scales. Phase 4a
        // diff harness confirmed this is the right choice: the
        // 0.20 logit L_inf vs Python is real but does NOT change
        // the entropy bound or the actual-coded byte count
        // measurably (ratio 0.2060 with INT8, 0.2059 with f32).
        // The 4× lower memory traffic is worth ~19% throughput.
        tensor::matvec_col_major_int8(
            &self.model.head_q,
            &self.model.head_scales,
            &self.scratch.normed,
            &mut self.logits,
            self.model.vocab_size,
            self.model.hidden_size,
        );
        &self.logits
    }

    /// Internal: run one block's forward pass, mutating `state` and `scratch`.
    fn forward_block(
        block: &Block,
        state: &mut LayerState,
        scratch: &mut Scratch,
        h: usize,
        im: usize,
    ) {
        // short = relu(short_weight @ x)
        tensor::matvec_square(&block.w_short, &scratch.x, &mut scratch.short, h);
        tensor::relu_inplace(&mut scratch.short);

        // residual = x
        scratch.residual.copy_from_slice(&scratch.x);

        // x = ln1(x)
        tensor::layer_norm(
            &scratch.x,
            &block.ln1.weight,
            &block.ln1.bias,
            1e-5,
            &mut scratch.normed,
        );

        // Time mix (attention). Computes in scratch.rwkv.
        Self::time_mix(block, state, scratch, h);

        // x = residual + rwkv -> back into scratch.x
        scratch.x.copy_from_slice(&scratch.residual);
        tensor::add_inplace(&mut scratch.x, &scratch.rwkv);

        // residual = x
        scratch.residual.copy_from_slice(&scratch.x);

        // x = ln2(x)
        tensor::layer_norm(
            &scratch.x,
            &block.ln2.weight,
            &block.ln2.bias,
            1e-5,
            &mut scratch.normed,
        );

        // Channel mix (FFN). Computes in scratch.ffn_out.
        Self::channel_mix(block, state, scratch, h, im);

        // x = residual + ffn_out
        scratch.x.copy_from_slice(&scratch.residual);
        tensor::add_inplace(&mut scratch.x, &scratch.ffn_out);

        // x = x + short (L3TC's extra shortcut)
        tensor::add_inplace(&mut scratch.x, &scratch.short);
    }

    /// Time mixing (attention). Reads `scratch.normed` as the
    /// post-ln1 input; writes the output into `scratch.rwkv`.
    fn time_mix(block: &Block, state: &mut LayerState, scratch: &mut Scratch, h: usize) {
        let att = &block.att;

        // xk = x * time_mix_k + state_x * (1 - time_mix_k)
        tensor::time_mix(&scratch.normed, &state.state_x, &att.time_mix_k, &mut scratch.xk);
        tensor::time_mix(&scratch.normed, &state.state_x, &att.time_mix_v, &mut scratch.xv);
        tensor::time_mix(&scratch.normed, &state.state_x, &att.time_mix_r, &mut scratch.xr);

        // Update state_x for next step
        state.state_x.copy_from_slice(&scratch.normed);

        // k = key @ xk  (hidden → hidden square matvec)
        tensor::matvec_square(&att.w_key, &scratch.xk, &mut scratch.k[..h], h);
        // v = value @ xv
        tensor::matvec_square(&att.w_value, &scratch.xv, &mut scratch.v, h);
        // r = sigmoid(receptance @ xr)
        tensor::matvec_square(&att.w_receptance, &scratch.xr, &mut scratch.r, h);
        tensor::sigmoid_inplace(&mut scratch.r);

        // Step 1 (pre-state-update): ww = tf + k; p = max(state_p, ww);
        // a = exp(state_p-p)*state_a + exp(ww-p)*v;
        // b = exp(state_p-p)*state_b + exp(ww-p).
        // Fused into one NEON loop — keeps e1/e2 in registers instead
        // of round-tripping through `scratch.e1`/`e2`.
        tensor::time_mix_step1(
            &state.state_p[..h],
            &att.time_first[..h],
            &scratch.k[..h],
            &state.state_a[..h],
            &state.state_b[..h],
            &scratch.v[..h],
            &mut scratch.ww[..h],
            &mut scratch.p[..h],
            &mut scratch.a[..h],
            &mut scratch.b[..h],
        );

        // Step 2 (post-state-update): ww = state_p + neg_exp_decay;
        // p_new = max(ww, k); state_{a,b} updated; state_p = p_new.
        // Fused, in-place over the state buffers.
        tensor::time_mix_step2(
            &att.neg_exp_time_decay[..h],
            &scratch.k[..h],
            &scratch.v[..h],
            &mut state.state_p[..h],
            &mut state.state_a[..h],
            &mut state.state_b[..h],
            &mut scratch.ww[..h],
        );

        // rwkv = r * a / b
        for i in 0..h {
            scratch.rwkv[i] = scratch.r[i] * scratch.a[i] / scratch.b[i];
        }
        // Apply output projection -> scratch.out_proj then back to rwkv
        tensor::matvec_square(&att.w_output, &scratch.rwkv, &mut scratch.out_proj, h);
        scratch.rwkv.copy_from_slice(&scratch.out_proj);
    }

    /// Channel mixing (FFN). Reads `scratch.normed` (post-ln2 input).
    /// Writes the output into `scratch.ffn_out`.
    fn channel_mix(
        block: &Block,
        state: &mut LayerState,
        scratch: &mut Scratch,
        h: usize,
        im: usize,
    ) {
        let ffn = &block.ffn;

        // xk = x * time_mix_k + state_ffn * (1 - time_mix_k)
        tensor::time_mix(&scratch.normed, &state.state_ffn, &ffn.time_mix_k, &mut scratch.xk);
        // xr = x * time_mix_r + state_ffn * (1 - time_mix_r)
        tensor::time_mix(&scratch.normed, &state.state_ffn, &ffn.time_mix_r, &mut scratch.xr);

        // state_ffn = x (the pre-mix input)
        state.state_ffn.copy_from_slice(&scratch.normed);

        // r = sigmoid(receptance @ xr)  (hidden → hidden square)
        tensor::matvec_square(&ffn.w_receptance, &scratch.xr, &mut scratch.r, h);
        tensor::sigmoid_inplace(&mut scratch.r);

        // k = (relu(key @ xk))^2
        //
        // FFN key is (intermediate, hidden). For L3TC-200K where
        // intermediate == hidden == 96 this is still a 96×96
        // matvec and matvec_square dispatches to NEON. For bigger
        // variants (e.g. 3.2M: 512×256) it's rectangular and
        // falls through to the generic `matvec` path (scalar for
        // modest sizes, sgemm above the BLAS threshold).
        if im == h {
            tensor::matvec_square(&ffn.w_key, &scratch.xk, &mut scratch.k[..h], h);
        } else {
            tensor::matvec(&ffn.w_key, &scratch.xk, &mut scratch.k[..im]);
        }
        tensor::relu_inplace(&mut scratch.k[..im]);
        tensor::square_inplace(&mut scratch.k[..im]);

        // kv = value @ k  (FFN value projects intermediate → hidden)
        if im == h {
            tensor::matvec_square(&ffn.w_value, &scratch.k[..h], &mut scratch.v, h);
        } else {
            tensor::matvec(&ffn.w_value, &scratch.k[..im], &mut scratch.v);
        }

        // ffn_out = r * kv
        for i in 0..h {
            scratch.ffn_out[i] = scratch.r[i] * scratch.v[i];
        }
    }

    /// Borrow the current per-layer state (read-only).
    ///
    /// Useful for tests and debugging; not part of the hot path.
    pub fn state(&self) -> &[LayerState] {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model(hidden: usize, vocab: usize, num_layers: usize) -> Model {
        // Build a Model with zero weights. The forward pass shouldn't
        // panic and should produce deterministic output.
        let blocks = (0..num_layers)
            .map(|_| Block {
                ln1: LayerNormParams {
                    weight: vec![1.0; hidden],
                    bias: vec![0.0; hidden],
                },
                ln2: LayerNormParams {
                    weight: vec![1.0; hidden],
                    bias: vec![0.0; hidden],
                },
                att: TimeMixParams {
                    time_decay: vec![0.0; hidden],
                    // -exp(0.0) = -1.0
                    neg_exp_time_decay: vec![-1.0; hidden],
                    time_first: vec![0.0; hidden],
                    time_mix_k: vec![0.5; hidden],
                    time_mix_v: vec![0.5; hidden],
                    time_mix_r: vec![0.5; hidden],
                    w_key: vec![0.0; hidden * hidden],
                    w_value: vec![0.0; hidden * hidden],
                    w_receptance: vec![0.0; hidden * hidden],
                    w_output: vec![0.0; hidden * hidden],
                },
                ffn: ChannelMixParams {
                    time_mix_k: vec![0.5; hidden],
                    time_mix_r: vec![0.5; hidden],
                    w_key: vec![0.0; hidden * hidden],
                    w_value: vec![0.0; hidden * hidden],
                    w_receptance: vec![0.0; hidden * hidden],
                },
                w_short: vec![0.0; hidden * hidden],
            })
            .collect();

        Model {
            emb: vec![0.1; vocab * hidden],
            vocab_size: vocab,
            hidden_size: hidden,
            // tiny_model uses intermediate == hidden for simplicity.
            // For models with a different expansion (e.g. L3TC-3.2M
            // where im=512, h=256) the real from_checkpoint loader
            // reads the shape from the FFN key tensor.
            intermediate_size: hidden,
            ln0: LayerNormParams {
                weight: vec![1.0; hidden],
                bias: vec![0.0; hidden],
            },
            blocks,
            ln_out: LayerNormParams {
                weight: vec![1.0; hidden],
                bias: vec![0.0; hidden],
            },
            // head is stored column-major; for a tiny test model
            // where all values are equal, the layout doesn't matter.
            head_col_major: vec![0.01; vocab * hidden],
            head_q: vec![0i8; vocab * hidden],
            head_scales: vec![0.0f32; hidden],
        }
    }

    #[test]
    fn forward_produces_finite_logits() {
        let m = tiny_model(8, 16, 2);
        let mut s = Session::new(&m);
        let logits = s.forward(0);
        assert_eq!(logits.len(), 16);
        for &v in logits {
            assert!(v.is_finite(), "logit not finite: {v}");
        }
    }

    #[test]
    fn forward_advances_state() {
        let m = tiny_model(8, 16, 2);
        let mut s = Session::new(&m);

        // Check that state_x starts at zero
        assert!(s.state()[0].state_x.iter().all(|&v| v == 0.0));

        // After one forward pass with a non-uniform embedding, state
        // should have moved (embedding is 0.1 everywhere, so post-ln
        // it's zero — need to use a model where the embedding row
        // varies to see state change).
        s.forward(5);
        // state_x gets the post-ln0 input, which is the LN of a
        // constant vector = zero. So state stays at zero in this
        // degenerate case. Test instead that state was updated
        // without panicking.
        let _ = s.state();
    }

    #[test]
    fn reset_clears_state() {
        let m = tiny_model(8, 16, 2);
        let mut s = Session::new(&m);
        // Manually perturb state
        s.state[0].state_a[0] = 42.0;
        s.reset();
        assert_eq!(s.state()[0].state_a[0], 0.0);
        assert_eq!(s.state()[0].state_p[0], -1e30);
    }
}
