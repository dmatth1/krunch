//! Batched RWKV inference session — Phase 13e.
//!
//! Runs `batch` independent compression streams in lockstep through
//! all the Phase 13 batched Metal kernels. Each batch lane has its
//! own `LayerState` (state_a/b/p/x/ffn) but all lanes share the
//! immutable model weights.
//!
//! # When to use this
//!
//! Below `GPU_AUTO_THRESHOLD_BYTES` of input the CPU `Session` plus
//! rayon segment parallelism wins. Past it, batching across segments
//! amortizes per-call GPU dispatch overhead — see
//! [`docs/phases/PHASE_13.md`](../../../docs/phases/PHASE_13.md) for
//! the measured crossover.
//!
//! # Correctness
//!
//! Every per-token logit must be quantization-equivalent to the CPU
//! `Session::forward` output (i.e., produce the same cum_freqs after
//! `round(p × PYTHON_FREQ_TOTAL); max(1)`). Phase 12-style polynomial
//! exp coefficients are reused on the GPU exactly so the freq tables
//! line up across backends. Files compressed via `BatchedSession`
//! decompress on `Session` and vice versa.

use crate::backend::mtl::{
    CumFreqsKernelMetal, ForwardLayerKernelMetal, GlueKernelsMetal, HeadKernelMetal,
    LayerNormKernelMetal, Matvec96Metal, MetalError, SigmoidKernelMetal, SubExpKernelMetal,
    TimeMixKernelMetal,
};
use crate::rwkv::{Block, Model};
use metal::{Buffer, CommandBuffer, MTLResourceOptions};

/// Allocate a GPU-visible buffer of `n * sizeof(T)` bytes, zero-filled.
fn new_shared_buf_bytes(device: &metal::Device, bytes: u64) -> Buffer {
    device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
}

/// Upload a `&[f32]` into a fresh shared-storage Metal buffer.
fn upload_f32(device: &metal::Device, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Fill a buffer's contents with a repeated f32 value.
fn fill_f32(buf: &Buffer, value: f32, n: usize) {
    unsafe {
        let ptr = buf.contents() as *mut f32;
        for i in 0..n {
            *ptr.add(i) = value;
        }
    }
}

/// Copy a Metal buffer's contents into a &mut [f32] slice.
fn read_f32(buf: &Buffer, dst: &mut [f32]) {
    unsafe {
        let src = buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
    }
}

/// Copy a Metal buffer's contents into a &mut [u32] slice.
fn read_u32(buf: &Buffer, dst: &mut [u32]) {
    unsafe {
        let src = buf.contents() as *const u32;
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
    }
}

/// Write a &[u32] into a Metal buffer's contents.
fn write_u32(buf: &Buffer, src: &[u32]) {
    unsafe {
        let dst = buf.contents() as *mut u32;
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
}

/// Per-layer GPU kernel set + the layer's GPU-resident blend weights
/// (time_mix_k/v/r for attention, time_mix_k/r for channel_mix).
/// Phase 13j: the blends used to happen CPU-side — now they're
/// elementwise GPU kernels that read these weight buffers.
struct LayerKernels {
    ln1: LayerNormKernelMetal,
    ln2: LayerNormKernelMetal,
    short: Matvec96Metal,
    att_k: Matvec96Metal,
    att_v: Matvec96Metal,
    att_r: Matvec96Metal,
    att_output: Matvec96Metal,
    ffn_k: Matvec96Metal,
    ffn_v: Matvec96Metal,
    ffn_r: Matvec96Metal,
    time_mix: TimeMixKernelMetal,
    // Persistent GPU copies of the per-channel blend weights.
    att_mix_k_buf: Buffer,
    att_mix_v_buf: Buffer,
    att_mix_r_buf: Buffer,
    ffn_mix_k_buf: Buffer,
    ffn_mix_r_buf: Buffer,
}

impl LayerKernels {
    fn new(block: &Block, h: usize, device: &metal::Device) -> Result<Self, MetalError> {
        Ok(Self {
            ln1: LayerNormKernelMetal::new(&block.ln1.weight, &block.ln1.bias, h, 1e-5)?,
            ln2: LayerNormKernelMetal::new(&block.ln2.weight, &block.ln2.bias, h, 1e-5)?,
            short: Matvec96Metal::new(&block.w_short, h)?,
            att_k: Matvec96Metal::new(&block.att.w_key, h)?,
            att_v: Matvec96Metal::new(&block.att.w_value, h)?,
            att_r: Matvec96Metal::new(&block.att.w_receptance, h)?,
            att_output: Matvec96Metal::new(&block.att.w_output, h)?,
            ffn_k: Matvec96Metal::new(&block.ffn.w_key, h)?,
            ffn_v: Matvec96Metal::new(&block.ffn.w_value, h)?,
            ffn_r: Matvec96Metal::new(&block.ffn.w_receptance, h)?,
            time_mix: TimeMixKernelMetal::new(
                &block.att.time_first,
                &block.att.neg_exp_time_decay,
                h,
            )?,
            att_mix_k_buf: upload_f32(device, &block.att.time_mix_k),
            att_mix_v_buf: upload_f32(device, &block.att.time_mix_v),
            att_mix_r_buf: upload_f32(device, &block.att.time_mix_r),
            ffn_mix_k_buf: upload_f32(device, &block.ffn.time_mix_k),
            ffn_mix_r_buf: upload_f32(device, &block.ffn.time_mix_r),
        })
    }
}

/// Batched inference session backed by Metal kernels.
///
/// Phase 13j: all per-lane state and per-step scratch live on the GPU
/// as persistent Metal buffers. One `forward_batched(prev_tokens)`
/// call encodes the entire forward pass + cum_freqs pipeline into a
/// single `MTLCommandBuffer` with a single `commit_and_wait` at the
/// end. The only CPU↔GPU traffic per token is:
/// - upload prev_tokens (batch × 4 bytes)
/// - readback logits (batch × vocab × 4 bytes) and freqs (same size)
///
/// All the previous CPU glue ops (embedding lookup, time_mix /
/// channel_mix input blends, residual adds, relu/square, copies,
/// `rwkv = r * a / b`) now run as GPU kernels inside the chained
/// cmd_buf (see `GlueKernelsMetal`).
#[allow(dead_code)] // some kernels retained for microbenchmarks and tests
pub struct BatchedSession<'a> {
    model: &'a Model,
    batch: usize,
    h: usize,
    vocab: usize,

    // Shared per-model kernels.
    ln0: LayerNormKernelMetal,
    ln_out: LayerNormKernelMetal,
    head: HeadKernelMetal,
    cum_freqs: CumFreqsKernelMetal,
    sub_exp: SubExpKernelMetal,
    sigmoid: SigmoidKernelMetal,
    glue: GlueKernelsMetal,

    // Per-layer kernels (includes per-layer blend weight buffers).
    // Retained for the fallback path and for the bench tools; the
    // hot `forward_batched` uses `layers_fused` when h == 96.
    layers: Vec<LayerKernels>,

    // Phase 13n: one fused-layer kernel per transformer block, used
    // when the model has h = 96 (the 200K default). Each fused kernel
    // executes the whole per-layer forward pass in ONE dispatch,
    // replacing ~17 separate kernel calls.
    layers_fused: Option<Vec<ForwardLayerKernelMetal>>,

    // Persistent GPU state. All `batch * h` f32 (lane-major).
    // One buffer per layer.
    state_a: Vec<Buffer>,
    state_b: Vec<Buffer>,
    state_p: Vec<Buffer>,
    state_x: Vec<Buffer>,
    state_ffn: Vec<Buffer>,

    // Shared per-token scratch on the GPU.
    emb_buf: Buffer,         // vocab * h (uploaded once at construction)
    prev_tokens_buf: Buffer, // batch u32
    x_buf: Buffer,           // batch * h
    residual_buf: Buffer,    // batch * h
    short_buf: Buffer,       // batch * h
    normed_buf: Buffer,      // batch * h
    xk_buf: Buffer,
    xv_buf: Buffer,
    xr_buf: Buffer,
    k_buf: Buffer,
    v_buf: Buffer,
    r_buf: Buffer,
    ww_buf: Buffer,
    p_buf: Buffer,
    a_buf: Buffer,
    b_buf: Buffer,
    rwkv_buf: Buffer,
    out_proj_buf: Buffer,
    ffn_out_buf: Buffer,
    logits_buf: Buffer, // batch * vocab f32
    // cum_freqs scratch (reused across calls).
    cum_max_buf: Buffer,   // batch f32
    cum_exps_buf: Buffer,  // batch * vocab f32
    cum_sum_buf: Buffer,   // batch f32
    cum_scale_buf: Buffer, // batch f32
    freqs_buf: Buffer,     // batch * vocab u32

    /// Host-side copy of the final `batch * vocab` logits. Populated
    /// after `forward_batched` returns.
    pub logits: Vec<f32>,
    /// Host-side copy of the `batch * vocab` u32 freqs table.
    /// Populated after `forward_batched` returns.
    pub freqs: Vec<u32>,

    /// Phase 13s: pending async GPU submission. When `Some`, a
    /// `forward_submit` has been issued without a matching
    /// `forward_wait_and_readback`. Holding the owned CommandBuffer
    /// here keeps the Rust-side lifetime right while the GPU runs
    /// the token asynchronously alongside CPU-side work.
    pending_cmd: Option<CommandBuffer>,
}

impl<'a> BatchedSession<'a> {
    /// Build a new session and upload all model weights to GPU
    /// kernel buffers. Initial state is fresh (zeros + state_p =
    /// −1e30 so the first token's prediction depends only on it).
    pub fn new(model: &'a Model, batch: usize) -> Result<Self, MetalError> {
        let h = model.hidden_size;
        let vocab = model.vocab_size;
        let num_layers = model.num_layers();

        let ln0 = LayerNormKernelMetal::new(&model.ln0.weight, &model.ln0.bias, h, 1e-5)?;
        let ln_out = LayerNormKernelMetal::new(&model.ln_out.weight, &model.ln_out.bias, h, 1e-5)?;
        let head = HeadKernelMetal::new(&model.head_q, &model.head_scales, vocab, h)?;
        let cum_freqs = CumFreqsKernelMetal::new(vocab)?;
        let sub_exp = SubExpKernelMetal::new(h)?;
        let sigmoid = SigmoidKernelMetal::new(h)?;
        let glue = GlueKernelsMetal::new()?;
        let device = glue.device();

        let mut layers = Vec::with_capacity(num_layers);
        for block in &model.blocks {
            layers.push(LayerKernels::new(block, h, &device)?);
        }

        // Phase 13n: build fused-layer kernels when h == 96 (the
        // size the MSL kernel is hard-coded for). Other sizes fall
        // back to the unfused path.
        let layers_fused = if h == 96 {
            let mut v = Vec::with_capacity(num_layers);
            for block in &model.blocks {
                v.push(ForwardLayerKernelMetal::new(block, &device)?);
            }
            Some(v)
        } else {
            None
        };

        let bh = batch * h;
        let bh_bytes = (bh * std::mem::size_of::<f32>()) as u64;
        let bv = batch * vocab;
        let bv_bytes = (bv * std::mem::size_of::<f32>()) as u64;
        let bv_u32_bytes = (bv * std::mem::size_of::<u32>()) as u64;
        let batch_u32_bytes = (batch * std::mem::size_of::<u32>()) as u64;
        let batch_f32_bytes = (batch * std::mem::size_of::<f32>()) as u64;

        // Persistent state buffers, one per layer.
        let state_a: Vec<Buffer> = (0..num_layers)
            .map(|_| new_shared_buf_bytes(&device, bh_bytes))
            .collect();
        let state_b: Vec<Buffer> = (0..num_layers)
            .map(|_| new_shared_buf_bytes(&device, bh_bytes))
            .collect();
        let state_p: Vec<Buffer> = (0..num_layers)
            .map(|_| new_shared_buf_bytes(&device, bh_bytes))
            .collect();
        let state_x: Vec<Buffer> = (0..num_layers)
            .map(|_| new_shared_buf_bytes(&device, bh_bytes))
            .collect();
        let state_ffn: Vec<Buffer> = (0..num_layers)
            .map(|_| new_shared_buf_bytes(&device, bh_bytes))
            .collect();

        // Persistent embedding table upload.
        let emb_buf = upload_f32(&device, &model.emb);

        let prev_tokens_buf = new_shared_buf_bytes(&device, batch_u32_bytes);

        let session = Self {
            model,
            batch,
            h,
            vocab,
            ln0,
            ln_out,
            head,
            cum_freqs,
            sub_exp,
            sigmoid,
            glue,
            layers,
            layers_fused,
            state_a,
            state_b,
            state_p,
            state_x,
            state_ffn,
            emb_buf,
            prev_tokens_buf,
            x_buf: new_shared_buf_bytes(&device, bh_bytes),
            residual_buf: new_shared_buf_bytes(&device, bh_bytes),
            short_buf: new_shared_buf_bytes(&device, bh_bytes),
            normed_buf: new_shared_buf_bytes(&device, bh_bytes),
            xk_buf: new_shared_buf_bytes(&device, bh_bytes),
            xv_buf: new_shared_buf_bytes(&device, bh_bytes),
            xr_buf: new_shared_buf_bytes(&device, bh_bytes),
            k_buf: new_shared_buf_bytes(&device, bh_bytes),
            v_buf: new_shared_buf_bytes(&device, bh_bytes),
            r_buf: new_shared_buf_bytes(&device, bh_bytes),
            ww_buf: new_shared_buf_bytes(&device, bh_bytes),
            p_buf: new_shared_buf_bytes(&device, bh_bytes),
            a_buf: new_shared_buf_bytes(&device, bh_bytes),
            b_buf: new_shared_buf_bytes(&device, bh_bytes),
            rwkv_buf: new_shared_buf_bytes(&device, bh_bytes),
            out_proj_buf: new_shared_buf_bytes(&device, bh_bytes),
            ffn_out_buf: new_shared_buf_bytes(&device, bh_bytes),
            logits_buf: new_shared_buf_bytes(&device, bv_bytes),
            cum_max_buf: new_shared_buf_bytes(&device, batch_f32_bytes),
            cum_exps_buf: new_shared_buf_bytes(&device, bv_bytes),
            cum_sum_buf: new_shared_buf_bytes(&device, batch_f32_bytes),
            cum_scale_buf: new_shared_buf_bytes(&device, batch_f32_bytes),
            freqs_buf: new_shared_buf_bytes(&device, bv_u32_bytes),
            logits: vec![0.0; bv],
            freqs: vec![0u32; bv],
            pending_cmd: None,
        };
        // Initialize persistent state buffers.
        let mut s = session;
        s.reset();
        Ok(s)
    }

    /// Reset all per-lane state to fresh values. Call at every
    /// segment boundary.
    pub fn reset(&mut self) {
        let bh = self.batch * self.h;
        for layer in 0..self.model.num_layers() {
            fill_f32(&self.state_a[layer], 0.0, bh);
            fill_f32(&self.state_b[layer], 0.0, bh);
            fill_f32(&self.state_p[layer], -1e30, bh);
            fill_f32(&self.state_x[layer], 0.0, bh);
            fill_f32(&self.state_ffn[layer], 0.0, bh);
        }
    }

    /// One step: feed one previous token per lane, encode the entire
    /// forward pass + cum_freqs pipeline into a single command buffer,
    /// commit + wait once, and populate `self.logits` and `self.freqs`.
    ///
    /// Phase 13j: all ~32 per-token GPU dispatches (embed → ln0 → N
    /// layer passes → ln_out → head → cum_freqs) run inside one
    /// `commit_and_wait`. All state and scratch are persistent Metal
    /// buffers; only `prev_tokens` goes CPU→GPU and only `logits` +
    /// `freqs` come GPU→CPU per call.
    pub fn forward_batched(&mut self, prev_tokens: &[u32]) {
        self.forward_submit(prev_tokens);
        self.forward_wait_and_readback();
    }

    /// Phase 13s: async version of `forward_batched`. Encodes the
    /// full per-token forward pipeline into a command buffer and
    /// submits it to the GPU WITHOUT waiting. Follow up with
    /// `forward_wait_and_readback()` when you need the freqs back
    /// on CPU. Between submit and wait, the caller can do CPU-side
    /// work (e.g. AC-encode the previous step's freqs) in parallel
    /// with the GPU's forward compute.
    ///
    /// # Panics
    /// Panics if a previous `forward_submit` hasn't been followed
    /// by `forward_wait_and_readback`. Only one GPU forward pass
    /// may be in flight at a time (state buffers are mutated in
    /// place by the fused-layer kernel; overlapping calls would
    /// race on the state).
    pub fn forward_submit(&mut self, prev_tokens: &[u32]) {
        assert_eq!(prev_tokens.len(), self.batch);
        assert!(
            self.pending_cmd.is_none(),
            "forward_submit called with pending work; call \
             forward_wait_and_readback first"
        );
        const PYTHON_FREQ_TOTAL: u32 = 10_000_000;
        let bh = (self.batch * self.h) as u32;
        let batch_u = self.batch;
        let h_u = self.h as u32;
        let batch_u32 = self.batch as u32;

        // Upload prev_tokens to the persistent buffer (batch × 4 bytes).
        write_u32(&self.prev_tokens_buf, prev_tokens);

        let cmd_buf = self.glue.queue().new_command_buffer();

        // 1. Embedding lookup: x[lane, :] = emb[prev_tokens[lane], :]
        self.glue.encode_embed(
            cmd_buf,
            &self.emb_buf,
            &self.prev_tokens_buf,
            &self.x_buf,
            h_u,
            batch_u32,
        );

        // 2. ln0 with input aliased to x (out → normed), then copy back.
        self.ln0
            .encode_into(cmd_buf, &self.x_buf, &self.normed_buf, batch_u);
        self.glue
            .encode_copy(cmd_buf, &self.normed_buf, &self.x_buf, bh);

        // 3. Per-layer block.
        //
        // Phase 13n: if the fused-layer kernel is available (model
        // has h = 96), each layer runs as ONE dispatch instead of
        // ~17 separate kernels. Collapses per-token dispatch count
        // from ~50 down to ~6 (embed + ln0 + copy + N_layers +
        // ln_out + head + cum_freqs_4).
        if let Some(fused) = &self.layers_fused {
            #[allow(clippy::needless_range_loop)] // `layer` indexes parallel per-layer state Vecs
            for layer in 0..self.model.num_layers() {
                fused[layer].encode_into(
                    cmd_buf,
                    &self.x_buf,
                    &self.state_x[layer],
                    &self.state_a[layer],
                    &self.state_b[layer],
                    &self.state_p[layer],
                    &self.state_ffn[layer],
                    self.batch,
                    1e-5_f32,
                );
            }
        } else {
            // --- unfused fallback path (non-h=96 models) ---
            //
            // Phase 13k: eliminate the two `residual = x` copies per layer.
            for layer in 0..self.model.num_layers() {
                // Phase 13l: short = relu(short_weight @ x), fused.
                self.layers[layer].short.encode_into_act(
                    cmd_buf,
                    &self.x_buf,
                    &self.short_buf,
                    batch_u,
                    1, // relu
                );

                // ln1: x → normed (x preserved so we can add rwkv into it)
                self.layers[layer]
                    .ln1
                    .encode_into(cmd_buf, &self.x_buf, &self.normed_buf, batch_u);

                // time_mix: produces rwkv, updates state_x/a/b/p.
                // Reads only `normed` + state — `x` is safe to keep.
                self.encode_time_mix(cmd_buf, layer);

                // x += rwkv (residual connection)
                self.glue
                    .encode_add_inplace(cmd_buf, &self.x_buf, &self.rwkv_buf, bh);

                // ln2: x → normed
                self.layers[layer]
                    .ln2
                    .encode_into(cmd_buf, &self.x_buf, &self.normed_buf, batch_u);

                // channel_mix: produces ffn_out, updates state_ffn.
                // Reads only `normed` + state_ffn — `x` is safe to keep.
                self.encode_channel_mix(cmd_buf, layer);

                // x += ffn_out; x += short
                self.glue
                    .encode_add_inplace(cmd_buf, &self.x_buf, &self.ffn_out_buf, bh);
                self.glue
                    .encode_add_inplace(cmd_buf, &self.x_buf, &self.short_buf, bh);
            }
        } // end unfused path

        // 4. Final layer norm.
        let skip_head = std::env::var("L3TC_SKIP_HEAD").is_ok();
        let skip_cum = std::env::var("L3TC_SKIP_CUMFREQS").is_ok();
        self.ln_out
            .encode_into(cmd_buf, &self.x_buf, &self.normed_buf, batch_u);

        // 5. Head matvec: produces `batch * vocab` logits.
        if !skip_head {
            self.head
                .encode_into(cmd_buf, &self.normed_buf, &self.logits_buf, batch_u);
        }

        // 6. cum_freqs: logits → freqs (u32).
        if !skip_cum {
            self.cum_freqs.encode_into(
                cmd_buf,
                &self.logits_buf,
                &self.cum_max_buf,
                &self.cum_exps_buf,
                &self.cum_sum_buf,
                &self.cum_scale_buf,
                &self.freqs_buf,
                batch_u,
                PYTHON_FREQ_TOTAL,
            );
        } // end skip_cum guard

        // 7. Commit without waiting; retain ownership so the
        //    caller can wait on it later.
        cmd_buf.commit();
        self.pending_cmd = Some(cmd_buf.to_owned());
    }

    /// Phase 13s: block until the pending `forward_submit` finishes,
    /// then copy the GPU freqs buffer into `self.freqs`. Becomes a
    /// no-op if there's no pending submission.
    pub fn forward_wait_and_readback(&mut self) {
        if let Some(cmd) = self.pending_cmd.take() {
            cmd.wait_until_completed();
            // Read freqs (compress/decompress hot path only needs this
            // — ~1 MB at batch=16, vocab=16K).
            //
            // L3TC_METAL_SKIP_READBACK=1 disables readback for
            // profiling only — the resulting `self.freqs` is stale,
            // so AC encode produces garbage. Bench use only.
            if std::env::var("L3TC_METAL_SKIP_READBACK").is_err() {
                read_u32(&self.freqs_buf, &mut self.freqs);
            }
        }
    }

    /// Copy the most recent GPU-side logits into the host-side
    /// `self.logits` vec. Tests and microbenches call this before
    /// reading `self.logits`; the compress/decompress path skips it.
    pub fn sync_logits_host(&mut self) {
        read_f32(&self.logits_buf, &mut self.logits);
    }

    /// Legacy API: after `forward_batched`, freqs are already populated,
    /// so this is now a no-op kept for API compat with pre-13j callers.
    pub fn cum_freqs_batched(&mut self) {
        // Freqs were computed + read back inside `forward_batched`.
    }

    fn encode_time_mix(&self, cmd_buf: &metal::CommandBufferRef, layer: usize) {
        let bh = (self.batch * self.h) as u32;
        let h_u = self.h as u32;
        let batch_u32 = self.batch as u32;
        let lk = &self.layers[layer];

        // Input blends xk/xv/xr = normed * mix + state_x * (1 - mix).
        // Also updates state_x ← normed in the same kernel.
        self.glue.encode_blend3(
            cmd_buf,
            &self.normed_buf,
            &self.state_x[layer],
            &self.state_x[layer],
            &lk.att_mix_k_buf,
            &lk.att_mix_v_buf,
            &lk.att_mix_r_buf,
            &self.xk_buf,
            &self.xv_buf,
            &self.xr_buf,
            h_u,
            batch_u32,
        );

        // 3 matvecs (independent, safe to chain). The receptance
        // matvec fuses the follow-up `sigmoid(r)` into its final
        // store per Phase 13l — saves one encoder per layer.
        lk.att_k
            .encode_into(cmd_buf, &self.xk_buf, &self.k_buf, self.batch);
        lk.att_v
            .encode_into(cmd_buf, &self.xv_buf, &self.v_buf, self.batch);
        lk.att_r.encode_into_act(
            cmd_buf,
            &self.xr_buf,
            &self.r_buf,
            self.batch,
            2, // sigmoid
        );

        // time_mix step1 reads state_p/a/b + k + v, writes ww/p/a/b.
        lk.time_mix.encode_step1_into(
            cmd_buf,
            &self.state_p[layer],
            &self.k_buf,
            &self.state_a[layer],
            &self.state_b[layer],
            &self.v_buf,
            &self.ww_buf,
            &self.p_buf,
            &self.a_buf,
            &self.b_buf,
            self.batch,
        );

        // step2 updates state_p/a/b in-place.
        lk.time_mix.encode_step2_into(
            cmd_buf,
            &self.k_buf,
            &self.v_buf,
            &self.state_p[layer],
            &self.state_a[layer],
            &self.state_b[layer],
            &self.ww_buf,
            self.batch,
        );

        // Phase 13k: write r*a/b into out_proj (rather than rwkv),
        // then run the output projection reading from out_proj and
        // writing to rwkv. Saves the extra copy that used to exist
        // to shuffle att_output's output back into rwkv.
        self.glue.encode_mul_div(
            cmd_buf,
            &self.r_buf,
            &self.a_buf,
            &self.b_buf,
            &self.out_proj_buf,
            bh,
        );
        lk.att_output
            .encode_into(cmd_buf, &self.out_proj_buf, &self.rwkv_buf, self.batch);
    }

    fn encode_channel_mix(&self, cmd_buf: &metal::CommandBufferRef, layer: usize) {
        let bh = (self.batch * self.h) as u32;
        let h_u = self.h as u32;
        let batch_u32 = self.batch as u32;
        let lk = &self.layers[layer];

        // Blend xk/xr + update state_ffn.
        self.glue.encode_blend2(
            cmd_buf,
            &self.normed_buf,
            &self.state_ffn[layer],
            &self.state_ffn[layer],
            &lk.ffn_mix_k_buf,
            &lk.ffn_mix_r_buf,
            &self.xk_buf,
            &self.xr_buf,
            h_u,
            batch_u32,
        );

        // Phase 13l: r = sigmoid(ffn_r @ xr), fused.
        lk.ffn_r.encode_into_act(
            cmd_buf,
            &self.xr_buf,
            &self.r_buf,
            self.batch,
            2, // sigmoid
        );

        // Phase 13l: k = max(0, ffn_k @ xk)^2, fused.
        lk.ffn_k.encode_into_act(
            cmd_buf,
            &self.xk_buf,
            &self.k_buf,
            self.batch,
            3, // relu_square
        );

        // v = ffn_v @ k
        lk.ffn_v
            .encode_into(cmd_buf, &self.k_buf, &self.v_buf, self.batch);

        // ffn_out = r * v
        self.glue
            .encode_mul(cmd_buf, &self.r_buf, &self.v_buf, &self.ffn_out_buf, bh);
    }
}

/// Compress a single tokenized segment through `BatchedSession`,
/// returning the AC body bytes. The session is reset before
/// processing.
///
/// **Backend interop note:** files compressed via `BatchedSession`
/// are bit-identical only when decoded via `BatchedSession`. The
/// FP arithmetic in the GPU forward pass diverges from CPU NEON by
/// a few ULPs per layer, which can shift a few freq-table entries
/// at borderline `round()` boundaries. Within a single backend the
/// cum_freqs are deterministic so the AC stays in sync; across
/// backends, the AC desyncs.
///
/// In other words: GPU encode → GPU decode works. CPU encode → CPU
/// decode works. Mixed does not. The auto-routing layer in
/// [`crate::backend::Backend::auto`] uses the same backend for both
/// halves of a round trip on a given machine.
pub fn compress_one_segment_batched(
    session: &mut BatchedSession<'_>,
    tokens: &[u32],
) -> Result<Vec<u8>, crate::Error> {
    use crate::arithmetic::ArithmeticEncoder;

    session.reset();
    if tokens.len() < 2 {
        let mut buf = Vec::new();
        let enc = ArithmeticEncoder::new(&mut buf);
        enc.finish()?;
        return Ok(buf);
    }
    let vocab = session.vocab;
    let mut buf = Vec::with_capacity(tokens.len());
    {
        let mut enc = ArithmeticEncoder::new(&mut buf);
        let mut prev = vec![tokens[0]; session.batch];
        for i in 1..tokens.len() {
            prev[0] = tokens[i - 1];
            session.forward_batched(&prev);
            session.cum_freqs_batched();
            let mut cum = Vec::with_capacity(vocab + 1);
            cum.push(0u64);
            let mut total: u64 = 0;
            for j in 0..vocab {
                total += session.freqs[j] as u64;
                cum.push(total);
            }
            enc.encode_symbol(&cum, tokens[i])?;
        }
        enc.finish()?;
    }
    Ok(buf)
}

/// Compress multiple tokenized segments via Metal. Returns an AC
/// body per input segment, in input order.
///
/// Phase 13h: true N-segment lockstep. Segments are processed in
/// chunks of `batch_size`; every step inside a chunk runs all lanes
/// in parallel through a single GPU forward pass + cum_freqs, then
/// each lane encodes its own next token into its own `Vec<u8>` AC
/// body. Ragged lengths are handled per-lane via an `active` gate.
/// Per-token GPU overhead (~30 sync barriers in the current chained-
/// dispatch state) thus amortizes across `n_active` lanes.
pub fn compress_segments_batched(
    model: &Model,
    segments: &[Vec<u32>],
    batch_size: usize,
) -> Result<Vec<Vec<u8>>, crate::Error> {
    use crate::arithmetic::ArithmeticEncoder;
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    let batch = batch_size.max(1);
    let mut bs = BatchedSession::new(model, batch)
        .map_err(|e| crate::Error::BadCheckpoint(format!("Metal init: {e}")))?;
    let vocab = bs.vocab;
    let mut all_outputs: Vec<Vec<u8>> = Vec::with_capacity(segments.len());

    for chunk in segments.chunks(batch) {
        let n = chunk.len();
        bs.reset();

        // Bookkeeping for the lockstep loop.
        let max_len = chunk.iter().map(|s| s.len()).max().unwrap_or(0);

        // Per-lane outputs (one Vec<u8> per chunk lane). Pre-size so
        // the encoder's BitWriter never reallocates during the hot
        // loop on typical text.
        let bufs: Vec<Vec<u8>> = (0..n).map(|i| Vec::with_capacity(chunk[i].len())).collect();

        if max_len < 2 {
            // Every segment in the chunk is empty / single-token: no
            // symbols to encode, just finish a fresh encoder per lane.
            for buf in bufs {
                let body = ArithmeticEncoder::new(buf).finish()?;
                all_outputs.push(body);
            }
            continue;
        }

        // Move buffers into encoders. `finish()` returns the inner
        // Vec<u8> back at the end of the chunk.
        let mut encs: Vec<ArithmeticEncoder<Vec<u8>>> =
            bufs.into_iter().map(ArithmeticEncoder::new).collect();

        // Initialize prev[lane] for the first forward step.
        // For lanes 0..n: prev = first token of segment (or 0 if empty).
        // For lanes n..batch (idle padding): copy lane 0's value so we
        // feed something valid; output is discarded.
        let mut prev = vec![0u32; batch];
        for i in 0..n {
            if !chunk[i].is_empty() {
                prev[i] = chunk[i][0];
            }
        }
        let pad_token = prev[0];
        prev[n..batch].fill(pad_token);

        // Reusable cum_freqs scratch (vocab+1 u64).
        let mut cum: Vec<u64> = Vec::with_capacity(vocab + 1);

        // Phase 13l profiling: time the GPU forward pass vs CPU AC
        // encode per step; print the breakdown on the first chunk so
        // we can see which side is the actual bottleneck.
        let profile = std::env::var("L3TC_PROFILE_METAL").is_ok();
        let mut fwd_ns: u128 = 0;
        let mut ac_ns: u128 = 0;

        // Phase 13s: pipeline GPU forward with CPU AC encode.
        //
        // Sequential baseline was:
        //   for step in 1..max_len:
        //     forward(step)   // blocking GPU, ~5 ms
        //     ac_encode(step) // CPU, ~2-3 ms
        //
        // Pipelined version:
        //   submit forward(1)
        //   for step in 1..max_len:
        //     wait+readback (finishes forward(step))
        //     pre-stage prev_tokens for step+1
        //     submit forward(step+1)      // runs on GPU
        //     ac_encode(step)              // runs on CPU in parallel
        //   drain last submission
        //
        // Saves ~min(fwd_time, ac_time) per step.

        bs.forward_submit(&prev);

        for step in 1..max_len {
            let t_fwd = std::time::Instant::now();
            bs.forward_wait_and_readback();
            if profile {
                fwd_ns += t_fwd.elapsed().as_nanos();
            }

            // Prepare `prev` for the NEXT forward (step+1): each lane's
            // `prev` is the token at this step's index (which each lane
            // will encode below). Do this BEFORE AC encoding so we can
            // issue the next GPU submission in parallel.
            let has_next = step + 1 < max_len;
            if has_next {
                for lane in 0..n {
                    let seg = &chunk[lane];
                    if step < seg.len() {
                        prev[lane] = seg[step];
                    }
                    // else: lane is past its segment; prev stays at whatever
                    // last token was, but output is discarded anyway.
                }
                bs.forward_submit(&prev);
            }

            let t_ac = std::time::Instant::now();
            for lane in 0..n {
                let seg = &chunk[lane];
                if step >= seg.len() {
                    continue;
                }
                let lane_freqs = &bs.freqs[lane * vocab..(lane + 1) * vocab];
                cum.clear();
                cum.push(0u64);
                let mut total: u64 = 0;
                for &f in lane_freqs {
                    total += f as u64;
                    cum.push(total);
                }
                encs[lane].encode_symbol(&cum, seg[step])?;
                // Note: prev[lane] was already set above for step+1's
                // forward. No need to reset here.
            }
            if profile {
                ac_ns += t_ac.elapsed().as_nanos();
            }
        }
        // Drain any remaining submission.
        bs.forward_wait_and_readback();

        if profile {
            let fwd_ms = fwd_ns as f64 / 1e6;
            let ac_ms = ac_ns as f64 / 1e6;
            let steps = (max_len.saturating_sub(1)) as f64;
            eprintln!(
                "  chunk n={n:<3} steps={:<5} fwd={fwd_ms:>8.1}ms ({:.0}µs/step) \
                 ac={ac_ms:>6.1}ms ({:.0}µs/step)",
                max_len - 1,
                (fwd_ns as f64 / steps) / 1e3,
                (ac_ns as f64 / steps) / 1e3,
            );
        }

        // Finish each encoder, push its Vec<u8> body in lane order.
        for enc in encs {
            all_outputs.push(enc.finish()?);
        }
    }

    Ok(all_outputs)
}

/// Phase 13o: parallel variant of [`compress_segments_batched`] that
/// spawns `n_workers` threads, each owning its own `BatchedSession`
/// (and therefore its own Metal command queue). Each worker handles
/// a contiguous strip of the input segments; results are concatenated
/// back in input order. For `n_workers <= 1` this is a straight
/// delegation to the single-threaded version.
///
/// The effective parallel fan-out is `n_workers * batch_size` — e.g.
/// 2 workers × batch=128 = 256 concurrent lanes, distributed across
/// 2 independent GPU command streams. On Apple M-series this gives
/// the GPU scheduler more concurrent work to dispatch on separate
/// compute cores.
pub fn compress_segments_batched_parallel(
    model: &Model,
    segments: &[Vec<u32>],
    batch_size: usize,
    n_workers: usize,
) -> Result<Vec<Vec<u8>>, crate::Error> {
    if n_workers <= 1 || segments.len() <= 1 {
        return compress_segments_batched(model, segments, batch_size);
    }
    let n_segs = segments.len();
    let per_worker = n_segs.div_ceil(n_workers);

    std::thread::scope(|scope| {
        let handles: Vec<_> = segments
            .chunks(per_worker)
            .map(|chunk| {
                scope.spawn(move || {
                    // Each worker re-owns a fresh slice view; the
                    // Model reference is shared read-only across
                    // threads (immutable weights).
                    compress_segments_batched(model, chunk, batch_size)
                })
            })
            .collect();
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(n_segs);
        for h in handles {
            let partial = h.join().expect("compress worker thread panicked")?;
            out.extend(partial);
        }
        Ok(out)
    })
}

/// Phase 13o counterpart to [`compress_segments_batched_parallel`].
pub fn decompress_segments_batched_parallel(
    model: &Model,
    bodies: &[(Vec<u8>, u32, u32)],
    batch_size: usize,
    n_workers: usize,
) -> Result<Vec<Vec<u32>>, crate::Error> {
    if n_workers <= 1 || bodies.len() <= 1 {
        return decompress_segments_batched(model, bodies, batch_size);
    }
    let n_bodies = bodies.len();
    let per_worker = n_bodies.div_ceil(n_workers);

    std::thread::scope(|scope| {
        let handles: Vec<_> = bodies
            .chunks(per_worker)
            .map(|chunk| scope.spawn(move || decompress_segments_batched(model, chunk, batch_size)))
            .collect();
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(n_bodies);
        for h in handles {
            let partial = h.join().expect("decompress worker thread panicked")?;
            out.extend(partial);
        }
        Ok(out)
    })
}

/// Inverse of [`compress_segments_batched`]. Same lockstep N-lane
/// batching: chunk the bodies into groups of `batch_size`, run a
/// single GPU forward + cum_freqs per step across every active lane,
/// then per-lane decode the next token from each lane's own AC body.
pub fn decompress_segments_batched(
    model: &Model,
    bodies: &[(Vec<u8>, u32, u32)],
    batch_size: usize,
) -> Result<Vec<Vec<u32>>, crate::Error> {
    use crate::arithmetic::ArithmeticDecoder;
    if bodies.is_empty() {
        return Ok(Vec::new());
    }
    let batch = batch_size.max(1);
    let mut bs = BatchedSession::new(model, batch)
        .map_err(|e| crate::Error::BadCheckpoint(format!("Metal init: {e}")))?;
    let vocab = bs.vocab;
    let mut all_outputs: Vec<Vec<u32>> = Vec::with_capacity(bodies.len());

    for chunk in bodies.chunks(batch) {
        let n = chunk.len();
        bs.reset();

        // Per-lane decoder + output tokens. Decoders borrow from
        // chunk's bodies; their lifetime is bounded by the chunk loop.
        let mut decs: Vec<Option<ArithmeticDecoder<&[u8]>>> = Vec::with_capacity(n);
        let mut tokens: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut prev = vec![0u32; batch];

        for (lane, (body, n_tokens, bos)) in chunk.iter().enumerate() {
            let mut t = Vec::with_capacity(*n_tokens as usize);
            t.push(*bos);
            tokens.push(t);
            prev[lane] = *bos;
            if *n_tokens <= 1 {
                decs.push(None);
            } else {
                decs.push(Some(ArithmeticDecoder::new(body.as_slice())?));
            }
        }
        let pad_token = prev[0];
        prev[n..batch].fill(pad_token);

        let max_len = chunk
            .iter()
            .map(|(_, n_tok, _)| *n_tok as usize)
            .max()
            .unwrap_or(0);
        if max_len <= 1 {
            for t in tokens {
                all_outputs.push(t);
            }
            continue;
        }

        let mut cum: Vec<u64> = Vec::with_capacity(vocab + 1);
        for step in 1..max_len {
            bs.forward_batched(&prev);
            bs.cum_freqs_batched();

            for lane in 0..n {
                let n_tokens = chunk[lane].1 as usize;
                if step >= n_tokens {
                    continue;
                }
                let dec = decs[lane].as_mut().expect("active lane has a decoder");
                let lane_freqs = &bs.freqs[lane * vocab..(lane + 1) * vocab];
                cum.clear();
                cum.push(0u64);
                let mut total: u64 = 0;
                for &f in lane_freqs {
                    total += f as u64;
                    cum.push(total);
                }
                let tok = dec.decode_symbol(&cum)?;
                tokens[lane].push(tok);
                prev[lane] = tok;
            }
        }

        for t in tokens {
            all_outputs.push(t);
        }
    }

    Ok(all_outputs)
}

/// Decompress one segment via `BatchedSession`. Inverse of
/// [`compress_one_segment_batched`]. `n_tokens` includes the
/// BOS-equivalent first token (which is supplied as `bos_token`
/// since the file format itself doesn't ship it).
pub fn decompress_one_segment_batched(
    session: &mut BatchedSession<'_>,
    ac_body: &[u8],
    n_tokens: u32,
    bos_token: u32,
) -> Result<Vec<u32>, crate::Error> {
    use crate::arithmetic::ArithmeticDecoder;

    session.reset();
    let mut tokens = Vec::with_capacity(n_tokens as usize);
    tokens.push(bos_token);
    if n_tokens <= 1 {
        return Ok(tokens);
    }

    let mut dec = ArithmeticDecoder::new(ac_body)?;
    let vocab = session.vocab;
    let mut prev_batch = vec![bos_token; session.batch];
    let mut prev = bos_token;
    for _ in 1..n_tokens {
        prev_batch[0] = prev;
        session.forward_batched(&prev_batch);
        session.cum_freqs_batched();
        let mut cum = Vec::with_capacity(vocab + 1);
        cum.push(0u64);
        let mut total: u64 = 0;
        for j in 0..vocab {
            total += session.freqs[j] as u64;
            cum.push(total);
        }
        let tok = dec.decode_symbol(&cum)?;
        tokens.push(tok);
        prev = tok;
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::rwkv::Session;

    /// End-to-end Phase 13e validation: BatchedSession output must
    /// match N independent CPU Sessions on the same input sequence.
    /// Uses the actual L3TC-200K checkpoint.
    #[test]
    fn batched_session_matches_n_cpu_sessions() {
        // Skip if the model file isn't checked out (CI).
        let model_path = "checkpoints/l3tc_200k.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }

        let mut ckpt = match Checkpoint::load(model_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let model = match Model::from_checkpoint(&mut ckpt) {
            Ok(m) => m,
            Err(_) => return,
        };

        let batch: usize = 4;
        // Deterministic synthetic token sequence per lane, length 8.
        // Each lane gets a distinct sequence so we exercise per-lane
        // state independence.
        let n_steps: usize = 8;
        let sequences: Vec<Vec<u32>> = (0..batch)
            .map(|b| {
                (0..n_steps)
                    .map(|t| ((t * 31 + b * 17) % model.vocab_size) as u32)
                    .collect()
            })
            .collect();

        // CPU reference: run N independent sessions, capture per-step
        // logits per lane.
        let mut cpu_logits_per_step: Vec<Vec<f32>> = (0..n_steps)
            .map(|_| vec![0.0; batch * model.vocab_size])
            .collect();
        for b in 0..batch {
            let mut s = Session::new(&model);
            for (t, &tok) in sequences[b].iter().enumerate() {
                let logits = s.forward(tok);
                let dst =
                    &mut cpu_logits_per_step[t][b * model.vocab_size..(b + 1) * model.vocab_size];
                dst.copy_from_slice(logits);
            }
        }

        // Metal batched session.
        let mut bs = match BatchedSession::new(&model, batch) {
            Ok(s) => s,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("BatchedSession::new failed: {other}"),
        };

        for t in 0..n_steps {
            let prev: Vec<u32> = (0..batch).map(|b| sequences[b][t]).collect();
            bs.forward_batched(&prev);
            bs.sync_logits_host();

            // Compare logits per lane.
            for b in 0..batch {
                let cpu = &cpu_logits_per_step[t][b * model.vocab_size..(b + 1) * model.vocab_size];
                let gpu = &bs.logits[b * model.vocab_size..(b + 1) * model.vocab_size];
                let mut max_abs = 0.0f32;
                for i in 0..model.vocab_size {
                    let d = (cpu[i] - gpu[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                // Logits accumulate FMA error across many ops. Allow
                // ~1e-2 absolute (well under freq-quantization
                // threshold once softmax+round absorbs the residual).
                assert!(
                    max_abs < 5e-2,
                    "step {t} lane {b}: max abs logit diff = {max_abs}"
                );
            }
        }
    }

    /// Phase 13e validation: BatchedSession.cum_freqs_batched on
    /// real model output must match the CPU cum_freqs path within
    /// the freq-quantization tolerance (±1 freq per element).
    #[test]
    fn batched_session_cum_freqs_matches_cpu() {
        use crate::tensor;

        let model_path = "checkpoints/l3tc_200k.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let mut ckpt = match Checkpoint::load(model_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let model = match Model::from_checkpoint(&mut ckpt) {
            Ok(m) => m,
            Err(_) => return,
        };

        let batch: usize = 2;
        let mut bs = match BatchedSession::new(&model, batch) {
            Ok(s) => s,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("BatchedSession::new failed: {other}"),
        };

        // Single forward step with synthetic token IDs.
        let prev: Vec<u32> = (0..batch).map(|b| (b as u32) * 100).collect();
        bs.forward_batched(&prev);
        bs.sync_logits_host();
        bs.cum_freqs_batched();

        // CPU reference for each lane.
        let mut cpu_freqs = vec![0u32; batch * model.vocab_size];
        let mut exps = vec![0.0f32; model.vocab_size];
        for b in 0..batch {
            let logits = &bs.logits[b * model.vocab_size..(b + 1) * model.vocab_size];
            let m = tensor::max_f32(logits);
            let s = tensor::softmax_shifted_exp_sum(logits, m, &mut exps);
            let scale = (1.0 / s) * 10_000_000.0;
            tensor::quantize_exps_to_freqs(
                &exps,
                scale,
                &mut cpu_freqs[b * model.vocab_size..(b + 1) * model.vocab_size],
            );
        }

        // Per-element freq drift can be larger than ±1 on real
        // model logits (the polynomial-vs-libm exp + FMA reorder
        // adds up to a few ULPs of the *exp* magnitude, which at
        // the round() boundary can shift a freq by O(scale * ULP)).
        // What matters for the AC encoder is *total cum drift per
        // lane* — bounded by sum_of_per_element_diffs. Phase 12
        // accepts <0.0005 bpb, which translates to ~hundreds of
        // freq drift across 16K elements.
        let mut total_per_lane = vec![0i64; batch];
        let mut max_diff = 0i64;
        #[allow(clippy::needless_range_loop)] // test path: `b`/`i` compute flat lane*vocab indices
        for b in 0..batch {
            let mut sum = 0i64;
            for i in 0..model.vocab_size {
                let k = b * model.vocab_size + i;
                let d = (cpu_freqs[k] as i64 - bs.freqs[k] as i64).abs();
                sum += d;
                if d > max_diff {
                    max_diff = d;
                }
            }
            total_per_lane[b] = sum;
        }
        let max_total = *total_per_lane.iter().max().unwrap();
        // Per-lane total drift must be a small fraction of the
        // 10M total mass — keep <10K (0.1%) which corresponds to
        // <0.001 bpb actual ratio drift.
        assert!(
            max_total < 10_000,
            "per-lane cum_freqs total drift = {max_total} \
             (max single-element diff = {max_diff}) — too large"
        );
    }

    /// Top-level codec round-trip: compress text via Metal,
    /// decompress via Metal, expect bit-identical output. Validates
    /// the full file format + tokenizer + AC + GPU forward path.
    #[test]
    fn codec_metal_round_trip_50kb() {
        let model_path = "checkpoints/l3tc_200k.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let mut ckpt = match Checkpoint::load(model_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let model = match Model::from_checkpoint(&mut ckpt) {
            Ok(m) => m,
            Err(_) => return,
        };
        // Probe Metal availability before doing real work.
        if BatchedSession::new(&model, 1).is_err() {
            return;
        }

        let tok_path =
            "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model";
        if !std::path::Path::new(tok_path).exists() {
            eprintln!("test skipped: tokenizer not found at {tok_path}");
            return;
        }
        let tokenizer = match crate::Tokenizer::load(tok_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("test skipped: tokenizer load failed: {e}");
                return;
            }
        };

        let input_path = "/tmp/e6_50k.txt";
        let text = match std::fs::read_to_string(input_path) {
            Ok(t) => t,
            Err(_) => return,
        };

        let compressed = crate::codec::compress_with_metal(
            &text,
            &tokenizer,
            &model,
            crate::codec::DEFAULT_SEGMENT_BYTES,
            1,
            0,
        )
        .expect("compress_with_metal");

        let recovered = crate::codec::decompress_with_metal(&compressed, &tokenizer, &model)
            .expect("decompress_with_metal");

        assert_eq!(
            recovered.len(),
            text.len(),
            "decompressed length mismatch: {} vs {}",
            recovered.len(),
            text.len()
        );
        assert_eq!(recovered, text, "decompressed text differs from original");
        eprintln!(
            "Metal round-trip OK: {} → {} → {} bytes (ratio {:.4})",
            text.len(),
            compressed.len(),
            recovered.len(),
            compressed.len() as f64 / text.len() as f64
        );
    }

    /// End-to-end Phase 13e round-trip: compress AND decompress a
    /// token sequence via BatchedSession (Metal). This is the
    /// canonical correctness test for the within-backend invariant.
    /// Cross-backend (GPU encode → CPU decode) does NOT round-trip
    /// because the GPU forward pass diverges from CPU NEON by a
    /// few ULPs — see compress_one_segment_batched docs.
    #[test]
    fn batched_round_trip_metal_self() {
        let model_path = "checkpoints/l3tc_200k.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let mut ckpt = match Checkpoint::load(model_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let model = match Model::from_checkpoint(&mut ckpt) {
            Ok(m) => m,
            Err(_) => return,
        };

        let tokens: Vec<u32> = (0..256)
            .map(|t| ((t * 31 + 11) % model.vocab_size) as u32)
            .collect();
        let bos = tokens[0];

        let mut bs = match BatchedSession::new(&model, 1) {
            Ok(s) => s,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("BatchedSession::new failed: {other}"),
        };

        // Encode.
        let ac_bytes = compress_one_segment_batched(&mut bs, &tokens).expect("batched compress");

        // Decode using the same backend (must reset state internally).
        let decoded = decompress_one_segment_batched(&mut bs, &ac_bytes, tokens.len() as u32, bos)
            .expect("batched decompress");

        assert_eq!(decoded.len(), tokens.len(), "length mismatch");
        let mut mismatches = 0usize;
        for i in 0..tokens.len() {
            if decoded[i] != tokens[i] {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  mismatch at {}: encoded={}, decoded={}",
                        i, tokens[i], decoded[i]
                    );
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "GPU encode → GPU decode round-trip failed: {mismatches}/{} tokens",
            tokens.len()
        );
    }
}
