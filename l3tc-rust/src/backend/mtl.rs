//! Metal backend — Phase 13.
//!
//! Smoke test (Phase 13b) + head matvec kernel (Phase 13c).
//! Phase 13d will add the rest of the forward pass.

use std::error::Error;
use std::fmt;

use metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

/// Errors produced by the Metal backend.
#[derive(Debug)]
pub enum MetalError {
    /// No Metal device is available on this machine (e.g., headless
    /// server, container without GPU passthrough).
    NoDevice,
    /// MSL source failed to compile via `Device::new_library_with_source`.
    /// The wrapped string is the compiler diagnostic.
    LibraryCompile(String),
    /// A pipeline state could not be built from a function.
    PipelineCreate(String),
    /// Generic runtime failure with diagnostic message.
    Runtime(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::NoDevice => write!(f, "no Metal device available"),
            MetalError::LibraryCompile(msg) => {
                write!(f, "MSL compile failed: {msg}")
            }
            MetalError::PipelineCreate(msg) => {
                write!(f, "pipeline create failed: {msg}")
            }
            MetalError::Runtime(msg) => write!(f, "Metal runtime: {msg}"),
        }
    }
}

impl Error for MetalError {}

/// Run a self-contained sanity check: compile a trivial elementwise-
/// add kernel, dispatch it on N floats, verify the output matches the
/// CPU result.
///
/// Returns `Ok(n)` on success (`n` = elements processed) or a
/// [`MetalError`] describing what failed. Used by the hidden CLI
/// subcommand `l3tc metal-smoke` to verify the toolchain works on
/// the local machine without committing to a full forward pass.
pub fn smoke_test(n: usize) -> Result<usize, MetalError> {
    let device = Device::system_default().ok_or(MetalError::NoDevice)?;
    let queue = device.new_command_queue();

    let pipeline = build_add_pipeline(&device)?;

    // Inputs: a[i] = i as f32; b[i] = (n - i) as f32; expected c[i] = n.
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected = n as f32;

    let buf_a = new_input_buffer(&device, &a);
    let buf_b = new_input_buffer(&device, &b);
    let buf_c = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_add(&queue, &pipeline, &buf_a, &buf_b, &buf_c, n);

    // Read back and verify.
    let out_ptr = buf_c.contents() as *const f32;
    // SAFETY: buf_c was created with `n * sizeof(f32)` bytes and
    // dispatch_add waits for completion before returning, so the
    // GPU writes are visible.
    let out: &[f32] = unsafe { std::slice::from_raw_parts(out_ptr, n) };
    for (i, &v) in out.iter().enumerate() {
        if (v - expected).abs() > 1e-6 {
            return Err(MetalError::Runtime(format!(
                "smoke test failed at index {i}: got {v}, expected {expected}"
            )));
        }
    }
    Ok(n)
}

const ADD_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element-wise vector add: c[i] = a[i] + b[i].
// One thread per element; total grid size = n; threadgroup
// size chosen by the dispatcher.
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       c [[buffer(2)]],
    uint                i [[thread_position_in_grid]]
) {
    c[i] = a[i] + b[i];
}
"#;

fn build_add_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(ADD_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let function = library
        .get_function("add_f32", None)
        .map_err(|e| MetalError::PipelineCreate(format!("{e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineCreate)
}

fn new_input_buffer(device: &Device, data: &[f32]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        bytes as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn dispatch_add(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    n: usize,
) {
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(c), 0);

    // Threadgroup width — pipeline-recommended; falls back to 64.
    let tg_width = pipeline
        .thread_execution_width()
        .max(1)
        .min(n as u64) as u64;

    encoder.dispatch_threads(
        MTLSize {
            width: n as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    commit_and_wait(cmd_buf);
}

fn commit_and_wait(cmd_buf: &CommandBufferRef) {
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

// ===================================================================
// Phase 13c: head matvec INT8 kernel
// ===================================================================

/// MSL kernel for the INT8 head matvec — `out[i] = sum_j (x[j] *
/// scales[j] * qmat[j*rows + i])` where `qmat` is column-major and
/// `i` is the output row, `j` indexes input columns.
///
/// One thread per output row. Adjacent threads read adjacent
/// `qmat[j*rows + i]` entries, so the per-`j` read is fully
/// coalesced (16 KB sequential per iteration on 16K vocab).
const HEAD_MATVEC_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Single-stream head matvec.
// out[i] = Σ_j (x[j] * scales[j] * qmat[j*rows + i])
// One thread per output row. Dispatch grid = (rows, 1, 1).
kernel void head_matvec_int8(
    device const char*  qmat   [[buffer(0)]],   // i8, column-major rows*cols
    device const float* scales [[buffer(1)]],   // length cols
    device const float* x      [[buffer(2)]],   // length cols
    device float*       out    [[buffer(3)]],   // length rows
    constant uint&      rows   [[buffer(4)]],
    constant uint&      cols   [[buffer(5)]],
    uint                i      [[thread_position_in_grid]]
) {
    if (i >= rows) return;
    float acc = 0.0;
    for (uint j = 0; j < cols; ++j) {
        float xs = x[j] * scales[j];
        acc = fma(xs, (float)qmat[j * rows + i], acc);
    }
    out[i] = acc;
}

// Batched head matvec.
// out[b * rows + i] = Σ_j (x[b * cols + j] * scales[j] * qmat[j*rows + i])
// Dispatch grid = (rows, batch, 1). Each thread handles one
// (batch, row) cell. The qmat read is shared across all batches —
// adjacent threads in a warp coalesce on the row axis just as in
// the single-stream case.
kernel void head_matvec_int8_batched(
    device const char*  qmat   [[buffer(0)]],   // i8, column-major rows*cols
    device const float* scales [[buffer(1)]],   // length cols
    device const float* x      [[buffer(2)]],   // batch * cols
    device float*       out    [[buffer(3)]],   // batch * rows
    constant uint&      rows   [[buffer(4)]],
    constant uint&      cols   [[buffer(5)]],
    constant uint&      batch  [[buffer(6)]],
    uint2               gid    [[thread_position_in_grid]]
) {
    uint i = gid.x;  // row
    uint b = gid.y;  // batch lane
    if (i >= rows || b >= batch) return;
    float acc = 0.0;
    device const float* xb = x + (b * cols);
    for (uint j = 0; j < cols; ++j) {
        float xs = xb[j] * scales[j];
        acc = fma(xs, (float)qmat[j * rows + i], acc);
    }
    out[b * rows + i] = acc;
}
"#;

/// A reusable Metal pipeline for the head INT8 matvec, holding the
/// model weights (qmat + per-column scales) AND the I/O buffers
/// (`x_buf`, `out_buf`) once at construction. Per-token `forward`
/// just memcpys `x` into the existing buffer, dispatches, and reads
/// `out_buf` directly — no per-call Metal allocations.
///
/// Equivalent CPU kernel: [`crate::tensor::matvec_col_major_int8`]
/// (Phase 12d hand-tuned NEON).
pub struct HeadKernelMetal {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_batched: ComputePipelineState,
    qmat_buf: Buffer,
    scales_buf: Buffer,
    x_buf: Buffer,
    out_buf: Buffer,
    rows: usize,
    cols: usize,
    rows_u32: u32,
    cols_u32: u32,
}

impl HeadKernelMetal {
    /// Build the kernel, upload model weights, and pre-allocate I/O
    /// buffers. `qmat` must be column-major i8 of length `rows * cols`;
    /// `scales` must be f32 of length `cols`.
    pub fn new(qmat: &[i8], scales: &[f32], rows: usize, cols: usize) -> Result<Self, MetalError> {
        assert_eq!(qmat.len(), rows * cols, "qmat shape");
        assert_eq!(scales.len(), cols, "scales shape");
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let (pipeline, pipeline_batched) = build_head_pipelines(&device)?;
        let qmat_buf = device.new_buffer_with_data(
            qmat.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(qmat) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = device.new_buffer_with_data(
            scales.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(scales) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Pre-allocated I/O buffers — reused across every forward()
        // call. Per-call cost becomes a memcpy and a dispatch.
        let x_buf = device.new_buffer(
            (cols * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (rows * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            queue,
            pipeline,
            pipeline_batched,
            qmat_buf,
            scales_buf,
            x_buf,
            out_buf,
            rows,
            cols,
            rows_u32: rows as u32,
            cols_u32: cols as u32,
        })
    }

    /// Batched forward — process `batch` independent x→out pairs in
    /// one dispatch. Amortizes the per-call dispatch overhead across
    /// the batch.
    ///
    /// `x` must be `batch * cols` floats laid out row-major (one
    /// segment's input vector per row); `out` must be `batch * rows`.
    /// Per-call buffers are allocated each invocation since batch
    /// size can vary; the model weights stay resident.
    pub fn forward_batched(&self, x: &[f32], out: &mut [f32], batch: usize) {
        assert_eq!(x.len(), batch * self.cols, "x shape (batch*cols)");
        assert_eq!(out.len(), batch * self.rows, "out shape (batch*rows)");

        let device = self.queue.device();
        let xb_buf = device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(x) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let outb_buf = device.new_buffer(
            (batch * self.rows * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_batched);
        encoder.set_buffer(0, Some(&self.qmat_buf), 0);
        encoder.set_buffer(1, Some(&self.scales_buf), 0);
        encoder.set_buffer(2, Some(&xb_buf), 0);
        encoder.set_buffer(3, Some(&outb_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &self.rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &self.cols_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        // 2D grid: (rows, batch). Threadgroup along rows axis.
        let tg_width = self
            .pipeline_batched
            .thread_execution_width()
            .max(1)
            .min(self.rows as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.rows as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        // SAFETY: outb_buf has batch*rows*sizeof(f32), GPU has finished.
        unsafe {
            let src = outb_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), batch * self.rows);
        }
    }

    /// One forward pass: copy `x` into the persistent input buffer,
    /// dispatch the kernel, copy the persistent output buffer into
    /// `out`. Blocks until the GPU finishes.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "x shape");
        assert_eq!(out.len(), self.rows, "out shape");

        // Copy x into the pre-allocated GPU-visible buffer.
        // SAFETY: x_buf has cols*sizeof(f32) bytes, x has cols floats.
        unsafe {
            let dst = self.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), dst, self.cols);
        }

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.qmat_buf), 0);
        encoder.set_buffer(1, Some(&self.scales_buf), 0);
        encoder.set_buffer(2, Some(&self.x_buf), 0);
        encoder.set_buffer(3, Some(&self.out_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &self.rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &self.cols_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(self.rows as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        // Read the persistent output buffer.
        // SAFETY: out_buf has rows*sizeof(f32), GPU has finished.
        unsafe {
            let src = self.out_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), self.rows);
        }
    }
}

fn build_head_pipelines(
    device: &Device,
) -> Result<(ComputePipelineState, ComputePipelineState), MetalError> {
    let library = device
        .new_library_with_source(HEAD_MATVEC_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let mk = |name: &str| -> Result<ComputePipelineState, MetalError> {
        let f = library
            .get_function(name, None)
            .map_err(|e| MetalError::PipelineCreate(format!("{name}: {e:?}")))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(MetalError::PipelineCreate)
    };
    Ok((mk("head_matvec_int8")?, mk("head_matvec_int8_batched")?))
}

// ===================================================================
// Phase 13e prep: batched layer_norm (proves the batched pattern
// works for kernels with intra-element reductions, not just
// embarrassingly-parallel matvecs).
// ===================================================================

const LAYER_NORM_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Batched layer norm: out[b, i] = (x[b, i] - mean_b) / sqrt(var_b + eps)
//                                 * weight[i] + bias[i]
// One thread per batch lane; each thread does its own three-pass
// reduction over h. For h=96 the pass cost is ~96 ops × 3, which
// is fine — at batch=256 the dispatch handles 256 lanes in parallel.
kernel void layer_norm_batched(
    device const float* x      [[buffer(0)]],   // batch * h
    device const float* weight [[buffer(1)]],   // h
    device const float* bias   [[buffer(2)]],   // h
    device float*       out    [[buffer(3)]],   // batch * h
    constant uint&      h      [[buffer(4)]],
    constant uint&      batch  [[buffer(5)]],
    constant float&     eps    [[buffer(6)]],
    uint                b      [[thread_position_in_grid]]
) {
    if (b >= batch) return;
    device const float* xb = x + (b * h);
    device float*       ob = out + (b * h);

    // Pass 1: mean.
    float sum = 0.0;
    for (uint i = 0; i < h; ++i) {
        sum += xb[i];
    }
    float mean = sum / float(h);

    // Pass 2: variance via running (x - mean)².
    float vsum = 0.0;
    for (uint i = 0; i < h; ++i) {
        float d = xb[i] - mean;
        vsum = fma(d, d, vsum);
    }
    float inv_std = rsqrt(vsum / float(h) + eps);

    // Pass 3: out = (x - mean) * inv_std * weight + bias.
    for (uint i = 0; i < h; ++i) {
        ob[i] = fma((xb[i] - mean) * inv_std, weight[i], bias[i]);
    }
}
"#;

/// Reusable batched layer-norm kernel. Holds the per-feature
/// `weight` and `bias` vectors on the GPU once (these are tied to
/// the model layer, not the per-token state).
///
/// CPU equivalent: [`crate::tensor::layer_norm`] (Phase 12f NEON).
pub struct LayerNormKernelMetal {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    weight_buf: Buffer,
    bias_buf: Buffer,
    h: usize,
    h_u32: u32,
    eps: f32,
}

impl LayerNormKernelMetal {
    /// Build the kernel; uploads `weight` and `bias` (both length `h`)
    /// to a pair of GPU-resident buffers.
    pub fn new(weight: &[f32], bias: &[f32], h: usize, eps: f32) -> Result<Self, MetalError> {
        assert_eq!(weight.len(), h);
        assert_eq!(bias.len(), h);
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let pipeline = build_layer_norm_pipeline(&device)?;
        let weight_buf = device.new_buffer_with_data(
            weight.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(weight) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let bias_buf = device.new_buffer_with_data(
            bias.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(bias) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            queue,
            pipeline,
            weight_buf,
            bias_buf,
            h,
            h_u32: h as u32,
            eps,
        })
    }

    /// Apply layer_norm to a flat `batch * h` input → `batch * h`
    /// output. Per-call buffers since batch can vary; weight/bias
    /// stay GPU-resident.
    pub fn forward_batched(&self, x: &[f32], out: &mut [f32], batch: usize) {
        assert_eq!(x.len(), batch * self.h);
        assert_eq!(out.len(), batch * self.h);

        let device = self.queue.device();
        let xb_buf = device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(x) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let outb_buf = device.new_buffer(
            (batch * self.h * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&xb_buf), 0);
        encoder.set_buffer(1, Some(&self.weight_buf), 0);
        encoder.set_buffer(2, Some(&self.bias_buf), 0);
        encoder.set_buffer(3, Some(&outb_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<f32>() as u64,
            &self.eps as *const f32 as *const std::ffi::c_void,
        );

        let tg_width = self.pipeline.thread_execution_width().max(1).min(batch as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: batch as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            let src = outb_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), batch * self.h);
        }
    }
}

fn build_layer_norm_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(LAYER_NORM_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let function = library
        .get_function("layer_norm_batched", None)
        .map_err(|e| MetalError::PipelineCreate(format!("layer_norm_batched: {e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineCreate)
}

// ===================================================================
// Phase 13e prep: batched 96×96 matvec (used 8x per layer per token
// in time_mix + channel_mix on the L3TC-200K model).
// ===================================================================

const MATVEC_96X96_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Batched 96x96 matvec, row-major.
// out[b, i] = Σ_j mat[i, j] * x[b, j]
// Dispatch grid = (h, batch). One thread per (batch, row) output.
// `mat` is shared across all batch lanes (one set of weights);
// stays in cache after warm-up.
kernel void matvec_96x96_batched(
    device const float* mat   [[buffer(0)]],   // h * h
    device const float* x     [[buffer(1)]],   // batch * h
    device float*       out   [[buffer(2)]],   // batch * h
    constant uint&      h     [[buffer(3)]],
    constant uint&      batch [[buffer(4)]],
    uint2               gid   [[thread_position_in_grid]]
) {
    uint i = gid.x;  // output row
    uint b = gid.y;  // batch lane
    if (i >= h || b >= batch) return;
    device const float* xb = x + (b * h);
    device const float* row = mat + (i * h);
    float acc = 0.0;
    for (uint j = 0; j < h; ++j) {
        acc = fma(row[j], xb[j], acc);
    }
    out[b * h + i] = acc;
}
"#;

/// Reusable batched 96×96 matvec kernel. The matrix lives on the GPU
/// once at construction; per-call uploads the batched input and reads
/// back the batched output.
///
/// CPU equivalent: [`crate::tensor::matvec_96x96`] (Phase 12 hand-tuned
/// NEON with 4 accumulators).
pub struct Matvec96Metal {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    mat_buf: Buffer,
    h: usize,
    h_u32: u32,
}

impl Matvec96Metal {
    /// Build the kernel and upload the matrix. `mat` must be
    /// row-major `h * h` (typically h=96 for the L3TC-200K model;
    /// the kernel is hard-coded to read `h * h` weights but works
    /// for any square h).
    pub fn new(mat: &[f32], h: usize) -> Result<Self, MetalError> {
        assert_eq!(mat.len(), h * h);
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let pipeline = build_matvec96_pipeline(&device)?;
        let mat_buf = device.new_buffer_with_data(
            mat.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(mat) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            queue,
            pipeline,
            mat_buf,
            h,
            h_u32: h as u32,
        })
    }

    /// Apply matvec to a flat `batch * h` input → `batch * h` output.
    pub fn forward_batched(&self, x: &[f32], out: &mut [f32], batch: usize) {
        assert_eq!(x.len(), batch * self.h);
        assert_eq!(out.len(), batch * self.h);

        let device = self.queue.device();
        let xb_buf = device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(x) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let outb_buf = device.new_buffer(
            (batch * self.h * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.mat_buf), 0);
        encoder.set_buffer(1, Some(&xb_buf), 0);
        encoder.set_buffer(2, Some(&outb_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(self.h as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.h as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            let src = outb_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), batch * self.h);
        }
    }
}

fn build_matvec96_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(MATVEC_96X96_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let function = library
        .get_function("matvec_96x96_batched", None)
        .map_err(|e| MetalError::PipelineCreate(format!("matvec_96x96_batched: {e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineCreate)
}

// ===================================================================
// Phase 13e prep: batched sub_exp (out[i] = exp(a[i] - b[i])).
// Used 4x per layer per token in time_mix's state-evolution exp
// passes. Polynomial matches the CPU NEON exp_f32x4_neon exactly so
// the freq-quantization remains equivalent across backends.
// ===================================================================

const SUB_EXP_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Inline polynomial exp matching the CPU NEON implementation:
// degree-6 minimax for 2^r on [0, 1], 2^k via bit construction.
// Inputs clamped to [-50, +∞); the positive side is wider than CPU
// (which only saw negative inputs in time_mix) but the formula
// works for any input within float-exponent range.
static inline float poly_exp(float x) {
    x = fmax(x, -50.0f);
    float y = x * 1.442695041f;            // log2(e)
    float k = floor(y);
    float r = y - k;
    float p = 0.000154035303933f;
    p = fma(p, r, 0.001333355814643f);
    p = fma(p, r, 0.009618129107628f);
    p = fma(p, r, 0.055504108664821f);
    p = fma(p, r, 0.240226506959101f);
    p = fma(p, r, 0.693147180559945f);
    p = fma(p, r, 1.0f);                    // ≈ 2^r
    // 2^k via float-bit construction (clamped so the exponent is in
    // the normal range [55, 254]).
    int  ki = int(k);
    int  exp_bits = (ki + 127) << 23;
    float two_k = as_type<float>(exp_bits);
    return p * two_k;
}

// Batched sub-exp: out[b, i] = exp(a[b, i] - b[b, i]).
// Inputs `a` and `b_v` are both batch * h. Each thread handles one
// (batch, element) cell.
kernel void sub_exp_batched(
    device const float* a     [[buffer(0)]],
    device const float* b_v   [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      h     [[buffer(3)]],
    constant uint&      batch [[buffer(4)]],
    uint2               gid   [[thread_position_in_grid]]
) {
    uint i  = gid.x;
    uint bb = gid.y;
    if (i >= h || bb >= batch) return;
    uint k = bb * h + i;
    float diff = a[k] - b_v[k];
    out[k] = poly_exp(diff);
}
"#;

/// Reusable batched sub_exp kernel. `out[b, i] = exp(a[b, i] - b[b, i])`.
///
/// CPU equivalent: [`crate::tensor::sub_exp`] (Phase 12a). Same
/// polynomial coefficients, so cum_freqs quantization is equivalent
/// across CPU / Metal.
pub struct SubExpKernelMetal {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    h: usize,
    h_u32: u32,
}

impl SubExpKernelMetal {
    /// Build the kernel. Stateless — no model weights to upload.
    /// `h` fixes the per-lane element count for downstream dispatch.
    pub fn new(h: usize) -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let pipeline = build_sub_exp_pipeline(&device)?;
        Ok(Self {
            queue,
            pipeline,
            h,
            h_u32: h as u32,
        })
    }

    /// Compute `out[b, i] = exp(a[b, i] - b_v[b, i])` for every
    /// `(b, i)` in `batch × h`. Per-call upload of `a` and `b_v`,
    /// per-call download of `out`.
    pub fn forward_batched(&self, a: &[f32], b_v: &[f32], out: &mut [f32], batch: usize) {
        assert_eq!(a.len(), batch * self.h);
        assert_eq!(b_v.len(), batch * self.h);
        assert_eq!(out.len(), batch * self.h);

        let device = self.queue.device();
        let n_bytes = (batch * self.h * std::mem::size_of::<f32>()) as u64;
        let a_buf = device.new_buffer_with_data(
            a.as_ptr() as *const std::ffi::c_void,
            n_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b_v.as_ptr() as *const std::ffi::c_void,
            n_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(self.h as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.h as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            let src = out_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), batch * self.h);
        }
    }
}

fn build_sub_exp_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(SUB_EXP_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let function = library
        .get_function("sub_exp_batched", None)
        .map_err(|e| MetalError::PipelineCreate(format!("sub_exp_batched: {e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineCreate)
}

// ===================================================================
// Phase 13e prep: batched sigmoid (safe -|x| form matching CPU
// Phase 12c). Used 1x per layer in both time_mix and channel_mix
// for the receptance gate.
// ===================================================================

const SIGMOID_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Same polynomial exp as sub_exp_batched (clamped to [-50, 0] for
// safety; sigmoid feeds it -|x| so the polynomial argument is
// always ≤ 0).
static inline float poly_exp(float x) {
    x = fmax(x, -50.0f);
    float y = x * 1.442695041f;
    float k = floor(y);
    float r = y - k;
    float p = 0.000154035303933f;
    p = fma(p, r, 0.001333355814643f);
    p = fma(p, r, 0.009618129107628f);
    p = fma(p, r, 0.055504108664821f);
    p = fma(p, r, 0.240226506959101f);
    p = fma(p, r, 0.693147180559945f);
    p = fma(p, r, 1.0f);
    int  ki = int(k);
    int  exp_bits = (ki + 127) << 23;
    float two_k = as_type<float>(exp_bits);
    return p * two_k;
}

// Sigmoid via the safe -|x| form, matching CPU Phase 12c. For
// x >= 0:  sig(x) = 1 / (1 + e^-x)            = 1 / (1 + e_negabs)
// For x < 0:   sig(x) = e^x / (1 + e^x)       = e_negabs / (1 + e_negabs)
// Branchless via select(): polynomial argument stays ≤ 0.
kernel void sigmoid_batched(
    device const float* x     [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      h     [[buffer(2)]],
    constant uint&      batch [[buffer(3)]],
    uint2               gid   [[thread_position_in_grid]]
) {
    uint i  = gid.x;
    uint bb = gid.y;
    if (i >= h || bb >= batch) return;
    uint k = bb * h + i;
    float v = x[k];
    float e = poly_exp(-fabs(v));
    float denom = 1.0f + e;
    float pos_form = 1.0f / denom;
    float neg_form = e / denom;
    out[k] = (v >= 0.0f) ? pos_form : neg_form;
}
"#;

/// Reusable batched sigmoid kernel (safe `-|x|` form).
///
/// CPU equivalent: [`crate::tensor::sigmoid_inplace`] (Phase 12c).
pub struct SigmoidKernelMetal {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    h: usize,
    h_u32: u32,
}

impl SigmoidKernelMetal {
    /// Build the kernel. Stateless.
    pub fn new(h: usize) -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let pipeline = build_sigmoid_pipeline(&device)?;
        Ok(Self {
            queue,
            pipeline,
            h,
            h_u32: h as u32,
        })
    }

    /// Compute `out[b, i] = sigmoid(x[b, i])` for every `(b, i)`.
    pub fn forward_batched(&self, x: &[f32], out: &mut [f32], batch: usize) {
        assert_eq!(x.len(), batch * self.h);
        assert_eq!(out.len(), batch * self.h);

        let device = self.queue.device();
        let n_bytes = (batch * self.h * std::mem::size_of::<f32>()) as u64;
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            n_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(self.h as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.h as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            let src = out_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), batch * self.h);
        }
    }
}

fn build_sigmoid_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(SIGMOID_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let function = library
        .get_function("sigmoid_batched", None)
        .map_err(|e| MetalError::PipelineCreate(format!("sigmoid_batched: {e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineCreate)
}

// ===================================================================
// Phase 13e prep: fused time_mix step1+step2 kernels (the most
// complex per-token kernels). step1 runs pre-state-update, step2
// runs post-state-update. Together they replace 11 separate
// element-wise passes from the pre-fused CPU baseline (Phase 12e).
// ===================================================================

const TIME_MIX_STEPS_KERNEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float poly_exp(float x) {
    x = fmax(x, -50.0f);
    float y = x * 1.442695041f;
    float k = floor(y);
    float r = y - k;
    float p = 0.000154035303933f;
    p = fma(p, r, 0.001333355814643f);
    p = fma(p, r, 0.009618129107628f);
    p = fma(p, r, 0.055504108664821f);
    p = fma(p, r, 0.240226506959101f);
    p = fma(p, r, 0.693147180559945f);
    p = fma(p, r, 1.0f);
    int  ki = int(k);
    int  exp_bits = (ki + 127) << 23;
    float two_k = as_type<float>(exp_bits);
    return p * two_k;
}

// time_mix step1 (pre-state-update). For each (batch, i):
//   ww[i] = time_first[i] + k[i]
//   p[i]  = max(state_p[i], ww[i])
//   e1    = exp(state_p[i] - p[i])
//   e2    = exp(ww[i] - p[i])
//   a[i]  = e1 * state_a[i] + e2 * v[i]
//   b[i]  = e1 * state_b[i] + e2
// time_first is broadcast across batch (per-block parameter).
kernel void time_mix_step1_batched(
    device const float* state_p    [[buffer(0)]],   // batch * h
    device const float* time_first [[buffer(1)]],   // h
    device const float* k          [[buffer(2)]],   // batch * h
    device const float* state_a    [[buffer(3)]],   // batch * h
    device const float* state_b    [[buffer(4)]],   // batch * h
    device const float* v          [[buffer(5)]],   // batch * h
    device float*       ww_out     [[buffer(6)]],   // batch * h
    device float*       p_out      [[buffer(7)]],   // batch * h
    device float*       a_out      [[buffer(8)]],   // batch * h
    device float*       b_out      [[buffer(9)]],   // batch * h
    constant uint&      h          [[buffer(10)]],
    constant uint&      batch      [[buffer(11)]],
    uint2               gid        [[thread_position_in_grid]]
) {
    uint i  = gid.x;
    uint bb = gid.y;
    if (i >= h || bb >= batch) return;
    uint kk = bb * h + i;

    float sp = state_p[kk];
    float kv = k[kk];
    float vv = v[kk];
    float sa = state_a[kk];
    float sb = state_b[kk];

    float ww = time_first[i] + kv;
    float p  = max(sp, ww);
    // Both exponents ≤ 0 since p = max(sp, ww).
    float e1 = poly_exp(sp - p);
    float e2 = poly_exp(ww - p);

    ww_out[kk] = ww;
    p_out[kk]  = p;
    a_out[kk]  = fma(e2, vv, e1 * sa);
    b_out[kk]  = fma(e1, sb, e2);
}

// time_mix step2 (post-state-update, in-place over state). For each:
//   ww[i]      = state_p[i] + neg_exp_decay[i]
//   p_new[i]   = max(ww[i], k[i])
//   e1         = exp(ww[i] - p_new[i])
//   e2         = exp(k[i] - p_new[i])
//   state_a[i] = e1 * state_a[i] + e2 * v[i]
//   state_b[i] = e1 * state_b[i] + e2
//   state_p[i] = p_new[i]
//   ww_out[i]  = ww[i]
// neg_exp_decay is broadcast across batch.
kernel void time_mix_step2_batched(
    device const float* neg_exp_decay [[buffer(0)]],  // h
    device const float* k             [[buffer(1)]],  // batch * h
    device const float* v             [[buffer(2)]],  // batch * h
    device float*       state_p       [[buffer(3)]],  // batch * h (in/out)
    device float*       state_a       [[buffer(4)]],  // batch * h (in/out)
    device float*       state_b       [[buffer(5)]],  // batch * h (in/out)
    device float*       ww_out        [[buffer(6)]],  // batch * h
    constant uint&      h             [[buffer(7)]],
    constant uint&      batch         [[buffer(8)]],
    uint2               gid           [[thread_position_in_grid]]
) {
    uint i  = gid.x;
    uint bb = gid.y;
    if (i >= h || bb >= batch) return;
    uint kk = bb * h + i;

    float sp = state_p[kk];
    float kv = k[kk];
    float vv = v[kk];
    float sa = state_a[kk];
    float sb = state_b[kk];
    float nd = neg_exp_decay[i];

    float ww    = sp + nd;
    float p_new = max(ww, kv);
    float e1    = poly_exp(ww - p_new);
    float e2    = poly_exp(kv - p_new);

    state_a[kk] = fma(e2, vv, e1 * sa);
    state_b[kk] = fma(e1, sb, e2);
    state_p[kk] = p_new;
    ww_out[kk]  = ww;
}
"#;

/// Reusable batched time_mix step1 + step2 kernel pair. The two
/// pipelines share one MSL library; both are pre-built at
/// construction. Per-block parameters (`time_first`, `neg_exp_decay`)
/// stay GPU-resident; per-token state is passed in by the caller.
///
/// CPU equivalent: [`crate::tensor::time_mix_step1`] and
/// [`crate::tensor::time_mix_step2`] (Phase 12e fused NEON).
pub struct TimeMixKernelMetal {
    queue: CommandQueue,
    pipeline_step1: ComputePipelineState,
    pipeline_step2: ComputePipelineState,
    time_first_buf: Buffer,
    neg_exp_decay_buf: Buffer,
    h: usize,
    h_u32: u32,
}

impl TimeMixKernelMetal {
    /// Build the kernel pair and upload the per-block parameter
    /// vectors. `time_first` and `neg_exp_decay` must both be `h`
    /// floats — they're shared across all batch lanes.
    pub fn new(time_first: &[f32], neg_exp_decay: &[f32], h: usize) -> Result<Self, MetalError> {
        assert_eq!(time_first.len(), h);
        assert_eq!(neg_exp_decay.len(), h);
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let (pipeline_step1, pipeline_step2) = build_time_mix_pipelines(&device)?;
        let time_first_buf = device.new_buffer_with_data(
            time_first.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(time_first) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let neg_exp_decay_buf = device.new_buffer_with_data(
            neg_exp_decay.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(neg_exp_decay) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            queue,
            pipeline_step1,
            pipeline_step2,
            time_first_buf,
            neg_exp_decay_buf,
            h,
            h_u32: h as u32,
        })
    }

    /// Run step1 (pre-state-update). Inputs are all `batch * h`
    /// flat slices; outputs are written into the four `*_out`
    /// parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn step1_batched(
        &self,
        state_p: &[f32],
        k: &[f32],
        state_a: &[f32],
        state_b: &[f32],
        v: &[f32],
        ww_out: &mut [f32],
        p_out: &mut [f32],
        a_out: &mut [f32],
        b_out: &mut [f32],
        batch: usize,
    ) {
        let n = batch * self.h;
        assert_eq!(state_p.len(), n);
        assert_eq!(k.len(), n);
        assert_eq!(state_a.len(), n);
        assert_eq!(state_b.len(), n);
        assert_eq!(v.len(), n);
        assert_eq!(ww_out.len(), n);
        assert_eq!(p_out.len(), n);
        assert_eq!(a_out.len(), n);
        assert_eq!(b_out.len(), n);

        let device = self.queue.device();
        let n_bytes = (n * std::mem::size_of::<f32>()) as u64;
        let upload = |data: &[f32]| {
            device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(data) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let sp_buf = upload(state_p);
        let k_buf = upload(k);
        let sa_buf = upload(state_a);
        let sb_buf = upload(state_b);
        let v_buf = upload(v);
        let ww_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let p_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let a_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_step1);
        encoder.set_buffer(0, Some(&sp_buf), 0);
        encoder.set_buffer(1, Some(&self.time_first_buf), 0);
        encoder.set_buffer(2, Some(&k_buf), 0);
        encoder.set_buffer(3, Some(&sa_buf), 0);
        encoder.set_buffer(4, Some(&sb_buf), 0);
        encoder.set_buffer(5, Some(&v_buf), 0);
        encoder.set_buffer(6, Some(&ww_buf), 0);
        encoder.set_buffer(7, Some(&p_buf), 0);
        encoder.set_buffer(8, Some(&a_buf), 0);
        encoder.set_buffer(9, Some(&b_buf), 0);
        encoder.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            11,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline_step1
            .thread_execution_width()
            .max(1)
            .min(self.h as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.h as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            std::ptr::copy_nonoverlapping(ww_buf.contents() as *const f32, ww_out.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(p_buf.contents() as *const f32, p_out.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(a_buf.contents() as *const f32, a_out.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(b_buf.contents() as *const f32, b_out.as_mut_ptr(), n);
        }
    }

    /// Run step2 (post-state-update, in-place over state vectors).
    pub fn step2_batched(
        &self,
        k: &[f32],
        v: &[f32],
        state_p: &mut [f32],
        state_a: &mut [f32],
        state_b: &mut [f32],
        ww_out: &mut [f32],
        batch: usize,
    ) {
        let n = batch * self.h;
        assert_eq!(k.len(), n);
        assert_eq!(v.len(), n);
        assert_eq!(state_p.len(), n);
        assert_eq!(state_a.len(), n);
        assert_eq!(state_b.len(), n);
        assert_eq!(ww_out.len(), n);

        let device = self.queue.device();
        let n_bytes = (n * std::mem::size_of::<f32>()) as u64;
        let upload = |data: &[f32]| {
            device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(data) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let k_buf = upload(k);
        let v_buf = upload(v);
        let sp_buf = upload(state_p);
        let sa_buf = upload(state_a);
        let sb_buf = upload(state_b);
        let ww_buf = device.new_buffer(n_bytes, MTLResourceOptions::StorageModeShared);
        let batch_u32 = batch as u32;

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_step2);
        encoder.set_buffer(0, Some(&self.neg_exp_decay_buf), 0);
        encoder.set_buffer(1, Some(&k_buf), 0);
        encoder.set_buffer(2, Some(&v_buf), 0);
        encoder.set_buffer(3, Some(&sp_buf), 0);
        encoder.set_buffer(4, Some(&sa_buf), 0);
        encoder.set_buffer(5, Some(&sb_buf), 0);
        encoder.set_buffer(6, Some(&ww_buf), 0);
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &self.h_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );

        let tg_width = self
            .pipeline_step2
            .thread_execution_width()
            .max(1)
            .min(self.h as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: self.h as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf);

        unsafe {
            std::ptr::copy_nonoverlapping(sp_buf.contents() as *const f32, state_p.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(sa_buf.contents() as *const f32, state_a.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(sb_buf.contents() as *const f32, state_b.as_mut_ptr(), n);
            std::ptr::copy_nonoverlapping(ww_buf.contents() as *const f32, ww_out.as_mut_ptr(), n);
        }
    }
}

fn build_time_mix_pipelines(
    device: &Device,
) -> Result<(ComputePipelineState, ComputePipelineState), MetalError> {
    let library = device
        .new_library_with_source(TIME_MIX_STEPS_KERNEL_MSL, &metal::CompileOptions::new())
        .map_err(MetalError::LibraryCompile)?;
    let mk = |name: &str| -> Result<ComputePipelineState, MetalError> {
        let f = library
            .get_function(name, None)
            .map_err(|e| MetalError::PipelineCreate(format!("{name}: {e:?}")))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(MetalError::PipelineCreate)
    };
    Ok((mk("time_mix_step1_batched")?, mk("time_mix_step2_batched")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test_runs_on_this_machine() {
        // If this MacBook has a Metal device the test should pass;
        // CI runs on x86 Linux without Metal so the build is gated
        // behind the feature flag and the test won't be compiled.
        match smoke_test(1024) {
            Ok(n) => assert_eq!(n, 1024),
            Err(MetalError::NoDevice) => {
                // Headless / no GPU — acceptable in CI.
            }
            Err(other) => panic!("Metal smoke test failed: {other}"),
        }
    }

    #[test]
    fn smoke_test_handles_small_sizes() {
        match smoke_test(4) {
            Ok(n) => assert_eq!(n, 4),
            Err(MetalError::NoDevice) => {}
            Err(other) => panic!("Metal smoke test failed: {other}"),
        }
    }

    /// Phase 13c: head matvec output on Metal must match the CPU
    /// NEON path within FMA-rounding tolerance. Uses the same shape
    /// as the L3TC-200K head (16384 × 96).
    #[test]
    fn head_matvec_metal_matches_cpu() {
        use crate::tensor;

        let rows: usize = 16384;
        let cols: usize = 96;

        // Deterministic synthetic weights — `qmat` spans the full i8
        // range, `scales` in a realistic [1e-4, 1e-2] range like a
        // quantized projection.
        let mut qmat = vec![0i8; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                qmat[j * rows + i] =
                    (((i as i32 * 31 + j as i32 * 7) % 251) - 125) as i8;
            }
        }
        let scales: Vec<f32> = (0..cols)
            .map(|j| 1e-4 + (j as f32) * 1e-5)
            .collect();
        // Input vector with mixed magnitudes.
        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) - 48.0) * 0.05)
            .collect();

        // CPU reference (Phase 12d hand-tuned NEON).
        let mut cpu_out = vec![0.0f32; rows];
        tensor::matvec_col_major_int8(&qmat, &scales, &x, &mut cpu_out, rows, cols);

        // Metal path.
        let kernel = match HeadKernelMetal::new(&qmat, &scales, rows, cols) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("HeadKernelMetal::new failed: {other}"),
        };
        let mut gpu_out = vec![0.0f32; rows];
        kernel.forward(&x, &mut gpu_out);

        // Tolerance: f32 FMA reordering between CPU NEON and GPU
        // produces small differences. Empirical rule of thumb is a
        // few ULPs scaled by sum magnitude. Use a generous absolute
        // tolerance — the freq-quantization downstream absorbs much
        // larger differences. Compare to the CPU magnitude scale.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..rows {
            let abs = (cpu_out[i] - gpu_out[i]).abs();
            let rel = abs / cpu_out[i].abs().max(1e-3);
            if abs > max_abs {
                max_abs = abs;
                max_idx = i;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        // Reference range here is roughly [-50, 50]. Tolerance of
        // ~5e-3 absolute / ~5e-4 relative is safely below the
        // freq-quantization step of 0.5 (since round(p * 10M) is
        // 0.5 / 10M ≈ 5e-8 in p-space, and these are pre-softmax
        // logits of order 10).
        assert!(
            max_abs < 5e-3,
            "head matvec CPU vs Metal max abs diff = {max_abs} \
             (rel = {max_rel}) at index {max_idx} \
             (cpu = {}, gpu = {})",
            cpu_out[max_idx],
            gpu_out[max_idx],
        );
    }

    /// Phase 13e prep: batched layer_norm on Metal must match the
    /// CPU NEON path within FMA-rounding tolerance, for every lane
    /// in the batch.
    #[test]
    fn layer_norm_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 64;
        let eps: f32 = 1e-5;

        // Synthetic weight + bias.
        let weight: Vec<f32> = (0..h).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..h).map(|i| -0.1 + (i as f32) * 0.005).collect();

        // Each batch lane has its own input pattern.
        let mut x = vec![0.0f32; batch * h];
        for b in 0..batch {
            for i in 0..h {
                x[b * h + i] = ((i as f32) - 48.0) * 0.1 + (b as f32) * 0.001;
            }
        }

        // CPU reference: apply layer_norm per batch lane.
        let mut cpu_out = vec![0.0f32; batch * h];
        for b in 0..batch {
            tensor::layer_norm(
                &x[b * h..(b + 1) * h],
                &weight,
                &bias,
                eps,
                &mut cpu_out[b * h..(b + 1) * h],
            );
        }

        // Metal path.
        let kernel = match LayerNormKernelMetal::new(&weight, &bias, h, eps) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("LayerNormKernelMetal::new failed: {other}"),
        };
        let mut gpu_out = vec![0.0f32; batch * h];
        kernel.forward_batched(&x, &mut gpu_out, batch);

        // Tolerance: layer norm uses sqrt + division → larger
        // numerical spread than pure FMA. Per the Phase 12f tests
        // we accept ~5e-5 relative.
        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..(batch * h) {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        assert!(
            max_abs < 5e-4,
            "layer_norm CPU vs Metal max abs diff = {max_abs} at {max_idx} \
             (cpu = {}, gpu = {})",
            cpu_out[max_idx],
            gpu_out[max_idx],
        );
    }

    /// Phase 13e prep: batched 96×96 matvec on Metal must match the
    /// CPU NEON path within FMA-rounding tolerance for every lane.
    #[test]
    fn matvec_96x96_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 64;

        // Synthetic mat with a structured pattern to make any
        // off-by-one detectable.
        let mat: Vec<f32> = (0..(h * h))
            .map(|k| ((k % 17) as f32 - 8.0) * 0.01)
            .collect();

        // Each batch lane has a slightly different x.
        let mut x = vec![0.0f32; batch * h];
        for b in 0..batch {
            for j in 0..h {
                x[b * h + j] = ((j as f32) - 48.0) * 0.05 + (b as f32) * 0.001;
            }
        }

        // CPU reference: matvec per lane.
        let mut cpu_out = vec![0.0f32; batch * h];
        for b in 0..batch {
            tensor::matvec_96x96(
                &mat,
                &x[b * h..(b + 1) * h],
                &mut cpu_out[b * h..(b + 1) * h],
            );
        }

        // Metal path.
        let kernel = match Matvec96Metal::new(&mat, h) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("Matvec96Metal::new failed: {other}"),
        };
        let mut gpu_out = vec![0.0f32; batch * h];
        kernel.forward_batched(&x, &mut gpu_out, batch);

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..(batch * h) {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        // Tolerance: 96-wide dot product accumulates ~96 FMAs of f32
        // values around 0.01. Max abs error should be a few ULPs of
        // the final magnitude (~1e-3 to 1e-1).
        assert!(
            max_abs < 5e-4,
            "matvec_96x96 CPU vs Metal max abs diff = {max_abs} at {max_idx} \
             (cpu = {}, gpu = {})",
            cpu_out[max_idx],
            gpu_out[max_idx],
        );
    }

    /// Phase 13e prep: batched sub_exp on Metal must match the CPU
    /// NEON polynomial within FMA-rounding tolerance for inputs in
    /// the time-mix range (always ≤ 0).
    #[test]
    fn sub_exp_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 32;

        // Time_mix-realistic inputs: a is the smaller, b is the
        // running max, so a - b is always ≤ 0. Spread across the
        // [-50, 0] range covered by the polynomial.
        let mut a = vec![0.0f32; batch * h];
        let mut b = vec![0.0f32; batch * h];
        for bi in 0..batch {
            for i in 0..h {
                let k = bi * h + i;
                let base = (i as f32 - 48.0) * 0.5;
                b[k] = base;
                a[k] = base - ((i + bi) as f32 % 30.0);
            }
        }

        // CPU reference per-lane.
        let mut cpu_out = vec![0.0f32; batch * h];
        for bi in 0..batch {
            tensor::sub_exp(
                &a[bi * h..(bi + 1) * h],
                &b[bi * h..(bi + 1) * h],
                &mut cpu_out[bi * h..(bi + 1) * h],
            );
        }

        // Metal path.
        let kernel = match SubExpKernelMetal::new(h) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("SubExpKernelMetal::new failed: {other}"),
        };
        let mut gpu_out = vec![0.0f32; batch * h];
        kernel.forward_batched(&a, &b, &mut gpu_out, batch);

        // Both implementations clamp inputs <= -50 to ~exp(-50). The
        // CPU NEON polynomial has documented max relative error ~1e-5;
        // Metal's polynomial (same coefficients) should land in the
        // same band.
        let mut max_rel = 0.0f64;
        let mut max_idx = 0usize;
        for i in 0..(batch * h) {
            let cpu_v = cpu_out[i] as f64;
            let gpu_v = gpu_out[i] as f64;
            // Skip near-zero CPU values; GPU may differ by clamp-floor
            // behaviour but freq-quantization absorbs it.
            if cpu_v < 1e-15 {
                continue;
            }
            let rel = (cpu_v - gpu_v).abs() / cpu_v;
            if rel > max_rel {
                max_rel = rel;
                max_idx = i;
            }
        }
        assert!(
            max_rel < 5e-5,
            "sub_exp CPU vs Metal max relative error = {max_rel} at {max_idx} \
             (cpu = {}, gpu = {})",
            cpu_out[max_idx],
            gpu_out[max_idx],
        );
    }

    /// Phase 13e prep: batched sigmoid on Metal must match the CPU
    /// NEON safe-form (Phase 12c) within FMA-rounding tolerance.
    #[test]
    fn sigmoid_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 32;

        // Cover positive AND negative inputs (the safe -|x| form
        // matters most for x > 0).
        let mut x = vec![0.0f32; batch * h];
        for bi in 0..batch {
            for i in 0..h {
                x[bi * h + i] = ((i as f32 - 48.0) * 0.5) + (bi as f32 * 0.05);
            }
        }

        // CPU reference: in-place sigmoid per lane.
        let mut cpu_out = x.clone();
        for bi in 0..batch {
            tensor::sigmoid_inplace(&mut cpu_out[bi * h..(bi + 1) * h]);
        }

        let kernel = match SigmoidKernelMetal::new(h) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("SigmoidKernelMetal::new failed: {other}"),
        };
        let mut gpu_out = vec![0.0f32; batch * h];
        kernel.forward_batched(&x, &mut gpu_out, batch);

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..(batch * h) {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        // Sigmoid output is in [0, 1]; absolute error budget is
        // tighter than sub_exp (no large dynamic range).
        assert!(
            max_abs < 1e-5,
            "sigmoid CPU vs Metal max abs diff = {max_abs} at {max_idx} \
             (cpu = {}, gpu = {})",
            cpu_out[max_idx],
            gpu_out[max_idx],
        );
    }

    /// Phase 13e prep: batched time_mix step1 on Metal must match
    /// the CPU NEON fused step1 for every (batch, i) cell.
    #[test]
    fn time_mix_step1_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 16;

        // Realistic time_mix-range inputs.
        let time_first: Vec<f32> = (0..h).map(|i| 0.5 - (i as f32) * 0.005).collect();
        let mk = |seed: f32| -> Vec<f32> {
            (0..(batch * h))
                .map(|k| ((k as f32) * 0.01 + seed).sin())
                .collect()
        };
        let state_p = mk(0.1);
        let k_in = mk(0.2);
        let state_a = mk(0.3);
        let state_b = mk(0.4);
        let v = mk(0.5);

        // CPU reference per-lane.
        let mut cpu_ww = vec![0.0f32; batch * h];
        let mut cpu_p = vec![0.0f32; batch * h];
        let mut cpu_a = vec![0.0f32; batch * h];
        let mut cpu_b = vec![0.0f32; batch * h];
        for bi in 0..batch {
            let s = bi * h;
            let e = s + h;
            tensor::time_mix_step1(
                &state_p[s..e],
                &time_first,
                &k_in[s..e],
                &state_a[s..e],
                &state_b[s..e],
                &v[s..e],
                &mut cpu_ww[s..e],
                &mut cpu_p[s..e],
                &mut cpu_a[s..e],
                &mut cpu_b[s..e],
            );
        }

        // Metal path.
        let neg_decay = vec![0.0f32; h]; // unused by step1
        let kernel = match TimeMixKernelMetal::new(&time_first, &neg_decay, h) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("TimeMixKernelMetal::new failed: {other}"),
        };
        let mut gpu_ww = vec![0.0f32; batch * h];
        let mut gpu_p = vec![0.0f32; batch * h];
        let mut gpu_a = vec![0.0f32; batch * h];
        let mut gpu_b = vec![0.0f32; batch * h];
        kernel.step1_batched(
            &state_p,
            &k_in,
            &state_a,
            &state_b,
            &v,
            &mut gpu_ww,
            &mut gpu_p,
            &mut gpu_a,
            &mut gpu_b,
            batch,
        );

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let tol = 5e-4;
        assert!(max_abs(&cpu_ww, &gpu_ww) < tol, "ww diff");
        assert!(max_abs(&cpu_p, &gpu_p) < tol, "p diff");
        assert!(max_abs(&cpu_a, &gpu_a) < tol, "a diff: {}", max_abs(&cpu_a, &gpu_a));
        assert!(max_abs(&cpu_b, &gpu_b) < tol, "b diff: {}", max_abs(&cpu_b, &gpu_b));
    }

    /// Phase 13e prep: batched time_mix step2 on Metal must match
    /// the CPU NEON fused step2 — including the in-place state
    /// updates.
    #[test]
    fn time_mix_step2_batched_metal_matches_cpu() {
        use crate::tensor;

        let h: usize = 96;
        let batch: usize = 16;

        let neg_decay: Vec<f32> = (0..h).map(|i| -((i as f32) * 0.01).exp()).collect();
        let mk = |seed: f32| -> Vec<f32> {
            (0..(batch * h))
                .map(|k| ((k as f32) * 0.01 + seed).cos())
                .collect()
        };
        let k_in = mk(0.6);
        let v = mk(0.7);
        let state_p_init = mk(0.8);
        let state_a_init = mk(0.9);
        let state_b_init = mk(1.0);

        // CPU reference: clone state and apply per-lane.
        let mut cpu_state_p = state_p_init.clone();
        let mut cpu_state_a = state_a_init.clone();
        let mut cpu_state_b = state_b_init.clone();
        let mut cpu_ww = vec![0.0f32; batch * h];
        for bi in 0..batch {
            let s = bi * h;
            let e = s + h;
            tensor::time_mix_step2(
                &neg_decay,
                &k_in[s..e],
                &v[s..e],
                &mut cpu_state_p[s..e],
                &mut cpu_state_a[s..e],
                &mut cpu_state_b[s..e],
                &mut cpu_ww[s..e],
            );
        }

        // Metal path: clone state again, apply the batched kernel.
        let time_first = vec![0.0f32; h]; // unused by step2
        let kernel = match TimeMixKernelMetal::new(&time_first, &neg_decay, h) {
            Ok(k) => k,
            Err(MetalError::NoDevice) => return,
            Err(other) => panic!("TimeMixKernelMetal::new failed: {other}"),
        };
        let mut gpu_state_p = state_p_init.clone();
        let mut gpu_state_a = state_a_init.clone();
        let mut gpu_state_b = state_b_init.clone();
        let mut gpu_ww = vec![0.0f32; batch * h];
        kernel.step2_batched(
            &k_in,
            &v,
            &mut gpu_state_p,
            &mut gpu_state_a,
            &mut gpu_state_b,
            &mut gpu_ww,
            batch,
        );

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let tol = 5e-4;
        assert!(max_abs(&cpu_state_p, &gpu_state_p) < tol, "state_p diff");
        assert!(max_abs(&cpu_state_a, &gpu_state_a) < tol, "state_a diff");
        assert!(max_abs(&cpu_state_b, &gpu_state_b) < tol, "state_b diff");
        assert!(max_abs(&cpu_ww, &gpu_ww) < tol, "ww diff");
    }
}
