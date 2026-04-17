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
}
