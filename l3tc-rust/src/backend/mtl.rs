//! Metal backend — Phase 13.
//!
//! Currently contains only the smoke test. Phase 13c will add the
//! head-matvec kernel; Phase 13d the rest of the forward pass.

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
}
