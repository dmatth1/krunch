//! CUDA backend for the RWKV-4-Pile-169M inference path (Tier-2).
//!
//! Gated behind the `cuda` cargo feature. Links the `cudarc` crate
//! (pure-Rust CUDA Driver API wrapper, no libtorch/libcudnn). The
//! fused WKV kernel source lives in `src/cuda/wkv.cu` and is compiled
//! to a `.ptx` blob at build time by `build.rs`. At runtime we load
//! the PTX via cudarc's NVRTC, resolve the `wkv_forward` symbol, and
//! launch it per-layer on each forward pass.
//!
//! Design notes:
//!
//! - All kernel launches go through a single `CudaContext` owning one
//!   `CudaDevice`. Multi-GPU is out of scope for Tier-2.
//! - Tensors are stored as `DevicePtr<half>` (fp16) — matches what
//!   ts_zip and the L3TC paper use and what our Spike 6 measurements
//!   validated on A10G.
//! - State buffers (`aa`, `bb`, `pp`) are allocated once and reused
//!   across chunks; the cost is O(hidden) per layer, not O(seq_len).

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::error::{Error, Result};

/// Embedded PTX produced by `build.rs` from `src/cuda/wkv.cu`.
///
/// The include path is relative to `OUT_DIR` — `build.rs` writes the
/// compiled PTX there and sets `cargo:rustc-env=WKV_PTX_PATH=...` so
/// we can pick it up here.
const WKV_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wkv.ptx"));

/// Handle to an initialized CUDA context + loaded WKV kernel module.
///
/// Cheap to clone (wraps `Arc`s). One instance per inference session.
pub struct CudaContextHandle {
    context: Arc<CudaContext>,
    wkv_module: Arc<CudaModule>,
}

impl CudaContextHandle {
    /// Initialize CUDA on the first visible device, load the WKV PTX.
    ///
    /// Returns `Error::Backend { .. }` if:
    /// - No CUDA device is visible
    /// - PTX load fails (kernel version mismatch, bad ABI, etc.)
    pub fn new() -> Result<Self> {
        let context = CudaContext::new(0).map_err(|e| Error::Backend {
            backend: "cuda",
            message: format!("CudaContext::new(0): {e}"),
        })?;

        let wkv_module = context
            .load_module(WKV_PTX.into())
            .map_err(|e| Error::Backend {
                backend: "cuda",
                message: format!("load wkv.ptx: {e}"),
            })?;

        Ok(Self { context, wkv_module })
    }

    /// Borrow the underlying CUDA context. Most callers shouldn't need
    /// this — prefer `launch_wkv_forward` below — but it's useful for
    /// direct memory management in tests.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Launch the fused WKV forward-pass kernel.
    ///
    /// # Parameters
    /// - `seq_len`: T
    /// - `channels`: C (== hidden_size)
    /// - `batch`: B (usually 1 for our per-chunk AC pattern; batching
    ///   across chunks is the stretch lever for multi-stream decode)
    /// - `w, u`: per-channel time-decay parameters, shape `(C,)` each
    /// - `k, v`: per-(B, T, C) projections
    /// - `y_out`: output buffer, shape `(B, T, C)`, pre-allocated by caller
    ///
    /// All inputs and outputs are device-resident fp16 (except `w`,
    /// `u` which the kernel promotes to fp32 internally — matches
    /// BlinkDL's impl).
    ///
    /// Validation of `B * C % min(C, 1024) == 0` and `T <= T_MAX` is
    /// the caller's job; kernel will misbehave or fault otherwise.
    pub fn launch_wkv_forward(
        &self,
        seq_len: i32,
        channels: i32,
        batch: i32,
        w: &cudarc::driver::CudaSlice<f32>,
        u: &cudarc::driver::CudaSlice<f32>,
        k: &cudarc::driver::CudaSlice<f32>,
        v: &cudarc::driver::CudaSlice<f32>,
        y_out: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<()> {
        let stream = self.context.default_stream();
        let kernel = self
            .wkv_module
            .load_function("wkv_forward")
            .map_err(|e| Error::Backend {
                backend: "cuda",
                message: format!("load wkv_forward symbol: {e}"),
            })?;

        // Grid/block matches BlinkDL's impl: (B, C/min(C,1024)) grid,
        // min(C,1024) block. Each block processes one (batch, channel)
        // stream across all T positions.
        let threads_per_block = channels.min(1024);
        let blocks_per_grid = (batch * channels) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_grid as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&batch);
        launch.arg(&seq_len);
        launch.arg(&channels);
        launch.arg(w);
        launch.arg(u);
        launch.arg(k);
        launch.arg(v);
        launch.arg(y_out);
        unsafe {
            launch.launch(cfg).map_err(|e| Error::Backend {
                backend: "cuda",
                message: format!("wkv_forward launch: {e}"),
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: verify we can initialize CUDA and load the PTX.
    /// Skipped on CI machines without a CUDA device.
    #[test]
    #[ignore = "requires CUDA device"]
    fn init_and_load_ptx() {
        let h = CudaContextHandle::new().expect("CUDA init");
        // Sanity: PTX is non-empty
        assert!(!WKV_PTX.is_empty());
        let _ = h.context();
    }
}
