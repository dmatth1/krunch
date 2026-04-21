//! Inference backend dispatch.
//!
//! Phase 13 introduces an opt-in GPU backend (Metal on Apple,
//! CUDA later) that runs the same RWKV forward pass with the
//! same file format and bit-equivalent cum_freqs as the CPU path.
//!
//! See [`docs/phases/PHASE_13.md`](../../../docs/phases/PHASE_13.md)
//! for the full plan and guardrails.
//!
//! # Design
//!
//! The hot path is `Session::forward(token) -> &[f32]`. Every backend
//! must produce logits whose freq-quantization (after
//! `round(p * PYTHON_FREQ_TOTAL); max(1)` in `codec::logits_to_cum_freqs_scratch`)
//! matches the CPU NEON path. Phase 4a + Phase 12 already validated
//! that the polynomial NEON exp + INT8 head matvec land within
//! that tolerance — any new backend just has to clear the same bar.
//!
//! Files compressed with one backend decompress with any other.
//! The file format is unchanged.
//!
//! # Backends
//!
//! - [`Backend::Cpu`] — always available, the Phase 12 NEON code path
//! - [`Backend::Metal`] — Apple Silicon GPU (gated behind the `metal`
//!   cargo feature)
//! - `Backend::Cuda` — NVIDIA GPU (future, gated behind `cuda`)
//!
//! # Use
//!
//! Pick a backend explicitly, or use [`Backend::auto`] which selects
//! the best one available given the input size:
//!
//! ```no_run
//! use l3tc::backend::Backend;
//!
//! let b = Backend::auto(2_000_000); // 2 MB input
//! // → Backend::Metal if compiled with --features=metal AND device
//! //   present AND input ≥ GPU_THRESHOLD; otherwise Backend::Cpu.
//! ```

#[cfg(feature = "metal")]
pub mod mtl;

#[cfg(feature = "metal")]
pub mod batched;

/// The set of inference backends this build was compiled to support.
///
/// `Cpu` is always present. `Metal` is present iff this crate was
/// built with `--features=metal`. Future variants (`Cuda`) follow
/// the same pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Phase 12 NEON / scalar CPU path. Always available.
    #[default]
    Cpu,

    /// Apple Metal GPU. Requires the `metal` cargo feature **and**
    /// a Metal-capable device at runtime.
    #[cfg(feature = "metal")]
    Metal,
}

/// Below this many input bytes, GPU init overhead exceeds the
/// throughput win. Auto-routing falls back to CPU on smaller inputs.
///
/// Empirically chosen — measured GPU init (Metal device + library
/// + first kernel dispatch) is ~50-100 ms on M-series.
///
/// To break even at the projected 5 MB/s GPU vs 0.17 MB/s CPU
/// throughput, the input has to be large enough that the per-input
/// GPU init cost amortizes. ~256 KB is the conservative floor.
pub const GPU_AUTO_THRESHOLD_BYTES: usize = 256 * 1024;

impl Backend {
    /// Pick the best backend available for an input of this size.
    ///
    /// - Below [`GPU_AUTO_THRESHOLD_BYTES`]: always CPU.
    /// - Above the threshold: GPU if compiled in **and** the device
    ///   is reachable at runtime; CPU otherwise.
    pub fn auto(input_bytes: usize) -> Self {
        if input_bytes < GPU_AUTO_THRESHOLD_BYTES {
            return Backend::Cpu;
        }

        #[cfg(feature = "metal")]
        {
            if metal_available() {
                return Backend::Metal;
            }
        }

        Backend::Cpu
    }

    /// Parse a backend selector string from the CLI. Returns `None`
    /// for unknown values; the caller should error out.
    ///
    /// Prefer this in internal call sites — it returns `Option` and
    /// avoids the friction of `FromStr::Err`. The `FromStr` impl
    /// below wraps this for external `str::parse::<Backend>()` users.
    pub fn parse_name(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Backend::Cpu),
            #[cfg(feature = "metal")]
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    /// Human-readable name for diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            #[cfg(feature = "metal")]
            Backend::Metal => "metal",
        }
    }
}

/// Error returned by `<Backend as FromStr>::from_str` for unrecognized
/// backend selector strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBackendError(pub String);

impl std::fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown backend: {:?}", self.0)
    }
}

impl std::error::Error for ParseBackendError {}

impl std::str::FromStr for Backend {
    type Err = ParseBackendError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_name(s).ok_or_else(|| ParseBackendError(s.to_string()))
    }
}

/// Probe for an available Metal device. Returns false if no GPU is
/// reachable (headless server, simulator, etc.).
#[cfg(feature = "metal")]
fn metal_available() -> bool {
    ::metal::Device::system_default().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_below_threshold_is_cpu() {
        assert_eq!(Backend::auto(1024), Backend::Cpu);
        assert_eq!(Backend::auto(GPU_AUTO_THRESHOLD_BYTES - 1), Backend::Cpu);
    }

    #[test]
    fn cpu_always_parseable() {
        assert_eq!(Backend::parse_name("cpu"), Some(Backend::Cpu));
        assert_eq!(Backend::parse_name("garbage"), None);
    }

    #[test]
    fn cpu_default() {
        assert_eq!(Backend::default(), Backend::Cpu);
    }
}
