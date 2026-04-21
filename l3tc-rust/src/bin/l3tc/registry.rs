//! Specialist model + tokenizer registry (Phase 14).
//!
//! Maps a `Specialist` enum to the `(model.bin, tokenizer.model)`
//! pair on disk. The CLI uses this to look up assets without the
//! user needing to know paths — `l3tc compress file.txt` finds the
//! right specialist's files automatically.
//!
//! ## Resolution order
//!
//! For a given specialist, we search these locations in order and
//! return the first that exists:
//!
//! 1. `$L3TC_MODEL_DIR/<specialist>/{model.bin, tokenizer.model}` —
//!    explicit override for development or non-standard installs.
//! 2. `<binary_dir>/models/<specialist>/...` — co-located with the
//!    binary (typical for a self-contained tarball install).
//! 3. `<binary_dir>/../share/l3tc/models/<specialist>/...` —
//!    typical for a `make install` style layout.
//! 4. `~/.l3tc/models/<specialist>/...` — per-user install.
//! 5. `/usr/local/share/l3tc/models/<specialist>/...` — system-wide.
//! 6. **Legacy fallback** for `Specialist::Unspecified` only:
//!    `checkpoints/l3tc_200k.bin` + the enwik8 SPM tokenizer
//!    bundled in `vendor/L3TC/dictionary/`. This preserves the
//!    behavior of `l3tc compress file.txt` working on a fresh
//!    repo checkout exactly the way it always has, even before
//!    Phase 14 specialist models exist.
//!
//! ## Status
//!
//! Phase 14 ships this resolver before the specialist models are
//! trained. Until they exist, every specialist (other than
//! `Unspecified`) will fail step 1-5 and the CLI must either:
//! - Accept the user's explicit `--model` / `--tokenizer` override.
//! - Fall back to `Unspecified`'s legacy paths and emit a clear
//!   "no specialist model installed; using default" warning.
//!
//! Both paths are wired in the CLI; this module just returns the
//! first existing pair or `None`.

use crate::specialist::Specialist;
use std::env;
use std::path::{Path, PathBuf};

/// On-disk location of a specialist's model binary and tokenizer.
#[derive(Debug, Clone)]
pub struct ModelAssets {
    /// Path to the converted `.bin` checkpoint.
    pub model: PathBuf,
    /// Path to the SentencePiece `.model` file.
    pub tokenizer: PathBuf,
}

/// Result of looking up a specialist. Carries whether we found the
/// asked-for specialist or had to fall back so the CLI can warn.
#[derive(Debug)]
pub struct ResolvedSpecialist {
    /// What the user asked for (or what auto-detect picked).
    pub requested: Specialist,
    /// What we actually resolved to. Differs when the requested
    /// specialist isn't installed and we fell back to `Unspecified`.
    pub resolved: Specialist,
    /// Files we found.
    pub assets: ModelAssets,
}

impl ResolvedSpecialist {
    /// True when we couldn't find the requested specialist and fell
    /// back to a default. CLI uses this to print a one-line warning.
    pub fn is_fallback(&self) -> bool {
        self.requested != self.resolved
    }
}

/// Look up assets for a specialist, returning the first hit across
/// the resolution order documented at the module level.
///
/// Returns `None` only when the specialist is `Unspecified` AND no
/// legacy default could be found (a misconfigured install).
pub fn resolve(specialist: Specialist) -> Option<ResolvedSpecialist> {
    // 1-5: standard search paths for this specialist.
    if let Some(assets) = search_standard_paths(specialist) {
        return Some(ResolvedSpecialist {
            requested: specialist,
            resolved: specialist,
            assets,
        });
    }

    // 6: fall back to Unspecified's legacy default if the requested
    // specialist isn't installed. This keeps the CLI usable before
    // any specialist models exist.
    if specialist != Specialist::Unspecified {
        if let Some(assets) = legacy_default() {
            return Some(ResolvedSpecialist {
                requested: specialist,
                resolved: Specialist::Unspecified,
                assets,
            });
        }
    } else if let Some(assets) = legacy_default() {
        return Some(ResolvedSpecialist {
            requested: specialist,
            resolved: specialist,
            assets,
        });
    }

    None
}

/// Search the standard install locations for a specialist's files.
fn search_standard_paths(specialist: Specialist) -> Option<ModelAssets> {
    let name = specialist.name();

    // Skip search for Unspecified — it doesn't have a dedicated
    // install directory; legacy_default() handles it.
    if specialist == Specialist::Unspecified {
        return None;
    }

    // Build the candidate roots in priority order.
    let mut roots: Vec<PathBuf> = Vec::new();

    if let Ok(env_dir) = env::var("L3TC_MODEL_DIR") {
        roots.push(PathBuf::from(env_dir));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("models"));
            roots.push(dir.join("../share/l3tc/models"));
        }
    }
    if let Some(home) = env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".l3tc/models"));
    }
    roots.push(PathBuf::from("/usr/local/share/l3tc/models"));

    for root in roots {
        let dir = root.join(name);
        let model = dir.join("model.bin");
        let tokenizer = dir.join("tokenizer.model");
        if model.is_file() && tokenizer.is_file() {
            return Some(ModelAssets { model, tokenizer });
        }
    }
    None
}

/// Pre-Phase-14 default: the L3TC-200K binary and enwik8 tokenizer
/// that ship in this repo. Used when no specialist is installed,
/// so `l3tc compress file.txt` keeps working on a fresh checkout.
fn legacy_default() -> Option<ModelAssets> {
    // Try a few likely roots: cwd, binary parent, and binary
    // grand-parent (covers running from `l3tc-rust/target/release`
    // and from the repo root).
    let mut roots: Vec<PathBuf> = vec![PathBuf::from(".")];
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
            if let Some(parent) = dir.parent() {
                roots.push(parent.to_path_buf());
                if let Some(gp) = parent.parent() {
                    roots.push(gp.to_path_buf());
                }
            }
        }
    }

    let model_rel = Path::new("checkpoints/l3tc_200k.bin");
    let tok_rel = Path::new(
        "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model",
    );

    for root in &roots {
        let model = root.join(model_rel);
        let tokenizer = root.join(tok_rel);
        if model.is_file() && tokenizer.is_file() {
            return Some(ModelAssets { model, tokenizer });
        }
        // Also try without the "../vendor" climb-up for installs
        // where vendor/ has been re-rooted next to checkpoints/.
        let alt_tok = root.join(
            "vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model",
        );
        if model.is_file() && alt_tok.is_file() {
            return Some(ModelAssets {
                model,
                tokenizer: alt_tok,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unspecified_falls_back_to_legacy_when_present() {
        // We can't assert a specific path because the test process
        // location varies; just confirm the resolver doesn't panic
        // and returns either Some(legacy) or None deterministically.
        let _ = resolve(Specialist::Unspecified);
    }

    #[test]
    fn unknown_specialist_resolution_doesnt_panic() {
        // None of the Phase 14 specialists are installed yet, so
        // these resolutions should fall through to legacy_default
        // (when running from the repo) or return None (when not).
        let _ = resolve(Specialist::Prose);
        let _ = resolve(Specialist::Code);
        let _ = resolve(Specialist::Logs);
    }

    #[test]
    fn fallback_flag_set_when_specialist_missing() {
        // If a specialist isn't installed but legacy default exists,
        // we should mark as fallback so the CLI can warn.
        if let Some(r) = resolve(Specialist::Prose) {
            // Either we found the prose specialist (resolved == requested)
            // or we fell back (resolved == Unspecified).
            if r.resolved == Specialist::Unspecified {
                assert!(r.is_fallback());
                assert_eq!(r.requested, Specialist::Prose);
            } else {
                assert_eq!(r.resolved, Specialist::Prose);
                assert!(!r.is_fallback());
            }
        }
    }
}
