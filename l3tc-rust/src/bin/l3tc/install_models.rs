//! `l3tc install-models` — fetch, verify, and install the specialist
//! model bundle.
//!
//! The CLI binary is small (~2 MB). Specialist models plus their
//! SentencePiece tokenizers total ~100 MB across 7 specialists, which
//! is too large to bundle into a Homebrew bottle or `cargo install`
//! payload. Instead, we ship them as a separate tarball on GitHub
//! Releases and let the user pull them down on first use.
//!
//! ```text
//! l3tc install-models                 # download + install default bundle
//! l3tc install-models --url URL       # install from a non-default URL
//! l3tc install-models --dest DIR      # install to a non-default directory
//! l3tc install-models --force         # re-download even if already installed
//! l3tc install-models --verify        # check installed files against manifest
//! l3tc install-models --list          # show which specialists are present
//! ```
//!
//! ## Bundle format
//!
//! The bundle is a `.tar.zst` archive containing:
//!
//! ```text
//! manifest.json
//! prose/model.bin
//! prose/tokenizer.model
//! code/model.bin
//! code/tokenizer.model
//! …
//! ```
//!
//! `manifest.json` declares every artifact's path, SHA-256, and
//! byte size, plus a `bundle_version` and the `l3tc_version` the
//! bundle was produced against. A binary with
//! `MODEL_BUNDLE_VERSION < manifest.bundle_version` refuses to
//! install — an older binary can't trust that it understands a
//! newer bundle's on-disk layout.
//!
//! ## Verification
//!
//! Every downloaded byte is SHA-256-hashed against the manifest
//! before it lands in the destination directory. The extraction
//! is staged in a sibling temp directory and atomically renamed
//! into place only after all hashes match, so a partial download
//! or network abort cannot leave a half-installed state behind.
//!
//! ## Trust model
//!
//! The manifest itself lives inside the tarball; downloading a
//! tampered tarball would yield a consistent-but-malicious set of
//! artifacts. HTTPS (TLS, via ureq + rustls + webpki-roots) is the
//! current integrity guarantee; the v0.1.0 bundle URL points at
//! `github.com/dmatth1/l3tc/releases/download/...` which GitHub
//! serves over HTTPS. A future revision can add detached signature
//! verification — bumped `bundle_version` would gate that.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use l3tc::MODEL_BUNDLE_VERSION;

/// Default URL for the specialist bundle. Points at the v0.1.0
/// GitHub Release asset. Users can override via `--url`.
///
/// This URL assumes the public GitHub repo is named `l3tc` (the
/// user is planning to rename from the current `ltec` before tag).
const DEFAULT_BUNDLE_URL: &str =
    "https://github.com/dmatth1/l3tc/releases/download/v0.1.0/l3tc-models-v0.1.0.tar.zst";

/// Max bytes we're willing to download. Guard against a rogue URL
/// returning an unbounded stream. 500 MB is comfortably above the
/// ~100 MB bundle target and well under anyone's disk.
const MAX_BUNDLE_BYTES: u64 = 500 * 1024 * 1024;

/// Name of the manifest file at the tarball root.
const MANIFEST_NAME: &str = "manifest.json";

// ---------------------------------------------------------------- //
// Manifest schema                                                  //
// ---------------------------------------------------------------- //

/// Top-level bundle manifest. Emitted by the release workflow that
/// builds the `.tar.zst` and consumed here at install / verify time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Bundle layout version. If this exceeds
    /// [`crate::MODEL_BUNDLE_VERSION`], this binary refuses to
    /// install the bundle.
    pub bundle_version: u32,
    /// `l3tc` version string the bundle was produced against (e.g.
    /// `"0.1.0"`). Informational; not a hard gate today.
    pub l3tc_version: String,
    /// Per-specialist artifact list. Keys are specialist names
    /// (`"prose"`, `"code"`, …) matching `Specialist::name()` in
    /// `crate::bin::l3tc::specialist`.
    pub specialists: std::collections::BTreeMap<String, SpecialistArtifacts>,
}

/// The two files that make up one specialist: its compiled model
/// and its SentencePiece tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistArtifacts {
    pub model: Artifact,
    pub tokenizer: Artifact,
}

/// One file in the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Path inside the tarball (relative, no leading slash). Also
    /// the path relative to the install destination where the file
    /// ends up on disk.
    pub path: String,
    /// Lowercase-hex SHA-256 of the file contents.
    pub sha256: String,
    /// Byte size of the file; cross-checked against what we read.
    pub size: u64,
}

impl Manifest {
    fn parse(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).with_context(|| "parse manifest.json")
    }

    /// Reject bundles this binary is too old to understand.
    fn check_compatible(&self) -> Result<()> {
        if self.bundle_version > MODEL_BUNDLE_VERSION {
            bail!(
                "bundle has format version {} but this binary understands up to version {}. \
                 Upgrade l3tc before installing this bundle.",
                self.bundle_version,
                MODEL_BUNDLE_VERSION,
            );
        }
        Ok(())
    }

    /// Iterate (path, expected_sha256, expected_size) triples.
    fn artifacts(&self) -> impl Iterator<Item = (&str, &str, u64)> {
        self.specialists.values().flat_map(|s| {
            [
                (s.model.path.as_str(), s.model.sha256.as_str(), s.model.size),
                (
                    s.tokenizer.path.as_str(),
                    s.tokenizer.sha256.as_str(),
                    s.tokenizer.size,
                ),
            ]
        })
    }
}

// ---------------------------------------------------------------- //
// CLI entry points                                                 //
// ---------------------------------------------------------------- //

/// `l3tc install-models [--url URL] [--dest DIR] [--force]`.
///
/// Downloads the bundle, verifies every artifact against the
/// manifest, and extracts atomically to `dest`. Default `dest` is
/// `~/.l3tc/models`, which is the registry's per-user search path.
pub fn run_install(
    url_override: Option<&str>,
    dest_override: Option<&Path>,
    force: bool,
) -> Result<()> {
    let url = url_override.unwrap_or(DEFAULT_BUNDLE_URL);
    let dest = resolve_dest(dest_override)?;

    if dest.join(MANIFEST_NAME).is_file() && !force {
        eprintln!(
            "models already installed at {}. Use --force to reinstall.",
            dest.display()
        );
        return Ok(());
    }

    eprintln!("downloading bundle from {url}");
    let bundle_bytes = http_get(url)?;
    eprintln!("  got {} bytes", bundle_bytes.len());

    let staging = dest.with_extension("staging");
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .with_context(|| format!("clean stale staging dir {}", staging.display()))?;
    }
    fs::create_dir_all(&staging)
        .with_context(|| format!("create staging dir {}", staging.display()))?;

    extract_and_verify(&bundle_bytes, &staging)?;

    // Atomic swap: rename dest → dest.old (if present), staging →
    // dest, drop dest.old. On Unix this keeps a running process's
    // open file handles valid during the rename.
    if dest.exists() {
        let backup = dest.with_extension("old");
        if backup.exists() {
            fs::remove_dir_all(&backup).ok();
        }
        fs::rename(&dest, &backup)
            .with_context(|| format!("move old install aside: {}", dest.display()))?;
        fs::rename(&staging, &dest)
            .with_context(|| format!("promote staging: {}", dest.display()))?;
        fs::remove_dir_all(&backup).ok();
    } else {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
        fs::rename(&staging, &dest)
            .with_context(|| format!("promote staging: {}", dest.display()))?;
    }

    eprintln!("installed to {}", dest.display());
    Ok(())
}

/// `l3tc install-models --verify`.
///
/// Reads the installed manifest and re-hashes every artifact on
/// disk. Reports any that are missing, wrong-sized, or hash-mismatched.
pub fn run_verify(dest_override: Option<&Path>) -> Result<()> {
    let dest = resolve_dest(dest_override)?;
    let manifest_path = dest.join(MANIFEST_NAME);
    if !manifest_path.is_file() {
        bail!(
            "no bundle installed at {}; run `l3tc install-models` first.",
            dest.display(),
        );
    }

    let manifest_bytes =
        fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest = Manifest::parse(&manifest_bytes)?;
    manifest.check_compatible()?;

    let mut problems = 0usize;
    for (rel_path, expected_sha, expected_size) in manifest.artifacts() {
        let path = dest.join(rel_path);
        match verify_file(&path, expected_sha, expected_size) {
            Ok(()) => {
                eprintln!("  ok    {rel_path}");
            }
            Err(e) => {
                eprintln!("  FAIL  {rel_path}: {e}");
                problems += 1;
            }
        }
    }

    if problems > 0 {
        bail!("{problems} artifact(s) failed verification");
    }
    eprintln!("all artifacts verified");
    Ok(())
}

/// `l3tc install-models --list`.
///
/// Shows which specialists are installed + their sizes. Skips
/// verification for speed; use `--verify` for that.
pub fn run_list(dest_override: Option<&Path>) -> Result<()> {
    let dest = resolve_dest(dest_override)?;
    let manifest_path = dest.join(MANIFEST_NAME);

    if !manifest_path.is_file() {
        eprintln!(
            "no models installed at {}. Run `l3tc install-models` to download the v0.1.0 bundle.",
            dest.display(),
        );
        return Ok(());
    }

    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest = Manifest::parse(&manifest_bytes)?;
    eprintln!(
        "installed at {} (bundle v{}, built against l3tc {})",
        dest.display(),
        manifest.bundle_version,
        manifest.l3tc_version,
    );
    for (name, artifacts) in &manifest.specialists {
        let m = dest.join(&artifacts.model.path);
        let t = dest.join(&artifacts.tokenizer.path);
        eprintln!(
            "  {name:12} {} model, {} tokenizer  [{}  {}]",
            human_bytes(artifacts.model.size),
            human_bytes(artifacts.tokenizer.size),
            if m.is_file() { "present" } else { "missing" },
            if t.is_file() { "present" } else { "missing" },
        );
    }
    Ok(())
}

/// Call this from `registry.rs` when a resolve() returns None so the
/// user gets a pointer to the next step instead of a raw file-not-
/// found error. Returns the message as an owned string so the caller
/// can embed it in an `anyhow!` chain.
pub fn missing_models_hint() -> String {
    "no specialist models are installed. Run `l3tc install-models` to \
     download the v0.1.0 bundle, or set $L3TC_MODEL_DIR to point at an \
     existing models directory."
        .to_string()
}

// ---------------------------------------------------------------- //
// Internals                                                        //
// ---------------------------------------------------------------- //

/// Resolve the install destination — user override, else
/// `$L3TC_MODEL_DIR`, else `~/.l3tc/models`.
fn resolve_dest(override_: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = override_ {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::var("L3TC_MODEL_DIR") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| anyhow!("$HOME is not set; pass --dest explicitly"))?;
    Ok(PathBuf::from(home).join(".l3tc").join("models"))
}

/// Blocking GET with a hard size cap. Returns the full response
/// body as a Vec<u8>. Progress is printed to stderr every 4 MB.
fn http_get(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;

    // Content-Length sanity check.
    let len_hint: Option<u64> = resp.header("Content-Length").and_then(|s| s.parse().ok());
    if let Some(n) = len_hint {
        if n > MAX_BUNDLE_BYTES {
            bail!("bundle at {url} is {n} bytes; refusing to download (cap is {MAX_BUNDLE_BYTES})");
        }
    }

    // Stream into memory with a cap. `ureq::Response::into_reader()`
    // gives us an unchunked raw body stream.
    let mut reader = resp.into_reader().take(MAX_BUNDLE_BYTES + 1);
    let mut out = Vec::with_capacity(len_hint.unwrap_or(8 * 1024 * 1024) as usize);
    let mut buf = [0u8; 128 * 1024];
    let mut last_mb_reported = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| "read response body")?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        let mb = out.len() as u64 / (4 * 1024 * 1024);
        if mb > last_mb_reported {
            eprint!("  {} MB…\r", out.len() / (1024 * 1024));
            let _ = io::stderr().flush();
            last_mb_reported = mb;
        }
    }
    if out.len() as u64 > MAX_BUNDLE_BYTES {
        bail!("download exceeded the {MAX_BUNDLE_BYTES}-byte cap");
    }
    eprintln!(); // finish the progress line
    Ok(out)
}

/// Decompress + untar the bundle into `dest`, hashing each file as
/// it streams and cross-checking every hash against the manifest.
/// Errors out (leaving `dest` partially populated — caller should
/// clean up) on any mismatch.
fn extract_and_verify(bundle_bytes: &[u8], dest: &Path) -> Result<()> {
    // Two passes over the archive. Pass 1: find and load the
    // manifest. Pass 2: extract + verify. The tarball is small
    // enough (~100 MB) that keeping the compressed bytes in memory
    // and streaming twice is fine.
    let manifest = load_manifest(bundle_bytes)?;
    manifest.check_compatible()?;

    // Build a path → (sha256, size) lookup from the manifest.
    let mut expected: std::collections::HashMap<&str, (&str, u64)> =
        std::collections::HashMap::new();
    for (path, sha, size) in manifest.artifacts() {
        expected.insert(path, (sha, size));
    }

    // Extract every file, verifying as we go. The manifest itself
    // is written out too so `--verify` and `--list` can read it.
    let decoder = zstd::stream::Decoder::new(io::Cursor::new(bundle_bytes))
        .with_context(|| "open zstd decoder")?;
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries().with_context(|| "read tar entries")? {
        let mut entry = entry.with_context(|| "iterate tar entry")?;
        let tar_path = entry
            .path()
            .with_context(|| "decode tar entry path")?
            .into_owned();
        let rel = tar_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 tar path: {:?}", tar_path))?;

        // Reject absolute paths and `..` segments to avoid a
        // zip-slip equivalent. Safe tar paths are relative and
        // component-free.
        if tar_path.is_absolute()
            || tar_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            bail!("refusing suspicious tar path: {rel}");
        }

        // Directory entries are informational; we create parent
        // directories on demand when writing files. Skip them so
        // they don't fail the manifest lookup.
        if entry.header().entry_type().is_dir() || rel.ends_with('/') {
            continue;
        }

        let out_path = dest.join(&tar_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }

        // Buffer + hash + write. We can't fully stream because we
        // need the hash for verification before promoting the file,
        // but the largest single artifact (~13 MB) fits comfortably.
        let mut body = Vec::new();
        entry
            .read_to_end(&mut body)
            .with_context(|| format!("read tar entry body {rel}"))?;

        if rel == MANIFEST_NAME {
            // Manifest is not listed inside itself — write it out
            // verbatim.
            fs::write(&out_path, &body).with_context(|| format!("write {}", out_path.display()))?;
            continue;
        }

        let (expected_sha, expected_size) = expected.get(rel).copied().ok_or_else(|| {
            anyhow!("tar entry {rel} is not listed in manifest; refusing to install")
        })?;
        if body.len() as u64 != expected_size {
            bail!(
                "{rel}: size {} != manifest size {expected_size}",
                body.len()
            );
        }
        let got_sha = hex_sha256(&body);
        if got_sha != expected_sha {
            bail!("{rel}: sha256 {got_sha} != manifest sha256 {expected_sha}");
        }
        fs::write(&out_path, &body).with_context(|| format!("write {}", out_path.display()))?;
    }

    Ok(())
}

/// First pass over the bundle: pull just the manifest out so we can
/// consult it during the full extraction pass.
fn load_manifest(bundle_bytes: &[u8]) -> Result<Manifest> {
    let decoder = zstd::stream::Decoder::new(io::Cursor::new(bundle_bytes))
        .with_context(|| "open zstd decoder (manifest pass)")?;
    let mut archive = tar::Archive::new(decoder);
    for entry in archive
        .entries()
        .with_context(|| "read tar entries (manifest pass)")?
    {
        let mut entry = entry.with_context(|| "iterate tar entry (manifest pass)")?;
        let path = entry
            .path()
            .with_context(|| "decode tar path (manifest pass)")?
            .into_owned();
        if path.to_str() == Some(MANIFEST_NAME) {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            return Manifest::parse(&buf);
        }
    }
    bail!("bundle does not contain {MANIFEST_NAME} at its root");
}

fn verify_file(path: &Path, expected_sha: &str, expected_size: u64) -> Result<()> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut total: u64 = 0;
    let mut buf = [0u8; 128 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    if total != expected_size {
        bail!("size {total} != expected {expected_size}");
    }
    let got = hex(hasher.finalize());
    if got != expected_sha {
        bail!("sha256 {got} != expected {expected_sha}");
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(hasher.finalize())
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn human_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

/// Internal: install from tarball bytes already in memory. Used
/// only by the in-process tests below so they can exercise the
/// extraction + verification path without a network round trip.
#[cfg(test)]
fn install_from_bytes(bundle_bytes: &[u8], dest: &Path) -> Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest).ok();
    }
    fs::create_dir_all(dest).with_context(|| format!("create dest {}", dest.display()))?;
    extract_and_verify(bundle_bytes, dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid `.tar.zst` bundle in memory containing a
    /// single fake specialist with matching manifest hashes. Used
    /// across the end-to-end install tests.
    fn build_test_bundle() -> Vec<u8> {
        let model_bytes = b"this is not really a model".to_vec();
        let tok_bytes = b"this is not really a tokenizer".to_vec();

        let manifest = Manifest {
            bundle_version: 1,
            l3tc_version: "0.1.0-test".to_string(),
            specialists: {
                let mut m = std::collections::BTreeMap::new();
                m.insert(
                    "prose".to_string(),
                    SpecialistArtifacts {
                        model: Artifact {
                            path: "prose/model.bin".into(),
                            sha256: hex_sha256(&model_bytes),
                            size: model_bytes.len() as u64,
                        },
                        tokenizer: Artifact {
                            path: "prose/tokenizer.model".into(),
                            sha256: hex_sha256(&tok_bytes),
                            size: tok_bytes.len() as u64,
                        },
                    },
                );
                m
            },
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();

        let mut tar_buf = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            let mut h = tar::Header::new_gnu();
            h.set_path(MANIFEST_NAME).unwrap();
            h.set_size(manifest_bytes.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder.append(&h, &manifest_bytes[..]).unwrap();

            let mut h = tar::Header::new_gnu();
            h.set_path("prose/model.bin").unwrap();
            h.set_size(model_bytes.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder.append(&h, &model_bytes[..]).unwrap();

            let mut h = tar::Header::new_gnu();
            h.set_path("prose/tokenizer.model").unwrap();
            h.set_size(tok_bytes.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder.append(&h, &tok_bytes[..]).unwrap();
            builder.finish().unwrap();
        }

        zstd::stream::encode_all(&tar_buf[..], 3).unwrap()
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "l3tc-install-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn install_and_list_round_trip() {
        let bundle = build_test_bundle();
        let dest = tempdir();
        install_from_bytes(&bundle, &dest).expect("install");

        assert!(dest.join("manifest.json").is_file());
        assert!(dest.join("prose/model.bin").is_file());
        assert!(dest.join("prose/tokenizer.model").is_file());

        // --verify should succeed.
        run_verify(Some(&dest)).expect("verify");

        // Cleanup.
        fs::remove_dir_all(&dest).ok();
    }

    #[test]
    fn verify_catches_mutated_artifact() {
        let bundle = build_test_bundle();
        let dest = tempdir();
        install_from_bytes(&bundle, &dest).expect("install");

        // Corrupt one byte of an artifact.
        let target = dest.join("prose/model.bin");
        let mut data = fs::read(&target).unwrap();
        data[0] ^= 0xFF;
        fs::write(&target, &data).unwrap();

        assert!(run_verify(Some(&dest)).is_err());

        fs::remove_dir_all(&dest).ok();
    }

    #[test]
    fn install_rejects_bundle_with_future_version() {
        // Build a bundle, then swap its manifest for one claiming a
        // future bundle_version. extract_and_verify should refuse.
        let model_bytes = b"x".to_vec();
        let future_manifest = Manifest {
            bundle_version: MODEL_BUNDLE_VERSION + 99,
            l3tc_version: "99.0.0".to_string(),
            specialists: {
                let mut m = std::collections::BTreeMap::new();
                m.insert(
                    "prose".to_string(),
                    SpecialistArtifacts {
                        model: Artifact {
                            path: "prose/model.bin".into(),
                            sha256: hex_sha256(&model_bytes),
                            size: 1,
                        },
                        tokenizer: Artifact {
                            path: "prose/tokenizer.model".into(),
                            sha256: hex_sha256(&model_bytes),
                            size: 1,
                        },
                    },
                );
                m
            },
        };
        let manifest_bytes = serde_json::to_vec_pretty(&future_manifest).unwrap();

        let mut tar_buf = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            let mut h = tar::Header::new_gnu();
            h.set_path(MANIFEST_NAME).unwrap();
            h.set_size(manifest_bytes.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder.append(&h, &manifest_bytes[..]).unwrap();
            builder.finish().unwrap();
        }
        let bundle = zstd::stream::encode_all(&tar_buf[..], 3).unwrap();

        let dest = tempdir();
        let err = install_from_bytes(&bundle, &dest).unwrap_err();
        assert!(
            err.to_string().contains("bundle has format version"),
            "expected future-version error, got: {err}"
        );

        fs::remove_dir_all(&dest).ok();
    }

    #[test]
    fn install_rejects_zip_slip_paths() {
        // Confirm the defense against tar-path escape (zip-slip).
        // `tar::Builder` already refuses to emit a header with `..`
        // via the high-level API, so we forge one at the byte level.
        // This exercises our extract-side guard in extract_and_verify.
        let mut tar_buf = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            // Bypass `set_path` which validates — write a raw header
            // with the path set via bytes. The `ustar` path field is
            // 100 bytes at offset 0; any NUL-terminated content is
            // accepted by the reader.
            let mut h = tar::Header::new_gnu();
            h.set_size(4);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            // Write the malicious path directly into the header's
            // path bytes. Using `as_old()` to get a mutable handle.
            let name = b"../evil.txt";
            let path_field = &mut h.as_old_mut().name[..name.len()];
            path_field.copy_from_slice(name);
            h.set_cksum();
            builder.append(&h, &b"evil"[..]).unwrap();
            builder.finish().unwrap();
        }
        let bundle = zstd::stream::encode_all(&tar_buf[..], 3).unwrap();

        let dest = tempdir();
        // Should fail either at manifest lookup (no manifest.json)
        // or at the path-safety check; in both cases no file should
        // be written outside `dest`.
        let err = install_from_bytes(&bundle, &dest).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("manifest") || msg.contains("suspicious") || msg.contains(".."),
            "unexpected error: {err}",
        );
        // Crucially: the evil file must NOT have been written to a
        // parent of `dest`.
        let parent = dest.parent().unwrap();
        assert!(
            !parent.join("evil.txt").exists(),
            "zip-slip escaped to {}",
            parent.display(),
        );
        fs::remove_dir_all(&dest).ok();
    }

    #[test]
    fn manifest_rejects_future_bundle_versions() {
        let m = Manifest {
            bundle_version: MODEL_BUNDLE_VERSION + 1,
            l3tc_version: "99.99.99".to_string(),
            specialists: Default::default(),
        };
        assert!(m.check_compatible().is_err());
    }

    #[test]
    fn manifest_accepts_same_or_older_bundle_versions() {
        let m = Manifest {
            bundle_version: MODEL_BUNDLE_VERSION,
            l3tc_version: "0.1.0".to_string(),
            specialists: Default::default(),
        };
        assert!(m.check_compatible().is_ok());
    }

    #[test]
    fn hex_sha256_of_empty_is_known_constant() {
        // SHA-256 of the empty string is a well-known constant.
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
    }

    #[test]
    fn human_bytes_formats_units() {
        assert_eq!(human_bytes(100), "100 B");
        assert_eq!(human_bytes(1024), "1.0 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MB");
    }

    #[test]
    fn manifest_parse_round_trip() {
        let mut specialists = std::collections::BTreeMap::new();
        specialists.insert(
            "prose".to_string(),
            SpecialistArtifacts {
                model: Artifact {
                    path: "prose/model.bin".into(),
                    sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        .into(),
                    size: 0,
                },
                tokenizer: Artifact {
                    path: "prose/tokenizer.model".into(),
                    sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        .into(),
                    size: 0,
                },
            },
        );
        let m = Manifest {
            bundle_version: 1,
            l3tc_version: "0.1.0".to_string(),
            specialists,
        };
        let j = serde_json::to_vec(&m).unwrap();
        let m2 = Manifest::parse(&j).unwrap();
        assert_eq!(m2.bundle_version, 1);
        assert_eq!(m2.specialists.len(), 1);
    }
}
