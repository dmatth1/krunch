//! Mixture-of-Specialists routing (Phase 14).
//!
//! Each `Specialist` variant maps to one trained model file plus
//! its tokenizer. The compressor either auto-detects the right
//! specialist from the input bytes or accepts an explicit override
//! from the CLI; the choice is recorded in the file header so the
//! decompressor can auto-load the matching model without user input.
//!
//! Phase 14 is in build-out: the routing/header plumbing ships first
//! so the file format is stable. Specialist *models* are trained as
//! a separate workstream; until they exist, every compress call uses
//! `Specialist::Unspecified` (model_id == 0), which preserves the
//! existing single-model behavior. New `.l3tc` files written before
//! Phase 14 specialists land therefore decompress with whatever model
//! the user passes on the CLI, identical to pre-Phase-14 behavior.

use std::fmt;

/// Identifies which trained specialist model a `.l3tc` file was
/// compressed with. Stored as a single byte at offset 6 of the
/// file header (one of the previously-reserved bytes — see
/// `codec.rs`).
///
/// `Unspecified` (0) means "no specialist routing was applied at
/// compress time — decompressor should use whatever model the user
/// loads." This is the value all pre-Phase-14 files have, and the
/// value Phase 14 itself emits until specialist models are trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Specialist {
    /// Pre-Phase-14 file or fallback when no specialist was selected.
    /// Decompressor uses the model the user supplies.
    Unspecified = 0,
    /// English natural language: prose, books, articles, docs.
    Prose = 1,
    /// Source code across major languages (Python, JS/TS, Java,
    /// C/C++, Go, Rust, Ruby, etc.).
    Code = 2,
    /// Configuration and structured data formats: YAML, JSON,
    /// TOML, XML, INI, .env.
    Structured = 3,
    /// System and application logs: syslog, nginx/apache, app logs,
    /// JSON logs, stack traces.
    Logs = 4,
    /// Tabular data: CSV, TSV.
    Tabular = 5,
    /// Markup and documentation: HTML, Markdown, LaTeX.
    Markup = 6,
    /// Generalist fallback model (Phase 11's balanced 51 GB corpus
    /// model) — used when detection confidence is low or content is
    /// genuinely mixed.
    Fallback = 7,
}

impl Specialist {
    /// Decode a model_id byte. Unknown IDs (1..=127 not covered above
    /// or any 128+) round-trip back as `Unspecified` so old
    /// decompressors don't choke on a future-Phase-14 file: they'll
    /// fall back to the user-supplied model.
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Unspecified,
            1 => Self::Prose,
            2 => Self::Code,
            3 => Self::Structured,
            4 => Self::Logs,
            5 => Self::Tabular,
            6 => Self::Markup,
            7 => Self::Fallback,
            // Reserved range 8..=127 for future Tier-2 specialists
            // (sql, diff, multi-lang prose, etc.); 128..=255 reserved
            // for OOB user-supplied models. Anything we don't know
            // about decodes as Unspecified.
            _ => Self::Unspecified,
        }
    }

    /// Encode as the byte stored in the file header.
    pub fn to_byte(self) -> u8 {
        self as u8
    }

    /// Short human-readable name. Stable; used for CLI args
    /// (`--model=prose` etc.) and the verbose mode banner.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Prose => "prose",
            Self::Code => "code",
            Self::Structured => "structured",
            Self::Logs => "logs",
            Self::Tabular => "tabular",
            Self::Markup => "markup",
            Self::Fallback => "fallback",
        }
    }

    /// Parse a CLI string like "prose" or "auto" or "unspecified".
    /// "auto" maps to `Unspecified`; the CLI layer uses that as a
    /// signal to run detection rather than ship a literal
    /// `Unspecified` to the header.
    pub fn from_cli_name(s: &str) -> Option<Self> {
        match s {
            "auto" | "unspecified" | "none" => Some(Self::Unspecified),
            "prose" => Some(Self::Prose),
            "code" => Some(Self::Code),
            "structured" | "config" | "configs" | "json" | "yaml" => Some(Self::Structured),
            "logs" | "log" => Some(Self::Logs),
            "tabular" | "csv" | "tsv" => Some(Self::Tabular),
            "markup" | "html" | "markdown" | "md" => Some(Self::Markup),
            "fallback" | "generalist" | "general" => Some(Self::Fallback),
            _ => None,
        }
    }
}

impl fmt::Display for Specialist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Detection result: the chosen specialist plus a confidence in
/// `[0.0, 1.0]`. The CLI's verbose mode prints both.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub specialist: Specialist,
    pub confidence: f32,
}

/// Auto-detect the right specialist from the first few KB of input.
///
/// Heuristic ordering matters: stronger signals (literal tag prefixes,
/// strict structural markers) come before weaker ones (statistical
/// shape detectors). When no rule fires above its threshold, returns
/// `Specialist::Fallback` so the generalist model picks up the slack.
///
/// **Cost budget:** a few hundred microseconds at most. Detection
/// must not be a meaningful fraction of the per-file compression
/// cost. We aim for `<1 ms` on a 4 KB sample.
///
/// **What this does NOT do:**
/// - Parse the input — only counts characters and matches prefixes.
/// - Read past the first 4 KB — the rest of the file is the
///   compressor's job.
/// - Decide UTF-8 validity — that's the codec's `raw_store`
///   fallback's job.
pub fn detect(sample: &[u8]) -> Detection {
    // Cap the sample we look at. Anything past 4 KB is wasted work
    // for routing.
    let n = sample.len().min(4096);
    let buf = &sample[..n];

    // Treat as text for pattern matching. Lossy is fine — we don't
    // need bit-perfect, just enough structure to recognize.
    let text = String::from_utf8_lossy(buf);

    // Short inputs don't justify model routing — fall back. The
    // overhead of model swap and per-segment setup eats any tiny
    // ratio gain on a sub-KB file.
    if buf.len() < 256 {
        return Detection {
            specialist: Specialist::Fallback,
            confidence: 0.50,
        };
    }

    // ---- Strong structural signals (high confidence) ----

    // LaTeX: `\documentclass`, `\begin{...}`, `\section{...}`. Checked
    // before markdown/prose so LaTeX's letter-heavy content doesn't
    // get misrouted to prose.
    if let Some(d) = detect_latex(&text) {
        return d;
    }

    // JSON-lines logs (timestamp-shaped per-line JSON) must run before
    // detect_json — otherwise a file full of `{"ts":...}\n{"ts":...}\n`
    // records gets labelled as generic structured JSON, not logs.
    if let Some(d) = detect_jsonl_logs(&text) {
        return d;
    }

    // YAML / Kubernetes manifest: starts with a known top-level key
    // OR has many `key: value` lines.
    if let Some(d) = detect_yaml(&text) {
        return d;
    }

    // TOML / INI: `[section]` headers + `key = value` lines. Also
    // covers bare-dotenv with many `KEY=VALUE` lines.
    if let Some(d) = detect_toml_or_ini(&text) {
        return d;
    }

    // JSON: starts with `{` or `[` and has balanced brackets.
    if let Some(d) = detect_json(&text) {
        return d;
    }

    // XML / HTML: starts with `<?xml`, `<!DOCTYPE`, or many tags.
    if let Some(d) = detect_xml_or_html(&text) {
        return d;
    }

    // CSV / TSV: consistent delimiter count across rows. Runs before
    // log detection because CSV rows frequently contain date-like
    // fields that would otherwise trigger the timestamp heuristic.
    if let Some(d) = detect_tabular(&text) {
        return d;
    }

    // System / app logs: timestamped lines, log levels, IP
    // patterns.
    if let Some(d) = detect_logs(&text) {
        return d;
    }

    // Markdown: headings, code fences, link syntax.
    if let Some(d) = detect_markdown(&text) {
        return d;
    }

    // Source code: many programming-language keywords / brace
    // density.
    if let Some(d) = detect_code(&text) {
        return d;
    }

    // ---- Weaker statistical signals ----

    // English prose: high letter ratio, sentence-like punctuation.
    if let Some(d) = detect_prose(&text) {
        return d;
    }

    // Default: generalist fallback at low confidence.
    Detection {
        specialist: Specialist::Fallback,
        confidence: 0.40,
    }
}

// -------------------- per-domain detectors -------------------- //
//
// Each returns `Some(Detection)` if its threshold is met,
// `None` otherwise. All are pure functions over the sample text.

fn detect_yaml(text: &str) -> Option<Detection> {
    // Strong prefix: Kubernetes / common YAML.
    let trimmed = text.trim_start();
    if trimmed.starts_with("apiVersion:") || trimmed.starts_with("kind:") {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.95,
        });
    }
    // `---` document separator: could be YAML, could be markdown
    // frontmatter. Only fire if the YAML extends beyond the opening
    // separator — i.e., the first 30 lines have several `key: value`
    // lines with no intervening `---\n` close (or we see close
    // followed by non-YAML body, in which case leave it to markdown).
    if trimmed.starts_with("---\n") || trimmed.starts_with("--- \n") {
        // Check for a frontmatter close within the first ~20 lines.
        // Frontmatter closes with another `---\n`. If found, the
        // remaining body is probably markdown; let that detector
        // handle it.
        let after = trimmed.trim_start_matches("---").trim_start_matches(' ');
        let after = after.trim_start_matches('\n');
        let close_idx = after.find("\n---\n").or_else(|| after.find("\n---"));
        let looks_frontmatter = close_idx
            .map(|i| {
                // Require close within the first 20 lines AND that
                // content exists after the close.
                let before = &after[..i];
                let line_count = before.lines().count();
                line_count <= 20 && (after.len() - i) > 200
            })
            .unwrap_or(false);
        if !looks_frontmatter {
            return Some(Detection {
                specialist: Specialist::Structured,
                confidence: 0.95,
            });
        }
        // Frontmatter case: fall through so markdown detector can
        // claim it.
    }

    // Statistical: many `^<indent>key: value` lines. We accept keys
    // containing `-` `.` `_` plus the usual alphanumerics so
    // real-world config keys like `X-Frame-Options:` and
    // `my.pkg.field:` count. YAML list continuation lines
    // (`- value`) count as YAML-shaped too, but only as secondary
    // evidence — a file of ONLY list markers is ambiguous (BGL logs,
    // bullet-heavy prose, etc.) and must also show `key: value` pairs
    // before we route it to structured.
    let lines: Vec<&str> = text.lines().take(100).collect();
    if lines.len() < 3 {
        return None;
    }
    let mut kv_lines = 0usize;
    let mut list_lines = 0usize;
    for line in &lines {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix("- ") {
            if !rest.is_empty() {
                list_lines += 1;
            }
            continue;
        }
        if let Some(colon_pos) = t.find(':') {
            let mut key = &t[..colon_pos];
            // Unquoted YAML keys and quoted (`"key":`) — strip outer
            // double/single quotes before checking.
            let is_quoted = key.len() >= 2
                && ((key.starts_with('"') && key.ends_with('"'))
                    || (key.starts_with('\'') && key.ends_with('\'')));
            if is_quoted {
                key = &key[1..key.len() - 1];
            }
            if key.is_empty() {
                continue;
            }
            let first_ok = key
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_');
            if !first_ok {
                continue;
            }
            if key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
            {
                kv_lines += 1;
            }
        }
    }
    let yaml_shape = kv_lines + list_lines;
    let ratio = yaml_shape as f32 / lines.len() as f32;
    // Require at least two `key: value` lines — a file of only `- ...`
    // list entries (like BGL-format logs "- 1234 ...") is not YAML.
    if ratio > 0.55 && kv_lines >= 2 {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.80,
        });
    }
    // Config files with multi-line string values (e.g., prompt
    // templates) score lower on the simple `key: value` ratio
    // because continuation lines aren't key-shaped. Accept these
    // at lower confidence if we see lots of absolute evidence
    // (many kv-lines AND list-lines in the first 100).
    if kv_lines >= 8 && list_lines >= 3 {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.70,
        });
    }
    None
}

fn detect_toml_or_ini(text: &str) -> Option<Detection> {
    // Looks for `[section]` headers AND `key = value` or `key=value`
    // lines. Matches both TOML and INI conventions. Also covers
    // .env / bare-assignment files where every line is `KEY=VALUE`
    // with no headers.
    let lines: Vec<&str> = text.lines().take(100).collect();
    if lines.len() < 3 {
        return None;
    }
    let mut header = 0usize;
    let mut kv = 0usize;
    for line in &lines {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        // [section] header
        if t.starts_with('[') && t.ends_with(']') && t.len() >= 3 {
            header += 1;
            continue;
        }
        // key = value or key=value
        if let Some(eq_pos) = t.find('=') {
            let key = t[..eq_pos].trim();
            if !key.is_empty()
                && key
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphabetic() || c == '_')
                && key
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
            {
                kv += 1;
            }
        }
    }
    let total = lines.len() as f32;
    // TOML-style (has headers + kvs) or .env (lots of kvs, no headers).
    if header >= 1 && kv >= 3 {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.90,
        });
    }
    if kv as f32 / total > 0.6 && kv >= 5 {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.80,
        });
    }
    None
}

fn detect_latex(text: &str) -> Option<Detection> {
    // Strong prefix markers are decisive.
    let trimmed = text.trim_start();
    if trimmed.starts_with("\\documentclass") || trimmed.starts_with("\\documentstyle") {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.95,
        });
    }
    // Statistical: count LaTeX-specific commands that markdown/prose
    // files wouldn't have.
    let markers = [
        "\\begin{",
        "\\end{",
        "\\section{",
        "\\subsection{",
        "\\chapter{",
        "\\usepackage{",
        "\\cite{",
        "\\ref{",
        "\\label{",
        "\\textbf{",
        "\\textit{",
        "\\emph{",
        "\\item ",
        "\\frac{",
        "\\sum_",
    ];
    let hits: usize = markers.iter().map(|m| text.matches(*m).count()).sum();
    if hits >= 3 {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.90,
        });
    }
    if hits >= 1 && text.contains("\\begin{") && text.contains("\\end{") {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.75,
        });
    }
    None
}

fn detect_jsonl_logs(text: &str) -> Option<Detection> {
    // Only fire when the file is a stream of per-line JSON records
    // with timestamp-ish fields. Per-line JSON without timestamps
    // (e.g. NDJSON data dumps) should fall through to detect_json.
    let lines: Vec<&str> = text.lines().take(40).collect();
    if lines.len() < 5 {
        return None;
    }
    let mut json_lines = 0;
    let mut ts_lines = 0;
    for line in &lines {
        let t = line.trim();
        if t.starts_with('{') && t.ends_with('}') && t.contains(':') {
            json_lines += 1;
            // Timestamp-shaped field name or ISO-8601 value.
            if t.contains("\"ts\"")
                || t.contains("\"timestamp\"")
                || t.contains("\"time\"")
                || t.contains("\"@timestamp\"")
                || t.contains("\"level\"")
                || t.contains("\"severity\"")
            {
                ts_lines += 1;
            }
        }
    }
    let total = lines.len() as f32;
    if json_lines as f32 / total > 0.7 && ts_lines as f32 / total > 0.5 {
        return Some(Detection {
            specialist: Specialist::Logs,
            confidence: 0.85,
        });
    }
    None
}

fn detect_json(text: &str) -> Option<Detection> {
    let trimmed = text.trim_start();
    let first = trimmed.chars().next()?;
    if first != '{' && first != '[' {
        return None;
    }

    // Count brace/bracket balance and JSON-ish tokens to make sure
    // it's not just a single character of noise. We don't fully
    // parse — just enough structure to be confident.
    let opens = trimmed.matches(['{', '[']).count();
    let closes = trimmed.matches(['}', ']']).count();
    let quotes = trimmed.matches('"').count();

    // For real JSON, expect parens to roughly balance (we may have
    // truncated mid-structure) and at least some quoted strings.
    let balance_ratio = if opens > 0 {
        (opens.min(closes) as f32) / (opens.max(closes) as f32)
    } else {
        0.0
    };
    if balance_ratio > 0.5 && quotes >= 4 && opens >= 2 {
        Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.90,
        })
    } else if first == '{' && quotes >= 2 {
        // Lower-confidence: looks like a JSON object but small or
        // truncated.
        Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.70,
        })
    } else {
        None
    }
}

fn detect_xml_or_html(text: &str) -> Option<Detection> {
    let trimmed = text.trim_start();
    if trimmed.starts_with("<?xml") {
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.95,
        });
    }
    let lower_start: String = trimmed
        .chars()
        .take(50)
        .collect::<String>()
        .to_ascii_lowercase();
    if lower_start.starts_with("<!doctype html") || lower_start.starts_with("<html") {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.95,
        });
    }

    // Tag density: count `<word ...>` occurrences in first 4 KB.
    let mut tag_count = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] == b'<' && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'/') {
            tag_count += 1;
        }
    }
    // Heuristic: more than ~1 tag per 100 bytes is markup-shaped.
    if tag_count > (text.len() / 100).max(10) {
        // Distinguish HTML-ish from XML-ish by common HTML tag names.
        let lower = text.to_ascii_lowercase();
        let html_tags = [
            "<div", "<span", "<p>", "<a ", "<img", "<head", "<body", "<script",
        ];
        let html_hits = html_tags.iter().filter(|t| lower.contains(*t)).count();
        if html_hits >= 2 {
            return Some(Detection {
                specialist: Specialist::Markup,
                confidence: 0.85,
            });
        }
        return Some(Detection {
            specialist: Specialist::Structured,
            confidence: 0.75,
        });
    }
    None
}

fn detect_logs(text: &str) -> Option<Detection> {
    let lines: Vec<&str> = text.lines().take(40).collect();
    if lines.len() < 5 {
        return None;
    }

    // Tightened after Phase 14 task 6 contamination audit (task 29):
    // the old "leveled > 0.4 OR ts > 0.5" rule swept in ~150 MB of
    // Dockerfiles, Python, and YAML — any file that just *mentions*
    // "ERROR" or "INFO" in string constants or comments. Real logs
    // always have timestamps; require timestamp AND (level keyword OR
    // IP-ish pattern) for the strong signal.
    let log_levels = [
        "INFO", "ERROR", "WARN", "DEBUG", "TRACE", "FATAL", "[INFO]", "[ERROR]",
    ];
    let mut leveled = 0;
    let mut timestamped = 0;
    let mut ip_like = 0;
    for line in &lines {
        if log_levels.iter().any(|lvl| line.contains(lvl)) {
            leveled += 1;
        }
        // Common timestamp shapes: ISO-8601 prefix, syslog "Mmm dd HH:MM:SS",
        // bracketed timestamp.
        if looks_timestamped(line) {
            timestamped += 1;
        }
        if has_ip_like(line) {
            ip_like += 1;
        }
    }
    let total = lines.len() as f32;
    let leveled_ratio = leveled as f32 / total;
    let ts_ratio = timestamped as f32 / total;
    let ip_ratio = ip_like as f32 / total;

    // Strong: majority timestamped AND (level keyword OR IP pattern).
    if ts_ratio > 0.5 && (leveled_ratio > 0.2 || ip_ratio > 0.2) {
        return Some(Detection {
            specialist: Specialist::Logs,
            confidence: 0.85,
        });
    }
    // Medium: mixed timestamp + level signal.
    if ts_ratio > 0.3 && leveled_ratio > 0.3 {
        return Some(Detection {
            specialist: Specialist::Logs,
            confidence: 0.65,
        });
    }
    // Bare stack traces: no timestamps but the literal trace-frame
    // markers are very distinctive. Covers Python tracebacks and
    // Java/JS/Go stack traces.
    let trace_markers = [
        "Traceback (most recent call last):",
        "at java.",
        "at org.",
        "at com.",
        "\n    at ",
        "\n\tat ",
        " File \"",
        "ValueError:",
        "TypeError:",
        "RuntimeError:",
        "KeyError:",
        "IndexError:",
        "AttributeError:",
        "goroutine ",
        "panic:",
    ];
    let trace_hits: usize = trace_markers.iter().map(|m| text.matches(*m).count()).sum();
    if trace_hits >= 3 {
        return Some(Detection {
            specialist: Specialist::Logs,
            confidence: 0.80,
        });
    }
    None
}

fn has_ip_like(line: &str) -> bool {
    // Loose IPv4: four dot-separated 1-3 digit groups anywhere in the
    // line. Doesn't validate octet range; a string of "999.999.999.999"
    // still returns true, which is fine — we only need a shape cue.
    let bytes = line.as_bytes();
    let mut parts = 0usize;
    let mut run = 0usize;
    for &b in bytes {
        if b.is_ascii_digit() {
            run += 1;
            if run > 3 {
                run = 0;
                parts = 0;
            }
        } else if b == b'.' {
            if (1..=3).contains(&run) {
                parts += 1;
                run = 0;
            } else {
                parts = 0;
                run = 0;
            }
        } else {
            if parts == 3 && (1..=3).contains(&run) {
                return true;
            }
            parts = 0;
            run = 0;
        }
    }
    parts == 3 && (1..=3).contains(&run)
}

fn looks_timestamped(line: &str) -> bool {
    // ISO-8601 prefix: 4 digits + '-' + 2 digits + '-'
    let bytes = line.as_bytes();
    if bytes.len() >= 10
        && bytes[..4].iter().all(|b| b.is_ascii_digit())
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[7] == b'-'
    {
        return true;
    }
    // Spark / short-year slash prefix: "YY/MM/DD HH:MM:SS ..."
    // (2 digits + '/' + 2 digits + '/' + 2 digits + ' ' + 2 digits ':').
    if bytes.len() >= 17
        && bytes[..2].iter().all(|b| b.is_ascii_digit())
        && bytes[2] == b'/'
        && bytes[3..5].iter().all(|b| b.is_ascii_digit())
        && bytes[5] == b'/'
        && bytes[6..8].iter().all(|b| b.is_ascii_digit())
        && bytes[8] == b' '
        && bytes[9..11].iter().all(|b| b.is_ascii_digit())
        && bytes[11] == b':'
    {
        return true;
    }
    // Syslog: "Mmm dd HH:MM:SS"
    if bytes.len() >= 15 && bytes[..3].iter().all(|b| b.is_ascii_alphabetic()) && bytes[3] == b' ' {
        return true;
    }
    // Bracketed timestamp anywhere in the first ~160 chars. Covers
    // both `[2024-01-01 ...]` at start and Apache/nginx access-log
    // format `... [02/Jun/2026:22:39:14 +0000] ...`. Looser than a
    // start-of-line check so nginx lines count as timestamped.
    //
    // Careful with char boundaries: `line[..160]` can panic on
    // non-ASCII input if byte 160 is mid-codepoint. Walk back to the
    // previous char boundary before slicing.
    let mut head_end = line.len().min(160);
    while head_end > 0 && !line.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let head = &line[..head_end];
    if let Some(ob) = head.find('[') {
        if let Some(cb_rel) = head[ob..].find(']') {
            let inner = &head[ob + 1..ob + cb_rel];
            let colons = inner.matches(':').count();
            let slashes = inner.matches('/').count();
            let has_digit = inner.chars().any(|c| c.is_ascii_digit());
            if has_digit && (colons >= 2 || (colons >= 1 && slashes >= 1)) {
                return true;
            }
        }
    }
    // BGL-style: line starts with "- <10-digit-unix-epoch>" or
    // "<TAG> <10-digit-unix-epoch>" where TAG is an ALPHA token like
    // APPREAD, KERNTERM, etc. BGL splits event categories into a
    // leading token; real logs always follow it with a Unix epoch.
    let space_at = bytes.iter().position(|&b| b == b' ');
    if let Some(sp) = space_at {
        let rest = &bytes[sp + 1..];
        if rest.len() >= 10 && rest[..10].iter().all(|b| b.is_ascii_digit()) {
            let prefix = &bytes[..sp];
            let prefix_ok = prefix == b"-"
                || (!prefix.is_empty() && prefix.iter().all(|b| b.is_ascii_uppercase()));
            if prefix_ok {
                return true;
            }
        }
    }
    false
}

fn detect_tabular(text: &str) -> Option<Detection> {
    // Check a larger window (up to 40 lines) so 2-column CSVs with
    // short rows have enough evidence to fire.
    let lines: Vec<&str> = text.lines().take(40).filter(|l| !l.is_empty()).collect();
    if lines.len() < 5 {
        return None;
    }

    // Try common delimiters; CSV is dominant, TSV second.
    for &delim in &[',', '\t', ';', '|'] {
        let counts: Vec<usize> = lines.iter().map(|l| l.matches(delim).count()).collect();
        let max = *counts.iter().max().unwrap_or(&0);
        if max == 0 {
            continue;
        }
        // Majority of rows have the same delimiter count. 2+ is strong
        // evidence; a consistent 1-per-row is weaker (covers 2-column
        // CSVs) and needs a higher agreement ratio to avoid matching
        // `key: value` prose.
        let mode = counts.iter().filter(|&&c| c == max).count();
        let ratio = mode as f32 / lines.len() as f32;
        if max >= 2 && ratio > 0.7 {
            return Some(Detection {
                specialist: Specialist::Tabular,
                confidence: 0.85,
            });
        }
        if max == 1 && ratio > 0.85 && lines.len() >= 8 {
            // 2-column CSV: `name,value\nfoo,1\nbar,2\n...`. Extra
            // evidence required because single-delim per line can
            // also match `key=value` configs (but detect_toml_or_ini
            // runs first, so by here we've ruled out those).
            return Some(Detection {
                specialist: Specialist::Tabular,
                confidence: 0.70,
            });
        }
    }
    None
}

fn detect_markdown(text: &str) -> Option<Detection> {
    let lines: Vec<&str> = text.lines().take(60).collect();
    if lines.len() < 3 {
        return None;
    }
    // Heading lines: `#` followed by space, or `##`/`###` at line
    // start. Python single-`#` comments like `# TODO` also match
    // `starts_with('#')`, so we require a ` ` or another `#` right
    // after to cut down false positives from code comments.
    let heading = lines
        .iter()
        .filter(|l| {
            let b = l.as_bytes();
            b.len() >= 2 && b[0] == b'#' && (b[1] == b'#' || b[1] == b' ')
        })
        .count();
    let fence = lines.iter().filter(|l| l.starts_with("```")).count();
    let bullet = lines
        .iter()
        .filter(|l| l.starts_with("- ") || l.starts_with("* ") || l.starts_with("+ "))
        .count();
    // Markdown-specific link syntax `](url)` — rare outside markdown.
    let link_like = text.matches("](http").count()
        + text.matches("](#").count()
        + text.matches("](/").count()
        + text.matches("](.").count();
    // Bold/italic syntax specific to markdown.
    let emph = text.matches("**").count() + text.matches(" *").count();

    // Require at least one markdown-specific signal (fence OR link-
    // like syntax OR heavy emphasis). `# comments` + `- bullets`
    // alone match too many Python/shell scripts.
    let has_specific = fence >= 1 || link_like >= 2 || emph >= 3;
    let shape_signals = (heading >= 2) as u32
        + (fence >= 1) as u32
        + (bullet >= 3) as u32
        + (link_like >= 2) as u32;

    if has_specific && shape_signals >= 2 {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.85,
        });
    }
    if has_specific && (heading + bullet + fence) > 3 {
        return Some(Detection {
            specialist: Specialist::Markup,
            confidence: 0.70,
        });
    }
    None
}

fn detect_code(text: &str) -> Option<Detection> {
    // Strong keyword markers. These forms with surrounding whitespace
    // are distinctive enough to not confuse with prose (they still
    // appear in prose but each hit is rare). Keeping the leading
    // space is important — `return ` without it matches English
    // "return home", `from ` matches "from the", etc.
    let strong_markers = [
        " def ",
        " class ",
        " function ",
        " return ",
        " import ",
        "#include",
        " public ",
        " private ",
        " static ",
        " void ",
        " fn ",
        " let ",
        " const ",
        " var ",
        " async ",
        " await ",
        " => ",
        "::",
        " struct ",
        " enum ",
        " impl ",
        " trait ",
        " interface ",
        "#define",
        "#!/usr/bin",
    ];
    let mut hits = 0;
    for kw in &strong_markers {
        if text.contains(kw) {
            hits += 1;
        }
    }

    // Python-specific indent-prefixed patterns. These ARE distinctive
    // — prose doesn't have many 4-space-indented `return ` or `if `
    // lines. We check only exact indent-+-keyword patterns, not the
    // bare keywords.
    let python_indent_markers = [
        "\n    return ",
        "\n    if ",
        "\n    elif ",
        "\n    else:",
        "\n    for ",
        "\n    while ",
        "\n    try:",
        "\n    except",
        "\n    with ",
        "\n    def ",
        "\n    class ",
        "\n    yield ",
        "\n        return ",
        "self.",
        "super().",
        "__init__",
        "__name__",
        "raise ValueError",
        "raise TypeError",
        "np.",
        "torch.",
        "pd.",
        "tf.",
        "nn.",
    ];
    let mut py_hits = 0;
    for kw in &python_indent_markers {
        if text.contains(kw) {
            py_hits += 1;
        }
    }

    // Brace / semicolon density (most non-Python code uses them).
    let braces = text.matches(['{', '}']).count();
    let semis = text.matches(';').count();
    let punct_ratio = (braces + semis) as f32 / text.len().max(1) as f32;

    // Minified-JS signal: many `){`, `;}`, `}(`, `})`. Nearly
    // unique to minified JS / one-line-packed source.
    let minified_js = text.matches("){").count()
        + text.matches(";}").count()
        + text.matches("}(").count()
        + text.matches("})").count();

    // Dotted-method calls `.word(`. Common in code, rare in prose.
    // Count approximately — don't need full parsing.
    let mut dotted_calls = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'.' && bytes[i + 1].is_ascii_alphabetic() {
            let end = (i + 32).min(bytes.len());
            let seg = &bytes[i + 1..end];
            let mut name_end = 0;
            for &b in seg {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    name_end += 1;
                } else {
                    break;
                }
            }
            if name_end > 0 && name_end < seg.len() && seg[name_end] == b'(' {
                dotted_calls += 1;
                i += 1 + name_end + 1;
                continue;
            }
        }
        i += 1;
    }

    // Minified JS has very high brace density and many `){` patterns.
    if minified_js >= 5 && punct_ratio > 0.05 {
        return Some(Detection {
            specialist: Specialist::Code,
            confidence: 0.90,
        });
    }
    // Require either strong lexical signal (keyword hits) OR strong
    // structural signal (py indent markers OR brace density with some
    // lexical context). Prose text occasionally has 1-2 of the
    // strong markers; hitting 3+ is characteristic of code.
    if hits >= 3 || py_hits >= 3 || (hits >= 2 && punct_ratio > 0.02) {
        return Some(Detection {
            specialist: Specialist::Code,
            confidence: 0.85,
        });
    }
    if hits >= 2
        || (py_hits >= 2 && hits >= 1)
        || (py_hits >= 2 && punct_ratio > 0.015)
        || (py_hits >= 1 && dotted_calls >= 8)
        || (dotted_calls >= 15 && punct_ratio > 0.015)
    {
        return Some(Detection {
            specialist: Specialist::Code,
            confidence: 0.65,
        });
    }
    None
}

fn detect_prose(text: &str) -> Option<Detection> {
    // Letter-vs-other ratio. Prose is typically >70% letters
    // (including spaces, the count is over a-zA-Z). Code, configs,
    // and structured data usually fall below.
    let mut letters = 0usize;
    let mut total = 0usize;
    for c in text.chars() {
        if c.is_ascii() {
            total += 1;
            if c.is_ascii_alphabetic() {
                letters += 1;
            }
        }
    }
    if total < 100 {
        return None;
    }
    let letter_ratio = letters as f32 / total as f32;

    // Sentence punctuation density: periods + commas per word-like
    // token.
    let words = text
        .split(|c: char| c.is_whitespace())
        .filter(|s| !s.is_empty())
        .count();
    let sentence_punct = text.matches(['.', ',']).count();
    let punct_per_word = sentence_punct as f32 / words.max(1) as f32;

    if letter_ratio > 0.72 && punct_per_word > 0.03 {
        return Some(Detection {
            specialist: Specialist::Prose,
            confidence: 0.80,
        });
    }
    if letter_ratio > 0.65 {
        return Some(Detection {
            specialist: Specialist::Prose,
            confidence: 0.55,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_roundtrip() {
        for v in [
            Specialist::Unspecified,
            Specialist::Prose,
            Specialist::Code,
            Specialist::Structured,
            Specialist::Logs,
            Specialist::Tabular,
            Specialist::Markup,
            Specialist::Fallback,
        ] {
            assert_eq!(Specialist::from_byte(v.to_byte()), v);
        }
    }

    #[test]
    fn unknown_byte_is_unspecified() {
        assert_eq!(Specialist::from_byte(99), Specialist::Unspecified);
        assert_eq!(Specialist::from_byte(255), Specialist::Unspecified);
    }

    #[test]
    fn cli_name_parses() {
        assert_eq!(Specialist::from_cli_name("prose"), Some(Specialist::Prose));
        assert_eq!(
            Specialist::from_cli_name("auto"),
            Some(Specialist::Unspecified)
        );
        assert_eq!(
            Specialist::from_cli_name("json"),
            Some(Specialist::Structured)
        );
        assert_eq!(Specialist::from_cli_name("garbage"), None);
    }

    #[test]
    fn detect_yaml_kubernetes() {
        let s = b"apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\ndata:\n  k: v\n  k2: v2\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Structured);
        assert!(d.confidence >= 0.85);
    }

    #[test]
    fn detect_json_object() {
        let s =
            br#"{"name": "test", "items": [1, 2, 3], "nested": {"key": "value"}, "more": "stuff"}"#;
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Structured);
    }

    #[test]
    fn detect_html() {
        let s = b"<!DOCTYPE html>\n<html>\n<head><title>X</title></head>\n<body><div><p>hi</p></div></body>\n</html>\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Markup);
    }

    #[test]
    fn detect_syslog() {
        let s = b"Jan 12 10:00:00 host sshd[1234]: INFO accepted publickey for root\n\
                  Jan 12 10:00:01 host sshd[1234]: INFO session opened for user root\n\
                  Jan 12 10:00:02 host kernel: [12345.67] usb 1-1: new device\n\
                  Jan 12 10:00:03 host cron[5678]: INFO (root) CMD (run-parts /etc/cron.hourly)\n\
                  Jan 12 10:00:04 host systemd[1]: INFO Started Daily apt activities\n\
                  Jan 12 10:00:05 host kernel: [12346.78] usb 1-1: device descriptor read\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Logs);
    }

    #[test]
    fn detect_csv() {
        let s = b"name,age,city,country,score\nAlice,30,NYC,USA,95\nBob,25,LA,USA,87\nCarol,40,Paris,FR,72\nDave,35,Tokyo,JP,88\nEve,28,Berlin,DE,91\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Tabular);
    }

    #[test]
    fn detect_markdown_doc() {
        let s = b"# Title\n\nSome intro paragraph here with words.\n\n## Section\n\n- bullet one\n- bullet two\n- bullet three\n\n```rust\nfn main() {}\n```\n\nLinks like [click](https://example.com).\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Markup);
    }

    #[test]
    fn detect_python_code() {
        let s = b"import os\nimport sys\n\ndef main():\n    return 0\n\nclass Thing:\n    def __init__(self):\n        self.x = 1\n\n    def run(self):\n        return self.x\n\nif __name__ == '__main__':\n    sys.exit(main())\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Code);
    }

    #[test]
    fn detect_english_prose() {
        // ~512 bytes of natural-language sentences.
        let s = "The quick brown fox jumps over the lazy dog. Compression \
        algorithms have been a subject of intense research since the early days \
        of computing. From simple run-length encoding to sophisticated \
        learned compressors, the field has evolved dramatically. Modern \
        approaches combine statistical models with carefully tuned coders to \
        achieve remarkable density. Each generation of tools makes its own \
        tradeoffs between speed, ratio, and memory footprint. The right choice \
        depends entirely on the workload at hand. There are no universal \
        winners.";
        let d = detect(s.as_bytes());
        assert_eq!(d.specialist, Specialist::Prose);
    }

    #[test]
    fn detect_short_input_returns_fallback() {
        let s = b"hello";
        let d = detect(s);
        assert_eq!(d.specialist, Specialist::Fallback);
    }

    #[test]
    fn detect_latex_documentclass() {
        let s = b"\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n\\section{Intro}\nText here with \\cite{foo}.\n\\end{document}\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Markup);
        assert!(d.confidence >= 0.90);
    }

    #[test]
    fn detect_jsonl_with_timestamps_routes_to_logs() {
        let s = br#"{"ts": "2026-04-01T10:00:00Z", "level": "INFO", "svc": "api", "msg": "ok"}
{"ts": "2026-04-01T10:00:01Z", "level": "ERROR", "svc": "db", "msg": "fail"}
{"ts": "2026-04-01T10:00:02Z", "level": "WARN", "svc": "cache", "msg": "miss"}
{"ts": "2026-04-01T10:00:03Z", "level": "INFO", "svc": "worker", "msg": "done"}
{"ts": "2026-04-01T10:00:04Z", "level": "DEBUG", "svc": "auth", "msg": "token"}
{"ts": "2026-04-01T10:00:05Z", "level": "INFO", "svc": "api", "msg": "ok"}
"#;
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Logs);
    }

    #[test]
    fn detect_yaml_deep_indent_and_list_markers() {
        let s = b"  response:\n    headers:\n      Content-Type:\n      - application/json\n      X-Frame-Options:\n      - DENY\n      Content-Security-Policy:\n      - frame-ancestors 'none'\n    status:\n      code: 200\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Structured);
    }

    #[test]
    fn detect_python_interior_function() {
        // No top-level `def`/`import` — all mid-function body.
        let s = b"        elif _type == 1:\n            top, left, height, width = 0, 0, h//2, w\n        elif _type == 2:\n            top, left, height, width = h//2, 0, h//2, w\n        elif _type == 3:\n            top, left, height, width = 0, w//2, h, w//2\n    else:\n        target_area = (h*w)//2\n        width = np.random.randint(target_area//h, w)\n        return (top, left, height, width)\n        if self.transform is not None:\n            img = self.transform(img)\n";
        let mut padded = s.to_vec();
        padded.resize(512, b' ');
        let d = detect(&padded);
        assert_eq!(d.specialist, Specialist::Code);
    }

    #[test]
    fn detect_unknown_falls_back() {
        // Random-looking high-entropy bytes; no signal fires.
        let mut rng_like = Vec::with_capacity(512);
        for i in 0..512 {
            rng_like.push(((i * 31 + 17) % 256) as u8);
        }
        let d = detect(&rng_like);
        assert_eq!(d.specialist, Specialist::Fallback);
    }
}
