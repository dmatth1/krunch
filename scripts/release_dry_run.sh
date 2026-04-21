#!/usr/bin/env bash
# End-to-end simulation of cutting an l3tc release from a clean state.
#
# Exercises the whole v0.1.0 release pipeline locally BEFORE any tag
# is pushed. Intended as a pre-flight check — run it any time someone
# touches packaging/, .github/workflows/, or the install-models path.
#
# What it does, step by step:
#   1. Build `l3tc` release binary for the current host target.
#   2. Assemble a placeholder specialist bundle from the legacy
#      `l3tc_200k.bin` + SentencePiece tokenizer.
#   3. Run `packaging/build-model-bundle.sh` to produce the `.tar.zst`.
#   4. Serve the bundle from a local HTTP server on loopback.
#   5. Run `l3tc install-models --url http://127.0.0.1:…` into a
#      throwaway directory; assert it succeeds and the manifest
#      landed.
#   6. Run `l3tc install-models --verify`; assert every artifact's
#      SHA-256 matches.
#   7. For every file in `bench/v010_benchmark/manifest.tsv`:
#      compress, verify the produced `.l3tc` exists, decompress,
#      diff against the original byte-for-byte. Fail on any
#      mismatch.
#   8. Print a PASS/FAIL summary. Exit 0 only on a total pass.
#
# Usage: scripts/release_dry_run.sh [--keep-artifacts]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Make sure cargo is on PATH. rustup installs to $HOME/.cargo/bin by
# default, but shells started non-interactively (like CI or a script
# launched from another tool) often don't source ~/.cargo/env.
if ! command -v cargo >/dev/null 2>&1; then
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found on PATH. Install rustup or source ~/.cargo/env." >&2
    exit 1
fi

KEEP=0
FULL_BUNDLE=0
for arg in "$@"; do
    case "$arg" in
        --keep-artifacts) KEEP=1 ;;
        --full-bundle)    FULL_BUNDLE=1 ;;
        -h|--help)
            sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
            echo
            echo "Flags:"
            echo "  --full-bundle      exercise all 7 specialists (dress rehearsal)"
            echo "  --keep-artifacts   keep the staging dir on exit (debugging)"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

STAGE="$(mktemp -d -t l3tc-dry-run-XXXXXX)"
cleanup() {
    if [[ $KEEP -eq 0 ]]; then
        rm -rf "$STAGE"
    else
        echo "keeping artifacts in $STAGE"
    fi
    if [[ -n "${HTTP_PID:-}" ]]; then
        kill "$HTTP_PID" 2>/dev/null || true
        # Wait for the port to actually release. python -m http.server
        # can take a beat to tear down on a busy system; without this
        # wait a subsequent dry-run in the same shell session will
        # race the port back.
        wait "$HTTP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

PASS=()
FAIL=()

step() { printf '\n== %s ==\n' "$*"; }
ok()   { PASS+=("$*"); printf '  ok   %s\n' "$*"; }
bad()  { FAIL+=("$*"); printf '  FAIL %s\n' "$*"; }

# ---- step 1: build release binary --------------------------------- #

step "build release binary"
(
    cd "$REPO_ROOT/l3tc-rust"
    cargo build --release --quiet
)
BIN="$REPO_ROOT/l3tc-rust/target/release/l3tc"
if [[ -x "$BIN" ]]; then
    ok "release binary at $BIN"
else
    bad "binary not built at $BIN"
    exit 1
fi

# Smoke: --help parses clap output without exploding.
if "$BIN" --help >/dev/null 2>&1; then
    ok "\`l3tc --help\` returns cleanly"
else
    bad "\`l3tc --help\` exited non-zero"
fi

# ---- step 2: assemble placeholder specialist sources -------------- #

step "assemble placeholder specialist"
SRC_MODEL="$REPO_ROOT/l3tc-rust/checkpoints/l3tc_200k.bin"
SRC_TOK="$REPO_ROOT/vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
if [[ ! -f "$SRC_MODEL" ]]; then
    bad "missing legacy model $SRC_MODEL"
    exit 1
fi
if [[ ! -f "$SRC_TOK" ]]; then
    bad "missing legacy tokenizer $SRC_TOK"
    exit 1
fi

SPECS="$STAGE/specialists"
# In quick mode we stage just `prose`. In full-bundle mode we stage
# all 7 specialists (copies of the same legacy weights). Since the
# file contents are identical, their SHA-256s are identical too —
# that's a useful corner case to confirm the manifest schema doesn't
# accidentally enforce uniqueness. What we're stress-testing here is
# packaging + install + routing plumbing, NOT compression quality.
if [[ $FULL_BUNDLE -eq 1 ]]; then
    SPECIALIST_NAMES=(prose code structured logs tabular markup fallback)
else
    SPECIALIST_NAMES=(prose)
fi
for s in "${SPECIALIST_NAMES[@]}"; do
    mkdir -p "$SPECS/$s"
    cp "$SRC_MODEL" "$SPECS/$s/model.bin"
    cp "$SRC_TOK"   "$SPECS/$s/tokenizer.model"
done
ok "staged ${#SPECIALIST_NAMES[@]} specialist(s): ${SPECIALIST_NAMES[*]}"

# ---- step 3: build the .tar.zst bundle ---------------------------- #

step "build model bundle"
BUNDLE="$STAGE/l3tc-models-dry-run.tar.zst"
"$REPO_ROOT/packaging/build-model-bundle.sh" \
    --models-dir "$SPECS" \
    --version 0.1.0-dry-run \
    --out "$BUNDLE" \
    >"$STAGE/bundle-build.log" 2>&1
if [[ -s "$BUNDLE" ]]; then
    ok "bundle built ($(wc -c < "$BUNDLE" | tr -d ' ') bytes)"
else
    bad "bundle missing or empty; see $STAGE/bundle-build.log"
    cat "$STAGE/bundle-build.log"
    exit 1
fi

# ---- step 4: serve over loopback HTTP ----------------------------- #

step "serve bundle on loopback"
# Port collisions are possible (e.g. a previous dry-run left its
# `python3 -m http.server` lingering, or another user of 8877-class
# ports). Try a small deterministic range first, then fall back to
# letting the kernel assign a port.
PORT=""
for p in 8877 8878 8879 8880 8881 8882 8883 8884 8885; do
    # `bash /dev/tcp` returns success if the port IS reachable (i.e.
    # already in use). We want the opposite — a port that refuses.
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
        PORT=$p
        break
    fi
done
if [[ -z "$PORT" ]]; then
    # Last resort: ask Python to pick a free port. Slightly more
    # complex because we can't predict the URL until the server has
    # started and printed it, so we shell out to a tiny one-liner
    # that binds to 0 and reports the chosen port on stdout.
    PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi
if [[ -z "$PORT" ]]; then
    bad "could not find a free loopback port"
    exit 1
fi

(
    cd "$STAGE" && python3 -m http.server "$PORT" --bind 127.0.0.1 \
        >"$STAGE/http.log" 2>&1
) &
HTTP_PID=$!
# Poll for readiness — the server prints its banner before accepting.
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fs -o /dev/null "http://127.0.0.1:$PORT/"; then
        break
    fi
    sleep 0.2
done
if curl -fs -o /dev/null "http://127.0.0.1:$PORT/"; then
    ok "HTTP server up on 127.0.0.1:$PORT"
else
    bad "HTTP server did not start on 127.0.0.1:$PORT"
    exit 1
fi

BUNDLE_URL="http://127.0.0.1:$PORT/$(basename "$BUNDLE")"

# ---- step 5: install-models from loopback ------------------------- #

step "install-models via install-models subcommand"
INSTALL_DIR="$STAGE/install"
if "$BIN" install-models --url "$BUNDLE_URL" --dest "$INSTALL_DIR" \
        >"$STAGE/install.log" 2>&1; then
    ok "install-models exit 0"
else
    bad "install-models exit non-zero; log follows"
    cat "$STAGE/install.log"
    exit 1
fi
if [[ -f "$INSTALL_DIR/manifest.json" ]]; then
    ok "manifest.json landed at $INSTALL_DIR"
else
    bad "manifest.json missing at $INSTALL_DIR"
fi
missing_count=0
for s in "${SPECIALIST_NAMES[@]}"; do
    if [[ ! -f "$INSTALL_DIR/$s/model.bin" || ! -f "$INSTALL_DIR/$s/tokenizer.model" ]]; then
        bad "$s specialist artifacts missing"
        missing_count=$((missing_count + 1))
    fi
done
if [[ $missing_count -eq 0 ]]; then
    ok "all ${#SPECIALIST_NAMES[@]} specialist(s) present on disk"
fi

# Manifest sanity: all specialists named in-bundle match what we
# staged, and each entry has non-empty sha + nonzero size.
if command -v python3 >/dev/null 2>&1; then
    manifest_check="$(python3 - "$INSTALL_DIR/manifest.json" "${SPECIALIST_NAMES[@]}" <<'PY'
import json, sys
path, *expected = sys.argv[1:]
m = json.load(open(path))
ok = True
if sorted(m.get("specialists", {}).keys()) != sorted(expected):
    print(f"specialist set mismatch: manifest={sorted(m['specialists'])} expected={sorted(expected)}")
    ok = False
for name, s in m.get("specialists", {}).items():
    for kind in ("model", "tokenizer"):
        art = s.get(kind, {})
        if not art.get("sha256") or art.get("size", 0) <= 0:
            print(f"{name}.{kind}: bad sha/size = {art}")
            ok = False
print("PASS" if ok else "FAIL")
PY
)"
    if [[ "${manifest_check##*$'\n'}" == "PASS" ]]; then
        ok "manifest.json has expected specialists with valid sha/size"
    else
        bad "manifest.json sanity check failed: $manifest_check"
    fi
fi

# ---- step 6: install-models --verify ------------------------------ #

step "install-models --verify"
if "$BIN" install-models --verify --dest "$INSTALL_DIR" \
        >"$STAGE/verify.log" 2>&1; then
    ok "--verify reported all artifacts valid"
else
    bad "--verify failed; log follows"
    cat "$STAGE/verify.log"
fi

# ---- step 6b: install-models --list output sanity ----------------- #

step "install-models --list"
"$BIN" install-models --list --dest "$INSTALL_DIR" \
    >"$STAGE/list.log" 2>&1 || true
# The --list command prints one "  <name> …" line per specialist.
# Check every staged name appears.
list_missing=0
for s in "${SPECIALIST_NAMES[@]}"; do
    if ! grep -Eq "^[[:space:]]+${s}[[:space:]]" "$STAGE/list.log"; then
        bad "--list output missing $s; see $STAGE/list.log"
        list_missing=$((list_missing + 1))
    fi
done
if [[ $list_missing -eq 0 ]]; then
    ok "--list output contains all ${#SPECIALIST_NAMES[@]} specialist(s)"
fi

# ---- step 7: round-trip across the v010 benchmark corpus ---------- #

step "round-trip compress + decompress + diff on benchmark corpus"
MANIFEST="$REPO_ROOT/bench/v010_benchmark/manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
    bad "benchmark manifest not found at $MANIFEST (task 26 not landed?)"
else
    RT_DIR="$STAGE/roundtrip"
    mkdir -p "$RT_DIR"
    rt_total=0
    rt_fail=0
    # Auto-routing tracking (full-bundle mode only). The detector's
    # accuracy on the v010 corpus is a known quantity from task 15
    # (overall 95.8%, code class 85.3%); we assert an empirically-
    # safe ≥80% floor here rather than strict equality so the dry-
    # run flags regressions without being brittle. Files whose
    # manifest domain is "mixed" are excluded — they're deliberately
    # ambiguous inputs with no single "right" routing answer.
    route_total=0
    route_match=0
    route_mismatches=""
    while IFS=$'\t' read -r filename domain bytes sha source; do
        [[ "$filename" == "filename" ]] && continue
        [[ -z "$filename" ]] && continue
        src="$REPO_ROOT/bench/v010_benchmark/$filename"
        if [[ ! -f "$src" ]]; then
            bad "benchmark file missing: $filename"
            rt_fail=$((rt_fail + 1))
            continue
        fi
        rt_total=$((rt_total + 1))
        enc="$RT_DIR/$(basename "$filename").l3tc"
        dec="$RT_DIR/$(basename "$filename").out"
        if ! L3TC_MODEL_DIR="$INSTALL_DIR" "$BIN" compress \
                "$src" -o "$enc" >/dev/null 2>"$STAGE/compress.err"; then
            bad "compress failed: $filename"
            cat "$STAGE/compress.err" >&2
            rt_fail=$((rt_fail + 1))
            continue
        fi
        if ! L3TC_MODEL_DIR="$INSTALL_DIR" "$BIN" decompress \
                "$enc" -o "$dec" >/dev/null 2>"$STAGE/decompress.err"; then
            bad "decompress failed: $filename"
            cat "$STAGE/decompress.err" >&2
            rt_fail=$((rt_fail + 1))
            continue
        fi
        if ! cmp -s "$src" "$dec"; then
            bad "round-trip byte mismatch: $filename"
            rt_fail=$((rt_fail + 1))
            continue
        fi
        # Full-bundle mode: compare detector output to the manifest
        # domain, skipping "mixed" (intentionally ambiguous).
        if [[ $FULL_BUNDLE -eq 1 && "$domain" != "mixed" ]]; then
            det_json="$("$BIN" detect --json "$src" 2>/dev/null || echo '{}')"
            detected="$(printf '%s' "$det_json" | \
                sed -n 's/.*"specialist":"\([^"]*\)".*/\1/p')"
            route_total=$((route_total + 1))
            if [[ "$detected" == "$domain" ]]; then
                route_match=$((route_match + 1))
            else
                route_mismatches+="    $filename  expected=$domain  detected=$detected"$'\n'
            fi
        fi
    done < "$MANIFEST"
    if [[ $rt_fail -eq 0 ]]; then
        ok "round-trip: $rt_total files, all bit-identical"
    else
        bad "round-trip: $rt_fail / $rt_total files failed"
    fi
    # Auto-routing summary (full-bundle mode).
    if [[ $FULL_BUNDLE -eq 1 && $route_total -gt 0 ]]; then
        route_pct=$(( 100 * route_match / route_total ))
        # 80% floor: comfortably below task-15's measured 95.8% so
        # we're flagging real regressions, not noise in the weak
        # classes (code class sits at 85% per detection_eval.md).
        if [[ $route_pct -ge 80 ]]; then
            ok "auto-routing: $route_match / $route_total files matched domain (${route_pct}%)"
        else
            bad "auto-routing: only ${route_pct}% match (floor: 80%)"
        fi
        # Print the mismatches as a diagnostic, even when above the
        # floor — helps spot regressions that haven't crossed 80%.
        if [[ -n "$route_mismatches" ]]; then
            printf '  routing mismatches (informational):\n%s' "$route_mismatches"
        fi
    fi
fi

# ---- summary ------------------------------------------------------ #

printf '\n================ SUMMARY ================\n'
printf 'passed: %d\n' "${#PASS[@]}"
printf 'failed: %d\n' "${#FAIL[@]}"

if [[ ${#FAIL[@]} -ne 0 ]]; then
    printf '\nFailures:\n'
    for f in "${FAIL[@]}"; do printf '  - %s\n' "$f"; done
    exit 1
fi

printf '\ndry-run PASS — safe to push a release tag.\n'
