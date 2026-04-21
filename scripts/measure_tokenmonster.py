"""Quick bake-off: TokenMonster vs our SentencePiece per-domain SPMs.

Tests the claim that TokenMonster's "ungreedy branching" encoding
gives meaningfully more compact output (higher B/T) than BPE/unigram
at the same 16K vocab size.

We can't train custom TokenMonster vocabs without the Go `getalltokens`
binary, so we use the public pre-trained 16K vocabs:
- english-16000-balanced-v1 — vs our prose (+fallback, which also uses enwik8)
- code-16000-consistent-v1 — vs our code
- englishcode-16000-consistent-v1 — vs enwik8 across all domains

For each, we measure B/T on the concatenated v010_benchmark files per
domain. Higher B/T = more compact = better for our compression pipeline.

If TokenMonster wins by ≥10% on any domain, worth the Rust-side
decoder work post-v0.1.0. If it's within a few percent of our custom
SPMs, we stick with SentencePiece.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import tokenmonster as tm

tm.set_local_directory("/tmp/tokenmonster_cache")

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench" / "v010_benchmark"
SPM_DIR = REPO / "data" / "specialists"
ENWIK8_SPM = REPO / "vendor" / "L3TC" / "dictionary" / \
    "vocab_enwik8_bpe_16384_0.999" / "spm_enwik8_bpe_16384_0.999.model"

DOMAINS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]


def load_domain_bytes(domain: str) -> bytes:
    """Concatenate all files in a domain, excluding uncompressible."""
    parts: list[bytes] = []
    d = BENCH / domain
    if not d.exists():
        return b""
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith(("highentropy_", "random_", "hex_")):
            continue
        parts.append(f.read_bytes())
    return b"".join(parts)


def spm_bt(sp: spm.SentencePieceProcessor, data: bytes) -> float:
    text = data.decode("utf-8", errors="replace")
    ids = sp.EncodeAsIds(text)
    return len(data) / max(1, len(ids))


def tm_bt(vocab, data: bytes) -> tuple[float, bool]:
    """Returns (B/T, round_trip_ok)."""
    tokens = vocab.tokenize(data)
    bt = len(data) / max(1, len(tokens))
    decoded = vocab.decode(tokens)
    if isinstance(decoded, (bytes, bytearray)):
        round_trip = bytes(decoded) == data
    else:
        round_trip = decoded.encode("utf-8", errors="replace") == data
    return bt, round_trip


def main() -> None:
    print("=" * 82)
    print("TokenMonster 16K vs our SPM 16K — B/T on concatenated v010_benchmark")
    print("=" * 82)
    print()

    # Load our SPMs (currently in data/specialists/)
    our_spms: dict[str, spm.SentencePieceProcessor] = {}
    for d in DOMAINS:
        sp = spm.SentencePieceProcessor()
        sp.Load(str(SPM_DIR / d / "spm.model"))
        our_spms[d] = sp
    sp_enwik8 = spm.SentencePieceProcessor()
    sp_enwik8.Load(str(ENWIK8_SPM))

    # Load TokenMonster vocabs by absolute path (avoids network download;
    # vocabs were fetched to /tmp/tokenmonster_cache with curl).
    TM_CACHE = Path("/tmp/tokenmonster_cache")
    tm_vocabs = {
        "tm-english-16K-balanced": tm.load(str(TM_CACHE / "english-16000-balanced-v1.vocab")),
        "tm-code-16K-consistent": tm.load(str(TM_CACHE / "code-16000-consistent-v1.vocab")),
        "tm-englishcode-16K-consistent": tm.load(str(TM_CACHE / "englishcode-16000-consistent-v1.vocab")),
    }

    # Per-domain samples
    samples = {d: load_domain_bytes(d) for d in DOMAINS}
    print(f"{'domain':<11s}  {'bytes':>9s}    "
          f"{'own-SPM':>9s}  {'enwik8':>9s}  "
          f"{'tm-eng':>9s}  {'tm-code':>9s}  {'tm-mix':>9s}    {'best':<20s}")
    print("-" * 110)

    for d in DOMAINS:
        data = samples[d]
        if not data:
            continue

        own_bt = spm_bt(our_spms[d], data)
        enw_bt = spm_bt(sp_enwik8, data)
        tm_eng_bt, rt_eng = tm_bt(tm_vocabs["tm-english-16K-balanced"], data)
        tm_code_bt, rt_code = tm_bt(tm_vocabs["tm-code-16K-consistent"], data)
        tm_mix_bt, rt_mix = tm_bt(tm_vocabs["tm-englishcode-16K-consistent"], data)

        # pick the best
        candidates = {
            "own-SPM": own_bt,
            "enwik8": enw_bt,
            "tm-english": tm_eng_bt,
            "tm-code": tm_code_bt,
            "tm-mix": tm_mix_bt,
        }
        best_name = max(candidates, key=candidates.get)
        best_val = candidates[best_name]

        rt_note = ""
        if not rt_eng or not rt_code or not rt_mix:
            rt_note = " ⚠️ tm round-trip FAILED"

        print(f"  {d:<9s}  {len(data):>9d}    "
              f"{own_bt:>9.3f}  {enw_bt:>9.3f}  "
              f"{tm_eng_bt:>9.3f}  {tm_code_bt:>9.3f}  {tm_mix_bt:>9.3f}   "
              f"{best_name:<12s} ({best_val:.3f}){rt_note}")

    print()
    print("Current production SPM column is 'own-SPM' (enwik8 for prose/fallback,")
    print("domain-trained unigram for the other 5).")


if __name__ == "__main__":
    main()
