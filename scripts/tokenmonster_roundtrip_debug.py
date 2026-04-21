"""Diagnose why TokenMonster round-trip fails on some corpora.

Test each vocab's byte_fallback / unicode behavior to see if the
round-trip failure is:
  (a) Unicode NFC/NFKC normalization — possibly fixable with flags
  (b) Missing byte fallback for OOV chars — fixable by vocab choice
  (c) Lossy by design (tokenmonster drops some content) — not fixable
"""

from pathlib import Path
import tokenmonster as tm

tm.set_local_directory("/tmp/tokenmonster_cache")
REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench" / "v010_benchmark"

VOCABS = {
    "tm-english":       "/tmp/tokenmonster_cache/english-16000-balanced-v1.vocab",
    "tm-code":          "/tmp/tokenmonster_cache/code-16000-consistent-v1.vocab",
    "tm-englishcode":   "/tmp/tokenmonster_cache/englishcode-16000-consistent-v1.vocab",
    # Unfiltered variants — supposed to round-trip losslessly via
    # -include-256-bytes guarantee during training.
    "tm-code-unfilt":   "/tmp/tokenmonster_cache/code-16000-unfiltered-v1.vocab",
    "tm-code-unfilt-nc": "/tmp/tokenmonster_cache/code-16000-unfiltered-nocapcode-v1.vocab",
}


def first_diff(a: bytes, b: bytes) -> tuple[int, bytes, bytes]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            start = max(0, i - 20)
            end = min(n, i + 20)
            return i, a[start:end], b[start:end]
    return n, a[n:n+20], b[n:n+20]


def test(vocab_name: str, vocab_path: str, data: bytes, label: str):
    vocab = tm.load(vocab_path)
    tokens = vocab.tokenize(data)
    decoded = vocab.decode(tokens)
    recovered = bytes(decoded) if isinstance(decoded, (bytes, bytearray)) \
        else decoded.encode("utf-8", errors="replace")
    ok = recovered == data
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {vocab_name:<20s} vs {label:<12s} "
          f"input={len(data)} recovered={len(recovered)} "
          f"diff={len(data) - len(recovered):+d}")
    if not ok:
        idx, orig, rec = first_diff(data, recovered)
        print(f"      first byte diff at offset {idx}:")
        print(f"        original:  {orig!r}")
        print(f"        recovered: {rec!r}")


def main() -> None:
    # Sample bytes from each domain with known-failing round-trips.
    samples = {}
    for d in ["prose", "code", "markup", "structured"]:
        parts = []
        for f in sorted((BENCH / d).iterdir())[:2]:
            if f.is_file() and not f.name.startswith(("highentropy_", "random_", "hex_")):
                parts.append(f.read_bytes())
        samples[d] = b"".join(parts)[:50_000]  # 50 KB per domain is plenty

    print("round-trip test: TokenMonster vocabs on v010_benchmark domain samples")
    print("=" * 80)
    for d, data in samples.items():
        print(f"\n--- {d} (sample {len(data)} B) ---")
        for name, path in VOCABS.items():
            test(name, path, data, d)


if __name__ == "__main__":
    main()
