"""Cross-validate the 7 specialist SPMs on representative content.

Each specialist's tokenizer should achieve the BEST B/T on its own
domain's content. This sanity-checks that the per-domain SPMs actually
specialized (vs. all converging to a general-purpose distribution).

Also verifies round-trip encode/decode preserves UTF-8 for each SPM.

Usage:
    vendor/L3TC/.venv/bin/python scripts/specialist_spm_crossvalidate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

DOMAINS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]

SAMPLES = {
    "prose": "The cat sat on the mat. She opened the book and began to read a story about a dragon.",
    "code": "def parse_config(path):\n    with open(path) as f:\n        return json.load(f)\n\nif __name__ == '__main__':\n    main()",
    "structured": '{\n  "apiVersion": "v1",\n  "kind": "ConfigMap",\n  "metadata": {\n    "name": "example"\n  }\n}',
    "logs": "2024-01-15 14:32:18 [INFO]  User 'alice' logged in from 192.168.1.42\n2024-01-15 14:32:19 [ERROR] database: connection refused",
    "tabular": "id,name,email,created_at\n1,alice,alice@example.com,2024-01-01\n2,bob,bob@example.com,2024-01-02\n",
    "markup": "<html>\n<head><title>Example</title></head>\n<body>\n  <h1>Welcome</h1>\n  <p>This is <b>bold</b> text.</p>\n</body>\n</html>",
}


def main():
    base = Path("data/specialists")
    tokenizers = {}
    for d in DOMAINS:
        sp = spm.SentencePieceProcessor()
        sp.load(str(base / d / "spm.model"))
        tokenizers[d] = sp

    print("=== SPM cross-validation: B/T by (tokenizer, content) ===\n")
    header = "content \\ tokenizer".ljust(16) + "".join(d[:8].ljust(10) for d in DOMAINS)
    print(header)
    print("-" * len(header))

    # For each content type, encode with every tokenizer and report B/T.
    content_types = list(SAMPLES.keys())
    for content_domain in content_types:
        sample = SAMPLES[content_domain]
        sample_bytes = len(sample.encode("utf-8"))
        row = content_domain[:14].ljust(16)
        best_domain, best_bt = None, 0.0
        for d in DOMAINS:
            sp = tokenizers[d]
            ids = sp.encode_as_ids(sample)
            bt = sample_bytes / max(len(ids), 1)
            if bt > best_bt:
                best_bt = bt
                best_domain = d
            row += f"{bt:.2f}".ljust(10)
        row += f"  -> best: {best_domain}"
        print(row)
        # Diagnostic: did the specialist for this content win?
        if content_domain in DOMAINS and best_domain != content_domain:
            print(f"   NOTE: {content_domain} SPM did not win on {content_domain} content "
                  f"(winner: {best_domain})")

    # Roundtrip check
    print("\n=== Round-trip sanity ===")
    for d in DOMAINS:
        sp = tokenizers[d]
        # Use its own domain sample if available, else prose.
        sample = SAMPLES.get(d, SAMPLES["prose"])
        ids = sp.encode_as_ids(sample)
        decoded = sp.decode(ids)
        # Unigram SPMs round-trip losslessly when add_dummy_prefix=False
        # and remove_extra_whitespaces=False (our config). Anything else
        # is a red flag.
        match = sample == decoded
        print(f"  {d:12s}: {'OK' if match else 'MISMATCH'}  "
              f"({len(ids)} tok, {len(sample)} B)")
        if not match:
            print(f"    original: {sample[:80]!r}")
            print(f"    decoded:  {decoded[:80]!r}")


if __name__ == "__main__":
    main()
