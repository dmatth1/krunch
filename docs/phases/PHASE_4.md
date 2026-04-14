# Phase 4 — Close the ratio gap to Python L3TC  ✅ 4a SHIPPED, 4b SHIPPED

## 4a — Implementation diff vs Python L3TC  ✅ DONE

The "4 pp gap to Python" was a phantom. The L3TC paper's "0.1665 on enwik6" is the theoretical entropy lower bound (compressor.py returns `math.ceil(entropy_sum / 8)`; the actual AC-encode path is commented out). Our forward pass is bit-identical to Python's (max L_inf 3.81e-05 with f32 head). Our entropy bound on enwik6 at segment 2048 is **0.1643** -- better than Python's 0.1665.

The 4 pp difference between actual coded bytes (0.2060) and the entropy bound (0.1665) is AC framing overhead, not a model bug. See `docs/phase_4a_findings.md`.

## 4b — Close the gap from actual coded bytes to entropy bound  ✅ SHIPPED

Measured overhead breakdown on enwik6 at segment_bytes=4096 (1135 segments):

| Source | Bytes | % of overhead |
|---|---:|---:|
| Raw-fallback bytes (Persian/Arabic ZWNJ) | 21,354 | 50.0% |
| Segment header bytes (13 + 2x unks) | 17,373 | 40.7% |
| Unk payload bytes | 3,813 | 8.9% |
| AC body bytes vs entropy bound | 143 | 0.3% |
| File header/trailer/CRC | 28 | 0.07% |
| **Total overhead** | **42,711** | **4.27 pp above entropy bound** |

Key finding: the arithmetic coder is essentially optimal (143 bytes overhead across 1135 segments). The overhead was half raw-fallback, half segment headers.

Shipped: varint segment headers (4b1) + unk extraction replacing raw-fallback (4b2). File format bumped to v4.

## Key tools created

- `l3tc dump-logits`, `l3tc entropy-bound`, `l3tc audit` subcommands
- `scripts/dump_python_logits.py`, `scripts/diff_logits.py`, `scripts/compute_entropy.py`
- `docs/phase_4a_findings.md`, `docs/phase_4b_findings.md`

## Deferred

- Hybrid classical fallback for OOD inputs -- moved to Phase 8
- RWKV-v7 architecture -- Phase 5
- Broader training corpus -- Phase 11
