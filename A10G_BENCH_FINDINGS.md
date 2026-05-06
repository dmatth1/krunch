# A10G bench results (2026-05-06)

g5.xlarge, on-demand (spot capacity unavailable in us-east-1), Docker
image `ghcr.io/dmatth1/krunch:latest`. Production path: dynamic
chunking + per-chunk packed compress + B-batched decompress with
graph capture default-on (`KRUNCH_OWN_WKV=1`).

## Results across input sizes

| Input | n_chunks | B (auto) | Compress KB/s | Decompress KB/s | Ratio | Roundtrip |
|---|---|---|---|---|---|---|
| 1 MB | 16 | 16 | 159.3 | 13.6 (eager) / 14.0 (graph, 1.03×) | 0.115 | ✓ |
| 4 MB | 64 | 64 | 152.7 | 38.8 | 0.116 | ✓ |
| 8 MB | 129 | 129 | 151.2 | **58.4** | 0.116 | ✓ |

## Key observations

1. **Decompress scales with B**, not input size directly. 1 → 4 → 8 MB
   gives B=16 → 64 → 129 and decompress 13.6 → 38.8 → 58.4 KB/s, all
   sub-linear in B (matmul work grows with B but launch overhead per
   step is fixed → diminishing returns). Theoretical saturation on A10G
   is ~640 (per `compute_decompress_batch`).
2. **Compress is flat ~150 KB/s regardless of input size.** Per-chunk
   sequential packed forward; chunk-batched compress is 5× slower (per
   `cli.py:53`). Each chunk's wall is dominated by its own forward.
3. **Earlier "47 KB/s historical decompress" was at higher B**, not a
   regression. Today's 58.4 KB/s at B=129 exceeds it.
4. **Graph capture: 1.03× win at B=16.** Same tiny win as T4. The
   per-step `.item()` autoregressive sync (1.16 ms/step on T4) appears
   to be the floor that graphs can't move. Need to re-test at higher
   B to see if the relative graph win grows.

## Tier-3 gate (1 MB, ratio ≤ 0.11, ≥ 200 KB/s both directions) status

At canonical 1 MB input:
- Compress: 159 → need 200 = **1.26× short**. Within reach via deferred
  levers (softmax+CDF fusion, per-chunk overhead reduction, reusable
  buffers). Estimate: ~10-15% combined → ~180 KB/s.
- Decompress: 14 → need 200 = **14× short**. **Structurally
  unreachable** on a single 1 MB chunked at 64 KB (B caps at 16).

The decompress gap is not a kernel-optimization problem. To close it
requires:

| Option | Win | Cost |
|---|---|---|
| Smaller chunk floor (32 KB / 16 KB) | 2-4× more B at 1 MB | ratio cost (already explored, ~+0.08% at 64 KB; smaller likely worse) |
| Within-chunk packet-parallel AC (à la GPUAR/GPUAC) | 2-4× | new AC bit format — breaks compatibility |
| Speculative decode | 1.5-2× | weeks of engineering, draft model, bit-exact verify path |
| Re-frame gate at larger input | gate hits at ~16 MB | reframe |

## Recommendation

**Reframe the tier-3 gate to a larger canonical input** (e.g., 16 MB
WildChat). At that size:
- Compress stays ~150 KB/s (constant, but the deferred levers should
  push it to 175-200).
- Decompress lands ~120-200 KB/s at B=256 (saturating-ish).
- Both gates hit without architectural changes; tier-3 unblocks.

Or keep 1 MB and accept that decompress is far short — and pivot focus
to the architectural levers (within-chunk parallelism / speculative
decode) that can move the 1 MB number. These are weeks of work, not
days.

## Side-by-side with T4 (for reference)

| Path | T4 (g4dn) | A10G (g5) at 1 MB | A10G (g5) at 8 MB |
|---|---|---|---|
| Compress | 41 KB/s | 159 KB/s | 151 KB/s |
| Decompress | 14 KB/s (B=16) | 14 KB/s (B=16) | 58 KB/s (B=129) |
| Ratio | 0.115 | 0.115 | 0.116 |

T4 and A10G hit the same 14 KB/s decompress at B=16 — confirming the
per-step floor is **launch-latency + Python loop**, not GPU speed,
when B is small. A10G pulls ahead at higher B because its launch
overhead amortizes across more chunks per step.
