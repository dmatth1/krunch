# int8 weight quantization — design exploration (2026-05-02)

Step 3 of the ts_zip plan (V1_PLAN §"What to steal from peer codecs").
Quoted lift: **1.5-2×** on top of the persistent kernel. This is the
biggest single perf lever in the peer-codec notes, and the most
invasive — touches model loading, weight storage, and every matmul
kernel.

This doc scopes the work before we commit. After bf16 (1.16-1.31×
per-matmul, microbench done), int8 is the next-biggest lever.

## Reference — rwkv-cpp-accelerated `kernel_mm8_threec`

From `PEER_CODEC_NOTES.md` appendix:

```c
y[n] += x[k] * (uint8(w[k,n]) * r[k] + o[k])
```

Quantization scheme: **per-input-channel** symmetric. Each input channel
`k` has one fp32 scale `r[k]` and one fp32 offset `o[k]`. Weights stored
as `uint8`; dequant is inline in the matmul.

Algebraic restructure:

```
y[n] = sum_k x[k] * uint8(w[k,n]) * r[k]    +    sum_k x[k] * o[k]
     = sum_k (x[k]*r[k]) * uint8(w[k,n])    +    <bias_corr>
       └─ pre-scale x ──┘                        └─ scalar, broadcast across n ─┘
```

So the matmul is effectively `y[n] = (x_scaled @ W_uint8)[n] + bias_corr`,
where `x_scaled = x ⊙ r` and `bias_corr = sum_k x[k] * o[k]` is a single
scalar added uniformly to every output channel.

**Why per-input-channel and not per-output-channel?** Standard
quantization is per-output-channel (one scale per output, foldable into
the next layer's bias). Per-input lets the kernel pre-scale the input
once and then do a clean uint8 matmul, but it doesn't fold into bias as
cleanly. Bellard / rwkv-cpp-accelerated picked per-input — we should
match unless there's a specific reason not to (CDF determinism is
preserved either way; this is just a precision/perf tradeoff).

## Design choices

### 1. Storage layout

Per weight matrix `W: [K, N]` fp16, store:
- `W_uint8: [K, N]` uint8 — quantized weights
- `scale: [K]` fp16 — per-input-channel scale
- `offset: [K]` fp16 — per-input-channel offset

Memory: `K*N + 4*K` bytes vs `2*K*N` bytes fp16. For `K=N=768`:
`590KB + 3KB ≈ 593KB` vs `1.18MB` → **2× HBM reduction.**

For `K=768, N=3072`: `2.36MB + 3KB` vs `4.72MB` → 2× reduction.
For `K=3072, N=768`: `2.36MB + 12KB` vs `4.72MB` → 2× reduction.

Total model size (12 layers, 7 matrices each):
- fp16: ~150 MB
- uint8: ~75 MB

### 2. Calibration

Per input channel `k`, find scale and offset such that the dequantized
weights minimize MSE vs the original fp16 weights:

```python
# For each input channel k:
w_k = W_fp16[k, :]              # [N]
w_min, w_max = w_k.min(), w_k.max()
scale_k  = (w_max - w_min) / 255
offset_k = w_min
W_uint8[k, :] = round((w_k - offset_k) / scale_k).clamp(0, 255)
```

Asymmetric quantization per input channel. **Not** per-output —
deliberately matches rwkv-cpp-accelerated's choice for kernel simplicity.

One-shot calibration at model load. No per-input data needed (no PTQ
calibration set).

### 3. Kernel design — two paths

#### Path A: inline dequant + fp16 WMMA (recommended for v1)

Modify `det_matmul_tc_async.cu`:
1. Load uint8 weight tile (`TILE_K × TILE_N` bytes) via cp.async.
2. Per-input-channel scale + offset: small auxiliary tile (`TILE_K`
   fp16 each, 32 bytes for `TILE_K=16`).
3. In smem: dequant uint8 → fp16 elementwise:
   `W_smem[k][n] = float(W_u8[k][n]) * scale[k] + offset[k]`
4. Standard fp16 WMMA on the dequantized smem tile.
5. Pre-scale `x` once (outside matmul; `x_scaled = x ⊙ r` shape [M, K]).
6. Add bias correction scalar at the end.

**Win source**: HBM bandwidth (2× less weight data). cp.async pipelining
is unchanged. WMMA throughput unchanged.

**Cost**: dequant arithmetic per K-tile. With TILE_K=16, TILE_N=64,
that's 1024 ops per tile per block — negligible vs WMMA's 4096
mma_sync ops per tile.

**Predicted lift**: 1.5-1.8× (limited by dequant overhead + non-bandwidth
work).

#### Path B: int8 tensor cores (deferred to v2 if A is insufficient)

A10G has int8 mma.s8.s8.s32 → 2× throughput vs fp16. Would need:
- Quantize x to int8 too (per-step calibration — extra work)
- int8 mma → int32 accum
- Rescale int32 → fp32 with per-input scale + per-output bias
- More invasive; deferred unless Path A leaves perf on the table.

### 4. Bit-stability / model_id

Quantization changes weights → CDFs differ → bytes differ. This is a
**v2 model_id `RWKV169M_uint8`** ship.

Within v2 model_id, encoder + decoder must use the same uint8 weights
+ same dequant kernel order — easy by construction (use the calibration
output for both).

### 5. E2E ratio question

Quoted "1.5-2× speed, neutral ratio if works" assumes the quantization
preserves CDFs well enough that compressed size moves <1%. **Has not
been measured for our model + WildChat.** Required gate before shipping.

If ratio degrades >2-3%, consider:
- Per-output-channel quantization instead (more standard, possibly
  better-preserving)
- Skip int8 for layers with high outliers; mixed-precision
- 4-bit (bigger ratio hit) vs 8-bit + group quantization (smaller hit)

## Implementation plan

| step | scope | days |
|---|---|---|
| **1. Calibration script** — compute per-input-channel uint8 quantization for each layer matrix, save alongside fp16 weights | Python, one-shot | 0.5 |
| **2. det_matmul_tc_uint8.cu** — port of det_matmul_tc_async with inline dequant | CUDA | 1.5-2 |
| **3. Microbench uint8 vs fp16 / bf16** — same shape sweep as bf16 bench | Python on A10G | 0.5 |
| **4. Wire into layer_cpp.cpp routing** — toggle via KRUNCH_INT8=1 env, fall back to fp16 if quantized weights absent | C++ | 1 |
| **5. E2E ratio test** — encode WildChat sample with uint8, compare ratio vs current fp16 codec | Python | 0.5 |
| **6. v2 model_id wiring** — emit `RWKV169M_uint8` in blob header; update decoder routing | Python + format | 1 |

**Total: 4-5 days** before ship-ready. Vs bf16 which was 1 day.

## Open questions

1. **Per-output-channel vs per-input-channel** quantization. Bellard
   uses per-input. Does per-output preserve our specific model's logits
   better? Empirical question — cheap to A/B at calibration step.

2. **fp16 scale/offset vs fp32**. fp32 is precise but 8B/k extra in
   HBM. fp16 cuts that in half. Likely fine for our weight ranges
   (typical `~0.05` standard deviation per peer notes). Default fp16,
   confirm at calibration that the fp16-storage round-trip doesn't lose
   accuracy.

3. **Symmetric vs asymmetric** quantization. Asymmetric (offset != 0)
   uses the full 256-value range; symmetric fits in fewer bits but
   wastes range when weights aren't centered. RWKV layer weights are
   approximately zero-mean → symmetric should work, but asymmetric
   is the safe default and matches rwkv-cpp-accelerated.

4. **Dequant precision: dequant-to-fp16 vs dequant-to-bf16**. If we
   ship uint8 + bf16 matmul, that's a third stack: `uint8 weights →
   bf16 dequant in smem → bf16 WMMA`. Compounds the bf16 lift (1.16×
   on A10G micro) with the int8 bandwidth lift (~1.5-2×). Best total
   if both work; worst regression if either doesn't.

## Recommended next session

1. Write calibration script (`scripts/quantize_weights_uint8.py`).
   Emit `weights.uint8.bin` alongside the existing `weights.fp16.bin`.
2. Verify the quantization-roundtrip MSE is acceptable (weights look
   close enough). One-line: `(W - dequant(quantize(W))).abs().max()`.
3. Build first uint8 matmul kernel (Path A: inline dequant +
   fp16 WMMA). Standalone; not yet wired into layer_cpp.
4. Microbench uint8 vs fp16 on A10G: same 6-shape sweep as bf16. Goal:
   confirm 1.5-2× speed and acceptable numeric drift.

If steps 1-4 land and microbench shows the lift, proceed to wiring +
E2E. If kernel speed underwhelms (e.g., dequant overhead dominates,
<1.2×), reconsider Path B (true int8 tensor cores) or park int8 in
favor of other levers.

## Calibration validation (2026-05-02)

Built `scripts/calibrate_uint8_weights.py` and ran against the actual
RWKV-4-Pile-169M state dict. Per-input-channel scheme with `scale =
(max-min)/255, offset = min`. Results across all 12 layers × 7
quantizable matrices:

| matrix | avg max-abs err | avg rel-RMSE | worst rel-RMSE |
|---|---|---|---|
| att.key.weight | 1.17e-2 | 0.89% | 0.99% |
| att.value.weight | 1.11e-2 | 0.84% | 0.91% |
| att.receptance.weight | 8.56e-3 | 0.87% | 0.93% |
| att.output.weight | 2.36e-2 | 1.00% | 1.25% |
| ffn.key.weight | 7.84e-3 | 0.88% | 0.91% |
| ffn.value.weight | 1.24e-2 | 0.79% | 0.93% |
| ffn.receptance.weight | 1.16e-2 | 0.75% | 0.81% |

**All matrices quantize within 0.75-1.25% relative RMSE.** That's tight
— in the range where the resulting CDFs are very likely to round to
the same top-k symbols on text data, so compressed-size impact should
be small. (Final answer needs the E2E encode test.)

**Storage:** fp16 quantizable weights total 184 MB; uint8 + per-input
scale/offset (fp16) totals 92 MB → **1.99× reduction** confirmed.
(Total model is larger; quantizable layer weights are ~75% of fp16
weight bytes, so actual model storage drops 1.7× — still significant.)

`att.output.weight` is the worst-case (1.25% rel-RMSE). If E2E shows
this matrix as the precision-blocking one, can keep Ow as fp16 and
quantize the other 6 — gives ~85% of the storage win without the worst
of the precision hit. Standard mixed-precision fallback.

## Status

- ✅ Design scheme chosen (per-input-channel asymmetric, inline dequant
  + fp16 WMMA = "Path A")
- ✅ Calibration validated on real weights (~1% rel-RMSE, 2× storage)
- ✅ Kernel implementation (`det_matmul_tc_uint8.cu`)
- ❌ **Microbench: Path A is 1.4-2× SLOWER than fp16 on A10G** (see below)
- ⏸ E2E ratio test on WildChat — moot until perf path works
- ⏸ Wiring + v2 model_id — moot

## Microbench result — Path A is not viable as designed (2026-05-02)

A10G g5.2xlarge sm_86, on-demand. Quantization preserves output (max-abs
4.6-9.6e-3 vs fp16; quantization itself within design budget). But:

| shape (M·K·N) | fp16 | bf16 | uint8 | uint8 vs fp16 |
|---|---|---|---|---|
| 1024·768·768 | 0.065 ms | 0.052 ms | 0.094 ms | **0.69×** |
| 1024·768·3072 | 0.206 ms | 0.157 ms | 0.350 ms | **0.59×** |
| 1024·3072·768 | 0.245 ms | 0.194 ms | 0.366 ms | **0.67×** |
| 128·768·768 | 0.030 ms | 0.026 ms | 0.057 ms | **0.52×** |
| 128·768·3072 | 0.044 ms | 0.038 ms | 0.073 ms | **0.61×** |
| 128·3072·768 | 0.110 ms | 0.095 ms | 0.219 ms | **0.50×** |

**Predicted 1.5-2×; actual 0.50-0.69×.** Path A is the wrong design.

### Why Path A fails

The implemented kernel does:
1. cp.async load fp16 A → A_smem (good — overlaps with WMMA)
2. **Sync** load uint8 B → B_u8_smem (no overlap)
3. `__syncthreads`
4. Dequant B_u8_smem → B_smem fp16 (per-element multiply-add, 1024
   elements per K-tile, 48 K-tiles per output tile)
5. `__syncthreads`
6. WMMA on dequanted B_smem

Steps 2-5 sit on the critical path with no async hiding. The bf16 path
hides B's HBM-load latency behind WMMA via cp.async double-buffering;
my uint8 path inserts dequant + 2 extra block syncs inline. Net loss
~30 μs per matmul, which doubles M=128 totals and adds 50% to M=1024.

### Two ways forward, ordered by complexity

**Path A' (rework Path A):** Add cp.async for the uint8 B-load too,
restore double-buffering, and run dequant in compute phase overlapping
the next K-tile's cp.async. Estimated ~1-2 days. Best-case lift ~1.5×
(bandwidth ceiling); realistic ~1.2-1.4× given dequant overhead.

**Path B (int8 tensor cores):** A10G's `mma.s8.s8.s32` has 2× throughput
vs `mma.f16.f16.f32`. Inputs both quantized to int8 (need x quantized
per-step too — extra work), int32 accumulator, fp32 rescale at end.
Best-case 2× lift; ~3-5 days work. Higher engineering risk.

### Recommendation

**Park uint8 for v1.** bf16 already delivered 1.16-1.31× per matmul
with one day of work and bit-stable (bytes diverge but predictably).
uint8 needs another 1-2 days of kernel rework just to beat fp16, and
the ratio question (CDF drift on real text) hasn't been measured.

For v2 model_id, Path B is the more attractive uint8 direction: native
int8 tensor cores have a real 2× headroom. Path A' is incremental work
for an incremental win against bf16, which is already simpler.

**Files left in tree:**
- `det_matmul_tc_uint8.cu` (kernel, building, correct, slow)
- `bench_uint8_vs_fp16.py`
- `calibrate_uint8_weights.py`
- pybind `det_matmul_uint8` in `layer_cpp.cpp`

Keep them — useful starting points if v2 work returns to int8.
