# Peer-codec implementation notes — Bellard / NNCP / RWKV ecosystem

Notes for **T3.S1** (Step 1 of the ts_zip-informed plan in `V1_PLAN.md`).
Output of ~1 day of reading peer codecs to inform our persistent-kernel +
int8 work in Steps 2-3.

**TL;DR:**
- **Bellard's published material is not a useful primary source for
  implementation.** ts_zip + LibNC + ts_server + NNCP (papers) all
  document the WHAT (model, ratio, throughput) and WHY (architectural
  rationale), but never the HOW (kernel boundaries, state layout, int8
  layout, host-sync avoidance). LibNC + ts_zip are closed-source
  commercial. NNCP papers describe Transformer-XL training, not GPU
  kernel design.
- **The actionable patterns came from `harrisonvanderbyl/rwkv-cpp-accelerated`**
  — open-source 8-bit RWKV-4 in CUDA/HIP/Vulkan. Same model class as
  our RWKV-4-Pile-169M; same quantization target (uint8 weights);
  has the kernel patterns laid out in code we can read.
- **Single biggest takeaway:** uint8 weights with **per-input-channel**
  scale + offset, dequantized inline inside the matmul. 4× HBM read
  reduction on weights. Ships as a *new model_id* (bitstream changes),
  not transparent v1 change.

---

## 1. Sources read

| source | useful? | what's there |
|---|---|---|
| `bellard.org/ts_zip/` | ❌ low | Product page. Confirms RWKV-169M-v4, 8-bit weights, BF16 evaluation, **1 MB/s on RTX 4090**. No HOW. |
| `bellard.org/libnc/` + `libnc.html` | ❌ low | C tensor library, custom 500 MB CUDA chunk allocator, Ampere+ only (for hardware bf16), float32 + bfloat16 supported. No quantization details, no kernel layout. |
| `bellard.org/ts_server/` + `ts_server.html` | ❌ low | Server announces "custom 8/4/3-bit quantization" via `ncconvert -q bf8 / bf4`. Quantization API documented (CLI), but layout details not. |
| `bellard.org/nncp/nncp_v2.1.pdf` | ❌ low | NNCP Transformer-XL paper. Architecture (model + training), throughput on RTX 3090 (3.25 kB/s for enwik9 base, 1.94 kB/s large). **Note: NNCP v2 implemented in PyTorch**, not LibNC — predates ts_zip. |
| `harrisonvanderbyl/rwkv-cpp-accelerated` (`include/rwkv/cuda/rwkv.cu`, 729 lines) | ✅ **high** | Concrete kernel patterns. Project is "dead, try RWKV.CUH/HPP/CPP" but the int8 layout + 3-way fused matmul + WKV kernel is exactly what we need. |

---

## 2. What Bellard's published material confirms

**Throughput points:**
- ts_zip: **1 MB/s compress on RTX 4090** with RWKV-4-Pile-169M (8-bit weights, bf16 eval).
- NNCP v2 large (Transformer-XL): **1.94 kB/s on RTX 3090** for enwik9.
  *NNCP is much slower than ts_zip — different model class (Transformer-XL with retraining vs RWKV with fixed weights).*

**Architecture choices that matched ours:**
- 8-bit weight quantization, fp16/bf16 evaluation.
- LibNC custom allocator: 500 MB chunks. Bulk allocation amortizes
  cudaMalloc cost. (Our PyTorch-based path inherits torch's allocator
  which already does this — not a delta.)
- Ampere+ requirement for hardware bf16 — same constraint we just hit
  with cp.async (sm_80+). A10G qualifies.

**Architecture points where we differ:**
- Bellard uses **bf16 evaluation**. We use **fp16 + fp32 accumulation**
  (in `det_matmul_tc*`). bf16 has wider dynamic range, less precision —
  may help numerical stability of WKV recurrence at higher layer counts.
  Worth measuring on our model.
- Bellard supports 4-bit and 3-bit too. We have not explored these.

**Performance gap math (us vs ts_zip on same RWKV-4-169M):**
- ts_zip: 1 MB/s on RTX 4090 (~83 TFLOPS fp16, ~1 TB/s HBM)
- krunch: 0.137 MB/s on A10G (~125 TFLOPS fp16 peak via TC, but
  effective much less; ~600 GB/s HBM)
- Hardware-normalized (compute): ts_zip RTX 4090 ≈ 0.66× our A10G;
  hardware-normalized 7.3× absolute / 0.66 = **ts_zip is ~11× more
  efficient per-FLOP on the same model class**.

That 11× gap is engineering: int8 weights (4× HBM savings on the
~85% of layer cost that's matmul) + persistent kernel discipline
(eliminates per-op launch + state transfers). Both are achievable
in our codebase.

---

## 3. What Bellard does NOT document (and we shouldn't expect to)

- Inference kernel architecture (kernel boundaries, persistent kernels)
- Host-sync avoidance patterns
- State buffer layout in HBM
- Quantization layout details (per-channel vs per-tensor, zero-point
  storage, dequantization placement, accumulator precision)
- Operator fusion patterns
- Specific code-shape examples beyond `nc_add()` / `nc_mul()` API skeletons

This is consistent with closed-source commercial software. **Reading
LibNC source was not possible** — it ships as a `libnc_cuda.dll`,
not source. **No GitHub mirror found** in search.

**Implication for our plan:** the ts_zip section of V1_PLAN's "1 day
reading the source" can't actually deliver source-of-truth from
Bellard. Substitute: ~1 day reading the open-source equivalents
(`harrisonvanderbyl/rwkv-cpp-accelerated` + BlinkDL's RWKV-CUDA).

---

## 4. Patterns extracted from `rwkv-cpp-accelerated` (the actionable stuff)

Source: `include/rwkv/cuda/rwkv.cu` (729 lines), 8-bit RWKV-4 kernels
in CUDA/HIP/Vulkan. Targets the same model we use.

### 4.1 Int8 weight layout — `kernel_mm8_threec`

**Weight format:** `uint8_t w[K, N]` row-major.

**Per-input-channel scale + offset (fp32):**
- `r[j]` — scale, fp32, one per input channel j (length K)
- `o[j]` — offset (zero-point baked in), fp32, one per input channel j

**Dequantization formula (inline in matmul):**

```
y[k] += input[j] * (w[j*N + k] * r[j] + o[j])
```

**Critical detail:** scale + offset are **per-INPUT-channel** (j-axis),
not per-output-channel (k-axis). This is a SIMPLER scheme than typical
matrix-quantization libraries (which usually quantize the output axis).
Cost: less precision per output dim. Benefit: dequantization is
constant-time per-input-channel; scale/offset arrays are tiny (K floats
each, vs K*N for tensor-wide quant).

**Memory savings on weights:**
- Per layer matmul (n_embd=768, n_att/n_ffn): weights are 768×N halves
  in fp16 path = 1.5 MB (N=768) or 4.7 MB (N=3072).
- In uint8 path: 768×N bytes + 768 fp32 (3 KB) scale + 768 fp32 (3 KB)
  offset = 0.75 MB + 6 KB ≈ **0.5× of fp16**. 2× HBM savings on
  weight reads.
- Wait — that's 2× not 4×. Because uint8 is half of fp16 (2 bytes), so
  read traffic is 2× less. The "4× HBM savings" claim from ts_zip is
  vs fp32 baseline, not fp16. Our baseline is fp16 → 2× savings.

**Tile sizing (this implementation):**
- `MM8_ONE_TILE = 256` → blockDim.y = 256
- `MM8_ONE_JSPLIT = 16` → input dim split into 16 chunks
- Cross-chunk reduction via `atomicAdd` to output `y[k]`

This is a different tile geometry than our WMMA kernels. **Note:**
the `rwkv-cpp-accelerated` matmul is *scalar* (per-thread fp32
accumulation), NOT WMMA tensor-cores. We can do better by combining
int8 weights with WMMA accumulation (NVIDIA's mma.s8 instructions
exist for sm_80+).

### 4.2 3-way fused matmul (KVR-shaped)

`kernel_mm8_threec` does THREE matmuls in ONE kernel launch:

```cuda
for (j = j0; j < j1; ++j) {
    y_local  += xy[j +     token*N*3] * (w [j*N + k] * r [j] + o1[j]);
    y1_local += xy[j + N + token*N*3] * (w1[j*N + k] * r1[j] + o2[j]);
    y2_local += xy[j + N*2 + token*N*3] * (w2[j*N + k] * r2[j] + o3[j]);
}
```

- `xy` is shape `[tokens, N*3]` — **kx, vx, rx concatenated** in memory
- Three weight matrices `w, w1, w2` all loaded in the same iteration
- One launch, three results (`y, y1, y2`)

Our existing `det_matmul_tc_3way` does the same fusion but with
**fp16 weights and WMMA**, not int8. Combining 3-way fusion with int8
+ WMMA is a clear extension.

### 4.3 RWKV state buffer layout

```cuda
__global__ void kernel_wkvc_forward(
    const unsigned long long C,
    const float *__restrict__ k, v, r,
    double *__restrict__ y, _aa, _bb, _pp,
    const double *__restrict__ u, w,
    ...) {
    for (token = 0; token < tokenlength; token++) {
        for (c = 0; c < EMBBLOCK; c++) {
            unsigned long long stateoffset = ii + offset*C;          // PARALLEL mode
            // OR
            stateoffset = ii + offset*C + token*C*layers;             // SEQUENTIAL mode
            double aa = _aa[stateoffset];
            double bb = _bb[stateoffset];
            double pp = _pp[stateoffset];
            ...
        }
    }
}
```

**Key state-layout observations:**
- `aa, bb, pp` use **fp64 (double)** for state — different from BlinkDL
  (which uses fp32). Higher precision for the WKV recurrence pp/aa/bb
  accumulators reduces drift across long sequences. **Worth measuring
  if our fp32 state introduces drift over many tokens.**
- State layout is `[B][layers][C]` row-major: `stateoffset = ii + offset*C + token*C*layers`.
  Layer index `offset` in the middle. Allows easy state slicing per
  layer. Our `cpp_path.fresh_state_batched` uses
  `[B, n_embd]` per-layer separately — equivalent semantically, slightly
  more allocation overhead.
- Per-channel WKV math is **inline scalar** (one thread per channel).
  No tensor-core opportunity — WKV is inherently per-channel scalar.
  Same as our `wkv_kernel.cu`. **No win available here.**

### 4.4 Other kernels (small ops)

- `cuda_layernorm`: one block per token, EMBBLOCK=16 elements per
  thread per iter. Two-pass (mean + std pre-computed externally,
  then this kernel applies). **Pre-computed mean/std** is our own
  `fused_pre_attn`-style optimization, validates that approach.
- `cuda_relusquared`: simple elementwise.
- `sigmoid`: simple elementwise.
- `setx`, `cuda_memset`: state copies, trivial.

**No persistent kernel.** Each op is a separate kernel launch, like
our current `cpp_path`. So `rwkv-cpp-accelerated` does NOT inform the
"persistent kernel" half of the V1_PLAN ts_zip section. That part
remains theoretical / Bellard-attributed.

---

## 5. Implications for krunch

### 5.1 Confirmed direction (ts_zip section of V1_PLAN aligns)

- **Persistent kernel for layer step (Step 2)** — independently
  motivated by our own profiling (T3.7 found 200+ launches per token
  is real overhead). Bellard claims this is the right architecture
  but published source doesn't show it. **Implement based on our
  own profile data + standard CUDA techniques (cooperative_groups
  grid sync, our existing `rwkv_step_v2.cu` scaffold), not as a
  Bellard port.**
- **Int8 weight quantization (Step 3)** — directly informed by
  `rwkv-cpp-accelerated`. Layout is documented above; integration is
  ~1 week of focused work.

### 5.2 Concrete int8 implementation sketch

Per-input-channel uint8 weights with fp32 scale + offset, dequant
inline in matmul, fp32 accumulation:

1. **Weight conversion script** (Python, one-shot at init):
   - For each weight `W[K, N]` in fp16: compute per-input-channel
     min/max, derive `scale[j] = (max[j] - min[j]) / 255`,
     `offset[j] = min[j]`. Quantize:
     `W_q[j, k] = round((W[j, k] - offset[j]) / scale[j])`.
   - Store `W_q` as `uint8 [K, N]`, plus `scale[K]` and `offset[K]`
     as fp32. Roughly 0.5× HBM weight footprint vs fp16.

2. **CUDA kernel** (new file `det_matmul_tc_int8.cu`):
   - Same 64×64 / 4-warp / cp.async layout as
     `det_matmul_tc_async.cu`. Inputs: `A` fp16 [M, K], `W_q` uint8
     [K, N], `scale`/`offset` fp32 [K]. Output: fp16 or fp32 [M, N].
   - In K-loop, load fp16 A tile (cp.async), load uint8 B tile +
     fp32 scale/offset chunk (cp.async). Dequantize to fp16 in shared
     mem before WMMA, OR dequantize per-thread inline before mma.sync.
   - HBM saved: weight reads halved. Compute: same as fp16 WMMA.

3. **Bit-stability**: per-input-channel scale/offset is shape-invariant.
   Dequant math is deterministic. → **bit-identical across M**.

4. **Bitstream change**: int8 weights produce different probabilities
   than fp16 weights → different AC bitstream → **NEW model_id (v1.x)**.
   Per V1_PLAN's hard constraint: do not silently change v1 bitstream.
   Ship as `model_id=2` alongside `model_id=1` (current fp16). Image
   bundles both for backwards compat.

### 5.3 What `rwkv-cpp-accelerated` does NOT inform

- Persistent kernel discipline — it doesn't have one.
- WMMA / Tensor Core integration with int8 — it uses scalar fp32
  matmul. Combining int8 + TC is our own extension. NVIDIA's
  `mma.sync.s8` PTX gives int8 multiply with int32 accumulate, but
  fp16 input × int8 weight needs explicit dequantize-to-fp16 first
  (no native mixed-type mma exists for this combination).
- Arithmetic coding / CDF organization — not relevant to its codebase.

### 5.4 Open questions worth a focused experiment

1. **Does int8 dequantize-then-fp16-WMMA actually beat fp16 WMMA?**
   Theoretical: 2× HBM weight savings, same compute. If memory-bound
   (which our profile suggests at the matmul level), yes. Microbench
   first before integrating.
2. **Per-input-channel vs per-output-channel scale**: which gives
   smaller drift on RWKV-4-169M's actual weights? Per-input-channel
   is what `rwkv-cpp-accelerated` uses; ts_zip's choice is undocumented.
   Worth measuring KL-divergence vs fp16 baseline on a 10 MB sample.
3. **Does fp64 WKV state (per `rwkv-cpp-accelerated`) measurably help
   numeric stability** vs our fp32 state on long contexts? Our 64 KB
   chunks have ~16K tokens. Drift over that length might be
   meaningful or negligible — measurement question.
4. **bf16 vs fp16 matmul**: Bellard uses bf16. We use fp16.
   bf16 has 8 exponent bits (vs fp16's 5) — better dynamic range,
   less precision. May help RWKV's long-tail logit values. A10G
   supports bf16 matmul natively. Worth a microbench.

### 5.5 Honest framing

- ts_zip's 7-11× absolute / hardware-normalized lead is engineering,
  not algorithmic magic. Replicable in our Apache-2.0 codebase.
- NOT replicable in v1 transparently — requires bitstream change
  (new model_id). v1 ships fp16 path; int8 ships as v1.x via new
  model_id alongside.
- Even matched on ratio + 2× compress speed, the krunch UX win
  remains: library-shaped + batch-deployable + Apache OSS. ts_zip
  is single-binary commercial.

---

## 6. Recommended next steps

In priority order, all bit-identical with fp16 baseline (no v1
bitstream change required):

| step | scope | predicted lift | days |
|---|---|---|---|
| **(a)** Persistent kernel for layer step (extend `rwkv_step_v2.cu` with WMMA + cp.async + B>1, finally landing the lever-4 work) | own design + cooperative_groups grid sync; reuse `det_matmul_tc_async.cu` patterns | 1.5–2× compress, 2–4× decompress | 5–7 |
| **(b)** bf16 microbench: replace fp16 in matmul path, measure ratio drift + speed | swap dtype in `det_matmul_tc*.cu`, run E2E | 1.05× speed, neutral ratio if works | 1 |
| **(c)** WKV state precision experiment: fp64 aa/bb/pp, measure drift | swap fp32 → fp64 in `wkv_kernel.cu`, run AC roundtrip on 1 GB sample | unknown — diagnostic | 1 |

### (b) results — A10G g5.2xlarge sm_86 (2026-05-02)

Built `det_matmul_tc_bf16.cu` (direct port of `det_matmul_tc_async.cu`
with `__nv_bfloat16`). Per-matmul speed and numeric drift vs fp32 ref:

| shape (M·K·N) | fp16 ms | bf16 ms | bf16 speedup | bf16 vs fp16 max-abs |
|---|---|---|---|---|
| 1024·768·768 | 0.065 | 0.052 | **1.24×** | 2.4e-3 |
| 1024·768·3072 | 0.206 | 0.157 | **1.31×** | 2.9e-3 |
| 1024·3072·768 | 0.244 | 0.194 | **1.26×** | 4.9e-3 |
| 128·768·768 | 0.030 | 0.026 | 1.15× | 2.0e-3 |
| 128·768·3072 | 0.044 | 0.038 | 1.16× | 2.4e-3 |
| 128·3072·768 | 0.110 | 0.095 | 1.16× | 4.9e-3 |

**Predicted 1.05× — actual 1.16-1.31×.** Better than expected. Compress
(M=1024) averages ~1.27× per-matmul; decompress (M=128) ~1.16×. If
matmul is ~70% of forward wall, E2E lift if all layer matmuls ported:
~1.18× compress, ~1.11× decompress.

**Numeric drift:** bf16 vs fp16 max-abs is 2-5e-3 (10× looser than fp16
vs fp32 — expected, bf16 has 7 mantissa bits vs fp16's 10). Whether
this stays ratio-neutral on real text needs an E2E encode test on
WildChat. Bytes diverge from v1 codec by definition → ship as v2
model_id `RWKV169M_bf16` if E2E ratio holds.

**Status:** microbench done. Wiring into the layer path (encode + decode
both use bf16 matmuls; weights stored bf16 instead of fp16) is the
follow-up. Predicate for proceeding: the E2E ratio test.

After (a) lands, run end-to-end on A10G; if compress < 200, do (b)+(c).
Only after that, consider int8 (Step 3) — which is bigger work AND
ships under new model_id.

---

## Appendix: copy of key kernel for reference

```cuda
// From rwkv-cpp-accelerated/include/rwkv/cuda/rwkv.cu (Apache-2.0).
// Three matmul accumulators in one launch, uint8 weights, fp32
// per-input-channel scale + offset, atomicAdd for cross-chunk reduction.
__global__ void kernel_mm8_threec(
    const unsigned long long N,
    const float *__restrict__ const xy,
    const uint8_t *__restrict__ const w,  const uint8_t *__restrict__ const w1, const uint8_t *__restrict__ const w2,
    const float   *__restrict__ const r,  const float   *__restrict__ const r1, const float   *__restrict__ const r2,
    const float   *__restrict__ const o1, const float   *__restrict__ const o2, const float   *__restrict__ const o3,
    float *__restrict__ const y, float *__restrict__ const y1, float *__restrict__ const y2,
    unsigned long long offset, unsigned long long tokenlength)
{
    for (unsigned long long token = 0; token < tokenlength; token++) {
        const unsigned long long k  = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned long long j0 = min(N, blockIdx.x       * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
        const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
        if (k < N) {
            float y_local = 0, y1_local = 0, y2_local = 0;
            for (long long j = j0; j < j1; ++j) {
                y_local  += xy[j        + token*N*3] * ((w [j*N + k + offset*N*N] * r [j + offset*N]) + o1[j + offset*N]);
                y1_local += xy[j + N    + token*N*3] * ((w1[j*N + k + offset*N*N] * r1[j + offset*N]) + o2[j + offset*N]);
                y2_local += xy[j + N*2  + token*N*3] * ((w2[j*N + k + offset*N*N] * r2[j + offset*N]) + o3[j + offset*N]);
            }
            atomicAdd(&y [k + token*N], y_local );
            atomicAdd(&y1[k + token*N], y1_local);
            atomicAdd(&y2[k + token*N], y2_local);
        }
    }
}
```
