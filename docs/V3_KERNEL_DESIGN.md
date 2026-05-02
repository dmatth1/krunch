# rwkv_step_v3: design notes

Persistent fused layer kernel for RWKV-4 stepped (T=1) batched (B≥16)
forward. Replaces v2's scalar-GEMV approach with WMMA-based matmul +
cp.async double-buffered K-loop, while preserving the cooperative-launch
phase decomposition (LN → premix → KVR → WKV → Ow → LN2 → ffn-premix →
ffn_R/K/V).

**Why v3 not v2:** v2 was correct (fp16 noise vs reference) but slower
than `cpp_path.forward_stepped_batched` because every matmul phase used
scalar per-thread fp32 GEMV. With B=128 (production decompress), WMMA
delivers ~2× over MW + cp.async and ~4× over scalar. Per peer-codec
notes: matmul is ~88% of layer-step cost on A10G; persistent kernel
discipline (1 launch per layer instead of ~16 ATen ops) eliminates
the remaining launch overhead.

**Why not just CUDA-Graph cpp_path:** that ~~would~~ does work (post
T3.7 own_wkv fix removes the dispatcher-unsafe op). It's lever 1b
(T3.13) — ~2 hour win, modest 10–15% decompress lift. v3 is the
bigger swing: 1 launch per layer × 12 layers per token + WMMA
matmul + integrated WKV + integrated LN/sigmoid/premix.
Predicted 1.5–2× over post-T3.7 cpp_path.

---

## Constraints

| | |
|---|---|
| Stepped only (T=1) | Decompress production path is `forward_stepped_batched`. Compress packed (T=N) is a v3 follow-up; needs different tile schedule. |
| Batched B ≥ 16 | WMMA M-tile is 16. B<16 wastes A reads. Production B=64–256, all qualify. |
| sm_80+ | cp.async needs sm_80. T4 (sm_75) falls back via `#if __CUDA_ARCH__ >= 800` guard; v3 not bench-able on T4. |
| Bit-correct vs reference | Within fp16 noise (~1e-2). NOT bit-equal to v2 or cpp_path; encoder + decoder must use SAME v3 kernel for AC roundtrip. |
| n_embd=768, n_ffn=3072 | RWKV-4-Pile-169M only. Hardcode constants. |

---

## Block + warp layout

**Grid:** `<<<6, 128>>>` cooperative.
**Per-block:** 4 warps × 32 threads = 128 threads.
**Channel ownership:** each block owns 128 contiguous channels (768 / 6).
Each warp owns 32 channels of those (128 / 4).

**Why 6 × 128 over 8 × 256 or 12 × 128:**
- Cooperative launch needs all blocks resident. T4 (40 SMs)
  comfortably fits any of these; A10G (80 SMs) fits all.
- Per-channel ops want 1 thread per channel × all B → 768 threads × B
  per-(b,c) cells. 6×128=768 matches channels exactly.
- For matmul phases, 4 warps per block × 6 blocks = **24 warps total**.
  Output tiles: 8 (M=B/16=128/16) × 48 (N/16=768/16) = 384 frags. Each
  warp does 384/24 = 16 frags. Manageable per-warp register footprint.
- For ffn_K (N=3072): 8 × 192 = 1536 frags / 24 warps = 64 frags per
  warp. Larger but still fits. Or break into multiple K-loop passes.

**Trade-off rejected: 8 warps per block × 256 threads.** Doubles
register pressure (8 warps × 16 frags × 8 elem/frag/thread = 1024
fp32 per thread for accumulators). T4 max regs/thread is 256 — won't
fit. 4 warps × 16 frags × 8 = 512 fp32 per thread × 32 threads = 4 KB
register file per warp; 4 warps = 16 KB; 6 blocks per SM = 96 KB.
A10G has 64K registers/SM = 256 KB. Fits ~2 blocks/SM at full
occupancy. Acceptable.

---

## Per-call HBM scratch

```cpp
struct V3Scratch {
    // LN1 mean/inv_std, broadcast across blocks via grid sync
    float ln1_mean[B_MAX];      // 256 floats
    float ln1_inv_std[B_MAX];

    // After premix: kx, vx, rx [B, C] each, fp16. Source for KVR matmul.
    __half kx[B_MAX * 768];     // B × C halves
    __half vx[B_MAX * 768];
    __half rx[B_MAX * 768];

    // After KVR: k, v fp32 [B, n_att]; r_pre fp16 [B, C]
    float  k_acc[B_MAX * 768];
    float  v_acc[B_MAX * 768];
    __half r_pre[B_MAX * 768];

    // After WKV: y [B, n_att] fp32. Then r * y in fp16 → ry buffer.
    float  y[B_MAX * 768];
    __half ry[B_MAX * 768];

    // After Ow + residual: x_after_att [B, C] fp16. Source for LN2.
    __half x_after_att[B_MAX * 768];

    // LN2 mean/inv_std
    float ln2_mean[B_MAX];
    float ln2_inv_std[B_MAX];

    // After FFN premix: ffn_kx, ffn_rx [B, C] fp16
    __half ffn_kx[B_MAX * 768];
    __half ffn_rx[B_MAX * 768];

    // After ffn_R: r_ffn [B, C] fp16 (post-sigmoid)
    __half r_ffn[B_MAX * 768];

    // ffn_K output: [B, n_ffn] fp16 (post-relu²)
    __half k_ffn[B_MAX * 3072];
};
```

For B_MAX=256: ~5 MB scratch. Allocated once at engine init, reused
per-layer call. State (att_xx, aa, bb, pp, ffn_xx) is mutated in
place by the kernel; passed via separate tensor args.

---

## Phase decomposition

Same 11 phases as v2, but matmul phases (3, 5, 8, 9, 10) now use
WMMA instead of scalar.

```
Phase 1: LN1 — block-local read of x, block reduces sum + sumsq for
         each batch element; broadcast mean/inv_std via shared. Write
         normalized to scratch. Per-block work: 128 channels × B
         elements each. Each thread does (B/threads_in_block) × C/4
         cells worth.

Phase 2: time-mix premix — reads xx + att_xx, writes kx/vx/rx to
         scratch. Per-(b,c) scalar arithmetic. Each thread covers 1
         channel × all B in a loop. Update att_xx state in place.

[grid.sync 1 — commit kx/vx/rx]

Phase 3: KVR matmul (3-way fused, WMMA).
         M = B (16-aligned), K = 768, N = 768. Each warp owns
         16 frags; 24 warps × 16 = 384 frags = M_tiles × N_tiles.
         Three weight matrices loaded together (Bellard-style 3-way
         fusion via shared kx/vx/rx layout) → 3× HBM read amortization.
         cp.async double-buffered K-loop. Outputs fp32 k, v; fp16 r_pre.
         Apply sigmoid to r_pre in same phase (fuse).

Phase 4: WKV — per-channel-per-batch sequential recurrence. Each thread
         handles all B elements of its channel (since recurrence is
         independent per channel). Read aa/bb/pp from state, update
         in place. Compute y. Write y to scratch.

[grid.sync 2 — commit y]

Phase 5: r * y elementwise → ry, then Ow matmul (WMMA).
         Multiply r_pre × y (post-sigmoid r times wkv y) into ry.
         Then Ow @ ry: M=B, K=768, N=768 → out_att fp16.
         Add residual: x + out_att → x_after_att, write to scratch.

[grid.sync 3 — commit x_after_att]

Phase 6: LN2 — same as LN1 but on x_after_att.

Phase 7: FFN premix — same shape as time-mix premix; writes ffn_kx,
         ffn_rx. Update ffn_xx state.

[grid.sync 4 — commit ffn_kx/ffn_rx]

Phase 8: ffn_R matmul + sigmoid (WMMA, M=B, K=768, N=768).

Phase 9: ffn_K matmul + relu² (WMMA, M=B, K=768, N=3072).
         Larger N — more frags per warp. Output reused only once
         (next phase) so write to scratch.

[grid.sync 5 — commit k_ffn]

Phase 10: ffn_V matmul + final residual (WMMA, M=B, K=3072, N=768).
          Add r_ffn × ffn_v + x_after_att → x_out.

End. Total grid syncs: 5. State writes (att_xx, aa, bb, pp, ffn_xx)
happen in-place, no scratch round-trip.
```

---

## Matmul phase implementation (3-way fused, WMMA + cp.async)

Reuses patterns from `det_matmul_tc_async.cu`:

**Per warp:**
```cpp
constexpr int FRAGS_PER_WARP_KVR = 16;  // 8 M-tiles × 2 N-tiles in M-major

// Each warp's accumulators: 3 sets (one per K, V, R), 16 frags each.
wmma::fragment<wmma::accumulator, 16, 16, 16, float>
    acc_k[FRAGS_PER_WARP_KVR],
    acc_v[FRAGS_PER_WARP_KVR],
    acc_r[FRAGS_PER_WARP_KVR];
// Total: 3 × 16 × 8 fp32/thread/frag = 384 fp32 per thread = 1.5 KB.
// Fits in registers.
```

**Tile schedule mapping:**

Warp `wid` (global) computes:
- `frag_id` ∈ [0, 16): `m_tile = (wid * 16 + frag_id) / N_tiles`,
  `n_tile = (wid * 16 + frag_id) % N_tiles`.
- For B=128 (M=8 tiles), N=768 (48 tiles): 384 frags.
  Warp 0: frags 0-15 = (m=0, n=0..15). Warp 1: frags 16-31 =
  (m=0, n=16..31). … Warp 7: m=0, n=112..127 (out of bounds for
  N=48?). Hmm need to revisit.

Actually for B=128, N=48 tiles, total frags = 8×48 = 384. With 24
warps, each does 16 frags. Walk frags as (M, N) → simpler: each warp
owns (m_tile_block, n_tile_warp) where m_tile_block = wid >> 1
(0..11, but only 8 valid) and n_tile_warp = (wid & 1) * 16 + frag_id.

Better: stripe assignment. Frag `f` for warp `w` → `(w * 16 + f)` mod
total_frags. Each warp does `frags_total / 24` frags scattered across
the output. Simpler: contiguous chunks.

**K-loop (cp.async + double buffer):**

```cpp
__shared__ __half A_smem[2][TILE_M * TILE_K];   // kx (or vx, rx)
__shared__ __half B_smem_K[2][TILE_K * TILE_N]; // Kw
__shared__ __half B_smem_V[2][TILE_K * TILE_N]; // Vw
__shared__ __half B_smem_R[2][TILE_K * TILE_N]; // Rw

// TILE_M = 64 or 128 (covers multiple m_tiles per block)
// TILE_K = 16
// TILE_N = 64 (covers 4 n_tiles per warp's frag span)

// Per K-iter:
// - cp.async A_smem[next] from kx (M tile rows, K=16)
//   ALSO cp.async vx → A_smem_v, rx → A_smem_r (3-way fusion uses
//   3 separate A buffers, since 3 different inputs per matmul).
//   → 6 cp.async issues per buffer (3 A + 3 B). Doubled buffers = 12.
//   Shared mem usage: 6 buffers × TILE_M × TILE_K × 2 bytes ≈ 12 KB
//   for 128×16 tiles. Fits.
// - mma_sync on current buffer × 3 weight matrices = 3 × 16 frags
//   per warp.
```

**Shared-mem budget per block:**
- 3 A buffers (kx, vx, rx) × 2 (double-buffered) × TILE_M × TILE_K × 2
  bytes = 3 × 2 × 128 × 16 × 2 = **24 KB**
- 3 B buffers (Kw, Vw, Rw) × 2 × 16 × TILE_N × 2 = 3 × 2 × 16 × 64 × 2
  = **12 KB**
- Total: ~36 KB. Fits in T4's 64 KB / A10G's 100 KB per-block budget.

**Why 3-way fusion:**
- HBM traffic for input is amortized 3× (kx, vx, rx loaded once,
  used 3 times — wait no, they're 3 DIFFERENT inputs).
- Actually the ammortization is on K-loop overhead: one K-tile load
  pass services 3 matmul accumulators. Saves 2× redundant cp.async
  for A-side (since K, V, R matmuls share K-axis traversal).
- Plus saves 2 grid.sync calls (vs separate K, V, R matmul phases).

---

## Per-channel phases (LN, premix, WKV, sigmoid, residual)

These are inherently scalar — no matmul opportunity. Each thread
handles 1 channel × loops over B. Pattern (per phase):

```cpp
const int c = bid * 128 + tid;  // 0..767
for (int b = 0; b < B; b++) {
    // ... compute for (b, c) ...
}
```

Reads from scratch (HBM with L2 cache); writes to scratch / state.
This is the dominant per-channel computation per layer step.

**Optimization opportunity:** for layer-norm reductions, each block's
128 threads cooperate on the same B values. Use shared-mem + warp
reduce within the block. Similar to v2's existing pattern.

---

## State buffer layout

In place mutation by kernel:
- `att_xx[B, C]` fp16 — last-token-xx, per-batch-per-channel
- `aa[B, n_att]` fp32 — WKV state numerator
- `bb[B, n_att]` fp32 — WKV state denominator
- `pp[B, n_att]` fp32 — WKV state log-max
- `ffn_xx[B, C]` fp16 — last-token-xx2

Same as cpp_path's `fresh_state_batched`. No layout change.

**Note from peer-codec read:** rwkv-cpp-accelerated uses fp64 for
aa/bb/pp. We use fp32. For long sequences (>1K tokens), this might
introduce drift. **Diagnostic experiment** parked separately
(T3.S2c) — not blocking v3.

---

## Bit-correctness contract

v3 will NOT be bit-equal to:
- v2 (different reduction order, scalar vs WMMA)
- cpp_path forward_stepped_batched (different fusion, different ATen op
  precision)

v3 WILL be bit-equal between:
- v3 stepped (T=1, B=128) and v3 packed-T variant (when the latter is
  built in subsequent sessions). Required for AC roundtrip. Achieved
  by using SAME WMMA kernel + SAME tile schedule + SAME K reduction
  order regardless of T or M.

---

## Implementation milestones

| step | scope | session |
|---|---|---|
| **(M0)** Design doc (this file) | DONE | session N |
| **(M1)** v3 KVR matmul phase (standalone, `det_matmul_tc_3way_async`) | DONE — 1.23× over 3 separate async at M=128, integrated into cpp_path with M≥256 gate. A10G E2E: compress 141 → 152 KB/s. | session N |
| **(M2)** v3 LN1 + premix phases (B-batched) | DONE — `rwkv_step_v3.cu` cooperative kernel with phases 1+2 implemented; max-abs 3.9e-3 vs torch ref at B=128 (fp16 noise band). Phases 3-10 stubbed. | session N+1 |
| **(M3)** v3 WKV phase (port from v2) | DONE — Phase 4 inline WKV ports v2 math (per-(b, channel) sequential recurrence). T4 g4dn unit test: y/aa/bb/pp max-abs 1e-7 to 7e-7 vs torch ref at B=128 (threshold 1e-3). | session N+2 |
| **(M4)** v3 Ow + LN2 + FFN-premix phases | DONE — Phase 5 (Ow scalar matmul + residual), Phase 6 (LN2), Phase 7 (ffn-premix) all in scalar form. T4 g4dn unit test: x_attn 4.9e-4, ffn_kx/rx 9.8e-4, ffn_xx 1.95e-3 vs torch ref at B=128 (threshold 5e-3). | session N+2 |
| **(M5)** v3 ffn_R/K/V phases | DONE — Phase 8 (ffn_K matmul + relu²; thread covers 4 outputs in N_FFN=3072), Phase 8b (ffn_R + sigmoid), Phase 9 (ffn_V matmul). T4 g4dn unit test: K_act 3.9e-3, R_act 4.9e-4, V_out 3.9e-3 vs torch ref at B=128 (threshold 5e-2). | session N+2 |
| **(M6)** Full layer step + bit-correctness vs ref | DONE — Phase 3 (KVR scalar matmul) + Phase 10 (final residual) implemented; v3 kernel is now end-to-end self-contained. T4 g4dn full-layer test: x_out 3.9e-3, att_xx 4.9e-4, ffn_xx 3.9e-3, aa/bb/pp 1-3e-3 vs torch ref at B=128 (threshold 5e-2). Kernel correctness validated; performance is M7+ work. | session N+2 |
| **(M7)** Wire into cpp_path.forward_stepped_batched | engine integration | DEFERRED — requires WMMA |
| **(M8)** A10G E2E gate measurement | tests/gpu.sh on g5.xlarge | DEFERRED — requires WMMA |

**Perf reality check (session N+2):** scalar v3 at B=128 measures 34.7 ms/layer
on T4 g4dn.xlarge (full M6 kernel, 4 scalar matmuls dominate). For 12-layer
model that's ~415 ms/step. cpp_path on A10G achieves ~3 ms/12-layer-step
equivalent (152 KB/s compress at 64K chunk → ~26 μs/token × 12 layers ÷ 128
tokens batched). **Scalar v3 is ~100× slower than the existing cpp_path baseline.**

The cooperative-kernel architecture's value (avoid HBM round-trip on
intermediates) saves only kx/vx/rx/ffn_kx-like *small* tensors. Matmul
outputs (k_acc/v_acc/ffn_K_act) hit HBM regardless of fusion. Without
WMMA matmul phases, v3 cannot compete with the multi-launch cpp_path
that already uses WMMA for matmuls.

**Forward path:**
- Treat v3 as a correctness reference (validates against torch full-layer
  forward).
- Implementing WMMA inside v3's cooperative layout requires reinterpreting
  the channel-thread block layout as warp-tile layout during matmul phases.
  Estimated effort: 2-3 sessions.
- Alternative: keep cpp_path as the perf path, focus T3 effort on the
  bigger compress/decompress gap drivers (e.g., T3.3 decode-side
  warp-cooperative + uint16 CDF — decompress is at 47/200 KB/s, the bigger
  gate gap).

This session: M0–M1.

---

## Open design questions (defer)

1. **Frag tile assignment**: contiguous (warp 0 = frags 0–15) vs
   strided. Contiguous gives better cache locality for B-tile loads;
   strided gives better cache locality for A-tile (input). Pick after
   microbench.

2. **Output staging**: fp32 accumulators directly to scratch (`y`,
   `k_acc`, `v_acc`) avoiding fp16 round-trip vs fp16-cast + scratch
   write. cpp_path uses fp16 for r_pre and fp32 for k/v. Match that.

3. **WMMA accumulator broadcast across phases**: keep the matmul output
   in registers and do per-channel ops on it directly, vs round-trip
   through scratch. Hard with WMMA's per-warp layout. Default: scratch
   round-trip.

4. **bf16 matmul (per peer-codec note 4.4)**: A10G supports bf16 WMMA.
   Worth trying once v3 is functional. Bigger dynamic range may help
   long-sequence stability. Defer to M9.
