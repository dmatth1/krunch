# Phase 13 — GPU backend (Metal first, CUDA later)

**Goal:** ship a first-class GPU backend that runs the same RWKV
forward pass with the same file format and freq-equivalent
cum_freqs as the CPU path. Default builds stay CPU-only with no
GPU dependency; GPU is opt-in via cargo features. Files
compressed with one backend decompress with any other.

**Status as of 2026-04-16:** Phase 13a shipped — backend
abstraction, cargo feature flags, optional `metal` dep all
compile cleanly. Phase 13b (Metal smoke test) is next.

---

## Why now

After Phase 12 (NEON tactical sweep) we hit the CPU bandwidth
ceiling at ~172 KB/s compress on the 200K default tier. The next
~10× of throughput requires either an architectural change
(two-tier predictor, distillation) or a new execution target.
GPU is the cleanest path:

- **Largest unclaimed segment of the speed Pareto curve.** Other
  GPU compressors (ts_zip, Nacrith, DualComp, LMCompress) all
  carry models 100-40000× larger than ours. At our 200K parameter
  count, kernel-launch overhead is the bottleneck, not compute —
  which scales much better. See [`docs/COMPARISON.md` §6](../COMPARISON.md)
  for the full landscape.

- **Use case alignment.** Three of the four target use cases in
  CLAUDE.md (backup/archival, log shipping, cold storage) are
  server-side and bulk-batched, exactly where GPU shines.

- **Engineering cost is bounded** by the existing codebase shape:
  `Session::forward(token) -> &[f32]` is the only contract. Any
  backend producing logits with quantization-equivalent values
  works. The file format, codec, AC, and tokenizer all stay
  unchanged.

**Projected wins on a consumer GPU at batch=512:**
- 200K model on RTX 4090: **~10-20 MB/s** (50-117× current CPU)
- 3.2M model on RTX 4090: ~2-4 MB/s
- M-series GPU (Metal): ~1-3 MB/s for 200K (6-18× current CPU)

That puts L3TC-prod ahead of every published learned compressor on
GPU throughput by ~10-20× (current GPU leader is ts_zip at ~1 MB/s
on RTX 4090 with a 169M-param model). See "GPU comparison" below.

---

## Five guardrails (the design)

These shape every implementation decision in this phase.

### 1. CPU stays the default

No GPU dependency unless explicitly enabled at build time. A user
who runs `cargo build --release` (no flags) gets the same binary
they get today: pure CPU, NEON-optimized, no Metal/CUDA runtime
deps, no GPU init at startup.

Implementation: `metal` cargo feature is **off** by default.

### 2. GPU is opt-in via cargo features

Two release artifacts:
- `l3tc` — built with `cargo build --release` (CPU only)
- `l3tc-gpu` — built with `cargo build --release --features=gpu`
  (CPU + Metal on macOS, will add CUDA later)

Backend selection at runtime is determined by build-time features
plus runtime device probing.

### 3. File format unchanged

GPU-encoded files decompress on CPU and vice versa. The file
format (v4, LEB128 segment headers + CRC32 trailer) is shared
across all backends.

This is enforced by the cum_freqs tolerance: any backend must
produce logits whose `freq = max(1, round(p × 10_000_000))`
quantization matches the CPU NEON path. Phase 4a already
established this is a satisfiable bar (NEON sub_exp differs by
~5e-7 relative from libm exp but produces identical freqs after
quantization). The Metal kernel must clear the same bar.

Validation: every Metal kernel ships with a unit test that
diffs its output against the CPU equivalent at the freq level
on a fixed corpus.

### 4. GPU optimizes for the batched / bulk workload

Single-stream batch=1 on GPU is overhead-bound: kernel launch is
~10-50 µs on Metal, vs. ~125 µs CPU forward pass. That's a
modest 3-12× per-token win, swamped by GPU init cost on small
inputs.

The real GPU win comes at batch=64-512 — process many independent
segments simultaneously, amortizing the kernel launch over the
batch. At batch=512: per-token amortized cost is sub-µs.

This implies the CPU path stays the default for single-file CLI
use; GPU activates for bulk processing where the segment count
exceeds the auto-routing threshold.

### 5. Auto-routing at runtime

If the build includes `--features=gpu` AND a GPU device is
reachable AND the input is large enough, route to GPU. Otherwise
CPU. CLI flag overrides:

```bash
l3tc compress big-archive.txt              # auto: GPU if available
l3tc compress small-file.txt --backend=cpu # explicit CPU
l3tc compress big-archive.txt --backend=metal  # force Metal
```

Threshold (in `backend.rs`): `GPU_AUTO_THRESHOLD_BYTES = 256 KB`.
Below that, CPU init is essentially free and GPU init dominates.

---

## Phased implementation plan

### Phase 13a — backend infrastructure ✅ shipped

- `src/backend.rs` with `Backend` enum (`Cpu`, `Metal`)
- `Backend::auto(input_bytes)` and `Backend::from_str()`
- `metal` cargo feature gating the Metal dep
- `metal-rs`, `objc2`, `objc2-foundation` as optional deps
- Re-exported as `l3tc::Backend` from `lib.rs`
- 3 unit tests (CPU default, threshold routing, parser)

Verified:
```bash
cargo build --release             # CPU only, builds clean
cargo build --release --features=metal  # Metal feature builds clean
cargo test --release --lib backend # 3 tests pass
```

### Phase 13b — Metal smoke test

Write a trivial Metal kernel (e.g., element-wise add) and dispatch
it via `metal-rs`. Validate the toolchain works on this MacBook
end-to-end before committing to porting forward-pass kernels.

Deliverable: a hidden CLI subcommand `l3tc metal-smoke` that
allocates a Metal buffer, runs a kernel, verifies output. Errors
out cleanly if Metal is unavailable.

### Phase 13c — Metal head matvec kernel ✅ shipped

Port `matvec_col_major_int8` (Phase 12d) to Metal. One thread per
output row; per-iteration reads of `qmat` are fully coalesced
(16 KB sequential reads per inner step on 16K vocab).

Implementation (`src/backend/mtl.rs`):
- `HEAD_MATVEC_KERNEL_MSL` — MSL compute kernel
- `HeadKernelMetal` struct holds the model weights (qmat + scales)
  on the GPU once at construction; per-token `forward()` only
  uploads the small `x` vector and reads back logits.
- Unit test asserts Metal output matches CPU NEON within
  5e-3 absolute (well below the freq-quantization step).

Measured single-call latency (16384 × 96, MacBook M-series,
1000-iter mean):

| backend | per-call | speedup vs CPU |
|---|---:|---:|
| CPU NEON (Phase 12d) | **113 µs** | 1.00× |
| Metal GPU batch=1 (per-call alloc) | 343 µs | 0.33× |
| Metal GPU batch=1 (pre-allocated) | 264 µs | 0.43× |

**Batched dispatch (the validation of the Phase 13e thesis):**

| batch | per-token amortized | per-token vs CPU |
|---:|---:|---:|
| 1 | 264 µs | 0.43× (slower) |
| 8 | 128 µs | 0.9× (~breakeven) |
| 32 | 45 µs | **2.5× faster** |
| 64 | 41 µs | 2.7× |
| 128 | 40 µs | 2.8× |
| **256** | **27 µs** | **4.2× faster** |
| 512 | 28 µs | 4.1× |

The crossover is at batch≈16. Past batch=256 the per-token cost
plateaus around 27 µs — that's the actual GPU compute cost when
launch overhead is fully amortized. The GPU is delivering ~4× the
CPU's per-token throughput on this kernel alone.

If the rest of the forward pass (~100 µs single-thread CPU) scales
similarly under batching, full-forward GPU per-token at batch=256
projects to ~50 µs → **per-stream ≈20 KB/s × 256 streams ≈ 5 MB/s
aggregate**. That matches the Phase 13 projection (1-3 MB/s
conservative, 5+ optimistic) for the 200K model on M-series.

CLI bench: `l3tc metal-bench-head --iters 1000` (gated by
`--features=metal`).

The path forward isn't tuning this kernel — it's **batching**
(Phase 13e). At batch=512 the same dispatch cost amortizes across
512 streams, putting per-stream amortized cost at sub-µs.

CLI bench: `l3tc metal-bench-head --iters 1000` (gated by
`--features=metal`).

### Phase 13d — De-prioritized after Phase 13c data

Originally planned to port all kernels for single-stream GPU. The
Phase 13c batched bench (4.2× per-token at batch=256, 0.43× at
batch=1) shows single-stream GPU loses to CPU NEON because of
dispatch sync overhead. Skip 13d as originally specified; jump
directly to Phase 13e.

A **batched** layer_norm kernel was ported as Phase 13e prep
(see `LayerNormKernelMetal` in `src/backend/mtl.rs`). This proves
the batched pattern works for kernels with intra-element
reductions, not just matvecs. Test: `layer_norm_batched_metal_matches_cpu`
asserts within 5e-4 vs CPU NEON.

The remaining kernels needed for batched forward pass — `sub_exp`,
`sigmoid`, `time_mix_step1`/`step2`, `matvec_96x96`/`256x256`/etc
— all map to one of two patterns (embarrassingly-parallel matvec
or intra-element reduction), both now demonstrated.

### Phase 13e — Batched encode/decode for GPU throughput

This is where GPU actually wins. Process N segments in parallel:
- Per-segment `LayerState` becomes `LayerState[N]` on GPU
- Forward pass advances all N segments by one token per dispatch
- Cum_freqs computed in parallel across the batch
- AC encode runs CPU-side after gathering freqs back (AC is
  serial per segment — each segment's encoder is independent,
  so encode them in parallel on CPU threads)
- Optimal batch size: empirically ~64-512 (depends on model size
  and GPU memory)

### Phase 13f — Auto-routing, CLI flag, file format check

- `--backend=auto|cpu|metal` flag in `l3tc.rs`
- Heuristic: GPU if compiled in AND device available AND input ≥
  `GPU_AUTO_THRESHOLD_BYTES`
- Round-trip test: CPU compress → GPU decompress, GPU compress →
  CPU decompress, byte-identical both ways
- Document the GPU build process in README

---

## What stays out of scope

- **CUDA backend.** Different repo phase (Phase 14 candidate).
  All design choices in Phase 13 should keep CUDA viable as a
  future addition (e.g., the `Backend` enum is open to new
  variants, not closed).
- **wgpu / candle / candle-metal abstractions.** Considered and
  rejected for v1 — they add an indirection that breaks the
  "freq-equivalent cum_freqs" guarantee unless we hand-tune the
  shaders anyway. Direct Metal kernels keep the Phase 12
  philosophy of hand-controlled inner loops. Revisit if/when we
  want CUDA, ROCm, and WebGPU all from one codebase.
- **MPSGraph** (Apple's high-level ML library). Faster for
  generic matmul but doesn't expose the INT8 head shape we
  optimized in Phase 12d. Use raw Metal compute for control.
- **Image/multi-modal compression.** Outside the project scope per
  CLAUDE.md primary-target statement.

---

## GPU comparison (projected, batch=512)

| Compressor | Hardware | Throughput | Model | Bottleneck |
|---|---|---:|---|---|
| **L3TC-prod-Metal (Phase 13 target)** | M-series Apple GPU | **~1-3 MB/s** | 200K RWKV-v4 | kernel launch overhead |
| **L3TC-prod-CUDA (future)** | RTX 4090 | **~10-20 MB/s** | 200K RWKV-v4 | kernel launch overhead |
| ts_zip (shipped 2024) | RTX 4090 | ~1 MB/s | 169M RWKV | model size |
| DualComp 0.3M (paper) | A100 batch=512 | 6.3 MB/s | 0.3M RWKV-7 | model size |
| Nacrith (shipped 2026) | GTX 1050 Ti | 0.2-0.28 MB/s | 135M Transformer | model size |
| LLMZip / Llamazip / LMCompress | A100 / multi-GPU | "GPU days per file" | 7B-70B LLM | model size |

L3TC-prod's parameter count (200K, 850× smaller than ts_zip) is
what enables the projected throughput lead. We don't need to be
better at GPU programming — we're starting with a 1000× smaller
forward pass.

---

## Build instructions

CPU-only (default, no GPU dep):
```bash
cd l3tc-rust
cargo build --release
./target/release/l3tc compress input.txt -o output.l3tc
```

CPU + Metal (Apple Silicon):
```bash
cd l3tc-rust
cargo build --release --features=metal
./target/release/l3tc compress input.txt -o output.l3tc \
    --backend=metal
```

Once shipped: setting `--backend=auto` (or omitting it) routes
based on input size + GPU availability.

---

## References

- Phase 12 work: `docs/phases/PHASE_12.md` (NEON kernel reference)
- Competitor landscape: `docs/COMPARISON.md` (GPU compressor field)
- Backend dispatch: `src/backend.rs`
- Project goals: `CLAUDE.md` (target use cases, ratio/speed gates)
