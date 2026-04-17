# Phase 13 — GPU backend (Metal first, CUDA later)

**Goal:** ship a first-class GPU backend that runs the same RWKV
forward pass on Apple Metal. Default builds stay CPU-only with no
GPU dependency; GPU is opt-in via cargo features.

**Status as of 2026-04-17:** **sub-phases 13a-13f shipped.** End-to-end
CLI compress/decompress via Metal works on real corpora with
bit-identical round-trip. Major correctness finding (bit) below.

**Phases 13g/13h/13i/13j shipped:** collapsing per-token GPU sync
overhead plus N-segment lockstep batching. Measured headline on
50 KB enwik6, L3TC-200K, M-series:

| phase | compress | change |
|---|---:|---:|
| pre-13h (lane-0 serial)        | 0.15 KB/s | baseline |
| 13h (8-lane lockstep)          | 1.12 KB/s | +7.5× |
| 13i (16-lane knee, sweep)      | 1.73 KB/s | +1.5× |
| **13j (GPU-resident + chained)** | **~5.0 KB/s** | **+2.9×** |

Cumulative **~33× over the original Phase 13e bring-up**, ratio
held at 0.1791 across all sub-phases, byte-identical round trip.
Still ~200× short of the 1 MB/s projection — the remaining gap is
per-step GPU wall time (each token still triggers ~40 encoder
dispatches inside the single command buffer; the dispatch-threads
setup overhead now dominates). Next levers are kernel fusion
(combine the elementwise glue kernels into one big per-token
kernel) and larger batch scaling on multi-MB inputs.

**Major caveat from Phase 13e:** files compressed via Metal must be
decompressed via Metal. Cross-backend interop (the original
guardrail) is not achievable because the GPU forward pass diverges
from CPU NEON by a few ULPs per layer, which shifts a handful of
freq-table entries at borderline `round()` boundaries. The file
header carries a `FLAG_GPU_ENCODED` bit (codec.rs) and the CLI
auto-routing reads it.

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

### 3. File format unchanged ⚠️ revised

**Original guardrail:** GPU-encoded files decompress on CPU and
vice versa.

**Actual outcome:** the file format itself is unchanged (v4,
LEB128 segment headers + CRC32 trailer), but a 1-bit flag
(`FLAG_GPU_ENCODED`) was added to the existing flags byte to
mark files produced by the GPU backend. Cross-backend interop
is NOT achievable: the GPU forward pass diverges from CPU NEON
by a few FP ULPs per layer, which at borderline `round()`
boundaries in cum_freqs shifts a handful of freq entries, which
desyncs the AC. Within a single backend the cum_freqs are
deterministic, so encode + decode using the same backend works
bit-identically.

CPU-encoded files (`FLAG_GPU_ENCODED` unset) decode on either
backend trivially because the CPU path is what produced them.
GPU-encoded files require a Metal-capable decoder.

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

### Phase 13e — Batched encode/decode for GPU throughput ✅ shipped

All seven batched Metal kernels covering the full forward pass +
cum_freqs are shipped, plus a `BatchedSession` orchestrator
(`src/backend/batched.rs`) and codec entry points
`compress_with_metal` / `decompress_with_metal` (codec.rs).

Kernels (`src/backend/mtl.rs`):
- `HeadKernelMetal` — INT8 head matvec, single + batched variants
- `LayerNormKernelMetal` — 3-pass mean/var/output reduction
- `Matvec96Metal` — square matvec for time_mix + channel_mix + short
- `SubExpKernelMetal` — `exp(a-b)` polynomial (matches CPU NEON
  coefficients exactly)
- `SigmoidKernelMetal` — safe `-|x|` form via `vbslq` select
- `TimeMixKernelMetal` — fused step1 + step2 (state evolution)
- `CumFreqsKernelMetal` — max + softmax + quantize_exps_to_freqs

Tests in `backend::mtl::tests` and `backend::batched::tests`
validate every kernel against its CPU equivalent within freq-
quantization tolerance (54 tests pass under `--features=metal`).

End-to-end test: `codec_metal_round_trip_50kb` — compresses the
50 KB enwik6 corpus via Metal, decompresses via Metal, asserts
bit-identical output. Result on a MacBook M-series:
    51200 bytes → 9172 bytes → 51200 bytes  ratio 0.1791

(Same ratio as CPU. Slow walltime: ~7 minutes due to the per-
segment serial design — proper N-segment batching is the next
follow-up; correctness is shipped.)

**FINDING (the design constraint that broke the original interop
guardrail):** GPU encode + CPU decode does NOT round-trip. The
GPU forward pass FP arithmetic diverges from CPU NEON by a few
ULPs per layer; at borderline `round()` boundaries in cum_freqs,
that shifts a few freq-table entries; the AC encoder/decoder
desync. Within a single backend the cum_freqs are deterministic
so encode/decode line up. Conclusion: **encode and decode with
the same backend.** The file header carries a `FLAG_GPU_ENCODED`
bit so decoders know which backend produced the file.

### Phase 13f — Auto-routing, CLI flag, file format check ✅ shipped

CLI now exposes the backend choice on both compress and decompress:

```bash
# Compress with the default CPU backend (current behaviour).
l3tc compress in.txt -o in.l3tc

# Opt into Metal (only on builds with --features=metal):
l3tc compress in.txt -o in.l3tc --backend=metal

# Decompress: --backend=auto (default) reads the file header's
# FLAG_GPU_ENCODED bit and picks the matching backend.
l3tc decompress in.l3tc -o in.txt
# auto-detected backend: metal

# Or force a specific backend explicitly:
l3tc decompress in.l3tc -o in.txt --backend=cpu
l3tc decompress in.l3tc -o in.txt --backend=metal
```

CPU-only builds error out clearly when given a GPU-encoded file:
"this build does not include Metal support — rebuild with
`--features=metal` to decompress GPU-encoded files".

Verified end-to-end: `cargo test --features=metal --lib` passes
all 56 tests. CLI smoke test: 4 KB enwik6 → Metal compress →
auto-detect Metal decompress → byte-identical.

### Phase 13g — Chained dispatch (eliminate per-kernel sync) 🚧 in progress

The Phase 13e bring-up shipped each Metal kernel as its own
`commit_and_wait` block. That made each kernel self-contained and
trivial to test, at the cost of one CPU↔GPU sync per kernel call.
For a 2-layer 96H forward pass that's roughly **30 GPU sync
barriers per token**, which dominates the per-token wall time
(~30 ms on M-series vs CPU's 0.18 ms — the ~5 MB/s aggregate
projected from the head microbench is gated on collapsing those
syncs).

The fix is to encode multiple stages into one `MTLCommandBuffer`
and call `commit_and_wait` once at the end of a logical pipeline.

**Step 1 — `CumFreqsKernelMetal::forward_batched` chained ✅ shipped**

The cum_freqs path was the cleanest place to validate the pattern:
3 kernels (max → softmax+sum → quantize) plus a tiny CPU-side
`scale = total / sum` step in between. Refactor:

- Promoted the scale step to its own MSL kernel
  (`scale_from_sum_batched` — one thread per lane). That removes
  the mandatory CPU stall between stages 2 and 3.
- All four encoders (max, softmax+sum, scale, quantize) now share
  one command buffer, with one `commit_and_wait` at the end.
- 3 syncs → 1 sync per cum_freqs call.

Correctness: existing `cum_freqs_batched_metal_matches_cpu` and
`batched_session_cum_freqs_matches_cpu` tests pass unchanged. End-
to-end `codec_metal_round_trip_50kb` round-trips bit-identically.
Wall-time on 8 KB Metal compress: 0.19 KB/s (vs ~0.15 KB/s prior;
small at this scope because cum_freqs is only ~3 of the 30 per-
token syncs).

**Step 2 — extend the pattern to the forward pass (deferred for now)**

The bigger remaining win requires that the rest of the forward pass
kernels chain too. Per-layer the natural groupings are:

- `time_mix`: 3 attention matvecs (att_k/att_v/att_r) + sigmoid
  + step1 + step2 + output projection — currently 8 separate
  commits, no CPU work between them except the elementwise
  `rwkv = r * a / b` after step2.
- `channel_mix`: ffn_r + sigmoid + ffn_k + (relu, square) + ffn_v
  — currently 5 commits.

Blocker: each kernel currently takes `&[f32]` slices and allocates
fresh GPU buffers per call. To chain into one cmd_buf we need:

1. `BatchedSession` to hold persistent Metal `Buffer` fields for
   per-token state (currently `Vec<f32>`).
2. Each kernel to expose a `dispatch_into(cmd_buf, &input_buf,
   &output_buf, ...)` API that encodes without committing.
3. A handful of CPU-side glue ops (embedding lookup, time_mix
   blends, relu/square, residual add) either move to GPU kernels
   too or stay CPU-side at the cost of mid-cmd-buf syncs.

Lower-priority now that 13h has unlocked most of the practical win;
revisit if the next throughput target requires it.

### Phase 13h — True N-segment lockstep ✅ shipped

`compress_segments_batched` and `decompress_segments_batched`
previously ran each segment serially through lane 0 of a
`BatchedSession`, leaving the other lanes idle (and giving up the
whole point of batching). 13h replaces both with chunked-lockstep
loops:

- Process segments in chunks of `batch_size` (default 8 from CLI).
- Per step inside a chunk, run a single `forward_batched` +
  `cum_freqs_batched` across all active lanes — one GPU dispatch
  sequence per step instead of N.
- Per-lane AC encoders own their own `Vec<u8>` outputs; per-lane
  AC decoders borrow each chunk's body slice. `ArithmeticEncoder`
  was already generic over `W: Write`, so no encoder refactor was
  needed — earlier comment to the contrary was a misread.
- Ragged segment lengths handled by skipping the per-lane encode
  step once `step >= seg.len()` while the GPU keeps running other
  lanes lockstep.

Wall-clock measurements on the 50 KB enwik6 / L3TC-200K combo:

| metric | pre-13h (lane-0 serial) | post-13h (8-lane lockstep) | speedup |
|---|---:|---:|---:|
| Metal compress | 0.15 KB/s | **1.12 KB/s** | **~7.5×** |
| Metal decompress | 0.15 KB/s | **1.12 KB/s** | **~7.5×** |
| Ratio | 0.1791 | 0.1791 | (held) |
| Round trip | byte-identical | byte-identical | ✅ |

Aligns with the expected `batch_size`× amortization. Larger batch
sizes (16, 32, 64) should keep scaling until per-call GPU compute
exceeds dispatch overhead — to be measured next. Auto-routing on
decompress reads `FLAG_GPU_ENCODED` from the file header and picks
the matching backend without user intervention.

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
