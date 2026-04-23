# Tier-2 integration plan (revised 2026-04-23)

Spike 6 validated the architecture:
- **RWKV-4-169m-pile** + custom WKV CUDA kernel on A10G fp16
- Measured: **ratio 0.111, compress 330-430 KB/s**
- Beats zstd by 34%, bzip3 by 24%, Spike-5 L3TC-12M by 50%

This plan integrates that substrate into our existing `l3tc-rust`
binary. **Pure Rust, no Python inference server.**

## Why pure Rust (not Python + HTTP hybrid)

An earlier draft of this plan proposed a Python FastAPI server
wrapping BlinkDL's RWKV-LM, with a thin Rust codec acting as HTTP
client. That was expedient, not principled. Rejected because:

- Two processes + IPC overhead for what should be one library call
- Python adds a whole deployment surface (venv, pytorch install, CUDA
  driver coupling at runtime) that the Rust binary already solves
- Inference server auth/lifecycle/health becomes an ops concern
- Doesn't match the rest of `l3tc-rust`'s design (one binary, no IPC)

## Why NOT candle (Option C, investigated + rejected 2026-04-23)

- Candle has **RWKV v5/v6/v7 only**, no v4 — our pretrained weights
  aren't loadable
- Candle has **no fused WKV kernel** — its RWKV forward is a Rust
  `for t_ in 0..t` loop over generic tensor ops, which collapses
  to the same ~1-30 KB/s Python-fallback speed we measured in v2/v3
- Using candle means porting RWKV-4 AND writing the WKV kernel — we
  get no meaningful head-start vs building from scratch

## Reference architecture: `cryscan/web-rwkv`

[cryscan/web-rwkv](https://github.com/cryscan/web-rwkv) is a pure-Rust
RWKV v4/5/6/7 inference library with WebGPU backend and its own fused
WGSL shaders for the recurrence. 347 stars, active. `Ai00-X/ai00_server`
(610 stars) is a production inference server built on it.

**We use web-rwkv as a reference, not a dependency.** Things we copy:

- Model forward-pass structure (time-mix, channel-mix, layer norm
  ordering, residual shape)
- State management pattern (fixed-size recurrent state per layer,
  cheap to copy/reset)
- How the fused recurrence is expressed as a single kernel call
  (WGSL in web-rwkv, CUDA in our port)
- Weight loading conventions (they load `.pth` directly)

Things we do differently:
- **Backend**: CUDA via `cudarc` instead of WebGPU via `wgpu`.
  Reasoning: Spike 6 measured CUDA directly (330 KB/s on A10G) and
  our production target is AWS g5. WebGPU portability is nice but
  adds another dependency layer we don't need. The CUDA backend lives
  alongside our existing `src/backend/mtl.rs` and `src/backend/batched.rs`.
- **Scope**: one model (RWKV-4-169m-pile), one task (arithmetic-coded
  compression). No streaming generation, no sampling, no multi-model
  serving.
- **Integration**: direct call from `NeuralRwkvCodec::encode/decode`,
  no HTTP layer.

## Proposed file layout in `l3tc-rust`

```
src/
├── rwkv.rs                   # EXISTING: L3TC-12M forward for legacy path
├── rwkv_v4_pile.rs           # NEW: RWKV-4-169m-pile forward (reference: web-rwkv)
├── backend/
│   ├── mod.rs                # EXISTING
│   ├── batched.rs            # EXISTING: CPU
│   ├── mtl.rs                # EXISTING: Apple Metal
│   └── cuda.rs               # NEW: CUDA via cudarc
├── cuda/
│   ├── wkv.cu                # NEW: fused WKV recurrence kernel
│   └── build.rs              # NEW: nvcc invocation at build time
├── weights_pth.rs            # NEW: .pth parser (pickle + torch tensor)
├── dispatcher.rs             # ADD: NeuralRwkvCodec struct
└── arithmetic.rs             # EXISTING: AC encoder/decoder
```

## Phased build

### Phase 0 — Backend scaffold + smoke test (2-3 days)

Goal: prove we can call a fused CUDA kernel from Rust and get a
correct matrix multiplication back.

- Add `cudarc` dependency to `Cargo.toml` (feature-gated behind `cuda`)
- New `src/backend/cuda.rs` with `CudaDevice` struct, `malloc/memcpy/launch`
  wrappers
- Trivial `.cu` kernel (vector add), compiled via `build.rs` using
  `cc` crate + nvcc, linked into the binary as a PTX/fatbin
- CI target: compiles on linux-x86_64-cuda12 (AWS target) + skip on
  macOS (Metal path stays)

### Phase 1 — Weight loader + model scaffold (3-4 days)

Goal: load `RWKV-4-Pile-169M-20220807-8023.pth` into Rust tensors,
run a dummy forward pass (any output is fine; correctness comes Phase 2).

- `src/weights_pth.rs`: parse PyTorch `.pth` pickle format. Use
  `serde-pickle` or port the ~200 lines of web-rwkv's `loader.rs`
- Struct `RwkvV4Pile169mWeights { emb, blocks[12], ln_out, head }`
- `src/rwkv_v4_pile.rs`: `RwkvV4Pile169m` with `forward(tokens) -> logits`.
  Implement time-mix, channel-mix, WKV call, LN, residual — but the
  WKV kernel is a STUB that returns zeros. Just wiring for now.

### Phase 2 — WKV fused kernel (5-7 days, load-bearing)

Goal: match BlinkDL's wkv_cuda.cu within ~10% on A10G.

- Port `BlinkDL/RWKV-LM/RWKV-v4/cuda/wkv_cuda.cu` into
  `l3tc-rust/src/cuda/wkv.cu`. It's ~200 lines; license is Apache-2.0
  so direct port with attribution is fine
- `build.rs` invokes nvcc at compile time, embeds PTX via `include_bytes!`
- Runtime: `cudarc::driver::CudaDevice` loads the PTX, launches the
  kernel with proper grid/block dims
- **Validation**: byte-exact logits on 1024-token sequence vs Python
  RWKV-LM on same weights/tokens (tolerance 1e-3 due to fp16)

### Phase 3 — Tokenizer + AC integration + end-to-end (3-4 days)

Goal: full compress/decompress roundtrip, ≥300 KB/s on A10G.

- Use `tokenizers` crate to load `20B_tokenizer.json` (GPT-NeoX BPE)
- Wire existing `arithmetic.rs` encoder with per-token probability
  distribution from `RwkvV4Pile169m::forward()`
- New `NeuralRwkvCodec { model: Arc<RwkvV4Pile169m>, tokenizer: Arc<Tokenizer> }`
  in `dispatcher.rs`
- CLI: `l3tc neural-compress --model rwkv-4-169m-pile input.txt -o out.bin`
- End-to-end roundtrip test: 100 MB WildChat-English, byte-exact
  decompress == input
- Throughput benchmark: on g5.xlarge, measure compress + decompress
  KB/s; compare to Spike 6 Python numbers (target 330 KB/s compress)

### Phase 4 — Dispatcher integration + service wiring (2-3 days)

- Dispatcher adds `CodecTag::NeuralRwkv = 0x07`, picks per-chunk vs
  zstd / bzip3 / lz4 based on shortest output
- Update `cdk/docker/compression/compression_worker.py` to pass
  `--neural-rwkv-model` path to the rust binary
- GPU-backed compression worker: CDK task definition switches to
  g5.xlarge EC2 (not Fargate; Fargate GPU is ~4× more expensive)
- CloudWatch EMF counters: `NeuralRwkvEncodeMs`, `NeuralRwkvBytesIn`,
  `NeuralRwkvBytesOut`, codec selection histogram

### Phase 5 — Production soak + validation

- Run 1 GB WildChat-English through the full pipeline, compare
  compressed size to Spike 6 projection (~110 MB at 0.111 ratio)
- Dispatcher selection histogram: neural wins on how many chunks?
  What's the fallback rate to zstd/bzip3?
- Decode throughput (not measured in Spike 6): confirm ≥150 KB/s
  per-stream, ≥300 KB/s with chunk-parallel batching
- Cost model: $/GB compressed vs current zstd baseline

## Time estimate

| phase | days | risk |
|---|---|---|
| 0: Backend scaffold | 2-3 | low |
| 1: Weight loader + scaffold | 3-4 | medium (pickle format) |
| 2: WKV CUDA kernel port | 5-7 | medium-high (first real CUDA work in this repo) |
| 3: Tokenizer + AC + E2E | 3-4 | low |
| 4: Dispatcher + service | 2-3 | low |
| 5: Production soak | 2 | low |
| **Total** | **17-23 days** | |

Phase 2 is the load-bearing risk. Mitigations:
- Keep the Python RWKV-LM path working as a fallback throughout
  (Spike 6 validates that runs at 330 KB/s)
- Per-phase go/no-go: if WKV kernel port stalls on correctness,
  we have the option to temporarily wrap RWKV-LM via PyO3 as a
  stopgap while we debug (rejected for production but acceptable for
  unblocking Phase 3)

## Out of scope for Tier-2

- Per-tenant LoRA fine-tuning (v2; raw zero-shot is good enough to
  ship)
- Multi-GPU or multi-model serving
- WebGPU backend (defer; CUDA covers our immediate target; revisit if
  we expand to Apple Silicon GPU servers or AMD)
- RWKV v5/v6/v7 support (v4-only; only swap if a measured ratio win
  justifies the port cost)

## Success criteria

Ship Tier-2 to dev when all of these hold:
- [ ] `NeuralRwkvCodec` in dispatcher, byte-exact roundtrip on 1 GB corpus
- [ ] Compress throughput ≥300 KB/s on g5.xlarge A10G (gate from Spike 6)
- [ ] Decompress throughput ≥150 KB/s on g5.xlarge A10G (per-stream)
- [ ] Compressed ratio ≤0.12 on WildChat-English 1 GB (near Spike 6's 0.111)
- [ ] Rust binary builds in CI on linux-x86_64-cuda12; graceful skip on
      macOS + CPU-only builds
- [ ] No Python runtime dependency in the production path

## Open questions to resolve early

- **`.pth` parsing**: is there a maintained `serde-pickle`-based loader
  we can drop in, or do we need to port web-rwkv's loader? Phase 1
  blocker.
- **`cudarc` maturity**: current version supports CUDA 12.x? PTX
  loading at runtime? Kernel launch ergonomics?  Investigate Phase 0.
- **Nvcc-in-build.rs on CI**: GitHub Actions linux runners don't have
  nvcc by default. Either add a CUDA setup step or use a prebuilt
  CUDA container image for CI.
