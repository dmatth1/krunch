# Deferred krunch optimizations (parked 2026-05-06)

These were identified during T4 phase profiling but not pursued because
each is too small in isolation to justify engineering effort. Captured
here so they're not re-discovered the next time someone profiles.

If we ever land 3+ of these in a single sprint, the cumulative win
matters; individually none does.

## Universal levers (help both T4 and A10G+)

### 1. Fuse cumsum into `det_softmax_cdf` kernel
**Where:** `krunch_ac/cuda/det_softmax_cdf.cu` writes per-symbol counts;
`cpp_path.softmax_cdfs_per_row` then runs `torch.cumsum` as a separate
kernel. Comment in the kernel (`det_softmax_cdf.cu:181-184`) notes the
serial in-kernel cumsum was 19× slower than torch's tuned scan, so it
was reverted. A *parallel* in-block prefix scan (Hillis-Steele or
Brent-Kung over the 256-thread block, V=50277 stripes) would beat
torch's launch-overhead-bound scan.
**Win:** ~5–8 % on both compress and decompress wall (per-row cumsum
is one of the bigger items in "outside layer-step" on the phase profile).
**Effort:** ~1–2 days. The block-scan over V=50277 with 256 threads is
the non-trivial part; needs careful boundary handling.
**Risk:** low; bit-exactness requires the same scan order as the current
cumsum, which is left-to-right serial — easy to preserve.

### 2. Skip `empty_cache()` and reuse `output_buf`/`ac_state` across compress chunks
**Where:** `inference.py:357 _compress_chunk_cpp` calls
`torch.cuda.empty_cache()` after every chunk. `output_buf` and `ac_state`
are also re-allocated each chunk.
**Win:** ~2–4 % compress (16 chunks × ~20–50 ms of empty_cache + alloc
overhead per chunk). Universal.
**Effort:** ~3 hours. Need an allocator-cache mechanism on the engine
side that sizes once for the largest chunk, reuses across calls; or
expose `compress_chunks_batched_eager(chunks)` that does the loop.
**Risk:** very low. Worst case is slightly higher VRAM headroom.

### 3. Batch tokenization
**Where:** `inference.py:226 compress_chunk` calls
`self._tokenizer.encode(text).ids` per chunk. The HuggingFace
`tokenizer.encode_batch([texts])` parallelizes internally and is much
faster than 16 sequential calls.
**Win:** ~1–2 % compress. Universal.
**Effort:** ~2 hours. Refactor `compress_chunk` to accept pre-tokenized
input, batch-tokenize at the chunking layer.
**Risk:** very low.

### 4. Multi-step / speculative decode
**Where:** `_decompress_chunks_batched_cpp` per-step `.item()` sync to
feed next token (autoregressive). Floor today is ~1.16 ms/step on T4
B=16 just from this.
**Win:** would amortize the sync over N tokens. Hard to estimate;
maybe 1.5–2× decompress depending on draft model accuracy.
**Effort:** weeks. Architectural change, requires a draft model + a
verification path that reconciles speculative tokens with the AC
bitstream.
**Risk:** high — verification has to be bit-exact or AC roundtrip breaks.

## T4-specific levers (not universal — A10G+ already has cp.async)

### 5. Lift `N >= 16384` gate so `det_matmul_tc_mw` covers layer matmul shapes
**Where:** `layer_cpp.cpp:753` head_shape predicate. Currently `mw`
(multi-warp 64×64 WMMA, sm_75-compatible) only routes for the head
matmul (N=50277). Layer matmul (N=768/3072) falls through to
`det_matmul_tc` (single-warp, 16×16) on T4.
**Win:** ~1.5–1.7× T4 compress (microbench). Zero on A10G+ —
cp.async takes precedence for sm_80+ at these shapes.
**Effort:** ~2 hours. One predicate change + bit-exactness verification
across M (`scripts/test_det_matmul_tc_mw.py` exists for this).
**Risk:** low — the kernel is already shipping for head_shape, just
exercising different N values.
**Worth it if:** T4 perf becomes a real product target (currently it
isn't — gate hardware is A10G+).

### 6. Build `det_matmul_tc_3way_mw` (multi-warp KVR fusion, no cp.async)
**Where:** would be a new file in `krunch_ac/cuda/`. Today's
`det_matmul_tc_3way` is single-warp WMMA (sm_75 compatible);
`det_matmul_tc_3way_async` is multi-warp + cp.async (sm_80+ only).
There's no multi-warp-without-cp.async variant.
**Win:** ~1.3–1.5× T4 KVR matmul (fraction of layer_step). Zero on A10G+.
**Effort:** ~2 days — write the kernel, microbench, M-stability tests.
**Worth it if:** combined with #5; otherwise too narrow.

## Already shipped, awaiting validation

- **Graph-capture decompress** (commit fd5d6bc) — wired and bit-exact
  on T4, neutral perf there. **Expected 3–4× on A10G+** but unmeasured.
  Pending A10G test run. Default-on under `KRUNCH_OWN_WKV=1`.
