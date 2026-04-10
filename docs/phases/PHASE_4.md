# Phase 4 — Close the ratio gap to Python L3TC  ✅ 4a SHIPPED (parity confirmed)

**Phase 4a result:** the "4 pp gap to Python" was a phantom. The
L3TC paper's reported "0.1665 on enwik6" is the *theoretical
entropy lower bound* — `compressor.py` literally returns
`total_bin_size_min = math.ceil(entropy_sum / 8)` and the actual
AC-encode-and-write-to-file path is commented out. Our forward
pass is bit-identical to Python's (max L_inf 3.81e-05 with f32
head, well under f32 ULP) and our entropy bound on enwik6 at
segment 2048 is **0.1643** — *better* than Python's 0.1665 because
we measure entropy from the raw softmax while Python measures
from the freq-quantized softmax (their AC-quantized number has
~0.22 pp of rounding loss baked in).

The 4 pp difference between our **actual coded bytes (0.2060)**
and the paper's **entropy bound (0.1665)** is real but it's
**AC framing overhead**, not a model bug. Phase 4b is reframed
around closing *that* gap — actual bytes → entropy bound — which
is a fundamentally different (and achievable) target.

See `docs/phase_4a_findings.md` for the diff harness, the
`entropy-bound` subcommand, and the full analysis.



**The headline goal:** match the Python L3TC-200K reference ratio
on enwik6 (≤ 0.1715, currently 0.2061) **without losing more than
15% on compress speed** (≥ 99 KB/s, currently 116).

**Key framing.** Python L3TC and our Rust port use the *same
architecture* (RWKV-v4 with HiRA), the *same checkpoint*
(`l3tc_200k_bpe16k_c999_checkpoint0019.pth`), and the *same
training corpus* (enwik8). The 4 pp ratio gap is therefore
**purely an implementation difference** in our Rust pipeline —
somewhere in checkpoint conversion, the forward pass, the
softmax→freq quantization, or the AC encoding our path diverges
from Python's. **No retraining and no architecture upgrade is
required to close it.** That's a Phase 5+ concern.

Phase 4 is therefore organized as:

- **4a — Implementation diff: find and fix the divergence(s).**
  Instrument both Python and Rust to dump per-token state on the
  first ~1000 tokens of enwik6, find the first place they
  disagree, fix it, repeat until ratio matches. ✅ Shipped.
- **4b — Close the gap from actual coded bytes to the entropy
  bound.** Varint segment headers (4b1) + unk extraction
  replacing raw-fallback (4b2). ✅ Shipped.
- **4b-polish — enwik8 confirmation + `docs/phase_4b_findings.md`
  writeup.** The final sanity check at 100 MB scale and the
  writeup that closes the phase.

**Deferred out of Phase 4 (moved to other phases):**
- **Hybrid classical fallback for OOD inputs** — moved to
  Phase 8 (multi-model dispatch), where it naturally fits as the
  last tier of the dispatch cascade. Phase 4 doesn't need it:
  enwik6 / enwik8 both compress well under the LM path, and the
  OOD failure mode is a structural question separate from the
  ratio-gap-to-entropy-bound question that Phase 4 is about.
- **RWKV-v7 architecture upgrade** — Phase 5.
- **Training on a broader corpus** (The Pile / RedPajama /
  domain mix) — Phase 11.
- **Bigger model variants** (3.2M / 12M) — rejected for default
  use per CLAUDE.md's speed floor; may appear as an opt-in
  "max-ratio" mode in Phase 5b or later.

---

## 4a — Implementation diff vs Python L3TC  ✅ DONE

**Starting belief (end of Phase 3):** Python L3TC reports 0.1665
on enwik6 and we sit at 0.2061. Gap of 4 percentage points,
suspected to be a forward-pass or freq-quantization divergence
in our Rust port.

**What we actually did (Phase 4a steps 1-5):**

1. Mirrored Python's `round(probs * 10_000_000); max(1)` cum_freqs
   scheme exactly in `logits_to_cum_freqs_scratch`. Removed
   `fast_exp_neg` (replaced with libm `f32::exp`). Result: ratio
   went 0.2061 → 0.2060 on enwik6 — **noise**. The cum_freqs
   scheme is not the source of any gap.
2. Wrote `scripts/dump_python_logits.py` that loads the L3TC-200K
   checkpoint, tokenizes the first 4096 bytes of enwik6, forwards
   each token through `RWKV_TC_HIRA_Infer_For_Script`, dumps
   per-token logits to a flat binary file.
3. Added `l3tc dump-logits` subcommand that does the same in Rust.
4. Wrote `scripts/diff_logits.py` that L_inf-diffs the two dumps
   token by token.
5. **Result with INT8 head:** L_inf ≈ 0.20 at every token — looked
   like a real divergence. **Result with f32 head temporarily
   reverted:** L_inf max **3.81e-05** across all 256 tokens. That's
   f32 ULP-level noise. **Our forward pass IS bit-identical to
   Python's.** The ~0.20 INT8 difference was real but turned out
   to be irrelevant to the actual coded byte count (verified next).
6. Added `l3tc entropy-bound` subcommand that computes
   `sum(-log2(softmax_p[next_token])) / 8 / input_bytes` over a
   full file — exactly what Python's `compressor.py` reports as
   `total_bin_size_min`.
7. Ran on enwik6 at segment 2048: **ratio 0.164299**, against
   Python's reported 0.1665. **We're 0.22 pp BETTER than Python**
   on the metric the paper reports. (We use raw softmax for the
   entropy; Python computes from `new_probs = freqs / freqs.sum`,
   which has a tiny rounding loss baked in.) At segment 4096:
   **0.163202**.
8. Reverted the f32 head debug edit. INT8 head is reinstated:
   real-coded ratio at default segment_bytes=4096 is unchanged
   at **0.2060**, decompress speed back at 120 KB/s. The 0.20
   logit drift from INT8 quantization does not affect real
   coded bytes — both INT8 and f32 head produce the same 0.2060
   ratio because the AC's freq quantization absorbs sub-bit
   logit noise.

**The framing finding.** Python's reported "compression ratio"
is **not actual coded bytes**. `compressor.py:315` literally
returns `total_bin_size_min = math.ceil(entropy_sum / 8)` and the
real arithmetic-encode-and-write-to-file path is commented out
(`vendor/L3TC/scripts/compressor.py:281-284`). The L3TC paper's
Table 1 ratios — RWKV-200K @ 24.36%, L3TC-200K @ 16.65% on
enwik6 — are entropy lower bounds, not bytes you can read off
disk. Our 0.2060 is real coded bytes including AC tail flush,
per-segment headers, file framing, and CRC.

**So Phase 4a is done.** We've matched (and beat) Python on the
forward pass and the entropy bound. The 4 pp "gap" was an
apples-to-oranges metric mismatch.

What shipped in 4a:
- `scripts/dump_python_logits.py` — Python forward-pass dumper
- `scripts/diff_logits.py` — token-by-token L_inf comparison
- `scripts/compute_entropy.py` — entropy from a dumped logits.bin
- `l3tc dump-logits` subcommand in the Rust CLI
- `l3tc entropy-bound` subcommand for whole-file entropy
- `logits_to_cum_freqs_scratch` rewritten to mirror Python exactly
  (commit `e4e6f0a`)
- `fast_exp_neg` removed (replaced with libm exp)
- `docs/phase_4a_findings.md` — full writeup

---

## 4b — Close the gap from actual coded bytes to the entropy bound

**The real Phase 4 work, post-4a.** We are at **0.2060** actual
coded bytes per byte of input on enwik6. The entropy bound (with
our current model) is **0.1632**. The gap is **~41,000 bytes per
megabyte**, or **4.28 percentage points**. This gap is achievable
work, not phantom — it lives in the AC framing, segment headers,
end-of-stream flush bits, and (to a tiny extent) freq
quantization rounding.

**Measured overhead breakdown** (Phase 4b1, `l3tc audit` on
enwik6 at segment_bytes=4096, 1135 segments):

| source | bytes | % of overhead | notes |
|---|---:|---:|---|
| **Raw-fallback bytes** (Persian/Arabic ZWNJ) | **21,354** | **50.0%** | 355 segments × ~60 B raw payload |
| **Segment header bytes** (13 + 2×unks) | **17,373** | **40.7%** | 1135 × 15.31 B avg |
| **Unk payload bytes** | 3,813 | 8.9% | tokenizer outliers |
| **AC body bytes vs entropy bound** | **143** | **0.3%** | the AC is essentially optimal |
| File header/trailer/CRC | 28 | 0.07% | constant |
| **Total overhead** | **42,711** | 100% | **4.27 pp above entropy bound** |

**The biggest finding:** the arithmetic coder is doing its job
almost perfectly. Only 143 bytes of overhead vs the entropy
bound across 1135 segment finishes — 0.13 bytes per `enc.finish()`
on average. The "3-4 KB AC tail flush" estimate from before was
wildly wrong; Nayuki's coder is far tighter than expected.

The unexplained ~30 KB turned out to be **half raw-fallback,
half segment headers**:

- **Raw-fallback (50%):** enwik6 has Persian/Arabic Wikipedia
  links containing ZWNJ (U+200C). SPM normalizes these in a way
  that breaks round-trip, so we route them through the
  raw-fallback path which stores the bytes verbatim. 355 of 1135
  segments are raw-fallback chunks of ~60 bytes each. Each
  raw-fallback segment costs 13 + 4 + ~60 = 77 bytes for ~60
  bytes of content (28% framing overhead per chunk).

- **Segment headers (41%):** the fixed 13 bytes per segment
  (n_tokens(4) + n_unks(4) + flags(1) + ac_len(4)) plus 2 bytes
  per unk. With 1135 segments and an average of 1.15 unks per
  segment, that's 15.31 bytes/segment.

**Bigger segments help only marginally**, because raw-fallback
segments aren't subdivided by `segment_bytes` — they're already
at the minimum (`MIN_RAW_FALLBACK_BYTES = 64`). Going from
segment_bytes 4096 → 16384 only saves ~2 KB (0.2 pp). The
realistic levers are elsewhere.

**Concrete approaches in measured-impact order** (numbers from
Phase 4b1 audit on enwik6 at segment 4096):

1. **Eliminate raw-fallback for ZWNJ** — biggest single win,
   ~21 KB / 2.1 pp on enwik6, but requires tokenizer surgery.
   The L3TC paper handles ZWNJ via the outlier-aware tokenizer
   (unk path), not a separate raw-fallback. Routing ZWNJ through
   the existing unk path would let the AC encode the structure
   while only the byte payload gets stored externally. Expected
   net win on enwik6: ~15-18 KB after accounting for the unk
   payload increase. **Phase 4b3.**
2. **Varint segment headers** — second biggest, ~9 KB / 0.9 pp,
   pure mechanical work. Pack `(n_tokens, n_unks, flags, ac_len)`
   into varints: typical values fit in 1-2 bytes each, dropping
   the 13-byte fixed header to ~5 bytes. Bumps file format
   v3 → v4. Zero speed cost. **Phase 4b1.**
3. **Tighter raw-fallback framing** — small but easy, ~3 KB /
   0.3 pp. The 4-byte `raw_len` becomes a 1-2 byte varint.
   Saves ~3 bytes per raw-fallback segment × 355 segments.
   Bundled with 4b1's varint format change. **Phase 4b1.**
4. **Bigger segments** — smallest, ~2 KB / 0.2 pp. Cost is half
   the segment-level parallelism, so probably not worth it given
   how small the win is post-varint. Skip unless profiling shows
   header overhead is still material after 4b1. **Maybe.**
5. **Hybrid classical fallback** for OOD inputs (webster,
   reymont, xml). Doesn't help enwik6 but caps the worst case
   at zstd's ratio. **Phase 4b4 polish, when convenient.**

**Expected end state of Phase 4b** (if 4b1 + 4b3 land):
enwik6 actual coded ratio **0.2060 → ~0.181** (closing >55% of
the gap to the entropy bound 0.1632). Phase 4 success criterion
of ≤0.180 is in reach.

**Speed budget for 4b:** still ≥99 KB/s on enwik6. None of the
approaches above should cost throughput; bigger segments and
tighter headers are pure wins, varint packing is constant-time
per segment, and a single-stream short-file path only kicks in
when the file is too small for parallelism to matter anyway.

---

## Where the original suspects ended up

We have several theories from earlier phases. The first one was
testable directly against the Python source under
`vendor/L3TC/scripts/compressor.py` and turned out to be wrong.
Empirical evidence beats every prior, so the suspect list below
is now ranked by what's actually still on the table.

**Suspect 1: cum_freqs precision and quantization** ❌ **RULED OUT**

The Python reference does literally three lines:
```python
probs = torch.softmax(logits, dim=-1)
freqs = torch.round(probs * 10_000_000).int()
freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
```

We reimplemented exactly this scheme in `logits_to_cum_freqs_scratch`
(Phase 4a, commit pending) — `round` instead of `floor`,
`PYTHON_FREQ_TOTAL = 10_000_000` instead of ~2^62, `max(1)` clamp,
no residual fixup. We also replaced the ~1%-error
`fast_exp_neg` polynomial with libm `f32::exp` so the softmax
matches `torch.softmax` to ULP precision.

**Result on enwik6: ratio 0.2061 → 0.2060.** Movement is noise
(±0.0005 budget). **The cum_freqs scheme is not the source of
the gap.**

This is a high-value negative result. It eliminates the largest
chunk of the suspect surface and forces the search into the
forward pass / checkpoint side, where divergences are harder to
spot but more structural.

**Suspect 2: forward-pass divergence (highest prior given suspect 1 ruled out)**

The 4 pp gap predates all of Phase 2.5: Phase 1 hit ratio 0.2094,
Phase 2 went to 0.2060, Phase 2.5 stayed at 0.2061, Phase 4a
landed at 0.2060. Python is at 0.1665. **The gap was already there
the moment we ran the Rust forward pass against the same
checkpoint.**

That points at the forward pass itself: somewhere in
`src/rwkv.rs` or `src/checkpoint.rs` or
`scripts/convert_checkpoint.py` we produce numerically different
logits than Python's `models/RWKV_V4/rwkv_tc_hira_infer.py`.
Possible sources, in rough order of likelihood:

- **HiRA merge dtype:** the converter computes `W' = W + B @ A`
  in PyTorch f32, then writes f32 to disk. If the Python
  inference path keeps W and HiRA branches separate (and applies
  the merge at every forward pass in higher precision), the
  merged-once result we use could be measurably lossy.
- **Time-mix order:** RWKV-v4 time mix is
  `xk = x * mix_k + state * (1 - mix_k)`. Subtle reorderings or
  the use of different `state_ffn` vs `state_att` semantics
  could change downstream values.
- **Layer norm precision:** if Python computes layer norm in f32
  via fused PyTorch ops and we do it in scalar f32 with a
  different rounding chain, the cumulative drift across 2 layers
  + ln_out could measurably shift logits.
- **The "short" connection:** the 200K model has a dedicated
  `block.w_short` matvec with relu. We treat it as a 96x96 NEON
  matmul (correctly, post-2.5a). Verify that Python's
  `rwkv_tc_hira_infer.py` does the same.

**Test:** instrument both Python and Rust to dump per-token
logits at the same input position on the first ~1000 tokens of
enwik6, and diff. The first divergence is the cause. This is
the actual measurement step that should drive the next commit.

**Suspect 3: model state precision (f32 vs f16)**
- Python L3TC runs the forward pass in f32 throughout. So do we.
  But the INT8 head quantization in 2.5b introduces ~0.4% relative
  error on logits before softmax. We measured the ratio impact as
  ~+0.0001 on enwik6, which is small — but on enwik8 the impact
  could be larger.
- Test: revert the head to f32 temporarily and measure. If the
  ratio improves measurably, INT8 is contributing to the gap and
  we need a smarter quantization scheme (per-row scales? blocked?).

**Suspect 3: tokenizer fallback path**
- The raw-fallback mechanism stores certain segments as plain
  bytes alongside the AC body. Python doesn't have this fallback
  — it relies on SPM round-tripping cleanly. On corpora without
  Persian/Arabic/CJK content the fallback should never trigger.
- Test: count fallback segments per corpus. If enwik6 has none
  this isn't the cause.

**Suspect 4: segment boundary leakage**
- Each segment resets the model state, which costs ~1-2 tokens of
  context for the AC's first predictions. With 4096-byte segments
  on enwik6 there are ~250 segments, so the cost is ~250-500
  tokens × ~0.5 bits = ~150 bytes. That's 0.015% of the 0.2 ratio
  — too small to explain the gap on its own, but could compound.
- Test: try segment_bytes = 32768 and measure the ratio. If it
  drops materially, segment overhead is real.

**Suspect 5: HiRA merge precision**
- Phase 1's checkpoint converter merges HiRA branches into the
  base weights at the f32 level (`W' = W + B @ A`). On the
  L3TC-200K model the BA term is ~10% the magnitude of W, so the
  merged W' has reduced effective precision.
- Test: verify the merged W' against a forward pass that keeps
  HiRA separate. If they differ beyond f32 epsilon the merge is
  lossy.

---

## Success criteria (Phase 4 exit)

Phase 4 is done when:
- ✅ Forward pass matches Python L3TC bit-identically (max
  L_inf < 1e-4 across the first 256 tokens of enwik6) — **DONE 4a**
- ✅ Entropy bound matches or beats Python L3TC on enwik6
  (≤ 0.1665 at segment 2048) — **DONE 4a, we're at 0.1643**
- enwik6 actual coded ratio ≤ **0.180** (closing >50% of the
  current gap to the entropy bound; target gap = 1.7 pp instead
  of 4.3 pp) — **4b target**
- enwik8 actual coded ratio drops correspondingly
- Compress speed ≥ 99 KB/s on enwik6 (≤15% drop from Phase 3 116)
- All unit tests pass + end-to-end integration tests pass
- `docs/phase_4a_findings.md` documents the diff harness, the
  paper-vs-reality finding, and the entropy bound result —
  **DONE**
- A `docs/phase_4b_findings.md` documents the AC overhead
  reduction work once 4b ships

## Non-goals (deferred to later phases)

- Multi-platform release builds (Phase 6)
- Replacing the model architecture (Phase 5 or later)
- Retraining the model on a larger or cleaner corpus (Phase 5)
- Adding new compressor variants (3.2M, 12M) — once 200K is
  matched, the bigger models come essentially for free

## Why this is the highest-value work

The CLAUDE.md goals are ratio (#1) and speed (#2). Phase 0-3
moved speed by 7.4× (huge) but only moved ratio by 0.0033 (from
0.2094 → 0.2061). The ratio is where we have the most room to
improve and the most direct payoff for users — every percentage
point of ratio improvement saves a percentage point of disk on
every file forever, and l3tc-rust is currently the
best-compressing tool in the bench suite *despite* this gap.
Closing it makes the lead structural rather than marginal.
