# Evaluation Guidelines

How to run compression benchmarks and avoid common pitfalls.

## Running the eval suite

```bash
# Full suite (skips files > N MB with --skip-large)
python3 scripts/run_eval_suite.py \
    --model l3tc-rust/checkpoints/<model>.bin \
    --tokenizer <path/to/spm.model> \
    --skip-large 6

# Single file (more detailed timing)
./l3tc-rust/target/release/l3tc compress bench/corpora/eval_suite/enwik6.txt \
    --model l3tc-rust/checkpoints/<model>.bin \
    --tokenizer <path/to/spm.model> \
    --verify --time -o /tmp/out.l3tc

# Per-phase profiling (runs single-threaded for clean timers)
./l3tc-rust/target/release/l3tc profile \
    --input bench/corpora/eval_suite/enwik6.txt \
    --model l3tc-rust/checkpoints/<model>.bin \
    --tokenizer <path/to/spm.model>
```

## Checkpoint conversion

The Rust binary requires `.bin` format. Training produces `.pth`.

```bash
cd vendor/L3TC && source .venv/bin/activate
python ../../l3tc-rust/scripts/convert_checkpoint.py \
    --input checkpoints/<name>.pth \
    --config config/l3tc/<config>.py \
    --output ../../l3tc-rust/checkpoints/<name>.bin
```

The converter applies HiRA merge (W = W + B @ A), squeezes
time_mix dims, renames ln0. The Rust binary auto-discovers
architecture from tensor shapes — no runtime config needed.

## Known pitfalls

### 1. Background CPU load invalidates speed measurements

**This is the #1 source of wrong numbers.** Any CPU-heavy
process running concurrently (e.g., HuggingFace dataset
streaming, Pile filtering, other compiles) will dramatically
reduce measured throughput.

Example: enwik6 with the 6L×32K model measured **9 KB/s**
with Pile filter running (84% CPU), vs **49 KB/s** on a clean
machine. A 5× difference from background load alone.

**Always check before benchmarking:**
```bash
# Kill competing processes or wait for them to finish
ps aux | grep -i "python\|pile\|download" | grep -v grep
# Verify low CPU before starting
top -l 1 | head -5
```

### 2. Profile mode is single-threaded by design

`l3tc profile` runs segments serially (no rayon) so per-phase
timers accumulate cleanly. It reports single-thread throughput.
The actual `compress` command uses rayon for segment-level
parallelism. Don't compare profile KB/s to compress KB/s.

Reference (6L×32K on M1 10-core, enwik6):
- profile: ~5 KB/s (single-thread)
- compress: ~49 KB/s (rayon, ~2.7 effective cores)

### 3. Small files underreport throughput

Rayon parallelism scales with segment count. At 4096-byte
segments:
- 11 KB file (c_source) → 3 segments → essentially single-threaded
- 150 KB file (fiction) → 38 segments → ~1.4× scaling
- 1 MB file (enwik6) → 244 segments → ~2.7× scaling
- 5 MB file → ~1220 segments → better scaling expected

Benchmark on files ≥ 1 MB for representative throughput.

### 4. Speed breakdown (6L × 96H × vocab 32K)

From `l3tc profile` on fiction.txt:

| phase | µs/step | % of time |
|-------|---------|-----------|
| forward pass (RWKV + head matvec) | 541 | 84.5% |
| cum_freqs (softmax + prefix sum) | 99 | 15.4% |
| AC encode | 0.26 | 0.04% |

The forward pass dominates. The head matvec (32K × 96 INT8)
is the single most expensive operation. AC encode/decode is
negligible.

### 5. Vocab size → speed relationship

| model | forward µs/step | cum_freqs µs/step | total µs/step |
|-------|-----------------|-------------------|---------------|
| 2L × 16K | 234 | 50 | 284 |
| 6L × 32K | 541 | 99 | 640 |
| ratio | 2.3× | 2.0× | 2.3× |

Per-step cost scales ~2.3× from 2L×16K to 6L×32K. But 32K
tokenizer produces ~10% fewer tokens per byte, so per-byte
cost is ~2.0×.

### 6. The `--verify` flag doubles wall time

`--verify` decompresses and compares after compression. The
decompressor has a linear search over cum_freqs (O(V) per
token) which is the one place binary search would help. For
benchmarking speed, run without `--verify` and verify
separately.

## Eval suite files

Located in `bench/corpora/eval_suite/`:

| file | type | size | notes |
|------|------|------|-------|
| enwik6.txt | Wikipedia | 1 MB | primary benchmark |
| webster.txt | dictionary | 41 MB | slow — skip for quick evals |
| fiction.txt | fiction | 150 KB | small, poor rayon scaling |
| json_api.txt | structured | 5 MB | synthetic JSON |
| nginx_log.txt | logs | 5 MB | synthetic nginx |
| python_source.txt | code | 1.1 MB | CPython stdlib |
| csv_data.txt | tabular | 5 MB | synthetic CSV |
| xml_silesia.txt | markup | 5 MB | Silesia corpus |
| c_source.txt | C code | 11 KB | too small for speed benchmarks |
| html.txt | HTML | 24 KB | too small for speed benchmarks |

## Reference results (epoch 6, 6L × 32K, clean machine)

| file | ratio | KB/s | notes |
|------|------:|-----:|-------|
| enwik6 | 0.339 | 49 | primary benchmark |
| python_source | 0.354 | ~30 | est. from clean run |
| fiction | 0.405 | ~20 | small file penalty |
| c_source | 0.600 | ~8 | 11 KB, no parallelism |
| html | 1.001 | — | worse than raw (OOD) |
