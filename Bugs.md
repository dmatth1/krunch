# Known bugs

Active production issues we've found but haven't fixed yet. Fix or
document each before v1.0.

---

## 1. W8A8 small-chunk roundtrip break on T4 (sm_75)

**Severity:** correctness — silent data corruption on small inputs.
**Surface:** any single-chunk compress/decompress on T4 with the
default `KRUNCH_INT8_W8A8=1`.

**Symptom.** `engine.compress_chunk(data)` produces an absurdly small
bitstream (~16-22 bytes regardless of input size from 360 B up to at
least 23 KB), and `engine.decompress_chunk(blob)` returns
free-running model text instead of the original bytes.

The compressed-size pattern reveals the failure mode: output grows
~1 byte per 2× input, i.e. log(N) bytes total. That's only possible
if the AC encoder sees `prob[symbol] ≈ 1.0` for every token, which
in turn means the encoder is being fed the model's argmax tokens
rather than the actual input tokens — or equivalently, the
encoder/decoder are running materially different forward passes.

| Input size | Encoded bytes | Ratio | Result |
|---|---|---|---|
| 360 B    | 16 | 0.044  | FAIL |
| 720 B    | 17 | 0.024  | FAIL |
| 1440 B   | 18 | 0.013  | FAIL |
| 2880 B   | 18 | 0.0063 | FAIL |
| 5760 B   | 19 | 0.0033 | FAIL |
| 11520 B  | 20 | 0.0017 | FAIL |
| 23040 B  | 22 | 0.0010 | FAIL |
| ≥ 64 KB  | ratio ≈ 0.116 | byte-exact ✅ |

**Why production didn't catch it.** `compute_chunk_size` floors at
64 KB. Every workload going through `krunch compress` on a file
≥ 1 MB hits ≥16 chunks of ~64 KB, all of which use the
`M >= 256` W8A8 *packed* path that's correct on T4. The bug only
fires on small, single-chunk inputs — i.e. the path no production
test exercised until `tests/unit/gpu/test_engine_roundtrip.py`.

**Workaround.** Set `KRUNCH_INT8_W8A8=0` to disable W8A8 entirely
on T4. All sizes work byte-exact in this mode. A10G+ (sm_80+) is
unaffected and keeps W8A8 on.

**Suspected root cause.** Encoder/decoder dispatch asymmetry on the
sm_75 fp16 path between `forward_packed_window` (used by encode at
M < SEQ_BATCH or for partial last-window) and the stepped decoder.
Note: `compute_decompress_batch` already returns reduced sat_B on
sm_75 (`n_sms * 2` instead of `n_sms * 8`), and
`KRUNCH_HEAD_ASYNC`/`KRUNCH_3WAY_ASYNC` already auto-disable on
sm<8. Something analogous likely needs to happen for the W8A8
dispatch threshold or packed-vs-stepped weight handling.

**Cheap-fix candidate (v1.0):** auto-disable `KRUNCH_INT8_W8A8` on
sm<8 at `cpp_path.py` import, mirroring the existing auto-disables
for the cp.async kernels. Loses W8A8 perf on T4 (~no impact since
T4 isn't a primary deployment target, ~41 KB/s either way).

**Proper fix (v1.x):** root-cause the encoder/decoder asymmetry on
sm_75 and patch the dispatch directly.

**Repro:**
```bash
docker run --rm --gpus all \
  -e KRUNCH_INT8_W8A8=1 \
  -v /tmp/gpu-tests:/gpu-tests \
  --entrypoint /opt/conda/bin/python \
  ghcr.io/dmatth1/krunch:latest \
  /gpu-tests/test_engine_cpp_roundtrip.py
# All 3 SAMPLES fail.
```

---

## 2. `tests/integration/gpu.sh` EXIT trap overwrites successful result.json on terminate failure

**Severity:** test infrastructure — produces misleading "FAIL"
output on a successful run if termination is blocked.

**Symptom.** Userdata script in `gpu.sh` runs the test, uploads a
real `result.json` (with all the metrics), then calls
`aws ec2 terminate-instances`. If termination fails (e.g.,
termination protection enabled, or the IAM role lacks
`ec2:TerminateInstances`), the script exits non-zero, the EXIT
trap fires, and overwrites the real result.json with:
```
{"all_gates_pass": false, "error": "user-data exited rc=255 — see setup.log"}
```
The host's `gpu.sh` poller then reports this stub instead of the
actual measurements.

**Fix.** EXIT trap should only write the failure stub if no
result.json has been uploaded yet (`aws s3 ls` check, then conditional
upload). One-line change in the trap body.

**Workaround.** SSH into the instance and read `/tmp/result.json`
or `/var/log/krunch-tier3.log` directly when termination fails.

---

## 3. pytest GPU suite cross-test state contamination

**Severity:** test infrastructure — not production.
**Surface:** running `tests/unit/gpu/` as a full pytest suite. Tests
that pass cleanly in isolation (e.g. `pytest test_engine_roundtrip.py`)
fail when invoked alongside the rest.

**Symptom.** `test_engine_roundtrip` (3 cases),
`test_compress_chunks_batch_equiv::test_batched_decompress_matches_per_chunk`,
and the 3 `test_batched_stepped` cases all fail with byte-exact
roundtrip mismatches when run as part of the full suite. Run
in isolation, all 7 pass.

**Root cause hypothesis.** Some piece of state survives between
tests despite the autouse fixture in `tests/unit/gpu/conftest.py`
that clears `cpp_path._BATCHED_STATE_CACHE`,
`_FULL_STEP_BUFS_CACHE`, `_FULL_STEP_GRAPH_CACHE`, and resets
W8A8/INT8_WEIGHTS/BF16 env vars. Likely candidates: cuBLAS
`set_cublas_pinned_algo()` global state, in-place mutations on
shared `model.w` tensors that survive cache resets, or CUDA
context state we're not clearing.

**Why production isn't affected.** `cli.py:cmd_decompress` picks ONE
of `decompress_chunks_batched` (B>=2) OR `decompress_chunk` (B=1)
per process invocation; the two paths are never mixed in a single
real run. The codec works correctly when pytest isolates each test.

**Workarounds:**
- For specific debugging: `pytest tests/unit/gpu/test_X.py` in isolation.
- Future fix: split GPU tests across pytest-forked subprocesses, or
  identify and reset the leaking global state (cuBLAS algo, model.w
  mutations, etc).

**Repro:**
```bash
docker run --rm --gpus all -v /tmp/gpu-tests:/gpu-tests \
  --entrypoint bash ghcr.io/dmatth1/krunch:latest \
  -c "pip install -q pytest && cd /gpu-tests && python -m pytest -v"
# 7 fail. Same tests pass with `pytest <file>.py` in isolation.
```
