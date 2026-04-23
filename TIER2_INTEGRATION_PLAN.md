# Tier-2 integration plan (revised 2026-04-23 v3)

**Design: Python+HTTP hybrid, not pure Rust.**

## Reasoning (revised)

An earlier draft proposed a pure-Rust port of BlinkDL's RWKV-LM
inference path. Investigation showed this was ~2-4 weeks of CUDA
engineering (cuBLAS bindings, LayerNorm + elementwise kernels,
orchestration, fp16 numerics debugging) for:

- **Performance gain: ~1-2%** (HTTP overhead is trivial vs the 200ms
  per-chunk kernel compute)
- **Cost savings: $5-20/month** at any realistic scale
- **Operational wins: real but limited** (single binary, no Python in
  prod path, simpler debugging)

Not worth 2-4 weeks of engineering for ~1% perf + marginal cost. The
pure-Rust plan is preserved at commit `af48db5` + `5c02671` if we
revisit for a concrete reason (Apple Silicon deployment, torch CVE
triage pain, cold-start SLO miss). The Phase 0 scaffold we already
committed stays in tree as useful documentation-of-intent.

## Design: Python inference server + thin Rust client

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│ Compressor worker           │     │ Inference server         │
│ (ECS task or EC2)           │     │ (same host, unix socket) │
│                             │     │                          │
│  l3tc-rust binary           │     │  Python + PyTorch + CUDA │
│  ┌────────────────────────┐ │     │                          │
│  │  Dispatcher            │ │     │  ┌────────────────────┐  │
│  │   - zstd / bzip3 / lz4 │ │     │  │ BlinkDL RWKV-LM    │  │
│  │   - NEW: NeuralRwkv    │─┼─────┼─►│ RWKV_GPT (fp16)    │  │
│  │     (HTTP client)      │ │ HTTP│  │ + WKV CUDA kernel  │  │
│  └────────────────────────┘ │     │  │ + constriction AC  │  │
│                             │     │  └────────────────────┘  │
└─────────────────────────────┘     └──────────────────────────┘
```

## Why this design

1. **We validated this exact path in Spike 6.** Python + RWKV-LM +
   custom WKV kernel measured 330-430 KB/s on A10G, ratio 0.111.
   That's our production baseline. No further architecture risk.
2. **Single process boundary = ~1-5ms overhead** per chunk. At 64 KB
   chunks and ~200ms compute, that's 1-2%. Noise.
3. **Python is the right language for the inference body** — torch,
   transformers, and the RWKV-LM inference code are all Python.
   Trying to Rust-port them is work that doesn't pay back.
4. **Rust dispatcher stays in Rust.** Chunk slicing, per-chunk
   winner-picks-all, blob format, classical codec fallbacks — all
   the stuff that benefits from Rust's speed and type safety stays
   where it is.
5. **One Python process per GPU worker** — not microservices,
   not cross-host. The Python server is a local inference daemon,
   not a distributed component.

## Phases

### Phase 1 — Python inference server (4-6 days)

Ship a working `cdk/docker/inference/server.py`:

- **Endpoints:**
  - `POST /compress` body: `application/octet-stream` → `application/octet-stream`
  - `POST /decompress` body: `application/octet-stream` → `application/octet-stream`
  - `GET /healthz` → readiness check (model loaded, kernel compiled)
- **Internals:**
  - Startup: load `RWKV-4-Pile-169M-20220807-8023.pth` + `20B_tokenizer.json`,
    trigger WKV kernel compile via warm-up forward pass
  - `/compress`: tokenize → forward → arithmetic-encode (constriction) →
    return bytes
  - `/decompress`: arithmetic-decode step-by-step against per-token
    forward passes → detokenize → return bytes
- **Dependencies:** pytorch (cu12.4), tokenizers, constriction, fastapi,
  uvicorn. Plus the system `ninja-build` package (see CLAUDE.md notes
  from Spike 6).

Deliverable: local roundtrip test passes on 100 MB WildChat-English,
100% byte-exact. Throughput measured ≥300 KB/s on A10G.

### Phase 2 — Rust `NeuralRwkvCodec` HTTP client (1-2 days)

- `NeuralRwkvCodec { endpoint: String, client: reqwest::blocking::Client }`
  in `l3tc-rust/src/dispatcher.rs`
- `encode` / `decode` POST raw bytes to the inference server, return
  raw bytes
- `CodecTag::NeuralRwkv = 8` already registered (committed)
- CLI flag: `--neural-rwkv-endpoint http://localhost:9000`

### Phase 3 — Service wiring (3-5 days)

- `cdk/docker/inference/Dockerfile`: CUDA base + pytorch + ninja-build +
  our Python server
- `cdk/docker/compression/compression_worker.py`: spawn inference
  server as subprocess at container start, pass its endpoint to the
  Rust dispatcher
- CDK `compression-stack.ts`: GPU-backed task definition. **EC2, not
  Fargate** — Fargate GPU is 3-4× more expensive and we don't need
  per-task elasticity. One g5.xlarge spot instance idles until traffic
  arrives, scales to zero when quiet.
- CloudWatch EMF: per-tenant counters for `NeuralBytesIn`,
  `NeuralBytesOut`, `NeuralEncodeMs`, `FallbackToClassical`

### Phase 4 — End-to-end validation (2-3 days)

On a running dev stack:
- 1 GB WildChat-English corpus through the full ingest path
- Measure: effective compress KB/s, ratio vs Spike 6's 0.111 projection,
  dispatcher codec-selection histogram (how often neural wins)
- Retrieval path: decompression KB/s, per-tenant isolation

Success criteria:
- Compress throughput ≥200 KB/s end-to-end (HTTP + chunk framing
  overhead should knock us from 330 to 200-ish)
- Decompress throughput ≥100 KB/s per-stream
- Ratio ≤0.12 on 1 GB corpus
- Zero byte-exactness failures across the 16k chunks

## Time estimate

**Revised total: 10-16 days** (was 17-23 for pure-Rust path).

| phase | days | risk |
|---|---|---|
| 1: Python inference server | 4-6 | low (all components already work in Spike 6) |
| 2: Rust HTTP client codec | 1-2 | low |
| 3: Service wiring + CDK | 3-5 | medium (first GPU-backed ECS task def) |
| 4: End-to-end validation | 2-3 | medium (decode throughput unverified) |

The big risk reduction vs pure-Rust is Phase 1 — instead of
re-implementing the forward pass, we wrap proven code.

## Pre-existing scaffold (committed, kept)

The Phase 0 Rust backend scaffold (commit `6cbba01`) stays in tree:
- `src/backend/cuda.rs` — CudaContextHandle
- `src/cuda/wkv.cu` — ported kernel
- `build.rs` — nvcc compile path
- Feature flags: `cuda`, `rwkv-v4-pile`

Not active in the production code path, but:
- Documents the pure-Rust route if we revisit
- ~1 day to delete if we want to prune (trivial)
- The weight converter `scripts/convert_rwkv_pth_to_safetensors.py`
  is still useful for any future Rust revisit

## Out of scope for Tier-2 v1

- Per-tenant LoRA fine-tuning (v2 — raw zero-shot works)
- Multi-GPU or multi-model serving
- ts_zip integration (licensing unclear; we have RWKV-LM which does
  the same thing, Apache-2.0)
- Batching requests server-side (v2 perf work)
