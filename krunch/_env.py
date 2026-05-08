"""
Single source of truth for KRUNCH_* environment variables.

This module documents every flag the krunch runtime honors: its default,
its bitstream impact, where it's read, and whether end users should
touch it. The actual values are read at the call sites (in
`krunch/cpp_path.py`, `krunch/kernels/rwkv/layer_cpp.cpp`,
`krunch/inference.py`, `krunch/chunking.py`, `krunch/job.py`); this
file only catalogs the contract.

Pinned by `tests/unit/test_env_flags.py` — adding a new KRUNCH_* env
read site without registering it here causes that test to fail.

Bitstream impact column:
- "yes": flipping this flag changes the bytes a compress would emit
  on a given input. A blob produced with this flag set must be
  decompressed with the same flag setting (and the same MODEL_ID).
- "no": flag only changes runtime/perf; output bytes identical.
- "internal-forced": the flag is force-set by krunch itself; users
  should never override it.

Tier column:
- "user": documented for end users (CLI flags, `--help`).
- "ops": deploy/distributed-mode flags (`KRUNCH_INPUT_LEN`, etc.).
- "internal": auto-derived per hardware; not user-facing.
- "tuning": exposed for power users / benchmarking.
- "debug": opt-in profiling / fallback paths.
"""

from typing import Literal, NamedTuple


class EnvFlag(NamedTuple):
    name: str
    default: str
    bitstream_impact: Literal["yes", "no", "internal-forced"]
    tier: Literal["user", "ops", "internal", "tuning", "debug"]
    summary: str


# ---------------------------------------------------------------------------
# The canonical table.
# ---------------------------------------------------------------------------
# Order: codec/bitstream-affecting first, then perf, then ops/distributed.

ENV_FLAGS: list[EnvFlag] = [
    # ---- Codec / bitstream-affecting -----------------------------------
    EnvFlag(
        "KRUNCH_DETERMINISTIC_MATMUL", "1", "internal-forced", "internal",
        "Force shape-invariant matmul kernels. Required for AC roundtrip; "
        "krunch sets this at module load — do not override.",
    ),
    EnvFlag(
        "KRUNCH_INT8_W8A8", "1", "yes", "tuning",
        "W8A8 production codec (int8 weights × int8 activations). "
        "Default-on. Set to 0 to fall back to fp16 baseline. "
        "Bytes differ between modes; both round-trip.",
    ),
    EnvFlag(
        "KRUNCH_BF16", "0", "yes", "debug",
        "Bf16 layer matmul (v2 codec, WIP). Switches weight dtype; "
        "encoder + decoder must use the same setting. Mutually exclusive "
        "with KRUNCH_INT8_W8A8 (must set =0) and KRUNCH_INT8_WEIGHTS "
        "(quantization layer ignores bf16 weights). Bytes diverge from "
        "fp16 codec → MODEL_ID bump on ship. NEXT-2 in "
        "TIER_3_OPTIMIZATION.md.",
    ),
    EnvFlag(
        "KRUNCH_INT8_WEIGHTS", "0", "yes", "debug",
        "Legacy int8-weights mode (quantize-then-dequant to fp16). "
        "Mutually exclusive with KRUNCH_INT8_W8A8. Kept opt-in for "
        "comparison runs; W8A8 supersedes it on the production path.",
    ),
    EnvFlag(
        "KRUNCH_OWN_WKV", "0", "yes", "internal",
        "Use krunch's graph-safe WKV kernel instead of BlinkDL's "
        "c10-dispatcher one. Encoder + decoder must agree. Auto-flipped "
        "to 1 when KRUNCH_DECOMPRESS_GRAPH=full.",
    ),
    EnvFlag(
        "KRUNCH_CHUNK_SIZE", "", "yes", "tuning",
        "Force a chunk size (bytes). Empty = auto from compute_chunk_size. "
        "Changes blob structure (different chunk count → different blob).",
    ),
    EnvFlag(
        "KRUNCH_TARGET_B", "32", "yes", "tuning",
        "Target decompress batch size. Drives compute_chunk_size's "
        "n_chunks ≈ 4 × target_B. Affects blob structure indirectly.",
    ),
    # ---- Perf / kernel routing -----------------------------------------
    EnvFlag(
        "KRUNCH_HEAD_ASYNC", "auto", "no", "internal",
        "cp.async + WMMA head matmul. Auto-detected: 1 on sm_80+, "
        "0 on sm_75 (kernels compile to no-ops). Set in cpp_path.py "
        "at module import.",
    ),
    EnvFlag(
        "KRUNCH_3WAY_ASYNC", "auto", "no", "internal",
        "cp.async + WMMA 3-way fused KVR matmul. Auto-detected like "
        "KRUNCH_HEAD_ASYNC. Disabled below sm_80.",
    ),
    EnvFlag(
        "KRUNCH_3WAY_ASYNC_M_MIN", "256", "no", "tuning",
        "Minimum M for 3-way async path; below this, fall back to sync "
        "3-way kernel.",
    ),
    EnvFlag(
        "KRUNCH_HEAD_MW", "1", "no", "tuning",
        "Multi-warp (4-warp 64×64 tile) head matmul kernel. Used on "
        "sm_75 fallback and large-M paths.",
    ),
    EnvFlag(
        "KRUNCH_HEAD_MW_M_MIN", "256", "no", "tuning",
        "Minimum M for the multi-warp head kernel; below this, fall "
        "back to single-warp tile.",
    ),
    EnvFlag(
        "KRUNCH_W8A8_ASYNC", "1", "no", "tuning",
        "cp.async-double-buffered W8A8 layer matmul on sm_80+. "
        "Disabled implicitly on sm_75 via the kernel guard.",
    ),
    EnvFlag(
        "KRUNCH_NO_3WAY", "0", "no", "debug",
        "Disable KVR fusion entirely; run K, V, R as 3 separate "
        "matmuls. For ablation only.",
    ),
    EnvFlag(
        "KRUNCH_CUBLAS_PINNED", "0", "no", "debug",
        "Use cuBLAS GemmEx with a pinned algorithm for layer matmul. "
        "Opt-in only — cuBLAS algos are not bit-stable across M, so "
        "this can break AC roundtrip. Kept for comparison.",
    ),
    EnvFlag(
        "KRUNCH_LAYERNORM_CUSTOM", "0", "no", "debug",
        "Use krunch's custom layer-norm kernel instead of at::layer_norm. "
        "Default-off; the custom kernel does not currently beat ATen. "
        "Candidate for removal — see docs/TIER_3_CLEANUP.md §2.1.",
    ),
    EnvFlag(
        "KRUNCH_DECOMPRESS_GRAPH", "auto", "no", "tuning",
        "CUDA-graph mode for batched decompress: 'full' (default if "
        "KRUNCH_OWN_WKV=1), 'per_layer', or 'eager'. W8A8 forces "
        "'eager' regardless.",
    ),
    EnvFlag(
        "KRUNCH_CPP_GRAPH", "0", "no", "debug",
        "Per-layer CUDA-graph capture for the B=1 stepped decompress path. "
        "Internal — not load-bearing on production B>=2 path.",
    ),
    EnvFlag(
        "KRUNCH_DECOMPRESS_BATCH", "1", "no", "tuning",
        "ThreadPoolExecutor fan-out for decompress (separate from the "
        "GPU lockstep batch). Default 1 (sequential).",
    ),
    EnvFlag(
        "KRUNCH_FORWARD_BATCH", "1024", "no", "tuning",
        "SEQ_BATCH for the packed-window encoder forward. Larger = "
        "better compute amortization, more peak VRAM.",
    ),
    # ---- Profiling -----------------------------------------------------
    EnvFlag(
        "KRUNCH_CPP_PROFILE", "0", "no", "debug",
        "Per-chunk timing breakdown (weights / state_init / forward / "
        "cdf / ac / copy). Logs at INFO.",
    ),
    EnvFlag(
        "KRUNCH_PHASE_PROFILE", "0", "no", "debug",
        "cudaEvents inside rwkv4_layer_step_cpp for in-kernel phase "
        "breakdown. Used to land T3.7 (gemm_fp16 routing fix).",
    ),
    # ---- Deploy / runtime config ---------------------------------------
    EnvFlag(
        "KRUNCH_MODEL_DIR", "/models", "no", "ops",
        "Directory holding RWKV-4-Pile-169M-*.pth and 20B_tokenizer.json. "
        "Baked into the Docker image.",
    ),
    # ---- Distributed `job` mode ----------------------------------------
    EnvFlag(
        "KRUNCH_MODE", "", "no", "ops",
        "`compress-part` / `decompress-part` / `finalize-compress` / "
        "`finalize-decompress`. Required by job.py.",
    ),
    EnvFlag(
        "KRUNCH_INPUT_URL", "", "no", "ops",
        "S3 URL of the per-worker input slice (job mode).",
    ),
    EnvFlag(
        "KRUNCH_OUTPUT_URL", "", "no", "ops",
        "S3 URL the worker uploads its result to (job mode).",
    ),
    EnvFlag(
        "KRUNCH_INPUT_LEN", "", "yes", "ops",
        "Total input length across the distributed job. Workers derive "
        "the global chunk size from this — must be identical across "
        "every worker, or chunks won't align and the blob is unreadable.",
    ),
    EnvFlag(
        "KRUNCH_PART_INDEX", "", "no", "ops",
        "0-based byte-range index for this worker.",
    ),
    EnvFlag(
        "KRUNCH_PART_COUNT", "", "no", "ops",
        "Total number of byte-range workers in this job.",
    ),
    EnvFlag(
        "KRUNCH_FINALIZE_OF", "", "no", "ops",
        "Manifest URL listing per-part output URLs. Read by the "
        "finalize-* modes to assemble the final blob.",
    ),
]


# Index by name for O(1) lookup.
BY_NAME: dict[str, EnvFlag] = {f.name: f for f in ENV_FLAGS}


def names() -> list[str]:
    """Sorted list of every registered KRUNCH_* flag name."""
    return sorted(BY_NAME.keys())


def is_bitstream_flag(name: str) -> bool:
    """True if flipping `name` changes the bytes encoders emit."""
    f = BY_NAME.get(name)
    return f is not None and f.bitstream_impact == "yes"
