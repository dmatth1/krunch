# Phase 4d — Port L3TC-3.2M into the Rust runtime  ✅ COMPLETE

Loaded and ran the L3TC-3.2M checkpoint end-to-end in l3tc-rust. See `docs/phase_4d_findings.md`.

## Key results

| Tier | Ratio (enwik6) | Compress KB/s | Params |
|---|---:|---:|---:|
| l3tc-rust 200K (default) | 0.1699 | 131 | 200K |
| l3tc-rust 3.2M (opt-in) | 0.1337 | 25.95 | 3.2M |

## What was required

- Made runtime dimension-agnostic: dispatch-at-runtime (Option B) for matvec kernel selection based on `model.hidden_size`
- Extended checkpoint loader to detect `intermediate_size` from FFN key tensor shape (3.2M has `intermediate_size=512` != `hidden_size=256`)
- Converted 3.2M checkpoint via existing `scripts/convert_checkpoint.py`
- No new NEON kernels needed; 256x256 attention matvecs fall back to scalar, INT8 col-major head AXPY is already dimension-agnostic

## 3.2M vs 200K structural differences

| Param | 200K | 3.2M |
|---|---:|---:|
| num_hidden_layer | 2 | 3 |
| hidden_size | 96 | 256 |
| intermediate_size | 96 | 512 |

## Decision

3.2M ships as opt-in "max ratio" tier. 200K remains default. Distillation (Phase 4e) explored next.
