# Phase 4e — Distillation for compression speed  ✅ CLOSED (failed)

Attempted to train a smaller student architecture via distillation from the 3.2M teacher to ship a faster default. The 4e3 experiment (1-layer 96-hidden student) ran end-to-end cleanly but missed both targets: 0.2871 ratio at 1.12x speedup vs the <=0.195 / >=2x bar. See `docs/phase_4e_findings.md`.

200K stays as default, 3.2M as opt-in, project moves to Phase 5/6/7.

## What shipped (infrastructure)

- **4e1** — `l3tc dump-teacher` CLI subcommand: parallel per-segment top-K softmax dump. 22 KB/s for 3.2M on enwik6.
- **4e2** — `scripts/distill_l3tc.py`: PyTorch training loop using L3TC's `RWKV_TC_HIRA` training class with CPU/MPS WKV replacement. Standard Hinton distillation loss: `(1-a) CE + a T^2 KL`.

## Key findings

- Same-shape distillation (4e2a) cannot improve speed -- inference cost is a function of architecture, not weights.
- 1-layer 96-hidden student (4e3) hit ratio 0.2871 -- the architecture's capacity floor is too high for the speed/ratio trade to be useful.
- The 200K architecture (2-layer 96-hidden) appears near the capacity floor for this task at this parameter count.

## Reference points

| Tier | Ratio (enwik6) | Compress KB/s | Params |
|---|---:|---:|---:|
| l3tc-rust 200K (default) | 0.1699 | 131 | 200K |
| l3tc-rust 3.2M (opt-in) | 0.1337 | 25.95 | 3.2M |
| 4e3 distilled (1-layer) | 0.2871 | ~147 | ~100K |
