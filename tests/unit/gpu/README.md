# tests/gpu — kernel-level correctness checks

Standalone scripts (not pytest-collected) that verify invariants of the
custom CUDA kernels and C++ inference path. They require:

- An NVIDIA GPU (sm_75+ for the cp.async paths)
- The `krunch_ac` extension built and importable
- Model weights at `/models/RWKV-4-Pile-169M-20220807-8023.pth`

Easiest way to run them is inside the published Docker image:

```bash
docker run --rm --gpus all \
  --entrypoint python \
  -v "$PWD/tests/gpu:/gpu_tests" \
  ghcr.io/dmatth1/krunch:latest \
  /gpu_tests/test_batched_stepped.py
```

These are **not in CI** (CI runs on `ubuntu-latest`, no GPU). They are
the regression-evidence for invariants cited inline in:

- `krunch/kernels/rwkv/layer_cpp.cpp` (layer-step routing, codec dispatch)
- `krunch/kernels/matmul/det_matmul_cublas.cu` (cuBLAS algo bit-stability)
- `krunch/cpp_path.py` (graph capture, batched-vs-stepped equivalence)
- `krunch/inference.py` (batched compress chunk lockstep)

If you're touching any of those code paths, run the corresponding
script in this directory before merging.
