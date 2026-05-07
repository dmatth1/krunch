"""Skip the whole tests/gpu/ tree when pytest is run without a GPU.

These scripts can also be invoked directly (`python test_X.py`) inside
the published Docker image — that path doesn't go through pytest and is
unaffected by this skip.

We skip at *collection* time (collect_ignore_glob) because the test
modules import torch + krunch_ac at top level and would crash the
collector otherwise.
"""
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

if not _HAS_CUDA:
    collect_ignore_glob = ["test_*.py", "bench_*.py", "rwkv4_step_ref.py"]
