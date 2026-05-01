"""Per-step profiling of the cpp_path decoder. Runs the model
forward + softmax + AC decode for N tokens, reports time per
component averaged."""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import time
import torch
import krunch_ac_cuda
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch import cpp_path
from krunch_ac.cdf import T as CDF_T


def main():
    RWKV = _load_rwkv()
    print("loading model...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    weights = cpp_path.init_weights(model, "cuda")
    state = cpp_path.fresh_state(weights)

    # Warmup 5 tokens (caches CUBLAS workspaces, allocator state)
    last = BOS_TOKEN
    for _ in range(5):
        logits = cpp_path.forward_stepped(weights, last, state)
        cdf = cpp_path.softmax_cdf_one_row(logits)
        last = int(torch.argmax(logits).item())
    torch.cuda.synchronize()

    # Time per component over 200 tokens
    N = 200
    t_fwd = 0.0
    t_cdf = 0.0
    t_decode = 0.0
    t_sync = 0.0

    # Fake bitstream just for decode_step (won't matter since we feed
    # it as input_buf; we're not measuring correctness here, just per-
    # call latency).
    in_buf = torch.zeros(64 * (N + 16), dtype=torch.uint8, device="cuda")
    ac_state = torch.zeros(4, dtype=torch.uint32, device="cuda")
    ac_state[1] = 0xFFFFFFFF
    out_sym = torch.empty(1, dtype=torch.int32, device="cuda")
    krunch_ac_cuda.decode_init(in_buf, ac_state)

    for _ in range(N):
        torch.cuda.synchronize(); t0 = time.time()
        logits = cpp_path.forward_stepped(weights, last, state)
        torch.cuda.synchronize(); t1 = time.time()
        cdf = cpp_path.softmax_cdf_one_row(logits)
        torch.cuda.synchronize(); t2 = time.time()
        krunch_ac_cuda.decode_step(cdf, in_buf, ac_state, out_sym)
        t3 = time.time()  # NO sync — measure dispatch
        tok = int(out_sym.item())  # this syncs
        t4 = time.time()
        last = tok if tok < 50000 else 0
        t_fwd += (t1 - t0)
        t_cdf += (t2 - t1)
        t_decode += (t3 - t2)
        t_sync += (t4 - t3)

    total = t_fwd + t_cdf + t_decode + t_sync
    print(f"per-token avg over N={N}:")
    print(f"  forward     {t_fwd/N*1000:6.2f} ms ({t_fwd/total*100:.1f}%)")
    print(f"  cdf         {t_cdf/N*1000:6.2f} ms ({t_cdf/total*100:.1f}%)")
    print(f"  decode_step {t_decode/N*1000:6.2f} ms ({t_decode/total*100:.1f}%)")
    print(f"  .item()sync {t_sync/N*1000:6.2f} ms ({t_sync/total*100:.1f}%)")
    print(f"  TOTAL       {total/N*1000:6.2f} ms")


if __name__ == "__main__":
    main()
