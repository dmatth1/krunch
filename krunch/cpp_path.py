"""Bit-exact C++ orchestration path for compress + decompress.

Wraps `krunch_ac_cuda.rwkv4_layer_step_cpp{,_t1}` + the deterministic
matmul kernel + per-row softmax/CDF so encoder (packed T) and decoder
(stepped T=1) produce byte-identical AC bitstreams.

`KRUNCH_DETERMINISTIC_MATMUL=1` and `KRUNCH_CPP_PATH=1` default ON as
of 2026-04-30 — bit-exact GPU AC roundtrip is the only correct path
on GPU. Set either to 0 to fall back to the legacy BlinkDL forward
(will FAIL roundtrip; kept for bench comparisons only).
"""
from __future__ import annotations

import os

# Default deterministic matmul ON before any C++ extension import.
# The C++ static-const reads this once at module load; setting it
# afterward has no effect.
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

N_LAYER = 12

_WEIGHTS_CACHE: dict[int, dict] = {}


def init_weights(model, device: str = "cuda") -> dict:
    """Extract per-layer weights once per (model, device). Cached by
    id(model)."""
    import torch
    key = id(model)
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]

    w = model.w

    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt:
            t = t.to(dtype=dt).contiguous()
        return t

    layers = []
    for i in range(N_LAYER):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
        layers.append([
            fix(w[bbb+'ln1.weight'], torch.float16),
            fix(w[bbb+'ln1.bias'], torch.float16),
            fix(w[att+'time_mix_k'].squeeze(), torch.float16),
            fix(w[att+'time_mix_v'].squeeze(), torch.float16),
            fix(w[att+'time_mix_r'].squeeze(), torch.float16),
            fix(w[att+'time_decay'], torch.float32),
            fix(w[att+'time_first'], torch.float32),
            fix(w[att+'key.weight'], torch.float16),
            fix(w[att+'value.weight'], torch.float16),
            fix(w[att+'receptance.weight'], torch.float16),
            fix(w[att+'output.weight'], torch.float16),
            fix(w[bbb+'ln2.weight'], torch.float16),
            fix(w[bbb+'ln2.bias'], torch.float16),
            fix(w[ffn+'time_mix_k'].squeeze(), torch.float16),
            fix(w[ffn+'time_mix_r'].squeeze(), torch.float16),
            fix(w[ffn+'key.weight'], torch.float16),
            fix(w[ffn+'value.weight'], torch.float16),
            fix(w[ffn+'receptance.weight'], torch.float16),
        ])

    bundle = {
        "layers": layers,
        "emb_w": fix(w['emb.weight'], torch.float16),
        "ln_out_w": fix(w['ln_out.weight'], torch.float16),
        "ln_out_b": fix(w['ln_out.bias'], torch.float16),
        "head_w": fix(w['head.weight'], torch.float16),
        "n_embd": int(model.args.n_embd),
        "n_att": int(model.args.n_att),
        "device": device,
    }
    _WEIGHTS_CACHE[key] = bundle
    return bundle


def fresh_state(weights: dict):
    """Initial RWKV-4 state (att_xx, aa, bb, pp, ffn_xx) per layer."""
    import torch
    device = weights["device"]
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    return (
        [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
    )


def forward_packed(weights: dict, input_ids, state):
    """Run all 12 layers packed (T = len(input_ids)). Returns
    logits [T, V]. Mutates state in place."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    device = weights["device"]
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    T = int(input_ids.shape[0])
    # Single-shot path: caller wants all logits at once. Used by tests
    # and the no-AC engine path. Keep for backwards compatibility.
    return forward_packed_window(weights, input_ids, state, off=0, n=T)


def forward_packed_window(weights: dict, input_ids, state, off: int, n: int):
    """Run the layer stack on a window [off, off+n) of input_ids. State
    is mutated in place — repeated calls with sequential, non-overlapping
    windows produce the same final state as one big call.

    Returns logits [n, V] (fp16). Caller is responsible for keeping VRAM
    bounded by choosing n: at V=50277 each window's output is
    n × 100 KB. For 1 MB chunks pick n ≤ 4096 to stay under ~400 MB."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long,
                                     device=weights["device"])
    ids_w = input_ids[off:off + n]
    T_w = int(ids_w.shape[0])
    x = emb_w[ids_w].view(1, T_w, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    x_flat = x.view(T_w, n_embd).contiguous()
    xn = F.layer_norm(x_flat, (n_embd,), weight=ln_out_w, bias=ln_out_b)
    return krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous())


_STEPPED_BUFS_CACHE: dict[int, list] = {}


def _get_stepped_bufs(weights: dict):
    """Static [1,1,n_embd] tensors used to host inputs/outputs for the
    graph-captured per-layer forward. One per layer boundary."""
    import torch
    key = id(weights)
    if key in _STEPPED_BUFS_CACHE:
        return _STEPPED_BUFS_CACHE[key]
    device = weights["device"]
    n_embd = weights["n_embd"]
    bufs = [torch.empty(1, 1, n_embd, dtype=torch.float16, device=device)
            for _ in range(N_LAYER + 1)]
    _STEPPED_BUFS_CACHE[key] = bufs
    return bufs


_GRAPH_CACHE: dict[int, list] = {}
_STEPPED_BATCHED_BUFS_CACHE: dict[tuple, list] = {}
_GRAPH_BATCHED_CACHE: dict[tuple, list] = {}


def _get_stepped_batched_bufs(weights: dict, B: int):
    """Static [B,1,n_embd] tensors for graph-captured batched stepped forward."""
    import torch
    key = (id(weights), B)
    if key in _STEPPED_BATCHED_BUFS_CACHE:
        return _STEPPED_BATCHED_BUFS_CACHE[key]
    device = weights["device"]
    n_embd = weights["n_embd"]
    bufs = [torch.empty(B, 1, n_embd, dtype=torch.float16, device=device)
            for _ in range(N_LAYER + 1)]
    _STEPPED_BATCHED_BUFS_CACHE[key] = bufs
    return bufs


def _capture_one_layer_graph_batched(weights, state, layer_idx, bufs):
    """Same as _capture_one_layer_graph but at B>1. State tensors are
    already [B, ...] shape — kernel dispatches batched path internally."""
    import torch
    import krunch_ac_cuda
    layer = weights["layers"][layer_idx]
    s0, s1, s2, s3, s4 = (state[k][layer_idx] for k in range(5))

    for _ in range(2):
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(
            bufs[layer_idx], s0, s1, s2, s3, s4, *layer)
        bufs[layer_idx + 1].copy_(out)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    cap_stream = torch.cuda.Stream()
    cap_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(g):
            out = krunch_ac_cuda.rwkv4_layer_step_cpp(
                bufs[layer_idx], s0, s1, s2, s3, s4, *layer)
            bufs[layer_idx + 1].copy_(out)
    torch.cuda.current_stream().wait_stream(cap_stream)
    return g


def forward_stepped_batched_graphed_v2(weights: dict, last_tokens, state):
    """Graph-captured B-batched stepped forward.

    REQUIRES: KRUNCH_DETERMINISTIC_MATMUL=1 + KRUNCH_OWN_WKV=1. With third-
    party rwkv ops, graph replay is non-deterministic (verified by
    test_graph_determinism.py). With our own kernels, replay is bit-exact
    (0.0 abs diff across 5 steps).

    Decompress 47/200 KB/s gate is bound by per-step ATen launch overhead
    on the stepped path (12 layers × ~10-20 ATen ops × Python dispatch).
    Graphs collapse that to one launch per layer per replay.
    """
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    n_embd = weights["n_embd"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    if not isinstance(last_tokens, torch.Tensor):
        last_tokens = torch.as_tensor(last_tokens, dtype=torch.long,
                                       device=weights["device"])
    B = int(last_tokens.shape[0])
    bufs = _get_stepped_batched_bufs(weights, B)

    cache_key = (id(weights), B)
    if cache_key not in _GRAPH_BATCHED_CACHE:
        # Capture one graph per layer. Snapshot+restore state so the
        # warmup (3 mutations per layer) doesn't desync with the caller.
        bufs[0].copy_(emb_w[last_tokens].view(B, 1, n_embd))
        snap = _snapshot_state(state)
        graphs = []
        for i in range(N_LAYER):
            _restore_state(state, snap)
            graphs.append(_capture_one_layer_graph_batched(weights, state, i, bufs))
        _restore_state(state, snap)
        _GRAPH_BATCHED_CACHE[cache_key] = graphs

    graphs = _GRAPH_BATCHED_CACHE[cache_key]
    bufs[0].copy_(emb_w[last_tokens].view(B, 1, n_embd))
    for i in range(N_LAYER):
        graphs[i].replay()
    xn = F.layer_norm(bufs[N_LAYER].view(B, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous())
    return logits  # [B, V]


def _snapshot_state(state):
    return [[t.clone() for t in state[k]] for k in range(5)]


def _restore_state(state, snap):
    for k in range(5):
        for i in range(N_LAYER):
            state[k][i].copy_(snap[k][i])


def _capture_one_layer_graph(weights, state, layer_idx, bufs):
    """Capture a CUDA graph for one layer's T=1 forward against the
    REAL state tensors. Mutates state once per warmup pass + once per
    capture (so 3 mutations total). Caller must snapshot/restore."""
    import torch
    import krunch_ac_cuda
    layer = weights["layers"][layer_idx]
    s0, s1, s2, s3, s4 = (state[k][layer_idx] for k in range(5))

    # Warm up cuBLAS workspaces / allocator state.
    for _ in range(2):
        out = krunch_ac_cuda.rwkv4_layer_step_cpp(
            bufs[layer_idx], s0, s1, s2, s3, s4, *layer)
        bufs[layer_idx + 1].copy_(out)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    cap_stream = torch.cuda.Stream()
    cap_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(g):
            out = krunch_ac_cuda.rwkv4_layer_step_cpp(
                bufs[layer_idx], s0, s1, s2, s3, s4, *layer)
            bufs[layer_idx + 1].copy_(out)
    torch.cuda.current_stream().wait_stream(cap_stream)
    return g


def forward_stepped_graphed_v2(weights: dict, last_token: int, state):
    """Properly graphed per-token forward.

    Captures via Python torch.cuda.graph context, snapshotting +
    restoring state around the warmup-and-capture sequence so the
    real state advances exactly once per replay.

    KNOWN BROKEN — same failing token sequence as v1. Most likely
    cause: cuBLAS workspaces or the rwkv::wkv_forward dispatcher
    lookup don't replay deterministically from a captured graph in
    this PyTorch build. Single-layer isolated test would confirm;
    deferred.
    """
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    n_embd = weights["n_embd"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]
    bufs = _get_stepped_bufs(weights)

    key = id(weights)
    if key not in _GRAPH_CACHE:
        # Capture all 12 layers. We use last_token's embedding as the
        # initial bufs[0] content so the capture sees realistic data.
        bufs[0].copy_(emb_w[last_token].view(1, 1, n_embd))
        snap = _snapshot_state(state)
        graphs = []
        for i in range(N_LAYER):
            # Restore state to the clean pre-call value before each
            # layer's capture so the graph is captured against the
            # correct starting state for layer i.
            _restore_state(state, snap)
            graphs.append(_capture_one_layer_graph(weights, state, i, bufs))
        # Restore one final time so the upcoming replay-loop sees
        # the same state the caller passed in.
        _restore_state(state, snap)
        _GRAPH_CACHE[key] = graphs

    graphs = _GRAPH_CACHE[key]
    # Set the input for this token.
    bufs[0].copy_(emb_w[last_token].view(1, 1, n_embd))
    for i in range(N_LAYER):
        graphs[i].replay()
    xn = F.layer_norm(bufs[N_LAYER].view(1, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous()).flatten()
    return logits


def forward_stepped_graphed(weights: dict, last_token: int, state):
    """Graph-captured per-layer T=1 forward.

    KNOWN BROKEN — DO NOT ENABLE FOR ROUNDTRIP. The underlying C++
    wrapper `rwkv4_layer_step_cpp_graphed` warms up the kernel
    twice + captures once on first invocation, all of which mutate
    the shared state tensors (att_xx, aa, bb, pp, ffn_xx) via
    copy_(). For per-token decoding that means state is advanced
    3× on the first token instead of 1×, corrupting subsequent
    decode steps. Confirmed by test_engine_cpp_roundtrip.py
    failing with KRUNCH_CPP_GRAPH=1.

    Fix path: either expose a separate `capture_layer_graph` +
    `replay_layer_graph` pair from C++ (so Python can snapshot/
    restore state around the warm-up), or reimplement the graph
    capture from Python using torch.cuda.graph against a non-
    state-mutating wrapper. Both ~half-day items.
    """
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]
    bufs = _get_stepped_bufs(weights)

    # Fill input buffer with the embedding of last_token.
    bufs[0].copy_(emb_w[last_token].view(1, 1, n_embd))
    for i in range(N_LAYER):
        krunch_ac_cuda.rwkv4_layer_step_cpp_graphed(
            i, bufs[i], bufs[i + 1],
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    xn = F.layer_norm(bufs[N_LAYER].view(1, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous()).flatten()
    return logits


def fresh_state_batched(weights: dict, B: int):
    """B-way batched RWKV state (att_xx[B,C], aa/bb/pp[B,n_att],
    ffn_xx[B,C]) per layer."""
    import torch
    device = weights["device"]
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    return (
        [torch.zeros(B, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
        [torch.zeros(B, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(B, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.full((B, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(B, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
    )


def forward_stepped_batched(weights: dict, last_tokens, state):
    """Run all 12 layers stepped (T=1) for B chunks in parallel.
    `last_tokens`: int32/long tensor [B] with each chunk's previous
    decoded token. Returns logits [B, V]. Mutates state in place.

    Bit-identical per-chunk to running forward_stepped sequentially
    on each chunk (verified via test_batched_stepped.py).
    """
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    if not isinstance(last_tokens, torch.Tensor):
        last_tokens = torch.as_tensor(last_tokens, dtype=torch.long,
                                       device=weights["device"])
    B = int(last_tokens.shape[0])
    x = emb_w[last_tokens].view(B, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    xn = F.layer_norm(x.view(B, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous())
    return logits  # [B, V]


def forward_stepped(weights: dict, last_token: int, state):
    """Run all 12 layers for one new token. Returns logits [V].
    Mutates state in place."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    device = weights["device"]
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    x = emb_w[last_token].view(1, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    xn = F.layer_norm(x.view(1, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous()).flatten()
    return logits  # [V]


def softmax_cdfs_per_row(logits_TxV):
    """Batched softmax + CDF via det_softmax_cdf kernel + torch.cumsum.
    Bit-identical between [T,V] and [1,V] invocation (verified) so
    encoder and decoder produce the same CDFs. The kernel writes
    counts; we scan with torch.cumsum (much faster than the kernel's
    own serial cumsum at V=50277)."""
    import torch
    import krunch_ac_cuda
    from krunch_ac.cdf import T as CDF_T
    counts = krunch_ac_cuda.det_softmax_cdf(logits_TxV.contiguous(), CDF_T)
    counts[:, 1:] = torch.cumsum(counts[:, 1:], dim=-1)
    return counts


def softmax_cdf_one_row(logits_V):
    """Single-row softmax + CDF for the stepped decoder path."""
    import torch
    import krunch_ac_cuda
    from krunch_ac.cdf import T as CDF_T
    counts = krunch_ac_cuda.det_softmax_cdf(
        logits_V.reshape(1, -1).contiguous(), CDF_T)
    counts[:, 1:] = torch.cumsum(counts[:, 1:], dim=-1)
    return counts[0].contiguous()


def cpp_path_enabled() -> bool:
    """Default ON as of 2026-04-30 — bit-exact GPU AC roundtrip is the
    only correct path. Set KRUNCH_CPP_PATH=0 to fall back to the
    legacy BlinkDL path (will FAIL roundtrip on GPU AC; kept for
    bench comparisons only)."""
    val = os.environ.get("KRUNCH_CPP_PATH", "1")
    return val == "1"


# Memory cost per cross-chunk batch slot (rough): RWKV-4-Pile-169M state
# (att_xx + aa/bb/pp + ffn_xx per layer × 12 layers) + 64 KB bitstream
# input + logits buffer + intermediate buffers. ~1.5 MB / slot includes
# headroom for cuBLAS workspace + allocator fragmentation.
_PER_BATCH_MEMORY_BYTES = 1.5 * 1024 * 1024


def compute_decompress_batch(n_chunks: int = -1, default: int = 64) -> int:
    """Pick cross-chunk batch B from runtime GPU specs + n_chunks.

    Replaces the prior static per-GPU table (which was extrapolated, not
    measured for non-T4 classes). Formula derived from kernel tile geometry:

      cp.async path (sm_80+, 64x64 tile, 4 warps):
        bottleneck matmul has N=768 -> 12 col-tiles. Want 1.5-wave
        coverage: ceil(B/64) * 12 ~= n_sms * 1.5  -> sat_B ~= n_sms * 8.

      legacy path (T4 sm_75, 16x16 single-warp):
        smaller per-tile work, more tiles -> sat_B ~= n_sms * 2.

    Result clamped by (a) n_chunks for the current call and (b) a memory
    budget that fits inside 80 % of VRAM.

    Predicted vs prior static table:
      T4 (40 SMs, sm_75):    ~80   (table: 64)
      A10G (80 SMs, sm_86):  ~640  (table: 128) -- table was 5x low
      L4 (58 SMs, sm_89):    ~464  (table: 128) -- table was 3.6x low
      L40S (142 SMs, sm_89): ~1136 (table: 256) -- table was 4.4x low
      A100 (108 SMs, sm_80): ~864  (table: 256) -- table was 3.4x low
      H100 (132 SMs, sm_90): ~1056 (table: 512) -- table was 2x low

    Override via env `KRUNCH_DECOMPRESS_BATCH=N`.
    """
    env = os.environ.get("KRUNCH_DECOMPRESS_BATCH")
    if env:
        try:
            B = max(1, int(env))
            return min(n_chunks, B) if n_chunks > 0 else B
        except ValueError:
            pass
    try:
        import torch
        if not torch.cuda.is_available():
            return default if n_chunks <= 0 else max(1, min(n_chunks, default))
        p = torch.cuda.get_device_properties(0)
        n_sms = int(p.multi_processor_count)
        if p.major >= 8:           # cp.async path (A10G/A100/L4/L40S/H100)
            saturation = n_sms * 8
        else:                       # T4 sm_75 legacy path
            saturation = n_sms * 2
        mem_cap = int(p.total_memory * 0.8 / _PER_BATCH_MEMORY_BYTES)
        upper = max(1, min(saturation, mem_cap))
    except Exception:
        upper = default
    if n_chunks > 0:
        return max(1, min(n_chunks, upper))
    return max(1, upper)


# Back-compat alias for callers that don't (yet) pass n_chunks.
def pick_decompress_batch(default: int = 64) -> int:
    return compute_decompress_batch(n_chunks=-1, default=default)
