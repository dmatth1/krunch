"""Bit-exact C++ orchestration path for compress + decompress.

Wraps `krunch_ac_cuda.rwkv4_layer_step_cpp{,_t1}` + the deterministic
matmul kernel + per-row softmax/CDF so encoder (packed T) and decoder
(stepped T=1) produce byte-identical AC bitstreams. This is the only
GPU AC path; `KRUNCH_DETERMINISTIC_MATMUL=1` is forced on at import.
"""
from __future__ import annotations

import os

# Default deterministic matmul ON before any C++ extension import.
# The C++ static-const reads this once at module load; setting it
# afterward has no effect.
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

# Auto-disable sm_80+ kernels (cp.async WMMA) on sm_75 and older (e.g. T4).
# Both det_matmul_tc_async and det_matmul_tc_3way_async have empty kernel
# bodies under #if __CUDA_ARCH__ >= 800 — invoking them on sm_75 leaves
# outputs uninitialized (garbage from at::empty), breaking the roundtrip.
# Must run before krunch_ac_cuda import so the C++ statics read correctly.
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _sm = _torch.cuda.get_device_properties(0).major
        if _sm < 8:
            os.environ.setdefault("KRUNCH_HEAD_ASYNC", "0")
            os.environ.setdefault("KRUNCH_3WAY_ASYNC", "0")
    del _torch, _sm
except Exception:
    pass

N_LAYER = 12

_WEIGHTS_CACHE: dict[int, dict] = {}


def _quantize_int8_per_output_channel(w):
    """Per-output-channel symmetric int8 quantization (no dequant).
    Returns (w_q [K, N] int8, scale [N] fp16). Pairs with the W8A8
    int8 WMMA kernel at integration time.

    Same recipe as _quantize_dequant_int8_per_output_channel — same
    rounding, same scale formula — but returns the raw int8 buffer
    + scale instead of dequantizing back to fp16. Used by init_weights
    when KRUNCH_INT8_W8A8=1 to populate the W8A8 weight bundle.
    """
    import torch
    if w.dim() != 2:
        raise ValueError(f"expected 2D matrix; got shape {tuple(w.shape)}")
    w_f = w.to(torch.float32)
    max_abs = w_f.abs().max(dim=0, keepdim=True).values  # [1, N]
    scale = (max_abs / 127).clamp(min=1e-8)              # [1, N]
    w_q = (w_f / scale).round().clamp(-127, 127).to(torch.int8)
    return w_q.contiguous(), scale.squeeze(0).to(torch.float16).contiguous()


def _quantize_dequant_int8_per_output_channel(w):
    """Per-output-channel SYMMETRIC int8 quantization SPIKE for the
    W8A8 path (Phase 2A, 2026-05-06). Validates quality before
    investing in the int8 WMMA kernel.

    For [K, N] matrix, quantizes per-column (per-output-channel):
        scale[j] = max(|W[:, j]|) / 127
        W_q[k, j] = round(W[k, j] / scale[j])  ∈ [-127, 127]
        W_dequant[k, j] = W_q[k, j] * scale[j]

    Per-OUTPUT-channel (vs Phase 1 spike's per-INPUT-channel) because
    in the W8A8 int8-WMMA path, the scale must be applied AFTER WMMA
    accumulates along K → scale must be constant across K, i.e. per-N.

    SYMMETRIC (no offset) because it's the standard W8A8 recipe — the
    inner WMMA is pure int8 × int8 → int32; output is multiplied by
    (scale_act[m] * scale_weight[n]) to recover fp16. With offsets,
    the math has an extra correction term.

    Same encoder/decoder application → same dequantized weights →
    bit-exact AC roundtrip per model_id.
    """
    import torch
    if w.dim() != 2:
        return w
    w_f = w.to(torch.float32)
    max_abs = w_f.abs().max(dim=0, keepdim=True).values  # [1, N]
    scale = (max_abs / 127).clamp(min=1e-8)              # [1, N]
    w_q = (w_f / scale).round().clamp(-127, 127)
    w_d = (w_q * scale).to(w.dtype)
    return w_d


def _quantize_dequant_uint8_per_input_channel(w):
    """Per-input-channel uint8 quantization SPIKE: quantizes a [K, N]
    fp16 matrix to uint8 then dequantizes back to fp16, simulating the
    numerical precision of an int8 weight × fp16 activation matmul kernel
    without actually building one yet.

    Quality test: does AC ratio degrade enough to disqualify int8?
    Encoder + decoder both apply the same quantize→dequant transform,
    so output bytes are bit-identical between them; the only question
    is ratio vs the fp16 baseline. If ratio holds → invest in the real
    int8 kernel. If ratio tanks → revert.

    Per-row (per-input-channel) scheme matches Bellard / harrisonvanderbyl
    /rwkv-cpp-accelerated:
        for i in [0, K):
            mn = min(W[i, :]); mx = max(W[i, :])
            scale[i] = (mx - mn) / 255
            offset[i] = mn
            W_q[i, :] = round((W[i, :] - mn) / scale[i])  ∈ [0, 255]
            W_dequant[i, :] = W_q[i, :] * scale[i] + offset[i]
    """
    import torch
    if w.dim() != 2:
        return w
    w_f = w.to(torch.float32)
    mn = w_f.min(dim=1, keepdim=True).values
    mx = w_f.max(dim=1, keepdim=True).values
    scale = ((mx - mn) / 255).clamp(min=1e-8)
    w_q = ((w_f - mn) / scale).round().clamp(0, 255)
    w_d = (w_q * scale + mn).to(w.dtype)
    return w_d


def init_weights(model, device: str = "cuda") -> dict:
    """Extract per-layer weights once per (model, device). Cached by
    id(model).

    KRUNCH_BF16=1 converts the 7 layer matmul weights (Kw/Vw/Rw/Ow +
    ffn_Kw/Vw/Rw) to bf16 — gemm_fp16 in layer_cpp.cpp detects bf16
    weight dtype and routes through the bf16 cp.async WMMA kernel.
    Bytes diverge from the fp16 codec → v2 model_id territory; encoder
    + decoder must both set the flag for the AC roundtrip to hold.

    KRUNCH_INT8_WEIGHTS=1 quantizes all matmul weights (KVR, Ow, FFN
    R/K/V) + emb + head to per-input-channel uint8, then dequantizes
    back to fp16 in-place. Bytes diverge from the fp16 codec → v2
    model_id (set 2 in the blob header, encoder + decoder must agree).
    Phase 1 spike: validated ratio quality (+0.15% on WildChat T4).

    KRUNCH_INT8_W8A8=1 uses per-OUTPUT-channel SYMMETRIC int8 quant
    (matches the W8A8 int8-WMMA kernel recipe). Phase 2A spike
    validates ratio for the W8A8 path before building the kernel.
    Mutually exclusive with KRUNCH_INT8_WEIGHTS=1.
    """
    import torch
    key = (id(model),
           os.environ.get("KRUNCH_BF16", "0"),
           os.environ.get("KRUNCH_INT8_WEIGHTS", "0"),
           os.environ.get("KRUNCH_INT8_W8A8", "1"))
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]

    w = model.w
    use_bf16 = os.environ.get("KRUNCH_BF16") == "1"
    use_int8 = os.environ.get("KRUNCH_INT8_WEIGHTS") == "1"
    use_w8a8 = os.environ.get("KRUNCH_INT8_W8A8", "1") == "1"
    if use_int8 and use_w8a8:
        raise ValueError("Set at most one of KRUNCH_INT8_WEIGHTS / KRUNCH_INT8_W8A8")

    def fix(t, dt=None, quantize=False):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt:
            t = t.to(dtype=dt).contiguous()
        if quantize:
            if use_int8:
                t = _quantize_dequant_uint8_per_input_channel(t).contiguous()
            elif use_w8a8:
                t = _quantize_dequant_int8_per_output_channel(t).contiguous()
        return t

    def matmul_dtype():
        return torch.bfloat16 if use_bf16 else torch.float16

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
            fix(w[att+'key.weight'],        matmul_dtype(), quantize=True),
            fix(w[att+'value.weight'],      matmul_dtype(), quantize=True),
            fix(w[att+'receptance.weight'], matmul_dtype(), quantize=True),
            fix(w[att+'output.weight'],     matmul_dtype(), quantize=True),
            fix(w[bbb+'ln2.weight'], torch.float16),
            fix(w[bbb+'ln2.bias'], torch.float16),
            fix(w[ffn+'time_mix_k'].squeeze(), torch.float16),
            fix(w[ffn+'time_mix_r'].squeeze(), torch.float16),
            fix(w[ffn+'key.weight'],        matmul_dtype(), quantize=True),
            fix(w[ffn+'value.weight'],      matmul_dtype(), quantize=True),
            fix(w[ffn+'receptance.weight'], matmul_dtype(), quantize=True),
        ])

    bundle = {
        "layers": layers,
        # emb.weight is [vocab, n_embd] = [50277, 768] = 38M params (~22% of
        # the model). Quantize too for the spike — the embedding lookup just
        # reads a single row per token, so int8 dequant during read is
        # straightforward in a real kernel.
        "emb_w": fix(w['emb.weight'], torch.float16, quantize=True),
        "ln_out_w": fix(w['ln_out.weight'], torch.float16),
        "ln_out_b": fix(w['ln_out.bias'], torch.float16),
        # head.weight is [vocab, n_embd] = [50277, 768] = 38M params, the
        # other big chunk. Quantize for the spike.
        "head_w": fix(w['head.weight'], torch.float16, quantize=True),
        "n_embd": int(model.args.n_embd),
        "n_att": int(model.args.n_att),
        "device": device,
    }

    # W8A8 (Phase 2A Step 3 integration): also store raw int8 weights +
    # per-output-channel scales for the 7 layer matmul weights. These are
    # what `rwkv4_layer_step_cpp_w8a8` consumes; the existing fp16 layer
    # weights in `layers` (already dequant-of-int8 per `fix(quantize=True)`)
    # are kept for the decompress path which still uses the fp16 kernel
    # (W8A8 regresses at small M; M-fallback handled at the layer-step
    # level).
    if use_w8a8:
        # Order MUST match rwkv4_layer_step_cpp's matmul arg order:
        # 0:Kw, 1:Vw, 2:Rw, 3:Ow, 4:ffn_Kw, 5:ffn_Vw, 6:ffn_Rw.
        layers_w8a8_int8 = []
        layers_w8a8_scale = []
        for i in range(N_LAYER):
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."
            ints, scales = [], []
            for name in ('key.weight', 'value.weight',
                          'receptance.weight', 'output.weight'):
                wf = w[att+name].to(device).to(torch.float16).contiguous()
                wq, ws = _quantize_int8_per_output_channel(wf)
                ints.append(wq); scales.append(ws)
            for name in ('key.weight', 'value.weight', 'receptance.weight'):
                wf = w[ffn+name].to(device).to(torch.float16).contiguous()
                wq, ws = _quantize_int8_per_output_channel(wf)
                ints.append(wq); scales.append(ws)
            layers_w8a8_int8.append(ints)
            layers_w8a8_scale.append(scales)
        bundle["w8a8_int8"] = layers_w8a8_int8     # list of 12 lists of 7 int8 [K,N]
        bundle["w8a8_scale"] = layers_w8a8_scale   # list of 12 lists of 7 fp16 [N]

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

    # W8A8 dispatch (Phase 2A Step 3): compress packed forward at M >= 256
    # uses the int8 Tensor Core path (1.5-2.2× per-call speedup at our
    # shapes). Decompress (small M) keeps using the existing fp16 layer
    # step on the dequant-fp16 weights stored in `layers` — same effective
    # weights, so encoder W8A8 and decoder fp16 produce identical outputs
    # (modulo activation-quant noise; verified by AC roundtrip at
    # integration time).
    # Must mirror decoder's gate exactly (forward_stepped_batched line ~825)
    # for AC bit-exactness — encoder fp16 + decoder W8A8 produces different
    # logits and breaks roundtrip on partial windows.
    use_w8a8 = ("w8a8_int8" in weights)
    if use_w8a8:
        w8a8_int8 = weights["w8a8_int8"]
        w8a8_scale = weights["w8a8_scale"]
        # Per layer, layers[i][:7] are non-matmul (LN1+bias, tm_k/v/r,
        # time_decay, time_first), layers[i][7:11] are att matmuls
        # (Kw,Vw,Rw,Ow), layers[i][11:13] are LN2, layers[i][13:15] are
        # ffn_tm, layers[i][15:18] are ffn matmuls (Kw,Vw,Rw). The W8A8
        # function takes the non-matmul args + (W_q,W_s) tuples for the
        # 7 matmuls, in original arg-order: K, V, R, O, ffn_K, ffn_V, ffn_R.
        for i in range(N_LAYER):
            L = layers[i]
            ints = w8a8_int8[i]    # [Kw,Vw,Rw,Ow,ffn_Kw,ffn_Vw,ffn_Rw]
            scales = w8a8_scale[i]
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_w8a8(
                x.contiguous(),
                state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
                # ln1, time_mix, time_decay/first
                L[0], L[1], L[2], L[3], L[4], L[5], L[6],
                # K, V, R, O matmul tuples
                ints[0], scales[0], ints[1], scales[1],
                ints[2], scales[2], ints[3], scales[3],
                # ln2, ffn_time_mix
                L[11], L[12], L[13], L[14],
                # ffn K, V, R matmul tuples
                ints[4], scales[4], ints[5], scales[5], ints[6], scales[6],
            )
    else:
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


# =============================================================================
# Whole-step CUDA Graph (Week 1, 2026-05-06).
# Captures emb lookup → 12 layers → LN_out → head matmul → softmax + CDF →
# AC decode batched into ONE graph object per (model, B). Replays 1× per
# decompress step instead of 12 (per-layer graphs) or 18+ (eager).
#
# Per V1_PLAN tier-3 sprint plan: closes the gap to ts_zip-class throughput
# by collapsing the kernel-launch overhead that dominated decompress per-step
# wall on T4 / A10G. The KTransformers SOSP'25 pattern.
# =============================================================================

_FULL_STEP_BUFS_CACHE: dict[tuple, dict] = {}
_FULL_STEP_GRAPH_CACHE: dict[tuple, object] = {}


def _get_full_step_bufs(weights: dict, B: int):
    """Pointer-stable I/O buffers for the whole-step decompress graph.

    Graphs are pointer-bound; `last_input` and `out_syms` MUST live at the
    same address across replays. The graph reads `last_input` at the start
    of each step (caller updates with previous step's decoded tokens) and
    writes `out_syms` at the end (caller reads after replay).
    """
    import torch
    key = (id(weights), B)
    if key in _FULL_STEP_BUFS_CACHE:
        return _FULL_STEP_BUFS_CACHE[key]
    device = weights["device"]
    bufs = {
        "last_input": torch.empty(B, dtype=torch.long, device=device),
        "out_syms":   torch.empty(B, dtype=torch.int32, device=device),
    }
    _FULL_STEP_BUFS_CACHE[key] = bufs
    return bufs


def _eager_full_step(weights, last_input_long, ac_state, input_buf,
                      base_byte_offsets, state, out_syms, B: int):
    """One full decompress step: emb → 12 layers → ln_out → head → softmax+CDF
    → AC decode batched. Mutates state, ac_state, out_syms in place. Used for
    warmup before graph capture and as the eager fallback path."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    from krunch_ac.cdf import T as CDF_T

    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    x = emb_w[last_input_long].view(B, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    xn = F.layer_norm(x.view(B, n_embd), (n_embd,),
                       weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous())
    # softmax + CDF (per-row, bit-identical between encoder T-pack and
    # decoder B-batched paths because the kernel is shape-invariant).
    cdfs = krunch_ac_cuda.det_softmax_cdf(logits.contiguous(), CDF_T)
    cdfs[:, 1:] = torch.cumsum(cdfs[:, 1:], dim=-1)
    krunch_ac_cuda.decode_step_batched(
        cdfs, input_buf, base_byte_offsets, ac_state, out_syms)


def forward_step_full_graphed_v3(weights: dict, ac_state, input_buf,
                                   base_byte_offsets, state, B: int):
    """Capture & replay one full decompress step in a single CUDA graph.

    REQUIRES: `KRUNCH_OWN_WKV=1` (graph-safe WKV is prerequisite — the pip
    rwkv WKV op uses a c10 dispatcher lookup that doesn't replay
    deterministically; verified by tests/gpu/test_t31_graph_diagnostic.py).

    First call captures (snapshots state, runs 2 warmup passes, captures on
    a side stream, restores state). Subsequent calls replay the captured
    graph — no Python or ATen launch overhead per step beyond `g.replay()`.

    Reads `bufs['last_input']` (stable). Writes `bufs['out_syms']` (stable).
    Mutates `state[*][i]` and `ac_state` per replay.

    Caller pattern:

        bufs = _get_full_step_bufs(weights, B)
        bufs['last_input'].fill_(BOS_TOKEN)
        for t in range(max_T):
            out_syms_buf = forward_step_full_graphed_v3(
                weights, ac_state, input_buf, base_byte_offsets, state, B)
            decoded_tokens[:, t] = out_syms_buf
            bufs['last_input'].copy_(out_syms_buf)  # int32 → int64 cast
    """
    import torch

    bufs = _get_full_step_bufs(weights, B)
    # Cache key includes data ptrs so a graph captured against one set of
    # caller-passed buffers (ac_state, input_buf) is invalidated if the
    # caller passes different ones — would dangle on replay.
    cache_key = (id(weights), B, ac_state.data_ptr(), input_buf.data_ptr())

    if cache_key not in _FULL_STEP_GRAPH_CACHE:
        # Capture warmup mutates state / ac_state / last_input. Snapshot
        # so the first real replay sees the caller's clean starting state.
        snap_state = _snapshot_state(state)
        snap_ac = ac_state.clone()
        snap_li = bufs["last_input"].clone()

        # 2 warmup passes prime cuBLAS workspaces and allocator pool.
        for _ in range(2):
            _eager_full_step(weights, bufs["last_input"], ac_state,
                              input_buf, base_byte_offsets, state,
                              bufs["out_syms"], B)
        torch.cuda.synchronize()

        # Restore for the capture pass.
        _restore_state(state, snap_state)
        ac_state.copy_(snap_ac)
        bufs["last_input"].copy_(snap_li)

        g = torch.cuda.CUDAGraph()
        cap_stream = torch.cuda.Stream()
        cap_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(g):
                _eager_full_step(weights, bufs["last_input"], ac_state,
                                  input_buf, base_byte_offsets, state,
                                  bufs["out_syms"], B)
        torch.cuda.current_stream().wait_stream(cap_stream)

        # Final restore — capture mutated state once more.
        _restore_state(state, snap_state)
        ac_state.copy_(snap_ac)
        bufs["last_input"].copy_(snap_li)

        _FULL_STEP_GRAPH_CACHE[cache_key] = g

    _FULL_STEP_GRAPH_CACHE[cache_key].replay()
    return bufs["out_syms"]


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
    decode steps. Confirmed by tests/gpu/test_engine_cpp_roundtrip.py
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


# Cached state tensors per (weights, B) — required for CUDA-graph-captured
# decompress because graphs are bound to specific tensor data-pointer addresses.
# Re-creating state across decompress calls would dangle captured graphs.
_BATCHED_STATE_CACHE: dict[tuple, tuple] = {}


def fresh_state_batched_cached(weights: dict, B: int):
    """Get cached B-batched state (same tensor identity across calls), reset
    in place. Required when graph capture is used (graphs are pointer-bound;
    re-allocating state would dangle them). Bit-equivalent to fresh_state_batched
    on first call after reset."""
    import torch
    key = (id(weights), B)
    if key not in _BATCHED_STATE_CACHE:
        _BATCHED_STATE_CACHE[key] = fresh_state_batched(weights, B)
    state = _BATCHED_STATE_CACHE[key]
    # In-place reset to fresh values (matches fresh_state_batched return).
    for t in state[0]: t.zero_()  # att_xx (fp16)
    for t in state[1]: t.zero_()  # aa (fp32)
    for t in state[2]: t.zero_()  # bb (fp32)
    for t in state[3]: t.fill_(-1e30)  # pp (fp32)
    for t in state[4]: t.zero_()  # ffn_xx (fp16)
    return state


def forward_stepped_batched(weights: dict, last_tokens, state):
    """Run all 12 layers stepped (T=1) for B chunks in parallel.
    `last_tokens`: int32/long tensor [B] with each chunk's previous
    decoded token. Returns logits [B, V]. Mutates state in place.

    Bit-identical per-chunk to running forward_stepped sequentially
    on each chunk (verified via tests/gpu/test_batched_stepped.py).

    W8A8 dispatch (Phase 2A Step 3, 2026-05-06): when bundle has
    `w8a8_int8` (set by init_weights when KRUNCH_INT8_W8A8=1), uses the
    W8A8 layer step. Required for bit-exact AC roundtrip — encoder
    uses W8A8 packed; decoder must use SAME quant scheme to produce
    matching logits. Cost: W8A8 regresses at small M (microbench
    0.49× at M=16); the regression is the price of bit-exact W8A8.
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

    use_w8a8 = ("w8a8_int8" in weights)
    if use_w8a8:
        w8a8_int8 = weights["w8a8_int8"]
        w8a8_scale = weights["w8a8_scale"]
        for i in range(N_LAYER):
            L = layers[i]
            ints = w8a8_int8[i]
            scales = w8a8_scale[i]
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_w8a8(
                x.contiguous(),
                state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
                L[0], L[1], L[2], L[3], L[4], L[5], L[6],
                ints[0], scales[0], ints[1], scales[1],
                ints[2], scales[2], ints[3], scales[3],
                L[11], L[12], L[13], L[14],
                ints[4], scales[4], ints[5], scales[5], ints[6], scales[6],
            )
    else:
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
    own serial cumsum at V=50277; D1 Phase 0c fused-scan attempt
    regressed compress 25-30% vs this — torch.cumsum's thrust-based
    scan is very tuned; not worth replacing)."""
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
