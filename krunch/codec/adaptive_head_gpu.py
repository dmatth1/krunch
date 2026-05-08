"""
GPU adaptive head — bit-exact match to the CPU reference at fp64.

The CPU reference in `adaptive_head.py` uses numpy fp64 throughout for
deterministic encoder/decoder symmetry. This module mirrors that with
torch fp64 ops on CUDA so the encoder + decoder can run on-GPU without
shipping V=50K probabilities to host every token.

Critical invariant: for the same input sequence, the output of
`AdaptiveHeadGPU.adjust()` must byte-equal the CPU reference at every
step. Any divergence breaks AC roundtrip. Pinned by
`tests/unit/gpu/test_adaptive_head_parity.py`.

The op order here MUST match `adaptive_head.py::adjust` exactly:

    log_buf = log(probs + 1e-10)
    log_buf += bias
    log_buf -= log_buf.max()
    log_buf = exp(log_buf)
    log_buf /= log_buf.sum()

Likewise for `update`:

    grad = adjusted.copy()
    grad[actual_token] -= 1.0
    bias -= lr * grad

Float64 ops are slow on consumer GPUs (no fp64 tensor cores), but for
a [B, V]-sized vector at B<=256, V=50277, total FLOPs are dominated by
the kernel-launch overhead, not the compute. Measured: ~0.1 ms per
adjust+update at B=161 V=50K on A10G — negligible against the ~6ms /
step layer-step cost.

Per docs/TIER_3_OPTIMIZATION.md NEXT-3.
"""

from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from .adaptive_head import DEFAULT_LR


class AdaptiveHeadGPU:
    """Batched torch-fp64 adaptive head.

    State: bias ``[B, V]`` float64 on GPU. Encoder runs at B=1 (one
    bias state per chunk), decoder runs at B=n_chunks (one bias state
    per concurrent chunk). Both call adjust + update sequentially per
    token.

    Use-case asymmetry between encode and decode:

    - Encoder: probs come from `forward_packed_window` which emits
      a [T, V] block per layer-step batch. Encoder still has to call
      adjust + update per-token sequentially (bias depends on prior
      token's grad). bias here has shape [1, V].
    - Decoder: ``_decompress_chunks_batched_cpp`` already does one
      stepped forward per timestep over B chunks. bias here has
      shape [B, V] and updates B rows in lockstep.
    """

    def __init__(self, vocab_size: int, batch_size: int = 1,
                 lr: float = DEFAULT_LR, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("torch not available")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not (0.0 < lr < 1.0):
            raise ValueError(f"lr must be in (0, 1), got {lr}")
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = float(lr)
        self.device = device
        self.bias = torch.zeros((batch_size, vocab_size),
                                  dtype=torch.float64, device=device)

    def reset(self) -> None:
        """Zero out the bias state across all rows."""
        self.bias.zero_()

    def adjust(self, probs: "torch.Tensor") -> "torch.Tensor":
        """Apply adaptive bias.

        Args:
            probs: [V] or [B, V], any float dtype on GPU. Cast to fp64.

        Returns:
            adjusted: same batched shape as ``probs``, fp64. Sums to 1
            along the V axis (within fp64 rounding).
        """
        if probs.dim() == 1:
            assert probs.shape == (self.vocab_size,), \
                f"probs shape {tuple(probs.shape)} != ({self.vocab_size},)"
            assert self.batch_size == 1, \
                "1-D probs requires batch_size=1"
            x = probs.to(torch.float64).unsqueeze(0)  # [1, V]
            squeeze = True
        else:
            assert probs.shape == (self.batch_size, self.vocab_size), \
                f"probs shape {tuple(probs.shape)} != ({self.batch_size}, {self.vocab_size})"
            x = probs.to(torch.float64)
            squeeze = False

        # Op order MUST mirror adaptive_head.py::adjust.
        x = torch.log(x + 1e-10)
        x = x + self.bias
        x = x - x.amax(dim=-1, keepdim=True)
        x = torch.exp(x)
        x = x / x.sum(dim=-1, keepdim=True)

        if squeeze:
            return x.squeeze(0)
        return x

    def update(self, actual_token, adjusted_probs: "torch.Tensor") -> None:
        """Update bias after observing actual_token.

        Args:
            actual_token: int (B=1 case) or [B] long tensor on the same
                device as ``bias``.
            adjusted_probs: [V] or [B, V] fp64, the array returned from
                this step's adjust().
        """
        if adjusted_probs.dim() == 1:
            assert adjusted_probs.shape == (self.vocab_size,)
            adj = adjusted_probs.unsqueeze(0)
            if isinstance(actual_token, int):
                token_t = torch.tensor([actual_token], dtype=torch.long,
                                         device=self.device)
            else:
                token_t = actual_token.view(1).to(self.device).long()
        else:
            assert adjusted_probs.shape == (self.batch_size, self.vocab_size)
            adj = adjusted_probs
            assert not isinstance(actual_token, int), \
                "B>1 update needs a [B] long tensor of tokens"
            token_t = actual_token.to(self.device).long()
            assert token_t.shape == (self.batch_size,)

        # grad = adjusted; grad[i, actual_token[i]] -= 1
        grad = adj.clone()
        b_idx = torch.arange(grad.shape[0], device=self.device)
        grad[b_idx, token_t] -= 1.0
        self.bias -= self.lr * grad


def encode_with_adaptive_head_gpu(probs_seq, tokens, lr: float = DEFAULT_LR,
                                    device: str = "cuda"):
    """Roundtrip-friendly encoder helper for tests.

    Mirrors `adaptive_head.encode_with_adaptive_head` but on GPU.
    Returns (adjusted_seq [T, V] fp64, final_bias [V] fp64).
    """
    if torch is None:
        raise RuntimeError("torch not available")
    if probs_seq.dim() != 2:
        raise ValueError(f"probs_seq must be 2-D, got {tuple(probs_seq.shape)}")
    T, V = probs_seq.shape
    assert tokens.shape == (T,), f"tokens shape {tuple(tokens.shape)} != ({T},)"
    head = AdaptiveHeadGPU(V, batch_size=1, lr=lr, device=device)
    out = torch.empty((T, V), dtype=torch.float64, device=device)
    for t in range(T):
        adj_t = head.adjust(probs_seq[t])
        out[t] = adj_t
        head.update(int(tokens[t].item()), adj_t)
    return out, head.bias[0].clone()
