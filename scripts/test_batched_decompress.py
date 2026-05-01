"""End-to-end batched decompress.

Compress B independent chunks (single-stream encoder via cpp_path),
then decompress them all in parallel via cross-chunk batched stepped
forward + krunch_ac_cuda.decode_step_batched.

Asserts: byte-exact roundtrip per chunk, AND aggregate decompress
KB/s across the batch. Compares to single-stream decompress KB/s
(sum of B independent decompresses) to show the batching speedup.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import time
import struct
import torch
import krunch_ac_cuda
from krunch.inference import InferenceEngine, BOS_TOKEN
from krunch import cpp_path


def make_chunks(n_chunks: int, chunk_kb: int):
    base = (
        b"Krunch is a neural compression codec for text built around "
        b"a small RWKV-4 language model and a custom GPU arithmetic coder. "
        b"It targets large datasets where ratio matters more than latency. "
    )
    out = bytearray()
    while len(out) < chunk_kb * 1024:
        out.extend(base)
    one = bytes(out[:chunk_kb * 1024])
    return [one for _ in range(n_chunks)]


def main():
    chunk_kb = int(os.environ.get("CHUNK_KB", 8))
    B = int(os.environ.get("B", 8))

    eng = InferenceEngine()
    eng.load()
    weights = cpp_path.init_weights(eng._model, eng._device)
    device = eng._device

    chunks = make_chunks(B, chunk_kb)
    encoded = [eng.compress_chunk(c) for c in chunks]
    n_tokens_per_chunk = [struct.unpack(">II", e[:8])[1] for e in encoded]
    print(f"prepped B={B} chunks of {chunk_kb} KB; n_tokens per chunk = {n_tokens_per_chunk}",
          flush=True)
    bitstreams = [e[8:] for e in encoded]

    # ============ Sequential reference (one-chunk-at-a-time) ============
    skip_seq = os.environ.get("SKIP_SEQ") == "1"
    if not skip_seq:
        t0 = time.time()
        seq_decoded = [eng.decompress_chunk(e) for e in encoded]
        t_seq = time.time() - t0
    else:
        t_seq = float("nan")

    # ============ Batched: B chunks in parallel ============
    n_tok_max = max(n_tokens_per_chunk)
    # pad bitstreams to a common length so we can stack offsets cleanly
    max_bs_len = max(len(bs) for bs in bitstreams) + 64
    cat = bytearray()
    base_offsets = []
    for bs in bitstreams:
        base_offsets.append(len(cat))
        cat.extend(bs)
        cat.extend(b"\x00" * (max_bs_len - len(bs)))
    in_buf = torch.frombuffer(bytes(cat), dtype=torch.uint8).to(device)
    base_offsets_t = torch.tensor(base_offsets, dtype=torch.int32, device=device).contiguous()

    # Init B AC decoder states (each reads its own [PRECISION] header bits)
    DecodeState_size = 4  # uint32 fields: low, high, value, bit_offset
    ac_states = torch.zeros(B * DecodeState_size, dtype=torch.uint32, device=device)
    krunch_ac_cuda.decode_init_batched(in_buf, base_offsets_t, ac_states)

    # Init B-way RWKV state
    state = cpp_path.fresh_state_batched(weights, B)
    out_syms = torch.empty(B, dtype=torch.int32, device=device)

    # Decoded tokens accumulator
    decoded_toks = [[] for _ in range(B)]
    # Track which chunks are done (have decoded their full n_tokens)
    done = [False] * B

    last = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=device)
    from krunch_ac.cdf import T as CDF_T

    t0 = time.time()
    with torch.no_grad():
        for step in range(n_tok_max):
            logits = cpp_path.forward_stepped_batched(weights, last, state)  # [B, V]
            counts = krunch_ac_cuda.det_softmax_cdf(logits.contiguous(), CDF_T)
            counts[:, 1:] = torch.cumsum(counts[:, 1:], dim=-1)
            cdfs = counts.contiguous()  # [B, V+1]
            V = int(cdfs.shape[1]) - 1
            krunch_ac_cuda.decode_step_batched(cdfs, in_buf, base_offsets_t,
                                                ac_states, out_syms)
            toks_cpu = out_syms.cpu().tolist()
            for b in range(B):
                if not done[b] and len(decoded_toks[b]) < n_tokens_per_chunk[b]:
                    decoded_toks[b].append(toks_cpu[b])
                    if len(decoded_toks[b]) == n_tokens_per_chunk[b]:
                        done[b] = True
            # Feed each chunk's just-decoded token as next input
            last = out_syms.long()
            if all(done):
                break
    torch.cuda.synchronize()
    t_bat = time.time() - t0

    # Verify per chunk
    bat_decoded = []
    for b in range(B):
        text = eng._tokenizer.decode(decoded_toks[b])
        # original chunks all same; decode bytes capped to original length
        orig_len = len(chunks[b])
        bat_decoded.append(text.encode("utf-8")[:orig_len])

    n_pass = sum(1 for b in range(B) if bat_decoded[b] == chunks[b])
    total_in = sum(len(c) for c in chunks)
    if not skip_seq:
        print(f"\nseq decompress: {total_in / 1024 / t_seq:.1f} KB/s "
              f"({t_seq:.2f}s for {B} chunks)")
    print(f"bat decompress: {total_in / 1024 / t_bat:.1f} KB/s "
          f"({t_bat:.2f}s for {B} chunks)")
    if not skip_seq:
        print(f"speedup:        {t_seq / t_bat:.2f}×")
    print(f"\nbyte-exact: {n_pass}/{B} chunks PASS")

    raise SystemExit(0 if n_pass == B else 1)


if __name__ == "__main__":
    main()
