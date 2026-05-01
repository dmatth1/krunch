"""Multi-process decompress worker pool.

Each worker is a separate Python process with its own CUDA context, holding
its own model copy. Decompress chunks are dispatched round-robin (or
work-stealing) to workers; results are aggregated in input order.

Why processes, not threads: under our GPU-AC decompress path, every per-token
step ends in a `.item()` GPU sync that blocks the calling thread. Threads
also share one CUDA context + one Python GIL, so kernel launches and `.item()`
syncs serialize. Processes have independent CUDA contexts → real GPU-level
overlap. T4 measurements (2026-04-30): N=2 gives 1.81× aggregate decompress
throughput, N=4 gives 2.12×; A10G expected 4-8× by N=4-8.

Memory: ~500 MB GPU per worker (RWKV-4-Pile-169M fp16 + state + tokenizer).
T4 (16 GB) fits 8-12 workers; A10G (24 GB) fits 16+. We default to env-tuned
KRUNCH_DECOMPRESS_WORKERS (default 4 on GPU, 1 on CPU).
"""

from __future__ import annotations

import os
import sys
import logging
import multiprocessing as mp
from typing import Optional

logger = logging.getLogger(__name__)


def _worker_main(in_q: "mp.Queue", out_q: "mp.Queue") -> None:
    """Worker entrypoint. Loads its own InferenceEngine, then loops on the
    input queue. Each item is `(chunk_idx, encoded_bytes)`; output is
    `(chunk_idx, decoded_bytes_or_exc_str)`. Sentinel `None` ends the loop.
    """
    # Import inside the worker so the parent's stdout-redirect / library
    # state doesn't carry across the spawn boundary.
    from krunch.inference import InferenceEngine

    # Workers must not write to stdout — CLI piping reserves stdout for
    # the final blob. Redirect to stderr defensively.
    os.dup2(2, 1)

    engine = InferenceEngine()
    engine.load()
    out_q.put(("ready", os.getpid()))
    while True:
        item = in_q.get()
        if item is None:
            break
        idx, encoded = item
        try:
            decoded = engine.decompress_chunk(encoded)
            out_q.put((idx, decoded))
        except Exception as e:  # noqa: BLE001
            out_q.put((idx, RuntimeError(f"worker {os.getpid()} chunk {idx}: {e}")))


class DecompressWorkerPool:
    """Spawn N processes; dispatch decompress chunks; collect ordered results.

    Use as a context manager or call .close() when done. CUDA-safe via
    `mp.get_context('spawn')` — fork would inherit the parent's CUDA context
    and break (each worker needs its own).
    """

    def __init__(self, n_workers: int):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.n_workers = n_workers
        ctx = mp.get_context("spawn")
        self._in_q: "mp.Queue" = ctx.Queue()
        self._out_q: "mp.Queue" = ctx.Queue()
        self._workers: list = []
        logger.info("spawning %d decompress workers", n_workers)
        for _ in range(n_workers):
            p = ctx.Process(target=_worker_main, args=(self._in_q, self._out_q))
            p.daemon = True
            p.start()
            self._workers.append(p)
        # Wait for all workers to finish loading.
        ready = 0
        while ready < n_workers:
            tag, _ = self._out_q.get()
            if tag == "ready":
                ready += 1
        logger.info("all %d workers ready", n_workers)

    def decompress_chunks(self, encoded_chunks: list[bytes]) -> list[bytes]:
        """Dispatch all chunks to the pool, return decoded bytes in input
        order. Each `encoded_chunks[i]` is the AC-coded bytestring (with
        its mini-header) that `InferenceEngine.decompress_chunk` accepts.
        Matching the `decompress_chunks_batched` signature so either can
        be plugged in via `chunking.decompress_all`'s `neural_batch_fn`.
        """
        n = len(encoded_chunks)
        if n == 0:
            return []
        for i, enc in enumerate(encoded_chunks):
            self._in_q.put((i, enc))

        results: list = [None] * n
        got = 0
        while got < n:
            idx, decoded = self._out_q.get()
            if isinstance(decoded, Exception):
                raise decoded
            results[idx] = decoded
            got += 1
        return results

    def close(self) -> None:
        for _ in self._workers:
            self._in_q.put(None)
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> "DecompressWorkerPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def default_worker_count() -> int:
    """Default N from env, or 1 on CPU, or 4 on GPU."""
    env = os.environ.get("KRUNCH_DECOMPRESS_WORKERS")
    if env is not None:
        return max(1, int(env))
    try:
        import torch
        return 4 if torch.cuda.is_available() else 1
    except Exception:
        return 1
