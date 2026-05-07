"""End-to-end engine-level chunk roundtrip via the C++ path.

Exercises InferenceEngine.compress_chunk + decompress_chunk with
KRUNCH_CPP_PATH=1 + KRUNCH_DETERMINISTIC_MATMUL=1. Asserts the
recovered bytes match the input exactly.
"""
import os
os.environ["KRUNCH_DETERMINISTIC_MATMUL"] = "1"
os.environ["KRUNCH_CPP_PATH"] = "1"
os.environ.setdefault("RWKV_CUDA_ON", "1")

import time
from krunch.inference import InferenceEngine


SAMPLES = [
    b"The quick brown fox jumps over the lazy dog. " * 8,
    (b"This is a test of the krunch neural compression codec running with the "
     b"C++ orchestration path on both compress and decompress sides. We expect "
     b"a byte-exact roundtrip. ") * 4,
    b"def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n" * 16,
]


def main():
    eng = InferenceEngine()
    eng.load()

    all_pass = True
    for idx, data in enumerate(SAMPLES):
        t0 = time.time()
        encoded = eng.compress_chunk(data)
        t_enc = time.time() - t0

        t0 = time.time()
        decoded = eng.decompress_chunk(encoded)
        t_dec = time.time() - t0

        ratio = len(encoded) / len(data)
        ok = decoded == data
        all_pass &= ok
        print(f"sample {idx}: {len(data)} B → {len(encoded)} B "
              f"(ratio={ratio:.3f}) "
              f"enc={t_enc*1000:.1f}ms dec={t_dec*1000:.1f}ms "
              f"{'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            # First diverging byte
            n = min(len(decoded), len(data))
            for i in range(n):
                if decoded[i] != data[i]:
                    print(f"  first diff at byte {i}: "
                          f"got 0x{decoded[i]:02x} expected 0x{data[i]:02x}")
                    print(f"  context expected: {data[max(0,i-20):i+20]!r}")
                    print(f"  context decoded:  {decoded[max(0,i-20):i+20]!r}")
                    break
            if len(decoded) != len(data):
                print(f"  length: got {len(decoded)} expected {len(data)}")
    print("ALL PASS" if all_pass else "SOMETHING FAILED")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
