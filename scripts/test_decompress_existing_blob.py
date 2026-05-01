"""Parse an existing .krunch blob and decompress via the engine
batched path, comparing per-chunk output to the original input
bytes. Tells us whether the bug is in the engine batched path
(any caller hits it) or in the CLI/chunking wrapping (only the
CLI path hits it).

Usage:
    python test_decompress_existing_blob.py /tmp/x.krunch /tmp/sample.bin
"""
import os, sys, struct
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import InferenceEngine, decode_header, HEADER_SIZE
from krunch.chunking import CHUNK_SIZE


def parse_blob(blob_path):
    with open(blob_path, "rb") as f:
        blob = f.read()
    hdr = decode_header(blob)
    entries_bytes = blob[HEADER_SIZE:]
    pos = 0
    chunks = []
    for _ in range(hdr["n_chunks"]):
        orig_len, comp_len = struct.unpack(">II", entries_bytes[pos:pos + 8])
        pos += 8
        chunks.append((orig_len, entries_bytes[pos:pos + comp_len]))
        pos += comp_len
    return hdr, chunks


def main():
    blob_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/x.krunch"
    src_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sample.bin"

    hdr, chunks = parse_blob(blob_path)
    print(f"blob: {blob_path}  n_chunks={hdr['n_chunks']}  "
          f"orig_len={hdr['original_len']}")
    encs = [enc for _, enc in chunks]
    orig_lens = [ol for ol, _ in chunks]

    # Source slices for comparison (chunks are CHUNK_SIZE bytes each
    # except possibly the last one).
    with open(src_path, "rb") as f:
        src = f.read()
    src_chunks = [src[i * CHUNK_SIZE : i * CHUNK_SIZE + ol]
                   for i, ol in enumerate(orig_lens)]

    eng = InferenceEngine()
    eng.load()

    print("calling engine.decompress_chunks_batched...")
    decoded = eng.decompress_chunks_batched(encs)

    n_pass = 0
    first_bad = -1
    for i, (d, src_c) in enumerate(zip(decoded, src_chunks)):
        # decoded[i] is already truncated to orig_len[i] inside the engine
        if d == src_c:
            n_pass += 1
        elif first_bad < 0:
            first_bad = i
    print(f"byte-exact: {n_pass}/{len(decoded)}")
    if first_bad >= 0:
        d = decoded[first_bad]
        s = src_chunks[first_bad]
        n = min(len(d), len(s))
        for j in range(n):
            if d[j] != s[j]:
                print(f"first diff at chunk={first_bad} byte={j}: "
                      f"got 0x{d[j]:02x} expected 0x{s[j]:02x}")
                print(f"  context expected: {s[max(0,j-20):j+20]!r}")
                print(f"  context decoded:  {d[max(0,j-20):j+20]!r}")
                break

    raise SystemExit(0 if n_pass == len(decoded) else 1)


if __name__ == "__main__":
    main()
