"""Re-tokenize the full corpus with the new balanced unigram SPM.

Parallel version: uses N workers (default 10 for M1).

Pile (40 GB) is decoded from existing tokenized file (no re-stream).
Other 4 components (nick007x, lumees, logs, CSV) are tokenized
fresh from raw text. Output is written to per-worker files in
corpus_build/retokenized/, ready for concatenation.

Usage:
    python scripts/retokenize_corpus.py
"""
from __future__ import annotations

import multiprocessing as mp
import sentencepiece as spm
from pathlib import Path
import time
import sys
import os

OLD_SPM = 'tokenizer_pile_32k/spm_pile_bpe_32768.model'
NEW_SPM = 'tokenizer_balanced_32k/spm_balanced_unigram_32768.model'

OUT_DIR = Path('corpus_build/retokenized')
OUT_DIR.mkdir(exist_ok=True)

N_WORKERS = 10


def write_tokens(f, ids):
    f.write('2\n')
    for tid in ids:
        f.write(f'{tid}\n')


# --- Worker for raw text files ---

def _tokenize_text_worker(args):
    in_path, out_path, start_offset, end_offset, worker_id = args
    sp = spm.SentencePieceProcessor()
    sp.load(NEW_SPM)
    docs = 0
    tokens = 0
    t0 = time.time()
    bytes_read = start_offset  # track locally — never call f.tell() in hot loop
    with open(in_path, 'r', encoding='utf-8', errors='replace') as fin:
        with open(out_path, 'w') as fout:
            fin.seek(start_offset)
            if start_offset > 0:
                skip = fin.readline()  # skip partial line
                bytes_read += len(skip.encode('utf-8'))
            buf, buf_bytes = [], 0
            for line in fin:
                buf.append(line)
                buf_bytes += len(line)
                bytes_read += len(line.encode('utf-8'))
                if buf_bytes >= 65536:
                    text = ''.join(buf)
                    ids = sp.encode_as_ids(text)
                    if ids:
                        write_tokens(fout, ids)
                        tokens += len(ids)
                        docs += 1
                    buf, buf_bytes = [], 0
                if bytes_read >= end_offset:
                    break
            if buf:
                text = ''.join(buf)
                ids = sp.encode_as_ids(text)
                if ids:
                    write_tokens(fout, ids)
                    tokens += len(ids)
                    docs += 1
    return worker_id, docs, tokens, time.time() - t0


def tokenize_text_parallel(in_path, out_prefix, label):
    """Tokenize a raw text file with N parallel workers."""
    print(f'\n--- {label} (parallel × {N_WORKERS}) ---')
    print(f'  in: {in_path}')
    in_path = Path(in_path)
    if not in_path.exists():
        print(f'  SKIP: file missing')
        return 0
    file_size = in_path.stat().st_size
    chunk = file_size // N_WORKERS
    args = []
    for i in range(N_WORKERS):
        start = i * chunk
        end = file_size if i == N_WORKERS - 1 else (i + 1) * chunk
        out_path = OUT_DIR / f'{out_prefix}.part{i:02d}.txt'
        args.append((str(in_path), str(out_path), start, end, i))

    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_tokenize_text_worker, args)

    elapsed = time.time() - t0
    total_docs = sum(r[1] for r in results)
    total_tokens = sum(r[2] for r in results)
    total_size = sum((OUT_DIR / f'{out_prefix}.part{i:02d}.txt').stat().st_size
                     for i in range(N_WORKERS))
    print(f'  DONE: {total_docs:,} docs, {total_tokens:,} tokens, '
          f'{total_size/1e9:.2f} GB, {elapsed:.0f}s')
    return total_tokens


# --- Worker for Pile decode + re-encode ---

def _retokenize_pile_worker(args):
    in_path, out_path, start_offset, end_offset, worker_id = args
    old_sp = spm.SentencePieceProcessor()
    old_sp.load(OLD_SPM)
    new_sp = spm.SentencePieceProcessor()
    new_sp.load(NEW_SPM)
    docs = 0
    tokens = 0
    t0 = time.time()
    bytes_read = start_offset  # track locally — never call f.tell() in hot loop

    with open(in_path, 'r') as fin:
        with open(out_path, 'w') as fout:
            fin.seek(start_offset)
            # Align to next BOS (token "2") at line start
            if start_offset > 0:
                skip = fin.readline()  # skip partial line
                bytes_read += len(skip.encode('utf-8'))
                # Scan forward (line by line) until we find a "2" line
                for line in fin:
                    bytes_read += len(line.encode('utf-8'))
                    if line.strip() == '2':
                        # We've consumed this BOS — start a new doc
                        break

            current_doc = []
            past_end = False
            for line in fin:
                bytes_read += len(line.encode('utf-8'))
                try:
                    tid = int(line.strip())
                except ValueError:
                    continue
                if tid == 2:  # BOS = new doc
                    if current_doc:
                        try:
                            text = old_sp.decode_ids(current_doc)
                            if text:
                                new_ids = new_sp.encode_as_ids(text)
                                if new_ids:
                                    write_tokens(fout, new_ids)
                                    tokens += len(new_ids)
                                    docs += 1
                        except Exception:
                            pass
                        current_doc = []
                    if past_end:
                        # We finished the last doc that started before our
                        # end boundary; stop here
                        break
                    if bytes_read >= end_offset:
                        # Mark to stop after current doc completes
                        past_end = True
                else:
                    current_doc.append(tid)
            # Flush final doc
            if current_doc:
                try:
                    text = old_sp.decode_ids(current_doc)
                    if text:
                        new_ids = new_sp.encode_as_ids(text)
                        if new_ids:
                            write_tokens(fout, new_ids)
                            tokens += len(new_ids)
                            docs += 1
                except Exception:
                    pass
    return worker_id, docs, tokens, time.time() - t0


def retokenize_pile_parallel(in_path):
    """Decode + re-encode Pile with N parallel workers."""
    print(f'\n--- Pile decode + re-encode (parallel × {N_WORKERS}) ---')
    in_path = Path(in_path)
    file_size = in_path.stat().st_size
    chunk = file_size // N_WORKERS
    args = []
    for i in range(N_WORKERS):
        start = i * chunk
        end = file_size if i == N_WORKERS - 1 else (i + 1) * chunk
        out_path = OUT_DIR / f'pile.part{i:02d}.txt'
        args.append((str(in_path), str(out_path), start, end, i))

    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_retokenize_pile_worker, args)

    elapsed = time.time() - t0
    total_docs = sum(r[1] for r in results)
    total_tokens = sum(r[2] for r in results)
    total_size = sum((OUT_DIR / f'pile.part{i:02d}.txt').stat().st_size
                     for i in range(N_WORKERS))

    # Per-worker breakdown
    for r in results:
        wid, d, t, e = r
        print(f'  worker {wid}: {d:,} docs, {t:,} tokens, {e:.0f}s')

    print(f'  DONE: {total_docs:,} docs, {total_tokens:,} tokens, '
          f'{total_size/1e9:.2f} GB, {elapsed:.0f}s')
    return total_tokens


def main():
    print(f'Re-tokenizing corpus with new balanced unigram SPM 32K')
    print(f'Workers: {N_WORKERS}')
    print(f'Old SPM: {OLD_SPM}')
    print(f'New SPM: {NEW_SPM}')

    total_tokens = 0
    t_start = time.time()

    # 1. Pile decode + re-encode (parallel)
    total_tokens += retokenize_pile_parallel('corpus_build/train_pile_40gb_32k.txt')

    # 2-5. Tokenize raw text files (parallel)
    raw_files = [
        ('corpus_build/structured/code_diverse_real.txt', 'nick007x', 'nick007x diverse code'),
        ('corpus_build/structured/code_real.txt', 'lumees1', 'lumees structured 1'),
        ('corpus_build/structured/code_real_extra.txt', 'lumees2', 'lumees structured 2'),
        ('corpus_build/structured/logs_real.txt', 'logs', 'Zenodo logs'),
        ('corpus_build/structured/csv_real.txt', 'csv', 'data.gov CSV'),
    ]

    for in_path, prefix, label in raw_files:
        total_tokens += tokenize_text_parallel(in_path, prefix, label)

    elapsed = time.time() - t_start
    print(f'\n=== ALL DONE ===')
    print(f'Total tokens: {total_tokens:,}')
    print(f'Total wall time: {elapsed:.0f}s ({elapsed/60:.0f} min)')

    print(f'\nOutput files in {OUT_DIR}:')
    total_size = 0
    for f in sorted(OUT_DIR.glob('*.txt')):
        size = f.stat().st_size
        total_size += size
    print(f'  Total parts: {len(list(OUT_DIR.glob("*.txt")))}')
    print(f'  Total size:  {total_size/1e9:.2f} GB')


if __name__ == '__main__':
    sys.exit(main())
