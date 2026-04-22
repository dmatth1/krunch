# Hybrid codec design — model + zstd dispatcher

Status: design spec, pre-implementation. Written 2026-04-22 off the
back of Spike 2 findings (see `SPIKE_2_LOG.md`). Rewrite
post-discussion with user: throw out the idea of hand-rolling
dictionary + delta encoders. Reuse zstd as the traditional-
compression half, build only a per-chunk codec dispatcher.

## The insight

Spike 2 ran 8 variations of a small RWKV model on HDFS logs and
landed at best ratio 0.074 vs. zstd's 0.047. The pattern is clean:
on HDFS-class templated data, zstd's LZ + 128 MB window + Huffman
tail is near-optimal. A 200 K–25 M-param model can't beat it
through prediction alone, because the remaining bits are
dominated by repeated variable-value dictionary hits that zstd
serves for free. On less-templated data (JSON events, app logs,
prose) a small model wins because patterns are context-dependent.

**Conclusion**: don't build a codec that tries to be better than
zstd on zstd's home turf. Build a **codec dispatcher** that picks
between our model and zstd per chunk, and ships the smaller output
with a 2-bit codec tag.

The result is **"never worse than zstd-with-dict, strictly better
on most real data"** — which is a much stronger product statement
than any single-codec approach, and holds even on HDFS.

## What we reuse (not reinvent)

| component | reuse source | why |
|---|---|---|
| LZ dictionary coder | `libzstd` via `zstd` Rust crate | 10+ years battle-tested; 100+ MB/s; we'd do a worse job |
| Per-dataset dictionary | `zstd --train` / `ZSTD_trainFromBuffer()` | Existing API; takes a sample, emits a 100 KB dict; 2-3× better than default zstd on homogeneous data |
| Huffman tail | inside zstd | free |
| BWT + MTF option | `bzip2` library if we want a fourth codec | optional, case-by-case |
| Arithmetic coding | already in `l3tc-rust` for the model path | keep |
| SPM tokenizer | already wired | keep |
| Neural model | `l3tc-rust` RWKV-v4 runtime | keep |

## What we build

1. **Per-dataset dictionary training** — add a zstd dictionary training
   pass to our training pipeline. Takes the first ~100 MB of a
   customer's data, emits `v{N}.zstd_dict` alongside the `.pth` and
   `.tokenizer.model` artifacts.
2. **Codec dispatcher** at encode time: given a chunk (say 64 KB), run
   all available codecs, pick the one with shortest output, write a
   2-bit tag + the encoded bytes into the final blob.
3. **Decode-side dispatcher**: read the 2-bit tag per chunk, route to
   the matching decoder.
4. **New blob format** with per-chunk codec tagging (2 bits per chunk
   = negligible overhead at 64 KB chunks).
5. **Service-side plumbing**: the training-complete Lambda now puts
   both `.pth` AND `.zstd_dict` into DDB; the compression worker
   downloads both; the Rust codec uses both.

## The codec menu

```
Tag  Codec                    When it wins
───────────────────────────────────────────────────────────────
0x0  zstd --long=27 -22       Default zstd, plain mode.
                              Good fallback if dict isn't ready.
0x1  zstd --dict <trained>    Templated data with known vocab of
                              variable values (HDFS, Stripe audit
                              trails, nginx access). Usually
                              strictly better than 0x0 when a
                              dict exists.
0x2  model + AC               Context-dependent patterns where
                              prediction beats dictionary recall
                              (JSON events with free-text fields,
                              prose, app logs with error strings).
0x3  RESERVED                 future — bzip2 / PAQ-class / etc.
```

Each chunk picks its own tag independently. No global choice.

## Encode pseudocode

```rust
fn encode_blob(raw: &[u8], model: &RWKVModel, zstd_dict: &[u8]) -> Vec<u8> {
    let mut out = BlobHeader::new();
    for chunk in raw.chunks(64 * 1024) {
        let mut candidates = vec![];

        candidates.push((0x0, zstd_encode(chunk, /*dict=*/None)));
        candidates.push((0x1, zstd_encode(chunk, Some(zstd_dict))));
        candidates.push((0x2, model_encode(chunk, model)));
        // (optional: 0x3 bzip2)

        let (tag, encoded) = candidates.into_iter()
            .min_by_key(|(_, e)| e.len())
            .unwrap();

        out.push_chunk(tag, encoded);
    }
    out.finalize()
}
```

Decode is just the inverse dispatch. Round-trip correctness per
chunk is enforced by the library calls — zstd guarantees its own
round-trip; our model path already has a round-trip test in
`l3tc-rust`.

## Speed + cost budget

Worst case per chunk we run three encoders (zstd default, zstd+dict,
model). zstd runs ~100+ MB/s so the zstd passes are effectively
free. The model pass is the slow one at ~150 KB/s compress
(l3tc-rust on L4 single-stream). For a 64 KB chunk that's ~0.4 s
of model work.

Net encode throughput on mixed data:
- Chunks where zstd wins (short circuit after zstd finishes first):
  ~100 MB/s chunk throughput.
- Chunks where model wins: ~150 KB/s (model-bound).
- Mixed corpus: aggregate ~1-10 MB/s depending on model-vs-zstd
  mix.

For async compression (the customer never waits), that's fine.

Decode is similar but only runs the chosen codec per chunk, so
decode throughput = whichever codec was picked. On the L4 GPU with
batching, ~5-10 MB/s effective for the model path. Well inside the
1 MB/s single-stream floor.

## Guarantees this buys

- **"Never worse than zstd-with-dict"**: for every chunk, the
  dispatcher has the zstd-dict option available, so the worst case
  is a chunk where zstd wins and we emit exactly what zstd would
  have emitted + 2 bits tag. Header + tag overhead is < 0.01% of a
  reasonably-sized blob.
- **"Strictly better than zstd"** on data where the model beats
  zstd on enough chunks to overcome the header cost. Measurably so
  on variable-content corpora.
- **No per-customer engineering.** The same codec dispatcher ships
  to every customer; it just picks per-chunk.

## What this means per-corpus

| corpus | expected winner | why |
|---|---|---|
| HDFS logs | `zstd --dict` on most chunks | highly templated; dict covers the repeats; model rarely wins |
| nginx access logs (with full query strings) | mixed, model-leaning | URL params + user agents are context-dependent |
| JSON API events w/ variable payloads | model on most chunks | free-text descriptions + structured fields; good prediction target |
| Stripe-style audit trails | mixed | structured fields zstd-friendly, free-text fields model-friendly |
| Prose (docs, articles) | model on almost all chunks | zstd's n-gram approach is weak on natural language |
| Binary blobs | zstd / raw | model has no business here |

## Engineering plan

| step | days | notes |
|---|---|---|
| Wire `zstd` crate into `l3tc-rust` | 0.5 | already a dependency; just expose it |
| Add `ZSTD_trainFromBuffer()` call to training container entrypoint | 0.5 | emits `v{N}.zstd_dict` |
| Per-chunk codec dispatcher + 2-bit tag in blob header | 1.5 | `l3tc-rust/src/codec.rs` + new blob version |
| Update metadata JSON to record which codecs are available + winning-chunk histogram | 0.5 | useful for ops to see where model is contributing |
| Decode-side dispatcher | 1 | mirror of encode side |
| Round-trip tests on HDFS + 1 JSON corpus + 1 prose corpus | 1 | must be bit-exact; no exceptions |
| Benchmark: total ratio, chunk-by-chunk codec distribution, decode throughput | 0.5 | writes up the numbers |
| Wire into service: Fargate worker downloads dict; training pipeline emits dict | 1 | CDK + worker changes |

**Total: ~1 week, 1 engineer.** An order of magnitude less than
the original (hand-rolled encoders) design. Keeps the neural
runtime simple — it only has to encode bytes to bits, same as
today; the dispatcher is separate plumbing.

## Open decisions

1. **Chunk size.** 64 KB is a reasonable default (balances per-chunk
   overhead vs. coder efficiency). Might tune per-corpus later.
2. **Whether to add `0x3 bzip2`** as a fourth option. Could help on
   extremely sorted / structured data. Low-cost to add; nice-to-have.
3. **Dictionary refresh cadence.** If a dataset drifts, the zstd
   dict may stale. Need policy for when to retrain (parallel to the
   model retrain logic).
4. **Dictionary size.** zstd's default is 110 KB; customers with
   huge vocabularies of variable values might want larger. Bench on
   real data.

## What this does NOT claim

- We are **not** beating the theoretical entropy of HDFS. Nobody
  can.
- We are **not** claiming our neural model is better than zstd on
  templated log data in isolation — it isn't, and that's fine.
- We are **not** eliminating the need for zstd as a dependency — we
  are doubling down on it.

The product claim is **"we pick the best compressor per chunk of
your data, automatically, from a menu that includes zstd-with-a-
dictionary-trained-on-your-data + a neural model trained on your-
data."** That is strictly better than any of the components on
their own on any mixed corpus.
