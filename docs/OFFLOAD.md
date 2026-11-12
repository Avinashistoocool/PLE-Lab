# PLE offload engine (`ple_lab/offload/`)

Per-Layer Embeddings (PLE, Gemma 3n) add a table `E_ple ∈ R^{V × L·p}` that every layer reads one slice of.
Unlike every other weight, the rows needed are known **before** the forward pass: they depend only on the
token ids. So the table never has to sit in accelerator memory; only `L·p` numbers per token have to arrive
in time. This package demonstrates and measures that: the table lives in GPU memory (baseline), CPU RAM, or a
memory-mapped file (standing in for phone flash), optionally behind a small GPU row cache and/or a prefetcher.

## Plug-in point

`docs/CONTRACT.md`: any `nn.Module` assigned to `model.ple_table` with
`forward(ids: LongTensor[B,T]) -> Tensor[B,T,L*p]` (raw unscaled rows, on `ids.device`).

```python
from ple_lab import load_run
from ple_lab.offload import make_source, attach

model = load_run("runs/ple-small", "cuda", load_table=False)        # dense weights only
src = make_source("runs/ple-small", "cached-lru", "cuda", capacity_frac=0.05, backing="mmap-i8")
attach(model, src)                                                   # checks width/vocab, sets model.ple_table
logits, _ = model(ids)
print(src.stats())   # hits, misses, evictions, backing/H2D bytes, resident VRAM ...
```

`make_source(run_dir, kind, device, *, capacity=None, capacity_frac=None, backing="mmap-f16", freq=None,
prefetch=False, prefetch_depth=2, verify=True, out_dtype=torch.float16)`, kinds:
`gpu, cpu, cpu-pinned, mmap-f16, mmap-i8, cached-lru, cached-lfu, static-topk`.
Source tensors are plain attributes, not registered buffers, so `model.to()`/`state_dict()` ignore them.

## Design

All off-GPU sources share one pipeline (`TableSource.forward`):

```
ids ──► CPU ──► torch.unique(sorted, inverse, counts) ──► fetch(unique ids) ──► rows [n, L·p] on device
                                    │                                               │
                                    └── inverse (int32) ──H2D──► rows.index_select(inverse) ──► [B,T,L·p]
```

Each distinct row is read from the backing store and copied host→device **at most once per call**
(prefill of 256 TinyStories tokens has ≈ 120 distinct ids; a B=16 batch ≈ 1000 of 4096).

| source | where the table lives | per call |
|---|---|---|
| `GPUTableSource` | whole fp16 table in VRAM | `F.embedding` (baseline) |
| `CPUTableSource(pinned)` | fp16 tensor in CPU RAM | `index_select` of unique rows into a staging buffer (pinned → truly async `non_blocking` H2D; pageable → driver bounce copy), H2D of just those rows |
| `MmapTableSource("f16"\|"i8")` | `np.memmap` of `ple_table.*.bin` | `np.take` of the sorted unique rows (lazy page faults); i8 also reads the fp32 per-row scale, ships int8+scale (≈½ the bytes) and dequantises on the device: `row = q·scale` |
| `CachedTableSource(backing, capacity_rows, policy)` | fixed `[capacity, L·p]` GPU buffer + CPU slot map (`slot_of[V]`, `id_of[C]`) in front of any backing source | hits gathered from the buffer, misses fetched from the backing in one batch, then admitted |
| `Prefetcher(source)` | wraps any source | background thread (+ dedicated CUDA copy stream) fetches rows for ids queued with `prefetch(ids)`; `forward` consumes the oldest prefetch |

**Cache policies.** `lru` evicts least-recently-used slots. `lfu` keeps whole-history counts and only admits a
miss if it has been seen more often than the victim (TinyLFU-style admission so one-off ids cannot flush hot
rows). `static-topk` preloads the `capacity` most frequent ids from a frequency array (the bench uses
`train.bin` counts, never the val stream it measures on) and never evicts. Correctness invariants (all tested):
rows used by the current call are never evicted by that call, and misses are returned from the freshly fetched
rows, so a cache smaller than the unique ids of a single batch is still exact; duplicates inside a batch are one
lookup; `hit_rate` is token-weighted (a duplicate of a hit counts as a hit).

**Prefetcher limits** (deliberately simple): strictly FIFO; with `verify=True` the ids of each forward are
compared to the prefetched ones (one D2H copy) and on mismatch everything pending is drained and the call falls
back to a synchronous fetch, so it is always correct. One worker thread; Python work is GIL-serialised, so only
the numpy/torch copy kernels and the async H2D overlap. It cannot help autoregressive decode (the next token is
unknown until sampled) and only pays off when GPU compute per step exceeds the fetch time. On CUDA the copy
stream waits on an event recorded at `prefetch()` time, so it never overwrites cache slots still being read.

**Byte accounting** (`stats()`): `backing_bytes` (row payload read from RAM/mmap, + int8 scales),
`backing_page_bytes` (mmap: distinct 4 KiB pages touched — what a cold flash read actually costs),
`h2d_bytes` (row payload host→device), `h2d_index_bytes` (inverse map/slot indices), `resident_vram_bytes`
(table or cache buffer kept on the GPU), plus `tokens`, `unique_rows`, `hit_tokens`, `hit_unique`,
`miss_unique`, `inserts`, `evictions`, `hit_rate`.

## Cost model

Per token, PLE needs one row: `L·p·bytes_per_elem` (fp16: `2·L·p`; int8: `L·p + 4` for the scale).
A dense decoder reads *all* dense weights per decode step (`2·N_dense` bytes in fp16, amortised over the batch).

| model | L·p | PLE bytes/token f16 / i8 | table size f16 | dense params | dense bytes per decode step (f16) | ratio |
|---|---|---|---|---|---|---|
| ple-small (L=6, p=64) | 384 | 768 / 388 | 38.6 MB | 30.5 M | 61.0 MB | ≈ 79 000× |
| synthetic-big (L=12, p=256) | 3072 | 6144 / 3076 | 309 MB | 24.7 M | 49.5 MB | ≈ 8 000× |
| Gemma 3n E2B-like (262144 × 8960) | 8960 | 17.9 KB / 9.0 KB | 4.7 GB | – | – | – |

So the table can be large (here 6× the dense weights for synthetic-big) while the per-token traffic is tiny:
even with zero caching, 6 KB/token at 1 000 tok/s is 6 MB/s — far under PCIe (≈ 12–25 GB/s) or UFS flash
(≈ 1–4 GB/s sequential; random 4 KiB reads are the real limiter, hence `backing_page_bytes`). And because token
frequencies are Zipfian, a small cache absorbs most of it.

## Benchmark

```
python -m ple_lab.offload.bench --run runs/ple-small --device cuda   # + val loss gpu-f16 vs mmap-i8
python -m ple_lab.offload.bench --synthetic --device cuda            # builds runs/synthetic-big
python -m ple_lab.offload.bench --gemma3n-table --device cuda        # opt-in, 4.7 GB file; checks disk/RAM first
```

`--device` defaults to `cpu` (or `$PLE_DEVICE`) so that development never touches a GPU that a training job
is using; with `--device cpu` CUDA is hidden before anything can initialise it.

Workloads (ids from `data/tinystories/val.bin`, bf16 autocast on CUDA, warmup then `cuda.synchronize` around
every timed iteration):

* `prefill` – full model, B=1, T=256 (60 iters).
* `decode` – full model, 256 steps on a **sliding window** of the last 128 tokens of a contiguous stream, one new
  token per step (this GPT has no KV cache, so each step re-runs the window; tok/s counts new tokens).
* `decode-1` – **table-only**, the same 256-step stream, 1 id per step: exactly the lookup a KV-cached decoder
  makes per token. This isolates the PLE fetch latency.
* `batched` – full model, B=16, T=256 (20 iters).

Sources: `gpu, cpu, cpu-pinned, mmap-f16, mmap-i8`, LRU caches at 1/5/20 % of the vocab and static top-k at 5 %
(cache backing = `mmap-f16`), and `+pf` (prefetch on) for prefill/batched. Output: markdown table on stdout and
`results/offload/<run>-<timestamp>.json` (all raw stats included).

Column meaning: `table VRAM` = resident table/cache bytes; `peak alloc` = `torch.cuda.max_memory_allocated`
during the measured loop (includes dense weights + activations); `H2D B/tok` and `backing B/tok` are per
looked-up row position (for `decode` that is 128 positions per new token).

## Results (RTX 4050 Laptop 6 GB, Windows/WDDM, torch 2.11, bf16 autocast, 2026-11-12)

Raw JSON: `results/offload/ple-small-20261112-192148.json` and `results/offload/synthetic-big-20261112-192517.json`.
Cache backing is `mmap-f16` (warm OS page cache). `vs gpu` = throughput change relative to the all-in-VRAM table.

### runs/ple-small (trained, L=6, p=64, d=384; table 19.3 M params = 36.8 MiB fp16)

| workload | source | tok/s | vs gpu | p50 ms | p95 ms | table VRAM MB | peak alloc MB | hit rate | H2D B/tok | backing B/tok |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | gpu | 9,374 | +0% | 27.098 | 30.368 | 36.84 | 246.4 | - | 0 | 0 |
| prefill | cpu | 8,553 | -9% | 29.336 | 37.316 | 0.00 | 210.2 | - | 384 | 384 |
| prefill | cpu-pinned | 8,074 | -14% | 29.231 | 42.883 | 0.00 | 210.2 | - | 384 | 384 |
| prefill | mmap-f16 | 9,219 | -2% | 27.209 | 31.780 | 0.00 | 210.2 | - | 384 | 384 |
| prefill | mmap-i8 | 8,945 | -5% | 28.358 | 31.675 | 0.00 | 210.2 | - | 194 | 194 |
| prefill | lru-1% | 8,154 | -13% | 30.769 | 37.422 | 0.37 | 210.5 | 77.4% | 123 | 123 |
| prefill | lru-5% | 8,001 | -15% | 29.871 | 35.300 | 1.84 | 211.4 | 87.9% | 63 | 63 |
| prefill | lru-20% | 8,515 | -9% | 29.674 | 33.668 | 7.37 | 216.9 | 87.9% | 63 | 63 |
| prefill | topk-5% | 8,638 | -8% | 28.693 | 34.474 | 1.84 | 211.4 | 95.5% | 22 | 22 |
| prefill | mmap-f16+pf | 8,906 | -5% | 28.541 | 33.249 | 0.00 | 210.3 | - | 384 | 384 |
| prefill | lru-5%+pf | 8,206 | -12% | 30.217 | 37.522 | 1.84 | 211.5 | 87.9% | 63 | 63 |
| decode | gpu | 37 | +0% | 26.626 | 31.595 | 36.84 | 233.3 | - | 0 | 0 |
| decode | cpu | 35 | -7% | 27.835 | 34.348 | 0.00 | 197.0 | - | 467 | 467 |
| decode | cpu-pinned | 35 | -5% | 28.097 | 32.823 | 0.00 | 197.0 | - | 467 | 467 |
| decode | mmap-f16 | 34 | -8% | 28.967 | 34.238 | 0.00 | 197.0 | - | 467 | 467 |
| decode | mmap-i8 | 32 | -14% | 30.974 | 34.997 | 0.00 | 197.0 | - | 236 | 236 |
| decode | lru-1% | 34 | -8% | 28.991 | 33.350 | 0.37 | 197.4 | 99.6% | 3 | 3 |
| decode | lru-5% | 33 | -12% | 29.996 | 36.674 | 1.84 | 198.2 | 99.6% | 3 | 3 |
| decode | lru-20% | 34 | -8% | 28.778 | 33.589 | 7.37 | 204.7 | 99.6% | 3 | 3 |
| decode | topk-5% | 34 | -10% | 29.340 | 34.686 | 1.84 | 198.2 | 96.4% | 21 | 21 |
| decode-1 | gpu | 3,339 | +0% | 0.270 | 0.460 | 36.84 | 161.6 | - | 0 | 0 |
| decode-1 | cpu | 1,102 | -67% | 0.808 | 1.386 | 0.00 | 124.7 | - | 768 | 768 |
| decode-1 | cpu-pinned | 1,030 | -69% | 0.835 | 1.767 | 0.00 | 124.7 | - | 768 | 768 |
| decode-1 | mmap-f16 | 820 | -75% | 1.053 | 2.410 | 0.00 | 124.7 | - | 768 | 768 |
| decode-1 | mmap-i8 | 624 | -81% | 1.436 | 2.842 | 0.00 | 124.7 | - | 388 | 388 |
| decode-1 | lru-1% | 632 | -81% | 1.335 | 3.593 | 0.37 | 125.1 | 56.2% | 336 | 336 |
| decode-1 | lru-5% | 614 | -82% | 1.491 | 2.668 | 1.84 | 126.6 | 56.2% | 336 | 336 |
| decode-1 | lru-20% | 584 | -83% | 1.573 | 3.154 | 7.37 | 132.1 | 56.2% | 336 | 336 |
| decode-1 | topk-5% | 825 | -75% | 1.088 | 1.854 | 1.84 | 126.6 | 96.5% | 27 | 27 |
| batched | gpu | 128,699 | +0% | 31.398 | 36.041 | 36.84 | 641.4 | - | 0 | 0 |
| batched | cpu | 118,112 | -8% | 33.160 | 42.943 | 0.00 | 604.6 | - | 160 | 160 |
| batched | cpu-pinned | 111,743 | -13% | 34.530 | 44.871 | 0.00 | 604.6 | - | 160 | 160 |
| batched | mmap-f16 | 119,554 | -7% | 32.892 | 40.827 | 0.00 | 604.6 | - | 160 | 160 |
| batched | mmap-i8 | 111,214 | -14% | 35.432 | 44.232 | 0.00 | 604.6 | - | 81 | 81 |
| batched | lru-1% | 111,138 | -14% | 36.194 | 41.600 | 0.37 | 604.9 | 76.7% | 96 | 96 |
| batched | lru-5% | 110,252 | -14% | 36.799 | 41.323 | 1.84 | 607.2 | 94.8% | 27 | 27 |
| batched | lru-20% | 113,339 | -12% | 35.461 | 40.040 | 7.37 | 612.1 | 95.8% | 22 | 22 |
| batched | topk-5% | 112,484 | -13% | 35.855 | 41.548 | 1.84 | 607.2 | 96.4% | 20 | 20 |
| batched | mmap-f16+pf | 120,139 | -7% | 34.334 | 37.916 | 0.00 | 605.2 | - | 160 | 160 |
| batched | lru-5%+pf | 116,188 | -10% | 34.860 | 37.285 | 1.84 | 607.9 | 94.8% | 27 | 27 |

**Quality** (50 batches × 16 × 256 tokens of val.bin, fp32, identical windows): val loss with the gpu-fp16
table **2.0153**, with the mmap-int8 table **2.0154** (Δ = +0.00002). Int8 offloading costs nothing measurable.

### runs/synthetic-big (random init, L=12, p=256, d=256; table 154.5 M params = 294.8 MiB fp16)

| workload | source | tok/s | vs gpu | p50 ms | p95 ms | table VRAM MB | peak alloc MB | hit rate | H2D B/tok | backing B/tok |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | gpu | 5,453 | +0% | 46.561 | 52.862 | 294.75 | 473.7 | - | 0 | 0 |
| prefill | cpu | 4,908 | -10% | 50.536 | 66.092 | 0.00 | 178.9 | - | 3,074 | 3,074 |
| prefill | cpu-pinned | 5,359 | -2% | 46.903 | 55.121 | 0.00 | 178.9 | - | 3,074 | 3,074 |
| prefill | mmap-f16 | 4,758 | -13% | 54.359 | 59.737 | 0.00 | 178.9 | - | 3,074 | 3,074 |
| prefill | mmap-i8 | 4,183 | -23% | 60.376 | 68.813 | 0.00 | 178.9 | - | 1,539 | 1,539 |
| prefill | lru-1% | 5,030 | -8% | 49.909 | 59.709 | 2.95 | 181.9 | 77.4% | 984 | 984 |
| prefill | lru-5% | 4,686 | -14% | 51.174 | 60.870 | 14.74 | 193.7 | 87.9% | 504 | 504 |
| prefill | lru-20% | 4,436 | -19% | 57.093 | 65.502 | 58.95 | 237.9 | 87.9% | 504 | 504 |
| prefill | topk-5% | 4,387 | -20% | 58.138 | 63.326 | 14.74 | 193.7 | 95.5% | 180 | 180 |
| prefill | mmap-f16+pf | 4,583 | -16% | 56.254 | 62.399 | 0.00 | 179.8 | - | 3,074 | 3,074 |
| prefill | lru-5%+pf | 4,968 | -9% | 51.197 | 58.539 | 14.74 | 194.5 | 87.9% | 504 | 504 |
| decode | gpu | 18 | +0% | 54.805 | 62.754 | 294.75 | 459.4 | - | 0 | 0 |
| decode | cpu | 17 | -4% | 54.378 | 72.430 | 0.00 | 164.7 | - | 3,738 | 3,738 |
| decode | cpu-pinned | 18 | -2% | 54.313 | 71.969 | 0.00 | 164.7 | - | 3,738 | 3,738 |
| decode | mmap-f16 | 17 | -3% | 56.958 | 64.109 | 0.00 | 164.7 | - | 3,738 | 3,738 |
| decode | mmap-i8 | 19 | +6% | 55.315 | 67.215 | 0.00 | 164.7 | - | 1,872 | 1,872 |
| decode | lru-1% | 37 | +107% | 24.043 | 44.469 | 2.95 | 167.6 | 99.6% | 23 | 23 |
| decode | lru-5% | 40 | +119% | 23.136 | 38.621 | 14.74 | 179.4 | 99.6% | 23 | 23 |
| decode | lru-20% | 30 | +64% | 32.613 | 46.293 | 58.95 | 223.6 | 99.6% | 23 | 23 |
| decode | topk-5% | 41 | +129% | 22.396 | 36.630 | 14.74 | 179.4 | 96.4% | 165 | 165 |
| decode-1 | gpu | 7,538 | +0% | 0.105 | 0.276 | 294.75 | 398.2 | - | 0 | 0 |
| decode-1 | cpu | 2,162 | -71% | 0.341 | 1.070 | 0.00 | 103.4 | - | 6,144 | 6,144 |
| decode-1 | cpu-pinned | 1,594 | -79% | 0.558 | 1.373 | 0.00 | 103.4 | - | 6,144 | 6,144 |
| decode-1 | mmap-f16 | 1,107 | -85% | 0.680 | 1.736 | 0.00 | 103.4 | - | 6,144 | 6,144 |
| decode-1 | mmap-i8 | 1,171 | -84% | 0.676 | 1.577 | 0.00 | 103.4 | - | 3,076 | 3,076 |
| decode-1 | lru-1% | 1,046 | -86% | 0.946 | 1.763 | 2.95 | 106.4 | 56.2% | 2,688 | 2,688 |
| decode-1 | lru-5% | 1,058 | -86% | 0.853 | 1.856 | 14.74 | 118.2 | 56.2% | 2,688 | 2,688 |
| decode-1 | lru-20% | 1,088 | -86% | 0.852 | 2.004 | 58.95 | 162.4 | 56.2% | 2,688 | 2,688 |
| decode-1 | topk-5% | 906 | -88% | 1.052 | 1.709 | 14.74 | 118.2 | 96.5% | 216 | 216 |
| batched | gpu | 75,498 | +0% | 52.475 | 65.520 | 294.75 | 901.3 | - | 0 | 0 |
| batched | cpu | 88,946 | +18% | 46.989 | 50.644 | 0.00 | 606.5 | - | 1,278 | 1,278 |
| batched | cpu-pinned | 77,590 | +3% | 51.877 | 59.878 | 0.00 | 606.5 | - | 1,278 | 1,278 |
| batched | mmap-f16 | 77,842 | +3% | 50.345 | 62.320 | 0.00 | 606.5 | - | 1,278 | 1,278 |
| batched | mmap-i8 | 62,065 | -18% | 63.938 | 78.341 | 0.00 | 606.5 | - | 640 | 640 |
| batched | lru-1% | 82,799 | +10% | 46.417 | 82.008 | 2.95 | 609.5 | 76.7% | 772 | 772 |
| batched | lru-5% | 88,322 | +17% | 45.258 | 58.903 | 14.74 | 621.2 | 94.8% | 212 | 212 |
| batched | lru-20% | 99,249 | +31% | 42.356 | 52.635 | 58.95 | 665.5 | 95.8% | 172 | 172 |
| batched | topk-5% | 81,340 | +8% | 52.657 | 72.528 | 14.74 | 621.2 | 96.4% | 158 | 158 |
| batched | mmap-f16+pf | 53,390 | -29% | 76.884 | 80.726 | 0.00 | 611.8 | - | 1,278 | 1,278 |
| batched | lru-5%+pf | 49,077 | -35% | 83.609 | 86.039 | 14.74 | 626.5 | 94.8% | 212 | 212 |

A repeat of a subset two minutes later (`--workloads decode,batched`) gave — batched: gpu 161,618 tok/s,
cpu −23 %, mmap-f16 −27 %, lru-5 % −14 %, topk-5 % −9 %; decode: gpu 40 tok/s, cpu −80 %, lru-5 % −45 %,
topk-5 % +12 %. The gpu baseline itself moved from 75 k to 162 k tok/s between runs, so run-to-run variation
of ±30–50 % on the full-model rows is normal on this shared laptop GPU (see caveats): **only large effects in
the full-model rows are meaningful**; the byte, hit-rate and memory columns are deterministic.

## Interpretation

* **VRAM.** ple-small: the PLE table goes from 36.8 MiB to 0 (cpu/mmap) or 1.84 MiB (5 % cache, 88–96 % hit rate)
  at a −7…−15 % prefill/batched throughput change. synthetic-big: **294.8 MiB → 14.7 MiB with a 5 % cache**
  (static top-k 95.5–96.5 % hit rate; LRU 88–95 % on prefill/batched); peak allocated memory falls 474 → 194 MiB
  (prefill) and 901 → 621 MiB (batched) — the table was ~60 % of prefill VRAM. The throughput cost of the 5 %
  caches is in the 0–20 % range (−9 %/−14 % for batched in the repeat run).
* **Zipf makes small caches work.** A 1 % LRU cache (503 rows) already serves 77 % of prefill/batched tokens; a
  5 % static top-k (frequencies from train.bin, never the val stream) serves ~96 % on every workload. Going from
  LRU 5 % to 20 % barely helps (87.9 → 87.9 %, 94.8 → 95.8 %): the remaining misses are the long tail. In the
  sliding-window decode, 127 of 128 rows repeat each step, so every cache hits 99.6 %.
* **Bytes moved are tiny.** Uncached, a row is 768 B (ple-small) / 6 KB (synthetic-big) per token; per-call
  de-duplication halves that for a 256-token prefill and cuts it ~5× for B=16. A 5 % cache brings it to 27–63 B
  and 158–504 B per token, versus 61 MB / 49.5 MB of dense weights read per decode step: 4–5 orders of
  magnitude less.
* **int8** halves backing and H2D bytes (388 vs 768 B/token) at no quality cost, but on this GPU the extra
  dequant kernels make it slightly slower; it wins where bandwidth (flash), not kernel launches, is the limit.
* **Latency, not bandwidth, is the cost.** `decode-1` (table only, one id per step) exposes the fixed per-call
  overhead of an off-GPU lookup: 0.3–1.5 ms (CPU unique, numpy gather, one or two tiny H2D copies, scatter) vs
  0.1–0.3 ms for an on-GPU embedding. A cache does not help there: a hit still pays the CPU slot lookup and an
  index H2D, and Python/launch overhead dominates. In a real decoder this ~1 ms is either hidden behind the
  20–50 ms model step (the `decode` rows are within noise of gpu) or overlapped by fetching the row right after
  sampling while early layers run; a C++/CUDA-graph implementation would shrink it further.
* **Pinned vs pageable**: for these ≤ 1 MB transfers pinned staging buys nothing measurable (a per-call pinned
  allocation even costs a little); it matters for bulk preloads and larger batches.
* **Prefetch** was roughly neutral for ple-small and *hurt* synthetic-big batched (−29 %/−35 %): the worker thread
  competes for the GIL with the thread launching the model's kernels, and the de-duplicated fetch (~1 ms) is
  already small relative to compute. It is kept as a correct, tested building block; the honest conclusion is
  that for models this size, synchronous de-duplicated fetches are good enough.

## Caveats

* **Windows/WDDM**: kernel launches and H2D copies go through the WDDM scheduler, adding jitter and per-copy
  latency vs Linux/TCC (p95 is 1.1–3× p50). Other GPU clients on this laptop (desktop apps, a local LLM server)
  cause the large run-to-run swings noted above.
* **Page-cache warmth**: after first touch the 37 / 309 MB table files sit in the OS page cache (16 GB RAM), so
  `mmap-f16` ≈ `cpu`. Cold reads from real flash would cost per *page*: `backing_page_bytes` in the JSON is the
  number to feed into a flash model.
* `decode` has no KV cache, so its absolute tok/s is low and dominated by the 128-token forward; `decode-1`
  isolates the lookup. Both count new tokens only.
* synthetic-big is random-init: timings are representative, quality is not (quality measured on ple-small).
* The opt-in `--gemma3n-table` run (262144 × 8960, writes a 4.7 GB file) was not executed.


## Tests

`tests/test_offload.py` (tiny GPT saved to `tmp_path`): logits of every source/policy (incl. int8-backed
cache and prefetch) match the in-memory model over several calls (fp16 `atol=1e-2`; int8 `atol=5e-2`:
per-row symmetric quantisation error ≤ max|row|/254 per element, amplified by `sqrt(p)` and the network);
exact row equality for cpu/mmap; i8 rows within half a quantisation step; LRU hit/miss/eviction/byte
accounting by hand; cache capacity (3) far smaller than the ~35 unique ids of one batch for all three policies,
plus slot-map consistency; LFU admission keeps hot rows; Zipf stream → >50 % hit rate with a 10 % cache;
prefetch mismatch fallback; `attach` validation. CUDA cases are parametrised but opt-in
(`PLE_TEST_CUDA=1`), so `python -m pytest -q tests` never touches the GPU.
