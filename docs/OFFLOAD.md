# PLE offload engine (`ple_lab/offload/`)

Per-Layer Embeddings (PLE, Gemma 3n) add a table `E_ple ∈ R^{V × L·p}`. Each layer reads one slice of this table.
For all other weights, the model does not know the necessary data before the forward pass. For this table, the
model knows the necessary rows **before** the forward pass, because the rows depend only on the token ids.
Thus, the table does not have to stay in accelerator memory. Only `L·p` numbers for each token must arrive in time.

This package shows and measures this. The table can be in one of these locations:
- GPU memory (the baseline);
- CPU RAM;
- a memory-mapped file (this file has the function of phone flash).

Optionally, a small GPU cache of rows and/or a prefetcher can be in front of the table.

## Plug-in point

`docs/CONTRACT.md`: you can assign any `nn.Module` to `model.ple_table` if it has
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
prefetch=False, prefetch_depth=2, verify=True, out_dtype=torch.float16)`. The kinds are:
`gpu, cpu, cpu-pinned, mmap-f16, mmap-i8, cached-lru, cached-lfu, static-topk`.
The source tensors are plain attributes, not registered buffers. Thus, `model.to()`/`state_dict()` ignore them.

## Design

All sources that are not on the GPU use the same pipeline (`TableSource.forward`):

```
ids ──► CPU ──► torch.unique(sorted, inverse, counts) ──► fetch(unique ids) ──► rows [n, L·p] on device
                                    │                                               │
                                    └── inverse (int32) ──H2D──► rows.index_select(inverse) ──► [B,T,L·p]
```

For each call, the pipeline reads each distinct row from the backing store and copies it host→device **one time
or less**. A prefill of 256 TinyStories tokens has ≈ 120 distinct ids. A B=16 batch has ≈ 1000 distinct ids of 4096.

| source | where the table lives | per call |
|---|---|---|
| `GPUTableSource` | whole fp16 table in VRAM | `F.embedding` (baseline) |
| `CPUTableSource(pinned)` | fp16 tensor in CPU RAM | `index_select` of unique rows into a staging buffer (pinned → truly async `non_blocking` H2D; pageable → driver bounce copy), then H2D of only those rows |
| `MmapTableSource("f16"\|"i8")` | `np.memmap` of `ple_table.*.bin` | `np.take` of the sorted unique rows (lazy page faults). i8 also reads the fp32 scale of each row, sends int8+scale (≈½ the bytes) and dequantises on the device: `row = q·scale` |
| `CachedTableSource(backing, capacity_rows, policy)` | fixed `[capacity, L·p]` GPU buffer + CPU slot map (`slot_of[V]`, `id_of[C]`) in front of a backing source | gathers hits from the buffer, fetches misses from the backing in one batch, then admits them |
| `Prefetcher(source)` | wraps a source | a background thread (+ a dedicated CUDA copy stream) fetches rows for ids that `prefetch(ids)` puts in the queue; `forward` uses the oldest prefetch |

**Cache policies.**
- `lru` evicts the slots that were least recently used.
- `lfu` keeps counts for the full history. It admits a miss only if the id occurred more frequently than the victim. This is admission in the style of TinyLFU: ids that occur one time cannot remove hot rows.
- `static-topk` loads the `capacity` most frequent ids from a frequency array before the start, and it never evicts. The bench uses the counts of `train.bin`, never the val stream that it measures.

**Correctness invariants** (the tests examine all of them):
- A call never evicts rows that the same call uses.
- The cache returns misses from the rows that it fetched in the same call. Thus, a cache that is smaller than the unique ids of a single batch is still exact.
- Duplicates in a batch are one lookup.
- `hit_rate` is weighted by tokens (a duplicate of a hit counts as a hit).

**Prefetcher limits** (the design is intentionally simple):
- The prefetcher is strictly FIFO.
- With `verify=True`, each forward compares its ids to the prefetched ids (one D2H copy). If they are different, the prefetcher removes all pending work, and the call does a synchronous fetch. Thus, the result is always correct.
- There is one worker thread. The GIL lets only one thread do Python work at a time. Thus, only the numpy/torch copy kernels and the async H2D can occur at the same time as other work.
- The prefetcher cannot help autoregressive decode, because the next token is not known before sampling. It helps only when the GPU compute for each step takes more time than the fetch.
- On CUDA, the copy stream waits on an event that `prefetch()` records. Thus, the copy stream never writes over cache slots that a read still uses.

**Byte accounting** (`stats()`):
- `backing_bytes`: the row payload that the source reads from RAM/mmap, + int8 scales;
- `backing_page_bytes`: for mmap, the distinct 4 KiB pages that the source reads. This value shows the real cost of a cold read from flash;
- `h2d_bytes`: the row payload host→device;
- `h2d_index_bytes`: the inverse map/slot indices;
- `resident_vram_bytes`: the table or cache buffer that stays on the GPU;
- also `tokens`, `unique_rows`, `hit_tokens`, `hit_unique`, `miss_unique`, `inserts`, `evictions`, `hit_rate`.

## Cost model

For each token, PLE needs one row: `L·p·bytes_per_elem` (fp16: `2·L·p`; int8: `L·p + 4` for the scale).
A dense decoder reads *all* dense weights for each decode step (`2·N_dense` bytes in fp16, divided over the batch).

| model | L·p | PLE bytes/token f16 / i8 | table size f16 | dense params | dense bytes per decode step (f16) | ratio |
|---|---|---|---|---|---|---|
| ple-small (L=6, p=64) | 384 | 768 / 388 | 38.6 MB | 30.5 M | 61.0 MB | ≈ 79 000× |
| synthetic-big (L=12, p=256) | 3072 | 6144 / 3076 | 309 MB | 24.7 M | 49.5 MB | ≈ 8 000× |
| Gemma 3n E2B-like (262144 × 8960) | 8960 | 17.9 KB / 9.0 KB | 4.7 GB | – | – | – |

Thus, the table can be large (for synthetic-big, 6× the dense weights), but the traffic for each token is small.
Without a cache, 6 KB/token at 1 000 tok/s is 6 MB/s. This is much less than PCIe (≈ 12–25 GB/s) or UFS flash
(≈ 1–4 GB/s sequential). For flash, the real limit is random 4 KiB reads. For this reason, the stats include `backing_page_bytes`.
Also, the token frequencies are Zipfian. Thus, a small cache serves most of the traffic.

## Benchmark

```
python -m ple_lab.offload.bench --run runs/ple-small --device cuda   # + val loss gpu-f16 vs mmap-i8
python -m ple_lab.offload.bench --synthetic --device cuda            # builds runs/synthetic-big
python -m ple_lab.offload.bench --gemma3n-table --device cuda        # opt-in, 4.7 GB file; checks disk/RAM first
```

The default of `--device` is `cpu` (or `$PLE_DEVICE`). Thus, development never uses a GPU that a training job
uses. With `--device cpu`, the bench hides CUDA before a component can initialise it.

Workloads (ids from `data/tinystories/val.bin`, bf16 autocast on CUDA, warmup, then `cuda.synchronize` around
each timed iteration):

* `prefill`: full model, B=1, T=256 (60 iters).
* `decode`: full model, 256 steps on a **sliding window** of the last 128 tokens of a contiguous stream, with one new
  token for each step. This GPT has no KV cache, thus each step runs the full window again. tok/s counts only new tokens.
* `decode-1`: **table only**, the same 256-step stream, 1 id for each step. This is the same lookup that a decoder
  with a KV cache makes for each token. This workload isolates the latency of the PLE fetch.
* `batched`: full model, B=16, T=256 (20 iters).

Sources:
- `gpu, cpu, cpu-pinned, mmap-f16, mmap-i8`;
- LRU caches at 1/5/20 % of the vocab, and static top-k at 5 % (cache backing = `mmap-f16`);
- `+pf` (prefetch on) for prefill/batched.

Output: a markdown table on stdout and `results/offload/<run>-<timestamp>.json` (with all raw stats).

Column meaning:
- `table VRAM`: the bytes of the table/cache that stay on the GPU;
- `peak alloc`: `torch.cuda.max_memory_allocated` during the measured loop (with dense weights + activations);
- `H2D B/tok` and `backing B/tok`: the bytes for each row position that the source looks up. For `decode`, there are 128 positions for each new token.

## Results (RTX 4050 Laptop 6 GB, Windows/WDDM, torch 2.11, bf16 autocast, 2026-11-12)

Raw JSON: `results/offload/ple-small-20261112-192148.json` and `results/offload/synthetic-big-20261112-192517.json`.
The cache backing is `mmap-f16` (warm OS page cache). `vs gpu` is the change of throughput, relative to the table that is fully in VRAM.

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

**Quality** (50 batches × 16 × 256 tokens of val.bin, fp32, identical windows): the val loss with the gpu-fp16
table is **2.0153**. With the mmap-int8 table, it is **2.0154** (Δ = +0.00002). The int8 offload has no cost that we can measure.

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

Two minutes later, we ran a subset again (`--workloads decode,batched`). The results were:
- batched: gpu 161,618 tok/s, cpu −23 %, mmap-f16 −27 %, lru-5 % −14 %, topk-5 % −9 %;
- decode: gpu 40 tok/s, cpu −80 %, lru-5 % −45 %, topk-5 % +12 %.

The gpu baseline itself changed from 75 k to 162 k tok/s between the runs. Thus, on this shared laptop GPU, a
variation of ±30–50 % between runs is normal for the rows of the full model (see Caveats). **Only large effects in
the rows of the full model are meaningful.** The columns for bytes, hit rate and memory are deterministic.

## Interpretation

* **VRAM.**
  - ple-small: the PLE table goes from 36.8 MiB to 0 (cpu/mmap) or to 1.84 MiB (5 % cache, hit rate of 88–96 %). The change of the prefill/batched throughput is −7…−15 %.
  - synthetic-big: the table goes from **294.8 MiB to 14.7 MiB with a 5 % cache**. The hit rate is 95.5–96.5 % for static top-k, and 88–95 % for LRU on prefill/batched.
  - The peak allocated memory decreases from 474 to 194 MiB (prefill) and from 901 to 621 MiB (batched). The table was ~60 % of the prefill VRAM.
  - The cost of the 5 % caches to throughput is in the range of 0–20 % (−9 %/−14 % for batched in the second run).
* **Zipf lets small caches work.**
  - A 1 % LRU cache (503 rows) serves 77 % of the prefill/batched tokens.
  - A 5 % static top-k cache serves ~96 % on all workloads. Its frequencies come from train.bin, never from the val stream.
  - An increase of the LRU cache from 5 % to 20 % helps almost nothing (87.9 → 87.9 %, 94.8 → 95.8 %). The remaining misses are the long tail.
  - In the sliding-window decode, 127 of 128 rows occur again in each step. Thus, all caches have a hit rate of 99.6 %.
* **The system moves few bytes.**
  - Without a cache, a row is 768 B (ple-small) or 6 KB (synthetic-big) for each token.
  - De-duplication in each call decreases this by half for a 256-token prefill, and by ~5× for B=16.
  - A 5 % cache decreases it to 27–63 B and 158–504 B for each token.
  - A decode step reads 61 MB / 49.5 MB of dense weights. Thus, the PLE traffic is 4–5 orders of magnitude less.
* **int8** decreases the backing bytes and the H2D bytes by half (388 vs 768 B/token), with no cost to quality. But on this GPU, the dequant kernels make it a little slower. int8 is better when the limit is bandwidth (flash), not kernel launches.
* **The cost is latency, not bandwidth.**
  - `decode-1` (table only, one id for each step) shows the fixed overhead of a lookup that is not on the GPU. This overhead is 0.3–1.5 ms (CPU unique, numpy gather, one or two small H2D copies, scatter). An embedding on the GPU takes 0.1–0.3 ms.
  - A cache does not help there. A hit also needs the CPU slot lookup and an index H2D, and the overhead of Python and of launches is the largest part.
  - In a real decoder, the 20–50 ms model step can hide this ~1 ms (the `decode` rows are in the noise range of gpu). Or, the decoder can fetch the row immediately after sampling, while the first layers run.
  - An implementation in C++ or with CUDA graphs can decrease this overhead more.
* **Pinned vs pageable:** for these transfers of 1 MB or less, pinned staging gives no measurable benefit. A pinned allocation in each call even costs a little. Pinned memory is important for bulk preloads and larger batches.
* **Prefetch** had almost no effect for ple-small. It made synthetic-big batched *slower* (−29 %/−35 %).
  - The worker thread competes for the GIL with the thread that launches the kernels of the model.
  - The de-duplicated fetch (~1 ms) is already small, relative to the compute.
  - We keep the prefetcher as a correct, tested component. Our conclusion: for models of this size, synchronous de-duplicated fetches are sufficient.

## Caveats

* **Windows/WDDM:** kernel launches and H2D copies go through the WDDM scheduler. This adds jitter and latency to each copy, compared to Linux/TCC (p95 is 1.1–3× p50). Other GPU clients on this laptop (desktop apps, a local LLM server) cause the large changes between runs that this document shows above.
* **Warm page cache:** after the first read, the table files (37 / 309 MB) stay in the OS page cache (16 GB RAM). Thus, `mmap-f16` ≈ `cpu`. Cold reads from real flash cost for each *page*. Use the value `backing_page_bytes` in the JSON as the input for a flash model.
* `decode` has no KV cache. Thus, its absolute tok/s is low, and the 128-token forward takes most of the time. `decode-1` isolates the lookup. Both workloads count only new tokens.
* synthetic-big has random initial weights. Its timings are representative, but its quality is not. We measured the quality on ple-small.
* We did not run the optional `--gemma3n-table` benchmark (262144 × 8960, writes a 4.7 GB file).


## Tests

`tests/test_offload.py` saves a tiny GPT to `tmp_path` and examines these items:
- The logits of each source/policy (also an int8-backed cache and prefetch) are the same as the logits of the in-memory model, over several calls. The tolerance is fp16 `atol=1e-2` and int8 `atol=5e-2`. The error of per-row symmetric quantisation is ≤ max|row|/254 for each element, and `sqrt(p)` and the network make it larger.
- The rows for cpu/mmap are exactly equal. The i8 rows are in half of a quantisation step.
- The LRU hit/miss/eviction/byte accounting agrees with a manual calculation.
- For all three policies, the tests use a cache capacity (3) that is much smaller than the ~35 unique ids of one batch. The results are correct, and the slot map stays consistent.
- LFU admission keeps hot rows.
- A Zipf stream gives a hit rate of >50 % with a 10 % cache.
- The prefetcher uses the fallback when the ids are different.
- `attach` validates its input.

The CUDA cases are parametrised but optional (`PLE_TEST_CUDA=1`). Thus, `python -m pytest -q tests` never uses the GPU.
