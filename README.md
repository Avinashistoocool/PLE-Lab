# PLE Lab: Per-Layer Embeddings, an offload engine and an explorer

A small, fully measured reproduction of the **Per-Layer Embeddings (PLE)** idea from Google's Gemma 3n. A PLE table gives every transformer layer its own small vector for every token. Because the table is indexed by token ID, which is known before the forward pass, it can live off the GPU and be streamed in a few hundred bytes per token.

The project has three parts:

| Part | What it does | Docs |
|---|---|---|
| `ple_lab/` (model, data, training) | GPT with optional Gemma-3n-style PLE, trained on TinyStories | `docs/CONTRACT.md` |
| `ple_lab/offload/` (offload engine) | GPU / CPU / pinned / mmap-fp16 / mmap-int8 table sources, LRU/LFU/static-top-k GPU caches, a prefetcher, and a benchmark | `docs/OFFLOAD.md` |
| `explorer/` (PLE Explorer) | Web app for token vectors, neighbours, PCA maps, gate activations, per-layer ablations, baseline comparison | `docs/EXPLORER.md` |

## Results (RTX 4050 Laptop, 6 GB)

### Training (same dense config and token budget, one seed each)
| run | PLE dim | val loss | params (total / PLE table / dense) | tokens | train time |
|---|---:|---:|---|---:|---:|
| baseline-small | 0 | 2.0288 | 30.04M / 0 / 30.04M | 24.6M | 11.8 min |
| ple-small | 64 | 2.0198 | 49.80M / 19.32M / 30.48M | 24.6M | 15.1 min |

The 0.009 loss gap is within single-seed noise, so it isn't claimed as an improvement.

### Offload engine
- **Table VRAM:** on a 155M-param synthetic table (`runs/synthetic-big`), a 5% static-top-k GPU cache cut table VRAM from **294.8 MB to 14.7 MB (−95%)** with a **95.5–96.5% hit rate**. Peak allocation dropped from 474 to 194 MB for prefill. Throughput changed by roughly −9% to −20%; run-to-run noise on this shared laptop GPU is ±30%.
- **Data per token:** with a 5% cache, the bytes moved per token fall to 20–504 B. A decode step reads about 50–60 MB of dense weights.
- **Cache size:** because token frequencies are Zipfian, a 1% LRU cache already serves about 77% of tokens, and going from 5% to 20% adds almost nothing.
- **int8 tables:** storing the table as int8 instead of fp16 changes val loss from **2.01533 to 2.01535** (50 × 16 × 256 tokens).
- **Prefetching** did not help; the background thread contends for Python's GIL.

### Explorer findings on `ple-small` (25,500 val predictions)
- **Ablation:** zeroing all PLE token inputs raises loss by **+0.621 ± 0.008 nats**, and 75.5% of tokens get worse.
- **Per layer:** layer 0 matters most (+0.090). The single-layer deltas sum to only 0.246, so the layers partly back each other up.
- **Row norm vs. frequency:** row norms track log token frequency (**Spearman 0.87** over tokens seen in training). Rare rows stay near their initial values.
- **Neighbours:** full-row nearest neighbours are semantic (" happy" → glad, lucky, cheerful; " dog" → puppy, cat, kitten).

Full tables are in `docs/OFFLOAD.md` and `docs/EXPLORER.md`. Raw data is in `results/`.

## Reproduce
```powershell
pip install -r requirements.txt
python -m ple_lab.data                                   # TinyStories -> data/tinystories/*.bin
python -m ple_lab.train --name baseline-small --ple-dim 0  --max-iters 3000 --batch-size 16 --grad-accum 2
python -m ple_lab.train --name ple-small      --ple-dim 64 --max-iters 3000 --batch-size 16 --grad-accum 2
python scripts\compare_runs.py
python -m ple_lab.offload.bench --run runs/ple-small --device cuda
python -m ple_lab.offload.bench --synthetic --device cuda
python -m explorer.server --runs runs --port 8765          # open http://localhost:8765
python -m pytest -q tests
```

## Limitations
- One seed, a small model and 24.6M training tokens.
- Benchmarks were run on a shared laptop GPU under Windows/WDDM.
- mmap numbers reflect a warm OS page cache, not cold flash reads.
- The Gemma-3n-sized table (262k × 8960) benchmark is implemented (`--gemma3n-table`) but was not run.
