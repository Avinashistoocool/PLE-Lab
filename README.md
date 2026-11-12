# PLE Lab: Per-Layer Embeddings, an offload engine and an explorer

This project is a small reproduction of the **Per-Layer Embeddings (PLE)** idea from Google's Gemma 3n, with full measurements. A PLE table gives each transformer layer its own small vector for each token. The token ID indexes the table, and the token ID is known before the forward pass. Thus, the table can stay off the GPU, and the system can stream a few hundred bytes for each token.

The project has three parts:

| Part | What it does | Docs |
|---|---|---|
| `ple_lab/` (model, data, training) | A GPT with optional PLE in the style of Gemma 3n, trained on TinyStories | `docs/CONTRACT.md` |
| `ple_lab/offload/` (offload engine) | Table sources (GPU / CPU / pinned / mmap-fp16 / mmap-int8), GPU caches (LRU/LFU/static-top-k), a prefetcher and a benchmark | `docs/OFFLOAD.md` |
| `explorer/` (PLE Explorer) | A web app that shows token vectors, neighbours, PCA maps, gate activations, ablations for each layer and a comparison with the baseline | `docs/EXPLORER.md` |

## Results (RTX 4050 Laptop, 6 GB)

### Training (same dense config and token budget, one seed each)
| run | PLE dim | val loss | params (total / PLE table / dense) | tokens | train time |
|---|---:|---:|---|---:|---:|
| baseline-small | 0 | 2.0288 | 30.04M / 0 / 30.04M | 24.6M | 11.8 min |
| ple-small | 64 | 2.0198 | 49.80M / 19.32M / 30.48M | 24.6M | 15.1 min |

The loss gap of 0.009 is in the range of the noise of a single seed. Thus, we do not claim it as an improvement.

### Offload engine
- **VRAM of the table:** the test used a synthetic table with 155M parameters (`runs/synthetic-big`). A static-top-k GPU cache of 5% decreased the table VRAM from **294.8 MB to 14.7 MB (−95%)**, with a **hit rate of 95.5–96.5%**. For prefill, the peak allocation decreased from 474 to 194 MB. The throughput changed by about −9% to −20%. On this shared laptop GPU, the noise from one run to the next is ±30%.
- **Data for each token:** with a 5% cache, the system moves only 20–504 B for each token. A decode step reads about 50–60 MB of dense weights.
- **Cache size:** the token frequencies are Zipfian. Thus, a 1% LRU cache serves about 77% of tokens. An increase from 5% to 20% adds almost nothing.
- **int8 tables:** when you keep the table as int8 and not as fp16, the val loss changes from **2.01533 to 2.01535** (50 × 16 × 256 tokens).
- **Prefetching** did not help. The background thread competes for the GIL of Python.

### Explorer findings on `ple-small` (25,500 val predictions)
- **Ablation:** when all PLE token inputs are zero, the loss increases by **+0.621 ± 0.008 nats**, and 75.5% of tokens become worse.
- **For each layer:** layer 0 is the most important (+0.090). The sum of the single-layer deltas is only 0.246. Thus, the layers partly replace each other.
- **Row norm and frequency:** the row norms follow the log token frequency (**Spearman 0.87** over the tokens that occur in training). Rare rows stay near their initial values.
- **Neighbours:** the nearest neighbours of full rows are semantic (" happy" → glad, lucky, cheerful; " dog" → puppy, cat, kitten).

The full tables are in `docs/OFFLOAD.md` and `docs/EXPLORER.md`. The raw data is in `results/`.

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
- The results come from one seed, a small model and 24.6M training tokens.
- We ran the benchmarks on a shared laptop GPU with Windows/WDDM.
- The mmap numbers show a warm OS page cache. They do not show cold reads from flash.
- The code has a benchmark for a table of the size of Gemma 3n (262k × 8960, `--gemma3n-table`), but we did not run it.
