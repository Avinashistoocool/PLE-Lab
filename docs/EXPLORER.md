# PLE Explorer

An interactive web app for looking inside the Per-Layer Embedding (PLE) table and the PLE path of a
trained run. See `docs/CONTRACT.md` for the model/run format.

## Running it

```powershell
python -m explorer.server --runs runs --port 8765
# then open http://127.0.0.1:8765
```

Options:

| flag | default | meaning |
|---|---|---|
| `--runs` | `runs` | directory containing run directories (`runs/<name>/config.json`, ...) |
| `--port` / `--host` | `8765` / `127.0.0.1` | where to serve |
| `--device` | `cpu` | `cpu` or `cuda`. Keep `cpu` while a training job is using the GPU. |
| `--data` | `<repo>/data/tinystories/train.bin` | uint16 GPT-2 token file used for token frequencies |

* Runs are discovered on every request and loaded lazily. A run is reloaded when `config.json`,
  `dense.pt` or `ple_table.f16.bin` changes (mtime), so a run that the trainer rewrites is picked up
  automatically. A run that is still half-written gives a 400 error instead of crashing the server.
* Token frequencies are counted once from `--data` (about 72M tokens, a few seconds) and cached in
  `<runs>/.explorer_cache/freq_<path-hash>_<size>_<mtime_ns>.npy`. If the data file changes (size or
  mtime), the cache key changes, counts are recomputed and the stale cache file is deleted. Without the
  file, the app still works: frequency-dependent views fall back to token-id order and say so.
* No trained run yet? `python -m explorer.make_demo_run` writes `runs/demo-random` (random init).
* UI smoke test and screenshots (server must be running, uses the installed Edge through Playwright):
  `python -m explorer.screenshot_tour --url http://127.0.0.1:8765 --out explorer/screenshots`
* Findings: `python -m explorer.findings --run runs/ple-small --baseline runs/baseline-small`
  writes `results/explorer_findings.json` (see [Findings](#findings)).

Baseline runs (`ple_dim=0`) appear in the run picker with "(baseline)"; every PLE view then shows a
warning and the API returns HTTP 400 for them. They are used in the Compare tab.

## Conventions

* **Token part / slice**: the raw table row `E[t]` reshaped to `[L, p]` and multiplied by `√p`, i.e.
  exactly the token term that goes into `u_l = (E[t]_l·√p·mask_l + RMSNorm(W_ctx·x₀·d^-½)_l)/√2`.
  At initialisation (`std 0.02`) a slice has norm ≈ `0.02·p` and a full row ≈ `0.02·p·√L`.
* **Gate**: `mean |gelu(W_gate·h)|` where `h` is the residual stream of block `l` *after* its attention
  and MLP sublayers (the input of the PLE injection), not the block input.
* **Contribution ratio**: `‖RMSNorm(W_proj(gate ⊙ u_l))‖ / ‖h‖`, the size of the PLE update relative to the
  residual stream it is added to. Because of the final RMSNorm its numerator is roughly constant
  (≈ `√d` times the norm weight), so the ratio mostly falls as `‖h‖` grows with depth.
* **Loss** at position `i` is `−log p(token_i | tokens_<i)`; position 0 has no loss. Top-5 at position `i`
  are the guesses for token `i+1`.
* **Δloss** (ablation) `= ablated loss − normal loss`. Ablation zeroes the token part of the chosen
  layers (`ple_mask`); the context part `RMSNorm(W_ctx x₀)` stays. Positive = the token got harder to
  predict without PLE.
* Text longer than `block_size` tokens is truncated (with a warning banner); empty text is an error.

## Tabs

### Token
Type a token (`" the"`, note the leading space for word tokens) or `#id`. Multi-token text uses the
first token and lists the pieces.

* **Per-layer row norm**: `‖slice_l‖` per layer, with the init norm as a dashed line. Bars well above
  it were moved by training; rare tokens stay near it. Click a bar to show neighbours in that layer.
* **Layer similarity**: cosine between this token's slices of different layers, next to the average
  matrix over the 1000 most frequent tokens. High off-diagonal values mean the layers store similar
  information for the token; values near 0 mean every layer gets its own, unrelated vector.
* **Vectors**: the `L × p` slice heatmap (average-pooled if `p > 128`).
* **Nearest tokens**: cosine nearest neighbours in one layer slice (or the full row). Click a row to
  jump to that token. Similar slices mean that layer's PLE path receives the same signal for both.

### Map
2-D PCA of the slice of one layer (or the full row) for the N most frequent tokens. Colour by
frequency rank or norm, highlight a token with the search box, click a point to open it. Note that
two PCs typically explain a small fraction of the variance (shown in the header), so the picture is
only a rough view of the dominant directions (often frequency/norm).

### Analyze
Runs the PLE model on your text (traced forward that is checked against the model's own forward in
the tests).

* Token chips coloured by gate magnitude in a chosen layer, by loss, or by Δloss. Hover for per-layer
  gates and the next-token top-5; click to pin.
* **PLE contribution** per layer (mean ratio over the text) and **mean gate** per layer.
* **Ablate PLE** (all layers, or the ticked layers) re-runs with the token part zeroed, shows mean Δloss
  and the most PLE-dependent tokens. One short text is noisy; use it for intuition, use
  `explorer.findings` for averaged numbers.

### Stats
* Histogram (log counts) of slice norms per layer over the whole GPT-2 vocab.
* Full-row norm vs. training count (log x axis) for 3000 random tokens, with the mean norm per
  frequency bucket and the init norm. Rows of tokens that never occur in train.bin receive no gradient
  from the loss; any change there comes from the optimizer (e.g. weight decay).
* int8 quantisation error per layer of the per-row symmetric int8 table (`‖q·s − w‖/‖w‖`).
  The scale is set by the row's largest entry, so a layer whose slice is small relative to the rest of
  the row loses relatively more precision.

### Compare
Per-token loss of a baseline run vs. a PLE run on the same text; `diff = baseline − PLE` (red = PLE
better). Only meaningful when both runs saw the same data and budget; a single text is anecdotal.

### About
Short explanation of PLE and of the formulas above.

## API

`GET /api/runs`, `GET /api/token?run&q|id`, `GET /api/neighbors?run&id&layer&k`,
`GET /api/layer_similarity?run[&id][&top_n]`, `GET /api/projection?run&layer&n`, `GET /api/stats?run`,
`POST /api/analyze {run,text,mask}` (`mask`: `null`, `"all"` or a list of layers),
`POST /api/compare {text,baseline,ple}`. Errors are HTTP 400 `{"detail": "..."}`.

## Findings

From `runs/ple-small` (L=6, d=384, p=64, 3000 iters × 16×2×256 = 24.6M tokens, reported val loss 2.0198)
vs. `runs/baseline-small` (same budget, val 2.0288). Produced by
`python -m explorer.findings` (CPU, ~7 min); raw numbers in `results/explorer_findings.json`.
Ablation numbers: 100 random 256-token windows from `data/tinystories/val.bin` (seed 0) = 25,500
next-token predictions. ± is the standard error of the mean per-token Δ.

**Ablation (zeroing the PLE token part; the context part stays).**

| ablated | mean loss | mean Δloss | tokens worse |
|---|---:|---:|---:|
| none | 2.0636 | – | – |
| all layers | 2.6845 | **+0.621 ± 0.008** | 75.5% |
| layer 0 | 2.1534 | +0.090 ± 0.003 | 63.2% |
| layer 1 | 2.1070 | +0.043 ± 0.002 | 59.6% |
| layer 2 | 2.0914 | +0.028 ± 0.002 | 58.6% |
| layer 3 | 2.0928 | +0.029 ± 0.002 | 58.5% |
| layer 4 | 2.0881 | +0.025 ± 0.001 | 60.3% |
| layer 5 | 2.0947 | +0.031 ± 0.002 | 61.4% |

* Layer 0's PLE slice matters most (Δ +0.090, twice layer 1); layers 2–5 are similar (+0.025…+0.031).
* Removing all layers (+0.621) hurts far more than the sum of single-layer removals (+0.246): the
  layers partly back each other up. The trained model clearly *uses* the table: without it, it is much
  worse than the baseline that never had one (2.68 vs 2.07).
* On the same windows the baseline's loss is 2.0667, i.e. PLE is better by only 0.003 nats here
  (0.009 by the trainer's val estimate). At this scale/budget the extra 19.3M table parameters buy
  very little; this is one seed each, so the sign of that small gap is not established.

**Row norm vs. frequency.** Spearman ρ between full-row norm and log10(1+train count) is **0.649** over
all 50,257 GPT-2 tokens and **0.872** over the 22,446 tokens that occur in train.bin (per layer
0.79–0.81). Mean full-row norm: 3.133 for unseen tokens (init expectation 0.02·p·√L = 3.135, i.e.
untouched apart from weight decay), 4.016 for seen ones, 7.419 for the 1000 most frequent.

**Layer independence.** Mean cosine between different layers' slices over the 1000 most frequent
tokens is ≈ 0 (all |cos| ≤ 0.02): each layer gets its own, unrelated vector per token.

**Nearest neighbours (cosine, top 6).** Single slices are noisy (many unrelated sub-word pieces);
the full row is cleanly semantic:

| token | layer 0 | layer 5 | full row |
|---|---|---|---|
| " the" | ghost, The, tailed, istors, backgrounds | " The", The, cel, 117, sho | The, " The", nearest, local, the |
| " happy" | fun, bubbles, worry, wisely, reminder | ojure, seeds, rope, spoon, arms | glad, lucky, happier, laughing, cheerful, pleased |
| " dog" | cat, Run, prowess, puppy, Juda | uggets, puppy, tiger, party, farmer, Max | puppy, cat, kitten, tiger, horse, dogs |
| " ran" | runs, am, marched, hurried, ogged | runs, otype, payments, poem, hurried | runs, run, running, hurried, am, rushed |
| " Lily" | Israeli, addy, nie, princess, ita, Lilly | Grace, drum, Sarah, ipop, Tom | Tom, Anna, Mia, Sam, Jill, Ben |
| " said" | says, replied, mentioned, Story, complain | says, urden, recept, OST, declared | says, saying, say, replied, complained |

Layer 0 tends to keep morphological/lexical relatives (" ran"→" runs", " said"→" says", " dog"→" cat");
no single layer stands out as "the semantic one". Top cosines are only 0.45–0.67, so this is qualitative.

**int8 quantisation.** Per-row symmetric int8 gives relative L2 error 0.715–0.738% per layer
(mean max-abs error per slice 2.79e-4 on raw rows). Its effect on the loss was not measured here.

**PLE contribution ratio** `‖PLE update‖/‖h‖`, mean over the 25,500 positions, layers 0→5:
0.611, 0.459, 0.390, 0.361, 0.322, 0.261. Mean gate |gelu(W_gate h)|: 2.55, 3.24, 2.42, 2.08, 2.10, 2.19.
The declining ratio largely reflects the growing residual norm (the RMSNorm'd update has a roughly fixed
size), so it is not by itself evidence that late layers use PLE less; the ablation is the better measure.

**Limitations.** One training seed per model, a short budget (24.6M tokens), a single 6-layer config;
ablation only zeroes the token part (the context part is still injected); neighbour lists are
qualitative.
