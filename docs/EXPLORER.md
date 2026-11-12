# PLE Explorer

The PLE Explorer is an interactive web app. It shows the contents of the Per-Layer Embedding (PLE) table and
the PLE path of a trained run. For the model format and the run format, see `docs/CONTRACT.md`.

## Running it

```powershell
python -m explorer.server --runs runs --port 8765
# then open http://127.0.0.1:8765
```

Options:

| flag | default | meaning |
|---|---|---|
| `--runs` | `runs` | the directory that contains the run directories (`runs/<name>/config.json`, ...) |
| `--port` / `--host` | `8765` / `127.0.0.1` | the address of the server |
| `--device` | `cpu` | `cpu` or `cuda`. Keep `cpu` while a training job uses the GPU. |
| `--data` | `<repo>/data/tinystories/train.bin` | the uint16 GPT-2 token file for the token frequencies |

* The server finds the runs again on each request, and it loads a run only when necessary. The server loads a run
  again when `config.json`, `dense.pt` or `ple_table.f16.bin` changes (mtime). Thus, the server automatically
  uses a run that the trainer writes again. If a run is only half written, the server gives a 400 error and does not crash.
* The server counts the token frequencies one time from `--data` (about 72M tokens, a few seconds). It keeps the
  counts in `<runs>/.explorer_cache/freq_<path-hash>_<size>_<mtime_ns>.npy`. If the data file changes (size or
  mtime), the cache key changes. Then the server counts again and deletes the old cache file. If the data file
  is not available, the app continues to work. The views that need frequencies then use the token-id order, and they show this.
* If you do not have a trained run, `python -m explorer.make_demo_run` writes `runs/demo-random` (random initial weights).
* UI smoke test and screenshots: the server must run. The test uses the installed Edge through Playwright.
  `python -m explorer.screenshot_tour --url http://127.0.0.1:8765 --out explorer/screenshots`
* Findings: `python -m explorer.findings --run runs/ple-small --baseline runs/baseline-small`
  writes `results/explorer_findings.json` (see [Findings](#findings)).

Baseline runs (`ple_dim=0`) show "(baseline)" in the run picker. For these runs, each PLE view shows a
warning, and the API returns HTTP 400. The Compare tab uses the baseline runs.

## Conventions

* **Token part / slice**: the raw table row `E[t]`, with the shape changed to `[L, p]` and multiplied by `√p`. This is
  exactly the token term in `u_l = (E[t]_l·√p·mask_l + RMSNorm(W_ctx·x₀·d^-½)_l)/√2`.
  At initialisation (`std 0.02`), a slice has a norm of ≈ `0.02·p`, and a full row has a norm of ≈ `0.02·p·√L`.
* **Gate**: `mean |gelu(W_gate·h)|`. Here, `h` is the residual stream of block `l` *after* its attention
  and MLP sublayers (the input of the PLE injection). It is not the input of the block.
* **Contribution ratio**: `‖RMSNorm(W_proj(gate ⊙ u_l))‖ / ‖h‖`. This is the size of the PLE update, relative to the
  residual stream that receives it. Because of the final RMSNorm, the numerator is almost constant
  (≈ `√d` times the norm weight). Thus, the ratio mostly decreases when `‖h‖` increases with depth.
* **Loss** at position `i` is `−log p(token_i | tokens_<i)`. Position 0 has no loss. The top-5 at position `i`
  are the guesses for token `i+1`.
* **Δloss** (ablation) `= ablated loss − normal loss`. The ablation sets the token part of the selected
  layers to zero (`ple_mask`). The context part `RMSNorm(W_ctx x₀)` stays. A positive value means that the token was more
  difficult to predict without PLE.
* If a text is longer than `block_size` tokens, the app truncates it and shows a warning banner. An empty text gives an error.

## UI

The shell uses the "Specimen Ledger" identity. It shows the PLE table as the catalogue of token specimens of a naturalist.
- **Ledger** (light, default):
  - archival paper `#f2ebdd` with a printed graph-paper grid and a vignette;
  - ink `#1c1a16`;
  - vermilion `#d9381e` for actions, selection and PLE;
  - verdigris `#1f5c57` as the secondary accent.

  The cards are paper index cards. Each card has a catalogue number ("No. 03 · Token"), hairline ink rules and double rules under the titles.
- **Darkroom** (dark): charcoal `#15130f` with film grain, paper text `#e9e1cf`, safelight red `#ff5a36` and brass `#c9a227`.
- Fonts (local, SIL OFL):
  - **Newsreader** (display, headings, italic KPI numerals);
  - **Atkinson Hyperlegible** (UI text);
  - **Martian Mono** (tokens, ids, axis labels, tables).
- Colour maps:
  - Ledger uses a paper → verdigris → ink ramp for magnitudes, and verdigris ↔ paper ↔ vermilion for signed values.
  - Darkroom uses viridis, and brass ↔ charcoal ↔ safelight.
- Layer colours L0–L5: gamboge, cerulean, viridian, ochre, prussian blue, madder rose. These are botanical-plate pigments,
  derived from Okabe–Ito, and they are safe for colour-blind users. They are `--l0`…`--l5` in `style.css`. Ledger uses deeper pigments, and Darkroom uses brighter pigments.
- The page load has one staggered sequence. The masthead rule draws in, and the title and the KPIs move up.
  Then the cards follow, ~60 ms apart. Tab switches use a short ink wipe. `prefers-reduced-motion` stops all motion.
- The 3D map is always a dark "lightbox" plate, with the caption "Plate I".
- Token strings use a mono font, with `·` for a space and `⏎` for a newline.

The app uses no CDNs at runtime.

* **Themes:** Ledger (light) is the default. To change to Darkroom, use the sun/moon button, `T` or the palette. The app keeps the selection in `localStorage` (`ple.theme`).
* **Hash routes** (back/forward work):
  - `#token/<query>` (for example `#token/%20dog` or `#token/%23262`);
  - `#map`, `#analyze`, `#stats`, `#compare`, `#about`;
  - you can add `?run=<name>` to all routes.
* **Shortcuts:**

  | key | action |
  |---|---|
  | `1`–`6` | switch tab |
  | `/` | focus search (Token or Map) |
  | `Ctrl/Cmd+K` | command palette |
  | `Ctrl+Enter` | run Analyze |
  | `T` | toggle theme |
  | `?` | shortcuts overlay |
  | `Esc` | close |

* **Command palette:** with the palette, you can do these tasks:
  - go to a tab;
  - examine a token (type, for example, `dog`);
  - analyze the sample text or your text;
  - ablate all layers;
  - change the map between 2D and 3D;
  - change the theme;
  - change the run.
* **Feedback:**
  - errors and warnings show as toasts (they replace the old banner);
  - a progress bar at the top shows while requests run;
  - a skeleton shimmer fills the panels while they load;
  - the "?" button on each card gives an explanation of the chart.
* **Map 2D | 3D:** a segmented toggle. The default is 3D when WebGL is available. The app keeps the selection as `ple.mapmode`.
  - 3D mode loads `/static/map3d.js` (`mountMap3D`) only when necessary.
  - If this fails, a note shows, and the app shows the 2D PCA map.
* **Animated backdrop:** `prefers-reduced-motion` stops it. It also stops when the tab is hidden.

## Tabs

### Token
Type a token (`" the"`; word tokens start with a space) or `#id`. If the text has more than one token, the tab uses the
first token and shows a list of the pieces.

* **Per-layer row norm**: `‖slice_l‖` for each layer, with the initial norm as a dashed line. Training moved the bars that are much
  higher than this line. Rare tokens stay near it. Click a bar to show the neighbours in that layer.
* **Layer similarity**: the cosine between the slices of this token in different layers, next to the average
  matrix over the 1000 most frequent tokens. High values off the diagonal mean that the layers keep similar
  information for the token. Values near 0 mean that each layer gets its own, unrelated vector.
* **Vectors**: the heatmap of the `L × p` slices (with average pooling if `p > 128`).
* **Nearest tokens**: the nearest neighbours by cosine in one layer slice (or the full row). Click a row to
  go to that token. When two slices are similar, the PLE path of that layer receives the same signal for both tokens.

### Map
A 2-D PCA of the slice of one layer (or the full row) for the N most frequent tokens. You can colour the points by
frequency rank or by norm. You can highlight a token with the search box. Click a point to open it.
Usually, two PCs explain only a small part of the variance (the header shows this value). Thus, the picture
is only an approximate view of the dominant directions (frequently, frequency or norm).

### 3D map
An interactive three.js view (`explorer/static/map3d.js`). A copy of three.js r170 is in
`explorer/static/vendor/three/`, thus the view works fully offline. Standalone demo:
`http://127.0.0.1:8765/static/map3d-demo.html?run=ple-small`.

**What it shows.**
- The view shows the N most frequent tokens (1000/2000/5000).
- The layer views use a *shared* 3-D PCA. This PCA is fitted on the slices of all layers together (`[N·L, p]`). Thus, all layers use the same axes.
- The slider moves each token from its position in layer `l` to its position in layer `l+1` (800 ms, easeInOutCubic).
  ▶ plays a loop 0→5→full.
- The **Full row** view (default) is a separate PCA of the concatenated row.
- For the display, the view centres each view and scales it to the same RMS radius.
- Three PCs explain only ~3–4% of the variance of each slice. Thus, the map shows the dominant directions. The cosine neighbours in the card are exact.

* **Points**: soft glowing sprites. The size is proportional to the log frequency. You can colour the points by one of these values:
  - the k-means cluster (12 clusters on the full row, normalised to unit length). The legend shows the top tokens of each cluster. Click a cluster to isolate it;
  - the log frequency;
  - the norm of the slice or row (with a mark at the initial norm);
  - the layer with the largest slice norm.
* **Hover**: shows the token (`·` = space), the id, the frequency rank, the count, the cluster and the norm.
* **Click**: selects the token. The view then shows these items:
  - the path of the token through layers 0…L−1, as a glowing tube with numbered layer markers;
  - constellation lines to the 8 cosine neighbours of the token in the current layer;
  - an info card with the norm of each layer, the neighbour list and the neighbour overlap between consecutive layers. You can click a bar to go to that layer, and you can click the neighbours.
* **Double-click** (or *Open in token view*) calls `onTokenClick(id)`.
* **Search** (`" dog"`, `dog`, `#3290`, or a substring): the camera moves to the token, and the token pulses.
  The view uses the same PCA to place tokens that are not in the top N.
* Other features:
  - labels for the 40 most frequent tokens (they fade with distance);
  - cluster labels in the full-row view;
  - a starfield, distance fog, a grid/axis gizmo and optional bloom;
  - a slow automatic rotation when idle. It stops when you use the view, and it starts again after 6 s.
* `Esc` clears the selection. Rendering stops when the view is hidden (IntersectionObserver +
  `visibilitychange`). DPR ≤ 2. `prefers-reduced-motion` stops the automatic rotation and the tweens.

**Module API.**

```js
const m = await import('/static/map3d.js');
const map = await m.mountMap3D(containerEl, {
  run: 'ple-small',                        // required
  api: (path, body) => Promise<json>,      // optional, default: fetch + throw on HTTP error
  onTokenClick: id => {},                  // optional (double-click / card button)
  theme: { bg, fg, accent, accent2, muted } // optional, default: the container's CSS variables
});
await map.setRun(name); map.setLayer(3 /* or 'full' */); map.highlight(' dog'); map.resize(); map.destroy();
```

The module adds its own CSS (a ref-counted `<style id="m3d-style">`). It builds its overlay UI in
the container. The container must have a size, because the view fills it. `destroy()` stops the render loop. It also removes
these items:
- the listeners and observers;
- the GPU resources (geometries, materials, textures, render targets, the WebGL context);
- all DOM elements that the module created.

### Analyze
This tab runs the PLE model on your text. It uses a traced forward. The tests compare this traced forward with the own forward of the model.

* The token chips have a colour for the gate magnitude in a selected layer, for the loss or for the Δloss. Move the pointer over a chip to see the gates
  of each layer and the top-5 for the next token. Click a chip to pin it.
* **PLE contribution** for each layer (the mean ratio over the text) and **mean gate** for each layer.
* **Ablate PLE** (all layers, or the selected layers) runs the model again with the token part set to zero. It shows the mean Δloss
  and the tokens that depend most on PLE. One short text has much noise. Use it to get a general idea, and use
  `explorer.findings` for averaged numbers.

### Stats
* A histogram (log counts) of the slice norms of each layer, over the full GPT-2 vocab.
* The full-row norm against the training count (log x axis) for 3000 random tokens. The chart also shows the mean norm for each
  frequency bucket and the initial norm. The rows of tokens that never occur in train.bin receive no gradient
  from the loss. If these rows change, the optimizer causes the change (for example, weight decay).
* The int8 quantisation error for each layer of the int8 table (symmetric, one scale for each row) (`‖q·s − w‖/‖w‖`).
  The largest entry of the row sets the scale. Thus, a layer with a slice that is small, relative to the remaining part of
  the row, loses more precision.

### Compare
The loss for each token of a baseline run and of a PLE run on the same text. `diff = baseline − PLE` (red = PLE
is better). This comparison is meaningful only when the two runs used the same data and budget. A single text is only an example.

### About
A short explanation of PLE and of the formulas above.

## API

`GET /api/runs`, `GET /api/token?run&q|id`, `GET /api/neighbors?run&id&layer&k`,
`GET /api/layer_similarity?run[&id][&top_n]`, `GET /api/projection?run&layer&n`, `GET /api/stats?run`,
`GET /api/projection3d?run&n=2000&space=shared|per_layer` (points with `coords` for each layer, `full`,
`norms`, `cluster`, `max_layer`, and also `clusters` and the explained variance; `n` ≤ 5000),
`GET /api/trajectory?run&id&n&space&k=8` (the coordinates of one token for each layer and for the full row in the same
map, its `k` cosine neighbours among the tokens of the map for each layer, and `neighbor_overlap`),
`POST /api/analyze {run,text,mask}` (`mask`: `null`, `"all"` or a list of layers),
`POST /api/compare {text,baseline,ple}`, `GET /api/norm_dist?run[&bins&gx&aspect]` (norm histograms for seen/unseen tokens + hexbin), `GET /api/neighbor_graph?run&id[&layer=-1|l|all&k&k2]` (see `explorer/static/viz/README.md`). Errors are HTTP 400 `{"detail": "..."}`.

## Findings

These findings compare two runs:
- `runs/ple-small` (L=6, d=384, p=64, 3000 iters × 16×2×256 = 24.6M tokens, reported val loss 2.0198);
- `runs/baseline-small` (same budget, val 2.0288).

The command
`python -m explorer.findings` (CPU, ~7 min) makes them. The raw numbers are in `results/explorer_findings.json`.
The ablation numbers come from 100 random 256-token windows from `data/tinystories/val.bin` (seed 0). This gives 25,500
next-token predictions. ± is the standard error of the mean Δ for each token.

**Ablation (the PLE token part is set to zero; the context part stays).**

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

* The PLE slice of layer 0 is the most important (Δ +0.090, two times layer 1). Layers 2–5 are similar (+0.025…+0.031).
* The removal of all layers (+0.621) causes much more damage than the sum of the single-layer removals (+0.246). Thus, the
  layers partly replace each other. The trained model clearly *uses* the table. Without the table, the model is much
  worse than the baseline, which never had a table (2.68 vs 2.07).
* On the same windows, the loss of the baseline is 2.0667. Thus, PLE is better by only 0.003 nats here
  (0.009 by the val estimate of the trainer). At this scale and budget, the 19.3M additional table parameters give
  little benefit. There is only one seed for each model. Thus, the sign of this small gap is not certain.

**Row norm and frequency.** The Spearman ρ between the full-row norm and log10(1+train count) is **0.649** over
all 50,257 GPT-2 tokens. It is **0.872** over the 22,446 tokens that occur in train.bin (0.79–0.81 for each layer).
The mean full-row norm is:
- 3.133 for unseen tokens. The expected initial value is 0.02·p·√L = 3.135. Thus, only weight decay changed these rows;
- 4.016 for seen tokens;
- 7.419 for the 1000 most frequent tokens.

**Layer independence.** Over the 1000 most frequent tokens, the mean cosine between the slices of different layers
is ≈ 0 (all |cos| ≤ 0.02). Each layer gets its own, unrelated vector for each token.

**Nearest neighbours (cosine, top 6).** Single slices have much noise (many unrelated sub-word pieces).
The full row is clearly semantic:

| token | layer 0 | layer 5 | full row |
|---|---|---|---|
| " the" | ghost, The, tailed, istors, backgrounds | " The", The, cel, 117, sho | The, " The", nearest, local, the |
| " happy" | fun, bubbles, worry, wisely, reminder | ojure, seeds, rope, spoon, arms | glad, lucky, happier, laughing, cheerful, pleased |
| " dog" | cat, Run, prowess, puppy, Juda | uggets, puppy, tiger, party, farmer, Max | puppy, cat, kitten, tiger, horse, dogs |
| " ran" | runs, am, marched, hurried, ogged | runs, otype, payments, poem, hurried | runs, run, running, hurried, am, rushed |
| " Lily" | Israeli, addy, nie, princess, ita, Lilly | Grace, drum, Sarah, ipop, Tom | Tom, Anna, Mia, Sam, Jill, Ben |
| " said" | says, replied, mentioned, Story, complain | says, urden, recept, OST, declared | says, saying, say, replied, complained |

Layer 0 usually keeps morphological/lexical relatives (" ran"→" runs", " said"→" says", " dog"→" cat").
No single layer is clearly "the semantic one". The top cosines are only 0.45–0.67. Thus, this result is qualitative.

**int8 quantisation.** Symmetric int8 with one scale for each row gives a relative L2 error of 0.715–0.738% for each layer.
The mean max-abs error for each slice is 2.79e-4 on raw rows. We did not measure its effect on the loss here.

**PLE contribution ratio** `‖PLE update‖/‖h‖`, mean over the 25,500 positions, layers 0→5:
0.611, 0.459, 0.390, 0.361, 0.322, 0.261. Mean gate |gelu(W_gate h)|: 2.55, 3.24, 2.42, 2.08, 2.10, 2.19.
The ratio decreases mostly because the residual norm increases (the update after RMSNorm has an almost fixed
size). Thus, the ratio alone does not show that the last layers use PLE less. The ablation is a better measure.

**Limitations.**
- There is one training seed for each model.
- The budget is short (24.6M tokens).
- There is only one config (6 layers).
- The ablation sets only the token part to zero. The model continues to inject the context part.
- The neighbour lists are qualitative.
