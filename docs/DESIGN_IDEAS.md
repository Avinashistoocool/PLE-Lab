# PLE Explorer: design ideas

Research notes for upgrading the PLE Explorer's look and interaction (`explorer/static/`). This doc
changes no code. It is written for three agents: **(a) 3D map**, **(b) shell restyle** and
**(c) charts and interactions**.
Verified on 2026-11-12: npm versions come from `npm view`, file sizes from fetching the jsDelivr URLs, and colour contrast was computed against the proposed background.

---

## TL;DR

### Recommended visual direction: "Instrument Panel", a restrained sci-fi HUD on a near-black lab surface

Select **one** direction and use it everywhere. I recommend a calm, dark **scientific-instrument / HUD** look:
- near-black blue surfaces;
- 1-px hairline borders with subtle corner ticks;
- monospace numerals;
- **one cool accent (cyan) for interaction** and **one warm accent (amber) for "PLE / reference lines"**;
- data drawn in **perceptually uniform colour maps** (viridis for magnitudes, a blue↔orange diverging map for signed values and Δ);
- glow used **only on data**: bloom on the 3D points, the selected token and active marks. Chrome never glows.

Why not the alternatives:
- **Glassmorphism:** backdrop blur over a WebGL canvas costs GPU time and lowers chart contrast.
- **Neo-brutalism:** thick borders and flat primaries fight dense heatmaps.
- **Full terminal green-on-black:** monochrome breaks colour-encoded data.

The HUD direction suits the subject (an "instrument" that reads a model's table). It also keeps the
charts legible and gives the 3D map a natural home: a dark void with glowing points. The full token set is in [§6](#6-global-styling-the-instrument-panel-token-set).

### Top-8 picks (ranked by insight × wow ÷ effort)

| # | Upgrade | Tab | Effort | Wow |
|---|---|---|---|---|
| 1 | **Aligned layer-morph 3D map** (Procrustes-aligned PCA plus a "stack" mode where Z = layer) | Map | M | ★★★★★ |
| 2 | **Token trajectory ribbons** across the 6 layers, with fly-to on search | Map | M | ★★★★★ |
| 3 | **Neighbour constellation**: k-NN lines in 3D, with the 2D force graph as a fallback | Map/Token | S–M | ★★★★☆ |
| 4 | **Layer "fingerprint" radial glyph** (6 petals × 64 spokes) with an animated fill | Token | M | ★★★★☆ |
| 5 | **Gate river**: a streamgraph of gate magnitude per layer along the text, linked to the chips | Analyze | M | ★★★★★ |
| 6 | **Ablation "before → after" morph**: chips and bars animate from loss to ablated loss | Analyze | S | ★★★★☆ |
| 7 | **Ridgeline histograms** of slice norms per layer, with init-norm and seen/unseen split | Stats | S | ★★★★☆ |
| 8 | **Compare slope chart** (baseline → PLE loss per token) plus a diverging Δ bar | Compare | S | ★★★☆☆ |

Runners-up: hexbin norm-vs-frequency (#9), sparkline chips (#10), lasso → "selection drawer" (#11) and the command palette with token fly-to (#12, for the shell agent).

### Library shortlist (no build step; vendor into `explorer/static/vendor/`)

| Lib | Pinned URL | Size (min) | Licence |
|---|---|---|---|
| three.js core | `https://cdn.jsdelivr.net/npm/three@0.187.0/build/three.module.js` (+ `three.core.js`) | 659 KB + 1.47 MB unminified; jsDelivr `.min.js` ≈ 383 KB + 412 KB | MIT |
| OrbitControls | `https://cdn.jsdelivr.net/npm/three@0.187.0/examples/jsm/controls/OrbitControls.js` | 40 KB | MIT |
| d3 (UMD) | `https://cdn.jsdelivr.net/npm/d3@7.9.2/dist/d3.min.js` | 274 KB | ISC |
| regl-scatterplot | `https://esm.sh/regl-scatterplot@1.16.0/es2022/regl-scatterplot.bundle.mjs` (+ `regl`, `pub-sub-es`) | 96 + 123 + 2 KB | MIT |
| gsap (optional) | `https://cdn.jsdelivr.net/npm/gsap@3.15.1/dist/gsap.min.js` | 72 KB | "Standard no-charge" (not OSI) |

Full table with offline notes: [§2](#2-libraries-usable-without-a-build-step).

**Critical data caveat for agent (a):** read [§3.1](#31-the-layer-alignment-problem-must-read-for-the-3d-map) before building the layer morph.

---

## 1. Patterns borrowed from state-of-the-art tools

Each row lists one tool, the patterns worth stealing and how they map onto PLE Explorer.

### TensorFlow Embedding Projector
Sources: https://projector.tensorflow.org/ and the paper by Smilkov et al. 2016, https://arxiv.org/abs/1611.05469.
- **2D/3D toggle** for the same projection, with an auto-rotate orbit in 3D. → Give our Map a `2D | 3D` segmented switch. The 2D view can reuse the same camera, top-down orthographic.
- **Search box that selects and isolates** matches. "Isolate N points" re-runs the projection on the subset. → Lasso or search, then **"re-project selection"** with a new `/api/projection?ids=` endpoint.
- **Nearest-neighbour inspector:** a click on a point draws lines to its k neighbours and lists them in a side panel. You can select the distance metric (cosine vs Euclidean). → Our "constellation lines" (#3).
- **Bookmarks:** saved view states with a camera plus a selection, playable as a story. → "Snapshots" in the URL hash (#19).
- **Label by / colour by** drop-downs from metadata columns. → Colour by frequency rank, norm, layer, k-means cluster or "seen in train".

### Neuronpedia
Sources: https://www.neuronpedia.org/ (open source, Apache-2.0, https://github.com/hijohnnylin/neuronpedia) and its graph view at https://www.neuronpedia.org/gemma-2-2b/graph.
- **Feature dashboard layout:**
  1. a header with an ID and summary stats;
  2. **top activating text snippets with per-token highlight**;
  3. logit lists (positive/negative) and density histograms. → Our Token tab should read like a "feature card":
  - big glyph plus key numbers;
  - **"where this token's PLE gets gated hardest"**, which needs sample contexts (new endpoint);
  - the neighbour list;
  - a norm histogram that marks where this token sits.
- **Inline token highlighting with hover tooltips showing exact values.** We already do this in our chips; borrow their *muted base colour plus single-hue intensity*, which reads better than full viridis on chips.
- **Cosine-similarity and UMAP explorers** with lists you can add to. → Add a "pin to tray" for tokens across tabs.

### Anthropic feature browser and attribution graphs
Sources: Scaling Monosemanticity, https://transformer-circuits.pub/2024/scaling-monosemanticity/; Circuit Tracing methods, https://transformer-circuits.pub/2025/attribution-graphs/methods.html; and Biology, https://transformer-circuits.pub/2025/attribution-graphs/biology.html.
- **Graph layout:** the prompt runs along the bottom, layers stack upward and the output sits on top. Nodes are hoverable; **Cmd/Ctrl+Click builds a subgraph**; features can be grouped into **"supernodes"**. The methods paper says: "Nodes can be hovered over and clicked on … Subgraphs can also be constructed by using Cmd/Ctrl+Click".
  → **Analyze "layer × position grid"**: x = token position, y = layer 0..5 (bottom to top) and node size = gate. Edge-like vertical connectors could encode the contribution ratio. This is the same spatial grammar, applied to PLE injection points.
- **Feature-browser side panel:** a click on an item opens a persistent panel on the right with the details. Thus, the main view never navigates away. → Adopt globally: a **detail drawer** instead of switching tabs on click. Keep "open in Token tab" as an explicit button.
- **Pruning slider** (keep the top x% of edges by influence). → Use a "top-k neighbours" / "edge threshold" slider on the constellation and the force graph.

### Transformer Explainer (Polo Club)
Sources: https://poloclub.github.io/transformer-explainer/ and arXiv 2408.04619.
- **Live model in the loop:** type text and every view updates. A **temperature slider** reshapes the output distribution in real time. → Debounce re-analysis on textarea input, about 400 ms. CPU inference on a 6-layer model is fast enough.
- **Expand-on-click abstraction levels:** a click opens a block into its sub-operations, with a smooth zoom transition. → The About tab could become an **interactive diagram of the PLE equation**. Each term of `u_l = (E[t]_l·√p·mask_l + RMSNorm(W_ctx x₀))/√2` would be a clickable box showing that term's live values for the selected token.
- **Animated flow lines** (dashed strokes moving along a path) show data flowing between components. → Use them on the PLE diagram and the attribution-style grid.

### LLM Visualization (Brendan Bycroft, 3D)
Sources: https://bbycroft.net/llm and https://github.com/bbycroft/llm-viz (MIT).
- **Whole model as 3D blocks of cells**: each tensor is a slab of cubes, coloured by value. During a guided walkthrough, the camera flies between stages. → A **"PLE table slab"**: a 3D InstancedMesh of 6 × 64 cells for the current token, extruded by value. Fly the camera from the table → the slice → the gate → the residual stream.
- **Narrated, step-wise camera tours** with a progress timeline on the left. → Use the "Tour" mode in the shell (command palette: "Start tour").
- **Hover a cell → see its formula and inputs highlighted.**
  → A hover on a heatmap cell must highlight the same `(layer, dim)` in the radial glyph and the 3D slab (linked brushing).

### Nomic Atlas and its deepscatter engine
Sources: https://atlas.nomic.ai/, https://github.com/nomic-ai/deepscatter and the docs at https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/how-to-visualize-embeddings.
- **Automatic topic labels** at cluster centroids. The label density changes with zoom (more labels when you zoom in). → In 3D, label the k-means cluster centroids with the most frequent member token, and show the top-N labels by screen-space priority.
- **Lasso, then "tag"/filter, with a linked data table below the map.** → A lasso selection drawer (#11).
- **Smooth zoom with points scaling sub-linearly** so dense areas stay readable. → Point size ∝ `zoom^0.5`.

### Apple Embedding Atlas (closest modern analogue)
Sources: https://apple.github.io/embedding-atlas/ and https://github.com/apple/embedding-atlas (MIT; npm `embedding-atlas@0.26.0`, but it needs a bundler).
- **KDE density contours under the points** and **order-independent transparency**. → In 2D, draw `d3.contourDensity` isolines behind the scatter so dense regions read as "terrain".
- **Cross-filtering:** brushing a histogram filters the map and vice versa. → Brushing the Stats frequency axis would highlight those tokens in the Map.

### Distill.pub
Sources: Wattenberg et al. 2016, "How to Use t-SNE Effectively", https://distill.pub/2016/misread-tsne/; Activation Atlas, https://distill.pub/2019/activation-atlas/; Zoom In, https://distill.pub/2020/circuits/zoom-in/.
- **Small multiples with a parameter scrubber**, plus an honest caption. The caption tells what the projection can show and what it cannot show. → **Six mini-maps (L0–L5) in a row** above the big map, which doubles as a layer picker. Show the PCA explained variance prominently, as Distill would.
- **Activation Atlas grid aggregation:** put the 2D map into the bins of a grid. Render the averaged representative of each cell. → The hexbin "atlas" mode of the Map, where each hex is labelled by its most frequent token.
- **Margin notes and hover-defined terms.** → About tab and chart captions.

### BertViz
Source: https://github.com/jessevig/bertviz (Apache-2.0).
- **Model view:** a bird's-eye grid with a thumbnail for each layer × head. Click a thumbnail to enlarge it. → The **Analyze "layer strip"**: six thumbnail rows of gate-per-token, click to expand. This is the same small-multiple idea.
- **Head view:** lines between tokens, weighted by attention. A hover isolates a line. → In Compare, connect each token's baseline-loss and PLE-loss points (slope lines) and hover to isolate.

### Scrollytelling and other ML visualisations
- R2D3, "A visual introduction to machine learning": scroll-driven transitions of the same marks. http://www.r2d3.us/visual-intro-to-machine-learning-part-1/
- Jay Alammar's "Illustrated Transformer": consistent colour per tensor role. https://jalammar.github.io/illustrated-transformer/
- PAIR "Understanding UMAP": 3D↔2D morph of the same points with a scrubber. https://pair-code.github.io/understanding-umap/

What to borrow: **one mark set, many states**. Animate positions between states instead of redrawing. Give each tensor role a fixed colour: token slice = amber, context term = cyan, gate = violet.

---

## 2. Libraries usable without a build step

Sizes are the measured transfer of the jsDelivr file (minified unless noted). "Offline OK" means the file
has no further network imports once downloaded, or that the listed sibling files also have to be vendored.

| Library | Version / licence | URL (pinned) | Size | Offline once vendored? | Notes |
|---|---|---|---|---|---|
| **three.js** | 0.187.0 / MIT | `https://cdn.jsdelivr.net/npm/three@0.187.0/build/three.module.min.js` **and** `.../build/three.core.min.js` | 383 KB + 412 KB | ✅ Vendor **both** files. `three.module.js` does `import … from "./three.core.js"`, so keep the filenames (or the `.min` pair with a matching import map) | Use an import map so the addons resolve the `three` specifier |
| OrbitControls | three addons | `https://cdn.jsdelivr.net/npm/three@0.187.0/examples/jsm/controls/OrbitControls.js` | 40 KB | ✅ Imports only `'three'` | |
| UnrealBloomPass | three addons | `.../examples/jsm/postprocessing/UnrealBloomPass.js` | 15 KB | ✅ Also vendor `EffectComposer.js`, `RenderPass.js`, `OutputPass.js`, `Pass.js`, `ShaderPass.js`, `MaskPass.js`, `shaders/CopyShader.js`, `shaders/LuminosityHighPassShader.js`, `shaders/OutputShader.js` | About 10 small files, keep the `jsm/` tree layout |
| CSS2DRenderer | three addons | `.../examples/jsm/renderers/CSS2DRenderer.js` | 8 KB | ✅ | HTML labels pinned to 3D points (≤ 100 labels for performance) |
| Line2 / LineSegments2 | three addons | `.../examples/jsm/lines/LineSegments2.js` (+ `LineMaterial.js`, `LineSegmentsGeometry.js`) | 11 KB + | ✅ | Thick, anti-aliased lines for trajectories and constellations |
| troika-three-text | 0.53.1 / MIT | `https://cdn.jsdelivr.net/npm/troika-three-text@0.53.1/dist/troika-three-text.umd.min.js` | 92 KB | ⚠️ The UMD build expects global deps and a font fetch; prefer CSS2D or sprite labels | SDF text in 3D, optional |
| three-spritetext | 1.10.0 / MIT | `https://cdn.jsdelivr.net/npm/three-spritetext@1.10.0/dist/three-spritetext.min.js` | 9 KB | ✅ | Billboard labels |
| **regl-scatterplot** | 1.16.0 / MIT | ESM: `https://esm.sh/regl-scatterplot@1.16.0/es2022/regl-scatterplot.bundle.mjs` | 96 KB | ⚠️ It imports `/pub-sub-es@~3.0.0` and `/regl@~2.1.1`. Vendor `https://cdn.jsdelivr.net/npm/regl@2.1.1/+esm` (123 KB) and `https://cdn.jsdelivr.net/npm/pub-sub-es@3.0.0/+esm` (2 KB) and rewrite the 2 import paths, **or** use the UMD `.../regl-scatterplot@1.16.0/dist/regl-scatterplot.min.js` (111 KB) plus `regl.min.js` (85 KB) and a 2 KB pub-sub-es UMD as globals | Its README says up to 20 million points with fast lasso, animated point transitions and `colorBy`/`opacityBy`/`sizeBy` (https://github.com/flekschas/regl-scatterplot). This is the **best 2D 50k-point engine** |
| regl | 2.1.1 / MIT | `https://cdn.jsdelivr.net/npm/regl@2.1.1/dist/regl.min.js` | 85 KB | ✅ | |
| **d3** | 7.9.2 / ISC | UMD: `https://cdn.jsdelivr.net/npm/d3@7.9.2/dist/d3.min.js` | 274 KB | ✅ Single file, `window.d3` | **Do not** use `.../d3@7.9.2/+esm`. It loads about 30 module imports. The UMD file is the vendorable one. Covers scales, zoom, brush, force, contours, Delaunay, shapes (`d3.area`, `d3.stack`) and interpolators |
| d3-hexbin | 0.2.2 / BSD-3 | `https://cdn.jsdelivr.net/npm/d3-hexbin@0.2.2/build/d3-hexbin.min.js` | 2 KB | ✅ Attaches `d3.hexbin` | |
| d3-force-3d | 3.0.6 / MIT | `https://cdn.jsdelivr.net/npm/d3-force-3d@3.0.6/+esm` | small | ⚠️ ESM with deps; easier with 3d-force-graph | |
| **force-graph** (2D) | 1.52.0 / MIT | `https://cdn.jsdelivr.net/npm/force-graph@1.52.0/dist/force-graph.min.js` | 169 KB | ✅ Self-contained UMD (`ForceGraph`) | Canvas; ideal for the neighbour graph |
| **3d-force-graph** | 1.80.2 / MIT | `https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.2/dist/3d-force-graph.min.js` | 1.36 MB | ✅ It **bundles its own three.js copy** | Do not load it next to our three.js (duplicate three, about 1 MB wasted). Prefer building the constellation in our own three scene |
| deck.gl standalone | 9.4.1 / MIT | `https://cdn.jsdelivr.net/npm/deck.gl@9.4.1/dist.min.js` | 2.03 MB | ✅ Single UMD (`deck`) | `OrbitView` + `PointCloudLayer` handle 50k points easily. Overlaps with three; only worth it if agent (a) has not started on three |
| Plotly | 4.3.0 / MIT | `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.3.0/plotly.min.js` | 4.73 MB | ✅ | Heavy; `scatter3d` and `scattergl` work but look "Plotly". Not recommended |
| ECharts | 6.1.0 / Apache-2.0 | `https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.min.js` | 1.10 MB | ✅ | Polished themes, `themeRiver` (streamgraph) and parallel axes |
| ECharts-GL | 2.1.0 / MIT (BSD-3 in some metadata) | `https://cdn.jsdelivr.net/npm/echarts-gl@2.1.0/dist/echarts-gl.min.js` | 625 KB | ✅ Needs echarts loaded first | `scatter3D` and `bar3D` (a 3D layer×dim bar field). Ageing; not recommended alongside three |
| **GSAP** | 3.15.1 / "Standard no-charge" licence | `https://cdn.jsdelivr.net/npm/gsap@3.15.1/dist/gsap.min.js` (+ `dist/Flip.min.js`, 25 KB) | 72 KB | ✅ | Free unless end users are charged (https://gsap.com/standard-license). It is *not* OSI open source. Timelines plus Flip for chip/tab layout morphs |
| Motion (motion.dev) | 14.13.0 / MIT | `https://cdn.jsdelivr.net/npm/motion@14.13.0/dist/motion.js` | 187 KB (UMD) | ✅ | MIT alternative to GSAP: `animate()` with spring physics |
| Observable Plot | 0.6.19 / ISC | `https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.19/dist/plot.umd.min.js` | 207 KB | ✅ Needs the d3 UMD loaded first | Fast to author ridgelines, hexbin and dot plots as SVG |
| uPlot | 1.6.32 / MIT | `https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js` | 50 KB | ✅ | Tiny and fast line/bar charts with a cursor sync API |
| umap-js | 1.4.0 / MIT | `https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js` | 81 KB | ✅ | Better done server-side (`umap-learn` or sklearn TSNE) and cached |
| lil-gui | 0.21.0 / MIT | `https://cdn.jsdelivr.net/npm/lil-gui@0.21.0/dist/lil-gui.esm.min.js` | 29 KB | ✅ | Quick "visual settings" panel for the 3D map (bloom strength, point size) |
| Tweakpane | 4.0.5 / MIT | `https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js` | 149 KB | ✅ | Prettier lil-gui with a HUD aesthetic |
| ninja-keys | 1.2.2 / MIT | `https://cdn.jsdelivr.net/npm/ninja-keys@1.2.2/dist/ninja-keys.js` | 16 KB | ⚠️ It imports `lit`; a hand-rolled palette (about 80 lines) is easier offline | Command palette |
| Floating UI | 1.8.0 / MIT | `https://cdn.jsdelivr.net/npm/@floating-ui/dom@1.8.0/dist/floating-ui.dom.umd.min.js` | 10 KB | ⚠️ Also needs `@floating-ui/core` UMD | Tooltip positioning; the current tooltip is fine |
| sigma.js | 4.0.6 / MIT | `https://cdn.jsdelivr.net/npm/sigma@4.0.6/dist/sigma.min.js` | 385 KB | ⚠️ Needs graphology | Overkill for k ≤ 50 neighbours |
| Fonts | `@fontsource-variable/*` 5.3.0 / OFL-1.1 | `https://cdn.jsdelivr.net/npm/@fontsource-variable/inter@5.3.0/files/inter-latin-wght-normal.woff2` (48 KB), `.../jetbrains-mono@5.3.0/files/jetbrains-mono-latin-wght-normal.woff2` (40 KB) | 88 KB | ✅ Vendor the latin `woff2` and write your own `@font-face` | See §6 |

**Import map** for three (put it in `index.html` before any module script):
```html
<script type="importmap">{ "imports": {
  "three": "/static/vendor/three/three.module.min.js",
  "three/addons/": "/static/vendor/three/addons/" } }</script>
```
If you vendor the **min** pair, make sure that `three.module.min.js` imports `./three.core.min.js`.
The auto-minified file of jsDelivr keeps `./three.core.js`. Thus, **either rename the core file to `three.core.js` or
vendor the unminified pair** (659 KB + 1.47 MB). FastAPI serves them gzipped only if you add middleware.

**Recommended stack:**
- three (3D);
- d3 UMD (all 2D SVG/canvas charts, force, brush, zoom, contours, hexbin);
- optionally, regl-scatterplot (the 2D map at 50k points);
- Motion or GSAP (UI tweens).
Skip Plotly, ECharts and deck.gl: they duplicate capability, add 1–5 MB and impose their own look.

---

## 3. Ranked design upgrades (22)

Data available today (from `explorer/analysis.py`):
- `/api/token` → `norms[L]`, `vectors[L][≤128]`, `init_norm`, `frequency`;
- `/api/neighbors?layer&k` → `[{id, piece, sim}]`, with `layer=-1` for the full row;
- `/api/layer_similarity` → `matrix[L][L]` and `global[L][L]`;
- `/api/projection?layer&n` → `points[{id, piece, x, y, norm, freq}]` plus `explained_variance`, with n ≤ 10k and **an independent 2-component PCA per layer**;
- `/api/stats` → `hist{edges, counts[L][bins]}`, `norm_vs_freq{ids, norm, freq, pieces}`, `trend`, `quant`;
- `/api/analyze` → `gate[T][L]`, `contrib_ratio[T][L]`, `loss[T]`, `top5`, and `ablation.{loss, delta}`;
- `/api/compare` → `baseline.loss`, `ple.loss` and `diff`.

### 3.1 The layer-alignment problem (must read for the 3D map)

The findings in `docs/EXPLORER.md` show that the slices of different layers are **almost orthogonal**: mean |cos| ≤ 0.02. `projection()` fits a **separate PCA per layer** (`analysis.py:337-361`). As a result:
- PC axes of layer 0 and layer 3 are unrelated directions, and each has an arbitrary **sign**.
- A naive per-point tween from layer-l coordinates to layer-(l+1) coordinates looks like random scrambling.

To make the morph *meaningful*, add a new endpoint `/api/projection_stack?n&dims=3&align=procrustes`:
1. Compute `Y_l = PCA_l(X_l)` (3 components) per layer, for the **same token ids**.
2. **Orthogonal-Procrustes-align** each `Y_l` to `Y_{l-1}` (or to the full-row PCA):
   `U,S,Vt = svd(Y_l.T @ Y_ref); R = U@Vt; Y_l ← Y_l @ R`.
   This removes sign flips and rotations, so the remaining movement is real neighbourhood change.
3. Return a `Float32Array`-friendly layout:
   `{ids[n], pieces[n], freq[n], norm[n][L], pos[L][n][3], explained[L][3], cluster[n]}`.
   Send it as JSON (n = 5000 → about 1 MB), or better as `application/octet-stream` plus a JSON header.

An alternative "honest" mode is a **joint basis**: run PCA on the concatenation `[X_0; …; X_5]` stacked over tokens. All layers then share axes, and the morph shows each layer's cloud *drifting* in a common space. Because the layers are orthogonal, the clouds will occupy different regions, which is itself the insight. Offer both modes, labelled **"aligned (shape)"** and **"shared axes (position)"**.

Show the explained variance per layer as a small bar in the HUD. Two or three PCs of a 64-D slice explain little, so be honest about it.

---

### #1 Aligned layer-morph 3D map, with a "stack" mode · Map · **M** · Wow ★★★★★
- **Shows:** the n most frequent tokens as glowing points in 3D. A layer scrubber (L0…L5, with a play button) tweens the points between Procrustes-aligned layer projections. **Stack mode** places the six layer clouds along Z (z = layer × gap), like a deck of slides you can orbit.
- **Why it matters for PLE:**
  - It shows directly whether a token's *neighbourhood* survives from layer to layer.
  - Clusters that stay together mean that the layers encode the same grouping again.
  - Clusters that dissolve mean that each layer gets its own unrelated code. The cos ≈ 0 finding predicts this.
  - Stack mode makes "6 independent tables in one row" tangible.
- **Sketch** (three, InstancedBufferAttribute or Points with a custom shader):
```js
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos[0].flat(), 3));
geo.setAttribute('target',   new THREE.BufferAttribute(pos[1].flat(), 3));
geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
const mat = new THREE.ShaderMaterial({ uniforms:{ t:{value:0}, size:{value:3}, stack:{value:0}, layerA:{value:0} },
  vertexShader:`attribute vec3 target; attribute vec3 color; varying vec3 vC; uniform float t,size,stack,layerA;
    void main(){ vec3 p = mix(position, target, smoothstep(0.,1.,t)); p.z += stack*(layerA+t)*4.;
      vec4 mv = modelViewMatrix*vec4(p,1.); gl_PointSize = size*300./-mv.z; gl_Position = projectionMatrix*mv; vC=color; }`,
  fragmentShader:`varying vec3 vC; void main(){ float d=length(gl_PointCoord-.5); if(d>.5) discard;
    gl_FragColor=vec4(vC, smoothstep(.5,.2,d)); }`, transparent:true, depthWrite:false, blending:THREE.AdditiveBlending });
// layer scrubber: on change, swap position/target attributes and tween uniforms.t 0→1 over 900 ms (easeInOutCubic)
```
- **Data:** the new `/api/projection_stack` (§3.1). Cache it server-side per `(n, mode)`.

### #2 Token trajectory ribbons and fly-to on search · Map · **M** · Wow ★★★★★
- **Shows:** for selected tokens (search hits, lasso set or the top-20), a 6-vertex polyline through their per-layer positions in stack mode. The line is coloured from layer 0 (amber) to layer 5 (cyan), with a small arrowhead at L5. Searching a token makes the camera **fly** to it (a 1.2 s tween of `controls.target` and `camera.position`), pulse its point and draw its ribbon.
- **Why it matters for PLE:**
  - It visualises how one token's *relative* position changes across layers.
  - Long, jagged ribbons mean "this token is encoded differently in each layer".
  - Frequent function words (`" the"`, `"."`) can be compared with content words.
  - Combined with the norm encoding (ribbon width ∝ slice norm), the reader sees *where the token's signal is large*.
- **Sketch** (three `LineSegments2` + `LineMaterial`):
```js
function ribbon(i){ const pts=[]; for(let l=0;l<L;l++){ const p=pos[l][i]; pts.push(p[0],p[1],l*GAP); }
  const g=new LineGeometry(); g.setPositions(pts); g.setColors(layerColors.flat());
  return new Line2(g, new LineMaterial({ linewidth:2.5, vertexColors:true, transparent:true, opacity:.9 })); }
function flyTo(v){ const from={...camera.position}, tgt=new THREE.Vector3(...v);
  animate(0,1,{duration:1.2, ease:'easeInOut', onUpdate:t=>{ controls.target.lerpVectors(controls.target,tgt,t*.2);
    camera.position.lerpVectors(from, tgt.clone().add(new THREE.Vector3(0,0,8)), t); }}); }
```
- **Data:** the same `pos[L][n][3]`. Search uses the existing `/api/token?q=` to resolve an id, then looks up its index.

### #3 Neighbour constellation, in 3D and as a 2D force graph · Map + Token · **S–M** · Wow ★★★★☆
- **Shows:** clicking a point draws faint lines to its k cosine neighbours, and each line's opacity encodes the similarity.
  - Toggle **"in this layer" vs "full row"**: per-layer neighbours are noisy sub-word fragments, while full-row neighbours are semantic (" dog" → puppy, cat, kitten…).
  - Neighbours missing from the projected set appear as ghost labels at the edge.
  - Token tab variant: a **2D force graph** of the token, its k = 15 neighbours and *their* top-3 neighbours, with edges weighted by sim. A layer selector re-runs the layout with a smooth transition.
- **Why it matters for PLE:** the findings say exactly this: "single slices are noisy, full row is cleanly semantic". Flipping between a layer and the full row and watching the constellation **reorganise** makes the claim visible.
- **Sketch** (force-graph UMD, or `d3.forceSimulation` on SVG):
```js
const nb = await api(`/api/neighbors?run=${r}&id=${id}&layer=${l}&k=15`);
const second = await Promise.all(nb.neighbors.slice(0,8).map(n=>api(`/api/neighbors?run=${r}&id=${n.id}&layer=${l}&k=3`)));
const nodes=[{id, piece:nb.piece, root:true}, ...uniq(nb.neighbors, second)], links=[...];
ForceGraph()(el).graphData({nodes,links}).backgroundColor('rgba(0,0,0,0)')
  .linkWidth(l=>1+4*l.sim).linkColor(l=>`rgba(34,211,238,${l.sim})`)
  .nodeCanvasObject((n,ctx)=>drawPill(ctx,n.piece,n.x,n.y,n.root))
  .onNodeClick(n=>openToken(n.id)).d3Force('link').distance(l=>120*(1-l.sim));
```
  - New endpoint (optional, saves N+1 round-trips): `/api/neighbor_graph?id&layer&k&k2` returns nodes and links.

### #4 Layer fingerprint radial glyph · Token · **M** · Wow ★★★★☆
- **Shows:** a polar glyph with 6 angular **petals** (one per layer) × 64 spokes (dims). Spoke length = |value| and spoke colour = sign, on a diverging map. Each petal's outer arc is scaled to its slice norm, and a dashed ring marks the init norm.
  - On token change, the glyph **morphs** spoke by spoke (stagger 4 ms per spoke).
  - Small multiples of the glyph in the neighbours table form a fingerprint lineup: similar tokens look similar.
- **Why it matters for PLE:**
  - It gives one memorable "face" per token, encoding the whole `[6 × 64]` row at a glance.
  - Norm vs init is immediately visible: rare tokens are a tidy circle at the dashed ring.
  - Lining glyphs up beside the neighbour list shows *which layer* drives the similarity.
- **Sketch** (d3 + SVG or canvas):
```js
const ang = d3.scaleLinear().domain([0, L*p]).range([0, 2*Math.PI]);
const r = d3.scaleLinear().domain([0, vmax]).range([R0, R1]);
svg.selectAll('line').data(t.vectors.flat()).join('line')
  .attr('x1',(v,i)=>R0*Math.sin(ang(i))).attr('y1',(v,i)=>-R0*Math.cos(ang(i)))
  .transition().duration(prefersReduced?0:500).delay((v,i)=>i*2)
  .attr('x2',(v,i)=>r(Math.abs(v))*Math.sin(ang(i))).attr('y2',(v,i)=>-r(Math.abs(v))*Math.cos(ang(i)))
  .attr('stroke', v=>divergingBO(v/vmax));
// petal separators + layer labels + dashed init ring at r(init_norm/sqrt(p))
```
  - **Data:** existing `/api/token` `vectors` (≤ 128 dims per layer; p = 64 is exact).

### #5 Gate river (streamgraph) · Analyze · **M** · Wow ★★★★★
- **Shows:** x = token position, stacked layers = gate magnitude per layer (L0 at the bottom). The token pieces are printed along the x axis, and hovering a token draws a vertical cursor line linked to the chip row. Toggle between **gate**, **contribution ratio** and **gate × ratio** ("effective PLE"). Use offset `wiggle` for the aesthetic streamgraph or `zero` for an honest stacked area.
- **Why it matters for PLE:**
  - The data is `gate[T][L]` and today we only show one layer at a time or the mean.
  - The river shows *which layer* reads the PLE slice at *which word*. For example, layer 1's band might swell on names (" Lily", " Max") while layer 0's swells on punctuation.
- **Sketch** (d3):
```js
const keys = d3.range(L);
const series = d3.stack().keys(keys).offset(mode==='wiggle'?d3.stackOffsetWiggle:d3.stackOffsetNone)
  .order(d3.stackOrderInsideOut)(a.gate.map(g=>Object.fromEntries(keys.map(k=>[k,g[k]]))));
const x=d3.scalePoint().domain(d3.range(T)).range([0,W]), y=d3.scaleLinear().domain(d3.extent(series.flat(2))).range([H,0]);
const area=d3.area().x((d,i)=>x(i)).y0(d=>y(d[0])).y1(d=>y(d[1])).curve(d3.curveCatmullRom);
svg.selectAll('path').data(series).join('path').attr('d',area).attr('fill',(s,l)=>layerColor(l));
svg.on('pointermove', e=>{ const i=nearest(e); cursor.attr('x',x(i)); highlightChip(i); });
```
  - **Data:** the existing `/api/analyze` `gate` and `contrib_ratio`.

### #6 Ablation before → after morph · Analyze · **S** · Wow ★★★★☆
- **Shows:** after an ablation run, chips first render coloured by normal loss, then **animate to Δloss**.
  - Each chip grows a little vertical bar under it (height = Δ) with a spring overshoot.
  - The summary counter **ticks up** from the base loss to the ablated mean loss.
  - A "replay" button and a hold-to-compare key (press **Space** to show the un-ablated state, release for ablated) let the reader flip between states.
- **Why it matters for PLE:** comparing two states is how the eye detects change. Animating the transition shows *which tokens* jump. The findings list layer 0 as most important, which can be checked live by toggling layer boxes.
- **Sketch:**
```js
chips.forEach((c,i)=>{ c.style.setProperty('--bg0', seq(a.loss[i]/mx)); c.style.setProperty('--bg1', divBO(a.ablation.delta[i]/dmx));
  c.style.setProperty('--h', `${Math.max(0,a.ablation.delta[i])/dmx*18}px`); c.style.transitionDelay=`${i*12}ms`; });
root.classList.add('ablated');   // CSS: .chip{background:var(--bg0);transition:background .5s var(--ease-out)} .ablated .chip{background:var(--bg1)}
countUp($('an-summary-num'), a.mean_loss, a.ablation.mean_loss, 800);
document.addEventListener('keydown', e=>e.code==='Space' && root.classList.remove('ablated'));
```
  - **Data:** the existing `/api/analyze` with `mask`. Tip: when ablating, run unablated and ablated in one call (already done, since `loss` and `ablation.loss` are both returned).

### #7 Ridgeline histograms of slice norms · Stats · **S** · Wow ★★★★☆
- **Shows:** six overlapping density ridges (L0 on top), x = slice norm, with a shared dashed vertical line at the init norm.
  - Optional split per ridge: **seen in train** (filled) vs **unseen** (outline). This needs a new endpoint field.
  - Hovering a ridge shows its mean and the share of tokens > 2× init.
- **Why it matters for PLE:** in the screenshot, today's six overlaid log-count histograms are hard to compare. Ridgelines make the **bimodality** obvious: a spike at init (about 27k never-seen tokens) plus a trained tail. They also show which layers' tails are longest.
- **Sketch** (d3):
```js
const y = d3.scaleBand().domain(d3.range(L)).range([0, H]).paddingInner(-1.2);  // overlap
const h = d3.scaleLinear().domain([0, d3.max(s.hist.counts.flat(), c=>Math.log1p(c))]).range([0, -y.bandwidth()*2.2]);
s.hist.counts.forEach((cnt,l)=>{ g.append('path').datum(cnt).attr('transform',`translate(0,${y(l)+y.bandwidth()})`)
  .attr('d', d3.area().x((c,i)=>x((s.hist.edges[i]+s.hist.edges[i+1])/2)).y0(0).y1(c=>h(Math.log1p(c))).curve(d3.curveBasis))
  .attr('fill', layerColor(l)).attr('fill-opacity',.55).attr('stroke', layerColor(l)); });
```
  - **Data:** the existing `stats.hist`. The seen/unseen split needs `hist.counts_seen[L][bins]`, which is easy in `analysis.stats()` with `counts>0`.

### #8 Compare slope chart and diverging bar · Compare · **S** · Wow ★★★☆☆
- **Shows:**
  - **Left:** a slope chart with two vertical axes (baseline loss, PLE loss). There is one line per token, coloured by sign of the diff on the diverging map, with opacity ∝ |diff|. Hovering a token highlights its line and chip.
  - **Right:** a sorted diverging bar of `diff`, with tokens on the y axis and zero in the centre, plus a running-mean line.
  - A toggle switches between sequence order and sorted order, with an animated reorder (d3 transition on y).
- **Why it matters for PLE:** "PLE is better by 0.003 nats on average" hides a wide spread. The slope chart shows that most tokens barely move while a few swing hard, and sorting surfaces *which* (e.g. names, rare words).
- **Sketch:**
```js
const y=d3.scaleLinear().domain([0,d3.max([...c.baseline.loss,...c.ple.loss])]).range([H,0]);
svg.selectAll('line.s').data(c.tokens.map((t,i)=>({t,b:c.baseline.loss[i],p:c.ple.loss[i],d:c.diff[i]})).filter(d=>d.b!=null))
  .join('line').attr('x1',0).attr('x2',W).attr('y1',d=>y(d.b)).attr('y2',d=>y(d.p))
  .attr('stroke',d=>divBO(d.d/dmax)).attr('stroke-opacity',d=>.15+.85*Math.abs(d.d)/dmax);
// sorted bar: y=d3.scaleBand().domain(order); transition().duration(600) on reorder
```
  - **Data:** the existing `/api/compare`.

### #9 Hexbin density for norm vs frequency · Stats · **S** · Wow ★★★☆☆
- **Shows:** replace the 3000-point scatter with a **hexbin over all ~50k tokens** (log x frequency, y full-row norm), coloured by log count in viridis.
  - Overlay the mean-per-bucket trend line and the init line.
  - Hovering a hex shows its count and three example tokens.
  - Brushing an x-range sends those tokens to the Map highlight (cross-filter).
- **Why it matters for PLE:** the ρ = 0.872 relationship and the dense "unseen tokens at init" stripe at x = 0 become a clear shape. A 3k sample under-shows both.
- **Sketch:**
```js
const hb = d3.hexbin().x(d=>x(Math.log10(1+d.f))).y(d=>y(d.n)).radius(6).extent([[0,0],[W,H]]);
const bins = hb(points); const c = d3.scaleSequentialLog(d3.interpolateViridis).domain([1, d3.max(bins,b=>b.length)]);
g.selectAll('path').data(bins).join('path').attr('d', hb.hexagon()).attr('transform',b=>`translate(${b.x},${b.y})`).attr('fill',b=>c(b.length));
```
  - **Data:** a new `stats` option `?full=1` that returns `norm[V]` and `freq[V]` as compact arrays (about 50k × 2 numbers, roughly 600 KB JSON). The server could also pre-bin: `hexbin{x,y,count,examples}`.

### #10 Sparkline chips · Analyze · **S** · Wow ★★★☆☆
- **Shows:** each chip gets a 6-bar micro-sparkline below it. The sparkline shows the gate per layer, or the Δloss per layer when per-layer ablation is available. Draw it as a single inline SVG or a CSS gradient. The chip background still encodes the selected metric.
- **Why it matters for PLE:** the per-layer profile of each token is visible without a hover. Thus, patterns such as "names light up L1" are easy to see in the full sentence.
- **Sketch:**
```js
const spark = g => `<svg class="spark" viewBox="0 0 ${L*3} 10">${g.map((v,l)=>`<rect x="${l*3}" y="${10-10*v/gmax}" width="2" height="${10*v/gmax}" fill="${layerColor(l)}"/>`).join('')}</svg>`;
chip.innerHTML = `${esc(vis(t.piece))}${spark(a.gate[i])}`;   // .chip{display:inline-flex;flex-direction:column} .spark{height:6px;width:100%}
```

### #11 Lasso selection drawer · Map (2D and 3D) · **M** · Wow ★★★★☆
- **Shows:** Shift+drag lasso (regl-scatterplot has it built in; in 3D, project the points to screen space and test them against the lasso polygon). A drawer slides up showing:
  - the selection count;
  - the token list sorted by frequency;
  - the mean slice-norm per layer for the selection vs all tokens (bars);
  - a "mean layer-similarity" matrix for the selection;
  - buttons for "Re-project selection" and "Trace ribbons".
- **Why it matters for PLE:** it turns the map from a picture into a query tool. "What are these 200 tokens in the corner?" becomes "numbers and digits, with high L0 norm".
- **Sketch (3D):**
```js
const sel=[]; const v=new THREE.Vector3();
for(let i=0;i<n;i++){ v.fromArray(posAttr.array,i*3).project(camera);
  const sx=(v.x+1)/2*W, sy=(1-v.y)/2*H; if(d3.polygonContains(lassoPts,[sx,sy])) sel.push(i); }
drawer.open(sel.map(i=>ids[i]));   // POST /api/selection_stats {ids} -> {mean_norm[L], sim[L][L]}
```
  - **Data:** a new `POST /api/selection_stats` (an easy reuse of `_layer_cos` and norms).

### #12 Command palette with token fly-to (for the shell agent) · Global · **S** · Wow ★★★★☆
- **Shows:** Ctrl+K opens a palette. Fuzzy actions:
  - "go to token ␣dog";
  - "map: layer 3";
  - "analyze: ablate L0";
  - "toggle bloom";
  - "start tour".
  Typing a token offers live matches (client-side fuzzy over the 5k projected pieces, or a new `/api/search?q=` prefix search over the vocab). Selecting one routes to the current tab's natural action: on the Map it flies to the token, on Token it opens it.
- **Why it matters for PLE:** tokens are the primary key of everything here, so a global token picker is the most-used interaction.
- **Sketch:** `fuse`-style scoring is unnecessary; `piece.toLowerCase().includes(q)` plus rank by frequency is enough at 50k entries.

### #13 Bloom/glow as encoding · Map 3D · **S** · Wow ★★★★★
- **Shows:** UnrealBloomPass with a moderate strength (0.6–0.9), threshold about 0.2 and radius 0.4. Point brightness is tied to a data channel, so **only "hot" tokens glow**. The default channel is slice norm relative to init. The selected token gets an additional pulsing halo sprite.
- **Why it matters for PLE:** glow becomes the "trained-ness" channel. Untouched tokens stay dim at the init norm, while heavily trained frequent tokens shine. This is the norm-vs-frequency finding rendered as light.
- **Sketch:**
```js
const composer=new EffectComposer(renderer); composer.addPass(new RenderPass(scene,camera));
const bloom=new UnrealBloomPass(new THREE.Vector2(W,H), .8, .4, .2); composer.addPass(bloom); composer.addPass(new OutputPass());
// colour = viridis(rank) * (0.35 + 0.65*clamp((norm/init-1)/3,0,1))   // brightness = trained-ness
```
  - Reduced motion or low GPU: `bloom.enabled=false`, and compensate with a point-size bump.

### #14 Colour-by modes: frequency, cluster, layer and "seen" · Map · **S** · Wow ★★★☆☆
- **Modes:**
  - **Frequency rank:** viridis, log scale.
  - **k-means cluster:** k = 12 on the full row, computed server-side, using a categorical palette (§6).
  - **"Seen in train":** binary, cyan vs grey.
  - **Norm / init:** cividis.
  - **Layer:** in stack mode.
- **Why it matters for PLE:** cluster colouring from the *full row* applied to a *single-layer* layout shows whether semantic groups hold together in that layer. This is a direct visual test of "no single layer is the semantic one".
- **Data:** `/api/projection_stack` includes `cluster[n]` (sklearn KMeans, cached).

### #15 Six-layer small-multiples strip · Map · **S** · Wow ★★★☆☆
- **Shows:** above the big map, six 120 px 2D thumbnails (regl or canvas) of the aligned projections, which act as the layer picker. The hovered token is highlighted in all six at once (linked brushing).
- **Why it matters for PLE:** this is the Distill/BertViz "model view" pattern. Comparing six layouts side by side is better than animating between them for spotting *where* a token jumps.

### #16 Animated heatmap: sort, morph and difference · Token · **S** · Wow ★★★☆☆
- **Shows:**
  - The L × p heatmap gains a **"sort dims by |value|"** toggle (columns animate to their new positions).
  - A **"diff vs neighbour"** mode: click a neighbour row and the heatmap tweens to `E[t] − E[nb]`.
  - A **cross-fade** when switching tokens (canvas: lerp the values over 300 ms).
  - Cell hover highlights the matching spoke in the radial glyph (#4).
- **Why it matters for PLE:** the "diff vs neighbour" mode answers *in which layer* two neighbours actually agree. Sorting reveals whether a slice is spiky (a few dims dominate) or diffuse.
- **Sketch:**
```js
function tween(from,to,ms){ const t0=performance.now(); (function f(now){ const k=ease(Math.min(1,(now-t0)/ms));
  draw(from.map((row,l)=>row.map((v,d)=>v+(to[l][d]-v)*k))); if(k<1) requestAnimationFrame(f); })(t0); }
```

### #17 Layer × position grid (attribution-graph style) · Analyze · **M** · Wow ★★★★☆
- **Shows:** a grid with x = tokens and y = layers 0–5 from bottom to top. Each cell is a circle with area = gate and colour = contribution ratio. When ablation is on, an outline ring shows that layer's Δ. Hovering a column lights up the whole token path; clicking pins it in the side panel.
- **Why it matters for PLE:** it copies the attribution-graph layout of Anthropic (prompt at the bottom, depth upward). Thus, it shows "where in the network PLE is injected". It replaces the single-layer slider.
- **Data:** the existing `gate[T][L]` and `contrib_ratio[T][L]`. A per-layer Δ needs **per-layer ablation in one call**, with a new `mask:"each"` option that returns `delta[L][T]` (about L+1 forward passes; cheap).

### #18 Layer-similarity matrix as an animated chord or arc diagram · Token · **S** · Wow ★★★☆☆
- **Shows:** keep the matrix, but add a toggle to an **arc diagram**. The diagram has six layer nodes on a line. The thickness of each arc = |cos|, and its colour = sign. Animate from the token's matrix to the global average and back.
- **Why it matters for PLE:** "≈ 0 everywhere" is a boring heatmap but a striking *empty* arc diagram. Any token with strong cross-layer arcs is immediately an outlier worth investigating.

### #19 Shareable view state (URL hash snapshots) · Global · **S** · Wow ★★☆☆☆
- Encode `tab, run, token, layer, camera pose, selection ids` in `location.hash`. Add "Copy link" and **bookmarks** that replay as a story (the Embedding Projector pattern). The 3D camera pose is 6 floats.

### #20 PLE equation as a live, clickable diagram · About · **M** · Wow ★★★★☆
- **Shows:** the About formula rendered as boxes:
  - `E[t]_l` (amber), `W_ctx x₀` (cyan), `+`, `/√2`, `gate` (violet), `⊙`, `W_proj`, `RMSNorm`, `+ h`.
  - Animated dashed flow lines run between the boxes.
  - Choosing a token and a layer fills each box with a mini heatmap or norm of the live values. This needs a new `/api/trace?run&text&pos&layer` that returns the intermediate vectors, all of which `traced_forward` can already expose.
- **Why it matters for PLE:** it turns the explanation into an instrument and teaches the mechanism (the Transformer Explainer pattern).

### #21 Quantisation error "lens" · Stats · **S** · Wow ★★☆☆☆
- **Shows:** per-layer int8 relative error as a bullet chart (target band 0.7–0.75 %). Clicking a layer shows a mini heatmap of `q·s − w` for the current token, so the tiny error becomes visible as structured noise.
- **Why it matters for PLE:** it supports the "can the table live in int8 / flash?" argument.

### #22 3D "PLE table slab" · Token · **L** · Wow ★★★★★
- **Shows:** an extruded cell grid (bbycroft style) of the `[6 × 64]` row of the token (InstancedMesh, height = value, diverging colour). It floats above a dim "table" plane that represents the 50k rows. The camera dollies in from the full table to the row to a slice. Hovering a cell shows `(layer, dim, value)`.
- **Why it matters for PLE:** it physically conveys "token-ID-indexed lookup of a row of 6 slices". The fact that the table is only *indexed* is the core idea of PLE.
- **Effort:** L because of camera choreography; it can reuse agent (a)'s three scene.

---

## 4. Interaction and motion grammar (all tabs)

Make it consistent so the app feels like one instrument:
- **Hover = preview, click = pin, Shift/Ctrl-click = add to selection, Esc = clear.** Use the same rule in the map, chips, tables and graphs.
- **Linked highlighting:** a global `bus.emit('focus-token', id)` that every visible view listens to. It highlights the token in the map, the neighbour table, the chips and the glyph lineup.
- **Detail drawer** on the right (Anthropic and Neuronpedia pattern) instead of tab jumps. "Open in Token tab" stays an explicit button.
- **State transitions animate marks rather than re-render:**
  - positions tween 400–900 ms;
  - colours cross-fade 250 ms;
  - number counters tick 600–800 ms.
- **Skeletons** while loading (§6). Never a blank card or the global "loading…" box alone.
- **Keyboard:** `1–6` = tabs, `[`/`]` = previous/next layer, `/` = focus search, `Ctrl+K` = palette, `Space` (Analyze) = hold to view un-ablated, `R` (Map) = reset camera.

---

## 5. Accessibility and performance

### Rendering 50k points
- **3D (three):** use **one `THREE.Points` draw call** with a custom shader, with all attributes in typed arrays. 50k points is easy (500k is also fine). Do not use one Mesh for each point.
  - Layer morph: tween a **uniform `t`** on the GPU (#1). Never rewrite 50k positions per frame in JS.
  - Picking: `Raycaster` with `params.Points.threshold` over 50k points is OK on hover with a throttle. For faster picking, use a GPU picking pass (render ids as colours to a 1×1 render target). Another option is a spatial grid in screen space that you build again when the camera stops.
  - Labels: CSS2D for ≤ 60 labels (the top-frequency ones in view plus the hovered and selected tokens). Recompute on `controls.change`, throttled to 10 Hz.
  - Bloom costs about 2–4 ms per frame at 1080p. Set `renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))` when bloom is on.
  - **Render on demand:** only call `composer.render()` when the controls change, a tween is active or a hover changes. This saves battery and lets the tab idle.
- **2D:** regl-scatterplot (WebGL) handles 50k points with lasso and transitions. Canvas 2D handles about 10k comfortably *if* drawn in batches: bucket points by colour, use one `beginPath` per bucket and `rect` instead of `arc`. The current `scatter()` calls `arc` plus `fill` per point and does an O(n) nearest search on every mousemove; use `d3.Delaunay.find` (O(log n)) instead.
- **Data transfer:** for n ≥ 10k, send binary. One option is `Float32Array` with `Response.arrayBuffer()` with a small JSON header. Another is quantising positions to `Int16` (×1/32767 scale), which halves the size again. Cache per run, layer and n with an `ETag` derived from the run's mtime.
- **Server:** PCA and Procrustes for 6 layers × 50k × 64 takes well under a second with numpy. Cache as `.npy` in `runs/.explorer_cache/` exactly as the frequency cache does.

### Reduced motion
```css
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{ animation-duration:.001ms!important; animation-iteration-count:1!important; transition-duration:.001ms!important; scroll-behavior:auto!important; }
}
```
```js
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
// JS tweens: duration = reduce ? 0 : d. 3D: disable auto-rotate, bloom pulse, animated background; fly-to becomes a cut.
```
Reference: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion. Also add an in-app "Motion: full / reduced / off" toggle, because some users want calm without changing OS settings.

### Colour
- **Sequential magnitudes** (norm, gate, loss, frequency): **viridis** stops `#440154 #472d7b #3b528b #2c728e #21918c #28ae80 #5ec962 #addc30 #fde725`.
  - Use **cividis** when the user enables "colour-blind safe" mode: `#00224e #2a3f6d #575d6d #7d7c78 #a59c74 #d2c060 #fee838`. It is designed for deuteranopia and protanopia.
  - On a near-black background, start viridis at t ≈ 0.15–0.25 (`#3b528b`). Otherwise the lowest values vanish into the background.
- **Diverging values** (vector values, Δloss, diff): replace today's blue↔red with **blue ↔ neutral ↔ orange** (`#3b82f6 → #2a2f3d → #f97316`). This pair survives red-green colour blindness better than blue/red-on-dark, and the mid stop matches the surface so zero "disappears".
  - Alternative: a ColorBrewer PuOr-style ramp `#5e3c99 #b2abd2 #2a2f3d #fdb863 #e66101`.
  - **Turbo** (`#3d358b #4682f8 #1fc9dd #3cf58e #a1fd3d #e7d739 #fe932a #e2430a #a11201`) is good for the 3D "wow" mode where reading ordered magnitude matters less. It is not perceptually uniform, so never make it the default for quantitative reading.
- **Categorical (6 layers):** a fixed, colour-blind-aware set based on Okabe-Ito but tuned to dark backgrounds:

  | L0 | L1 | L2 | L3 | L4 | L5 |
  |---|---|---|---|---|---|
  | `#f5a524` | `#4fc3f7` | `#2dd4a7` | `#f472b6` | `#a78bfa` | `#e8e36b` |

  Use the same layer colour everywhere: bars, river bands, ridges, ribbons and checkboxes.
- **Never encode with colour alone:**
  - add the value to tooltips;
  - use dashed lines for reference values;
  - mark positive/negative bars with a ▲/▼ glyph in tables;
  - keep the text contrast that is checked below.
- Contrast against `--bg #0a0d14`:
  - text `#e6e9f2`: 16.0:1;
  - muted `#8a93a8`: 6.3:1;
  - accent cyan `#22d3ee`: 10.8:1;
  - amber `#fbbf24`: 11.6:1;
  - violet `#a78bfa`: 7.1:1;
  - faint `#5b6478`: 3.3:1, so use it only for non-essential decoration and disabled states.

### Other accessibility notes
- **Charts:** give each chart card `role="img"` plus an `aria-label` summary (e.g. "Layer 0 norm 2.31, 2.9× init…"). The canvas also needs a visually hidden data table (`<details>` "Show data").
- **Keyboard:** chips need `tabindex=0`, with arrows to move and Enter to pin. The 3D map needs keyboard orbit (arrow keys rotate, +/− zoom) and a "list view" fallback.
- **Focus:** add a visible focus ring (`--focus` below) on every interactive element. The current CSS only outlines inputs.
- **3D fallback:** provide a 2D view when WebGL is unavailable (`renderer` creation throws). Show a toast and fall back to the 2D map.

---

## 6. Global styling: the "Instrument Panel" token set

### Comparison of directions
| Direction | Look | Pros | Cons for this app |
|---|---|---|---|
| Glassmorphism | translucent blurred panels over a gradient or 3D background | current ("2024") look, pairs with an animated background | `backdrop-filter: blur` over a live WebGL canvas costs 2–6 ms/frame; low contrast hurts heatmaps; panels look muddy over dense points |
| Neo-brutalism | thick black borders, flat saturated blocks, hard offset shadows | playful, highly legible | fights perceptual colour maps; feels like a toy rather than an instrument |
| **Terminal / sci-fi HUD (restrained)** ✅ | near-black, hairlines, corner ticks, mono numerals, one glow accent | matches "instrument reading a model"; maximum data contrast; glow budget goes to the data; cheap to render | risk of cliché if overdone, so keep the decoration to corner ticks and a grid background only |

**Recommendation:** the restrained HUD, with **one** glass element allowed: the floating 3D-map overlay controls, which may use `backdrop-filter: blur(8px)` because they are small.

### Tokens (CSS custom properties)
```css
:root{
  /* surfaces */
  --bg:#0a0d14;            /* app background (near-black blue) */
  --bg-grid:#111624;       /* 1px background grid / dot pattern */
  --surface:#10141e;       /* cards */
  --surface-2:#161b28;     /* inputs, table hover, chips base */
  --surface-3:#1d2333;     /* popovers, drawer */
  --hairline:#232a3b;      /* borders */
  --hairline-strong:#2f3850;
  /* text */
  --text:#e6e9f2; --text-muted:#8a93a8; --text-faint:#5b6478;
  /* accents */
  --accent:#22d3ee;        /* interaction: focus, selection, links, active tab (cyan) */
  --accent-dim:#0e7490;
  --ple:#f5a524;           /* PLE token-part / reference lines / init norm (amber) */
  --gate:#a78bfa;          /* gate / contribution (violet) */
  --ctx:#4fc3f7;           /* context term W_ctx x0 */
  --pos:#f97316; --neg:#3b82f6; --zero:#2a2f3d;   /* diverging */
  --ok:#34d399; --warn:#fbbf24; --err:#fb7185;
  --focus:0 0 0 2px var(--bg),0 0 0 4px var(--accent);
  /* layers L0..L5 */
  --l0:#f5a524; --l1:#4fc3f7; --l2:#2dd4a7; --l3:#f472b6; --l4:#a78bfa; --l5:#e8e36b;
  /* radii */
  --r-xs:3px; --r-sm:6px; --r-md:10px; --r-lg:14px; --r-pill:999px;
  /* shadows (dark UI: shadow = depth + faint inner highlight) */
  --sh-1:0 1px 0 rgba(255,255,255,.03) inset, 0 1px 2px rgba(0,0,0,.4);
  --sh-2:0 1px 0 rgba(255,255,255,.04) inset, 0 8px 24px rgba(0,0,0,.45);
  --sh-3:0 1px 0 rgba(255,255,255,.05) inset, 0 20px 48px rgba(0,0,0,.6);
  --glow-accent:0 0 0 1px rgba(34,211,238,.35), 0 0 18px rgba(34,211,238,.25);
  --glow-ple:0 0 0 1px rgba(245,165,36,.35), 0 0 18px rgba(245,165,36,.22);
  /* type */
  --font-ui:"Inter Variable","Inter","Segoe UI",system-ui,sans-serif;
  --font-mono:"JetBrains Mono Variable","JetBrains Mono","Cascadia Mono",Consolas,monospace;
  --fs-xs:11px; --fs-sm:12px; --fs-md:14px; --fs-lg:18px; --fs-xl:26px; --fs-hero:40px;
  --tracking-caps:.08em;    /* small-caps labels: uppercase 11px mono, letter-spacing */
  /* spacing (4px grid) */
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px;
  /* motion */
  --ease-out:cubic-bezier(.16,1,.3,1);       /* expo-out: entrances, morphs */
  --ease-in-out:cubic-bezier(.65,0,.35,1);   /* camera, layer morph */
  --ease-spring:cubic-bezier(.34,1.56,.64,1);/* small overshoot: chips, toggles */
  --d-instant:80ms; --d-fast:150ms; --d-base:250ms; --d-slow:450ms; --d-morph:900ms; --d-camera:1200ms;
}
```

### Typography
- **Inter Variable** for UI text and **JetBrains Mono Variable** for tokens, numbers, axis ticks and small-caps HUD labels. Both are OFL-1.1.
- Vendor the latin `woff2` files (48 KB and 40 KB):
  - `https://cdn.jsdelivr.net/npm/@fontsource-variable/inter@5.3.0/files/inter-latin-wght-normal.woff2`
  - `https://cdn.jsdelivr.net/npm/@fontsource-variable/jetbrains-mono@5.3.0/files/jetbrains-mono-latin-wght-normal.woff2`
- Optional display face for the brand only: **Space Grotesk Variable** (`@fontsource-variable/space-grotesk@5.3.0`).
- Numbers: `font-variant-numeric: tabular-nums slashed-zero;`.
- Section labels: `font: 500 11px/1 var(--font-mono); text-transform: uppercase; letter-spacing: var(--tracking-caps); color: var(--text-muted)`, e.g. `PER-LAYER NORM · ‖E[t]ₗ·√p‖`.

### Signature HUD details (use sparingly)
- **Corner ticks** on cards: draw 8-px L-shaped brackets at the 4 corners with two layered `linear-gradient` backgrounds in `--hairline-strong`. On `:hover` they become `--accent` (150 ms).
- **Background:** a 24-px dot grid (`radial-gradient(circle, var(--bg-grid) 1px, transparent 1px)`) plus one slow radial "aurora" in `rgba(34,211,238,.05)` drifting over 40 s. Under reduced motion it is static.
- **Card header strip:** an uppercase mono label on the left and a live readout on the right (e.g. `L0 · 2.31 · ×2.9 INIT`).
- **Active tab:** text in `--text`, a 2-px underline in `--accent` that **slides** between tabs (one absolutely positioned bar, with `transform` transitions over `--d-base` using `--ease-out`).

### Micro-interactions
- **Buttons:**
  - primary is solid `--accent` with `#041016` text;
  - secondary is ghost with a hairline border;
  - `:active` applies `transform: translateY(1px)`;
  - hover adds `--glow-accent` (150 ms).
- **Chips:** hover lifts them 1 px with a slight scale of 1.04 using `--ease-spring`, and adds an outline in `--accent`. A pinned chip gets a persistent glow.
- **Numbers:** count up on change (600 ms), and the changed digit flashes `--accent` for 300 ms.
- **Tooltips:** fade plus 4-px slide from the cursor (`--d-fast`). Use `--surface-3` with a hairline border, a mono header row (token and `#id`) and key/value rows below.
- **Sliders:** the layer slider becomes a **segmented L0…L5 control** with layer-colour dots, plus arrow-key support.
- **Copy feedback:** clicking a token piece copies it to the clipboard and shows a toast "copied ␣dog".

### Loading skeleton style
```css
.skeleton{ position:relative; overflow:hidden; background:var(--surface-2); border-radius:var(--r-sm); }
.skeleton::after{ content:""; position:absolute; inset:0; transform:translateX(-100%);
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.06),transparent);
  animation:shimmer 1.4s var(--ease-in-out) infinite; }
@keyframes shimmer{ to{ transform:translateX(100%); } }
/* chart skeletons mimic their final shape: 6 bars for norms, a 6×N grid for heatmaps, a dotted cloud for maps */
@media (prefers-reduced-motion: reduce){ .skeleton::after{ animation:none; } }
```
HUD flavour: while the 3D map loads, show a **scanline progress** with mono text `PROJECTING 5 000 TOKENS · L0–L5 · PCA+PROCRUSTES` and a thin cyan bar.

---

## 7. Ideas for the parallel agents

**For (a), the 3D map agent:**
- **Do not morph raw per-layer PCA coordinates.** Align them (Procrustes) or use shared axes (§3.1). Otherwise the morph is noise from sign flips and rotations.
- One `Points` draw call with a custom shader, tweening a uniform `t` on the GPU, rendered on demand.
- Bloom brightness ∝ norm / init (#13): it makes "trained-ness" glow.
- Add stack mode (Z = layer) and trajectory ribbons (#2).
- k-NN constellation lines from `/api/neighbors` with a layer / full-row toggle (#3).
- Lasso by screen-space projection plus `d3.polygonContains` (#11).
- CSS2D labels for ≤ 60 tokens.
- Use the layer colours `--l0…--l5` for ribbons and HUD.
- Ship a 2D fallback, and respect reduced motion (no auto-rotate; fly-to becomes a cut).

**For (b), the shell agent:**
- Use the §6 token set.
- Command palette with token search (#12).
- Sliding tab underline.
- Dot-grid plus slow aurora background (static under reduced motion).
- Detail drawer and global `focus-token` bus (§4).
- Skeletons.
- URL-hash state (#19).
- Avoid full-page `backdrop-filter` over the WebGL canvas.

**For (c), the charts agent:** use the d3 UMD as the single 2D dependency. The priority order is: #5 gate river, #6 ablation morph, #4 radial glyph, #7 ridgelines, #8 slope chart, #9 hexbin, #10 sparkline chips, #16 heatmap modes, #17 layer×position grid and #18 arc diagram.

### New endpoints suggested (all optional, all cheap)
| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/projection_stack?run&n&mode=aligned\|shared&dims=3` | `ids, pieces, freq, norm[n][L], pos[L][n][3], explained[L][3], cluster[n]` (binary option) | #1 #2 #13 #14 #15 |
| `GET /api/neighbor_graph?run&id&layer&k&k2` | nodes and links | #3 |
| `POST /api/selection_stats {run, ids}` | `mean_norm[L], sim[L][L], top pieces` | #11 |
| `POST /api/analyze` with `mask:"each"` | `delta_by_layer[L][T]` | #10 #17 |
| `GET /api/stats?full=1` | `norm[V], freq[V]` or a pre-binned hexbin, plus `hist.counts_seen` | #7 #9 |
| `GET /api/search?run&q&limit` | vocab prefix/fuzzy matches by frequency | #12 |
| `GET /api/trace?run&text&pos&layer` | intermediate vectors for the PLE equation | #20 |

---

## 8. Sources
- TensorFlow Embedding Projector, https://projector.tensorflow.org/, and Smilkov et al., arXiv:1611.05469.
- Neuronpedia, https://www.neuronpedia.org/, with source at https://github.com/hijohnnylin/neuronpedia (Apache-2.0).
- Anthropic, *Scaling Monosemanticity*, https://transformer-circuits.pub/2024/scaling-monosemanticity/.
- Anthropic, *Circuit Tracing* (interface: hover/click, Ctrl+Click subgraphs, supernodes, pruning), https://transformer-circuits.pub/2025/attribution-graphs/methods.html.
- Anthropic, *On the Biology of a Large Language Model*, https://transformer-circuits.pub/2025/attribution-graphs/biology.html.
- Transformer Explainer, https://poloclub.github.io/transformer-explainer/, and arXiv:2408.04619.
- LLM Visualization, https://bbycroft.net/llm, with source at https://github.com/bbycroft/llm-viz (MIT).
- Nomic Atlas, https://atlas.nomic.ai/; deepscatter, https://github.com/nomic-ai/deepscatter.
- Apple Embedding Atlas (KDE contours, order-independent transparency, cross-filtering), https://apple.github.io/embedding-atlas/ and https://github.com/apple/embedding-atlas.
- Distill: https://distill.pub/2016/misread-tsne/, https://distill.pub/2019/activation-atlas/ and https://distill.pub/2020/circuits/zoom-in/.
- BertViz (head, model and neuron views), https://github.com/jessevig/bertviz.
- PAIR Understanding UMAP, https://pair-code.github.io/understanding-umap/; R2D3, http://www.r2d3.us/; Illustrated Transformer, https://jalammar.github.io/illustrated-transformer/.
- regl-scatterplot README (20M points, lasso, transitions, colorBy/opacityBy/sizeBy), https://github.com/flekschas/regl-scatterplot.
- three.js bloom example, https://threejs.org/examples/#webgl_postprocessing_unreal_bloom.
- 3d-force-graph, https://github.com/vasturiano/3d-force-graph.
- GSAP Standard License, https://gsap.com/standard-license.
- MDN `prefers-reduced-motion`, https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion.
- Package versions and licences: `npm view` on 2026-11-12. File sizes: measured jsDelivr/esm.sh downloads on 2026-11-12. Colour-map hex values: sampled from matplotlib's viridis, cividis and turbo.
