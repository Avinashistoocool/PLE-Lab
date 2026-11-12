# `static/viz/`: interactive charts

These files are standalone ES modules with no build step. They use d3 v7.9.0. The d3 copy is in `../vendor/d3/` (ISC licence), and the ESM shim `vendor/d3/d3.js` loads it.

```js
import * as viz from './viz/index.js';
```

## Conventions

* Each chart has the form `render(el, data, opts)` and returns `{update(data), destroy(), ...}`. The charts keep no global state. They share only one tooltip element and `viz.css`. The code adds `viz.css` to the page automatically.
* The colours come from the CSS custom properties in DESIGN_IDEAS §6: `--accent`, `--ple`, `--pos`, `--neg`, `--zero`, `--l0`…`--l5`, `--surface*`, `--hairline*` and `--text*`. Each property has a fallback value. Magnitudes use viridis. Signed values use blue ↔ orange.
* Motion lasts 300–1100 ms with easing. The charts do not show animation when `prefers-reduced-motion: reduce` is set, or when `<html data-motion="reduced|off">` is present.
* Charts with many marks (glyph, heatmap, hexbin) draw on canvas. The other charts use SVG. All charts change their size with their container.

## Charts

| export | file | data (endpoint) | extra opts | extra methods |
|---|---|---|---|---|
| `glyph(el, token, opts)` | glyph.js | `GET /api/token` | `size`, `vmax`, `onHover({layer,dim,value}\|null)`, `onClick` | `highlight(layer, dim)` |
| `heatmap2(el, token, opts)` | heatmap2.js | `GET /api/token` | `cellH`, `vmax`, `controls`, `onHover`, `onClick` | `highlight(layer, dim)`, `setSort(bool)` |
| `simMatrix(el, sim, opts)` | heatmap2.js | `GET /api/layer_similarity?id=` | `size`, `mode: token\|global\|diff`, `onHover` | `setMode()` |
| `forceGraph(el, graph, opts)` | forcegraph.js | **new** `GET /api/neighbor_graph?run&id&layer(-1\|0..L-1\|all)&k&k2` | `height`, `onNodeClick(node)`, `onHover` | `focus(id)` |
| `river(el, analysis, opts)` | river.js | `POST /api/analyze` | `height`, `mode: gate\|ratio\|effective`, `offset: wiggle\|zero`, `onHover(i)`, `onClick(i)` | `highlight(i)`, `setMode`, `setOffset` |
| `ablation(el, analysis, opts)` | ablation.js | `POST /api/analyze` with `mask` | `height`, `autoplay`, `onHover(i)`, `onClick(i)` | `highlight(i)`, `play()`, `show('before'\|'after')` |
| `sparkChips(el, analysis, opts)` | sparkchips.js | `POST /api/analyze` | `color: loss\|delta\|gate\|none`, `layer`, `spark: gate\|ratio`, `selected`, `onHover(i)`, `onClick(i)` | `highlight(i)`, `select(i)`, `setColor(mode, layer)` |
| `sparkSvg(values, vmax)` | sparkchips.js | n/a | n/a | returns an SVG string for your own chips |
| `ridgeline(el, dist, opts)` | ridgeline.js | **new** `GET /api/norm_dist?run[&bins=48]` | `height`, `scale: log\|sqrt\|linear`, `overlap`, `onHover` | `setScale` |
| `hexbin(el, dist, opts)` | hexbin.js | **new** `GET /api/norm_dist?run[&gx=36&aspect=0.5]` | `height`, `onClick(bin)` (each bin has `examples[{id,piece,freq,norm}]`), `onHover` | n/a |
| `slope(el, cmp, opts)` | slope.js | `POST /api/compare` | `height`, `order: seq\|sorted`, `onHover(i)`, `onClick(i)` | `highlight(i)`, `setOrder` |

### New endpoints

The code is in `explorer/analysis.py` and the tests are in `tests/test_explorer.py`.

* `GET /api/norm_dist?run&bins&gx&aspect` returns the items that follow. The server keeps them in a cache for each run.
  * `edges[B+1]`;
  * `layers[{layer, seen[B], unseen[B], mean, mean_seen, median, p99, frac_gt_2x_init}]`, where "seen" means a train.bin count > 0;
  * `init_norm`, `init_full_norm`, `n_seen` and `n_unseen`;
  * `hexbin{radius, aspect, x_domain, y_domain, bins[{x, y, n, mean_norm, mean_logf, examples}]}`. This item shows all of the about 50k tokens, with x = log10(1+count) and y = the norm of the full row. The coordinates are normalised to x ∈ [0,1] and y ∈ [0,aspect], with the d3-hexbin layout;
  * `trend`.
* `GET /api/neighbor_graph?run&id&layer&k=12&k2=2` returns:
  * `nodes[{id, piece, root, layers[], sim, shared}]`;
  * `links[{source, target, sim, layer, kind: root|peer}]`.

  `layer=all` takes the union of the top-k of each layer. Peer links connect each node to the top-k2 other nodes that are most similar to it, by the cosine of the full row.

## Suggested wiring in `app.js`

Create each chart one time. Then call `.update()` when you load new data. `api` and `q` are the existing helpers in app.js.

```js
import * as viz from './viz/index.js';
const V = {};

// Token tab (loadToken): add containers <div id="token-glyph">, <div id="token-graph">
V.glyph ? V.glyph.update(t) : (V.glyph = viz.glyph($('token-glyph'), t, { onHover: h => V.heat?.highlight(h?.layer ?? null, h?.dim) }));
V.heat  ? V.heat.update(t)  : (V.heat  = viz.heatmap2($('token-heat'), t, { onHover: h => V.glyph?.highlight(h?.layer ?? null, h?.dim) }));
V.sim   ? V.sim.update(sim) : (V.sim   = viz.simMatrix($('token-sim'), sim));   // replaces token-sim + global-sim heatmaps (toggle inside)
const g = await api(`/api/neighbor_graph?${q({ id: t.id, layer: $('nb-layer').value, k: 12 })}`);
V.graph ? V.graph.update(g) : (V.graph = viz.forceGraph($('token-graph'), g, { onNodeClick: n => openToken(n.id) }));

// Analyze tab (drawAnalysis): add <div id="an-river">, <div id="an-abl">; replace the an-chips innerHTML
const link = i => { V.chips?.highlight(i); V.river?.highlight(i); V.abl?.highlight(i); };
V.chips?.destroy(); V.chips = viz.sparkChips($('an-chips'), a, { color: $('an-color').value, layer: +$('an-layer').value,
  onHover: i => { V.river?.highlight(i); V.abl?.highlight(i); }, onClick: i => drawSelected(i) });
V.river ? V.river.update(a) : (V.river = viz.river($('an-river'), a, { onHover: i => { V.chips.highlight(i); V.abl?.highlight(i); }, onClick: i => drawSelected(i) }));
if (a.ablation) { V.abl?.destroy(); V.abl = viz.ablation($('an-abl'), a, { onHover: i => { V.chips.highlight(i); V.river.highlight(i); } }); }

// Stats tab (loadStats): stats-hist → ridgeline, stats-freq → hexbin
const nd = await api(`/api/norm_dist?${q({})}`);
V.ridge?.destroy(); V.ridge = viz.ridgeline($('stats-hist'), nd);
V.hex?.destroy();   V.hex   = viz.hexbin($('stats-freq'), nd, { onClick: b => openToken(b.examples[0].id) });

// Compare tab (runCompare): cmp-bars → slope; chips stay and link both ways
V.slope?.destroy(); V.slope = viz.slope($('cmp-bars'), c, { onHover: i => {/* highlight cmp chip i */} });
```

Call `destroy()` on every chart when the run changes.

## Demo and test harness

* Open `/static/viz-gallery.html`. This page shows all charts with live data for `runs/ple-small`.
* `python explorer/viz_gallery_check.py --url http://127.0.0.1:8773` uses Playwright with Edge. The script does these steps:
  1. It moves the pointer over the charts.
  2. It drags a node of the force graph.
  3. It looks for console errors and HTTP errors.
  4. It writes `explorer/screenshots/viz-*.png`.
