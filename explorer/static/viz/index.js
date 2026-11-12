// PLE Explorer chart modules. Every chart: render(el, data, opts) -> {update(data), destroy(), ...}.
// See ./README.md for data shapes, feeding endpoints and suggested wiring in app.js.
export { render as glyph } from './glyph.js';
export { render as heatmap2, similarity as simMatrix } from './heatmap2.js';
export { render as forceGraph } from './forcegraph.js';
export { render as river } from './river.js';
export { render as ablation } from './ablation.js';
export { render as sparkChips, sparkSvg } from './sparkchips.js';
export { render as ridgeline } from './ridgeline.js';
export { render as hexbin } from './hexbin.js';
export { render as slope } from './slope.js';
export { theme, viridis, diverging, layerColor, reducedMotion, tip, tipHtml, vis, esc, fmt } from './core.js';
