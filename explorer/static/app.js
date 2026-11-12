/*
 * PLE Explorer app shell.
 *
 * EXTENSION HOOKS (for viz modules such as charts.js additions or static/viz/*.js):
 *   Each tab has ONE render function that draws all of its charts from already-fetched data:
 *     renderTokenTab(t, sim)   Token tab   containers: #token-norms #token-sim #token-heat #token-glyph #token-graph #nb-table
 *     renderMap()              Map tab     containers: #map-plot (2D canvas), #map3d (3D, map3d.js)
 *     renderAnalyze()          Analyze tab containers: #an-chips #an-legend #an-river #an-abl #an-contrib #an-gate-mean
 *                                                      #an-sel-gate #an-sel-top5 #an-ablation-summary
 *     renderStats(s)           Stats tab   containers: #stats-hist #stats-freq #stats-quant
 *     renderCompare(c)         Compare tab containers: #cmp-chips #cmp-legend #cmp-bars
 *   Data loaders (loadToken, loadNeighbors, loadMap, runAnalyze, loadStats, runCompare) fetch and then
 *   call the render function. To add a chart: add a container with a stable id in index.html inside the
 *   matching tab, and draw into it at the end of the render function. Shared state lives in `state`
 *   (state.run, state.tokenId, state.analysis, state.selected, state.stats, state.compare).
 *   `bus` is a tiny event bus: bus.on('focus-token', id => ...), bus.on('theme', t => ...),
 *   bus.on('run', run => ...), bus.on('tab', tab => ...).  Re-rendering on theme change is automatic.
 *   Interactive charts from ./viz (see viz/README.md) are kept in `V`: created once, update()d on new data,
 *   destroy()ed on run change and theme change (killViz) so they re-read the CSS custom properties.
 */
import { barChart, heatmap, scatter, tooltip, esc, fmt, seq, div } from './charts.js';
import { toast, clearToasts, countUp, initPopovers, segmented, syncSegments, skeleton, tokHTML, visTok, inkFor,
  toggleTheme, currentTheme, onTheme, reducedMotion } from './shell/ui.js';
import { INFO } from './shell/info.js';
import { createPalette } from './shell/palette.js';
import { initAbout } from './shell/about.js';
import * as viz from './viz/index.js';

const $ = id => document.getElementById(id);
const TABS = ['token', 'map', 'analyze', 'stats', 'compare', 'about'];
const SAMPLE_TOKENS = [' the', ' happy', ' dog', ' ran'];
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const state = {
  runs: [], run: null, tab: 'token', tokenId: null, analysis: null, selected: null, stats: null, compare: null,
  dirty: new Set(['token', 'map', 'stats']),
};

export const bus = (() => {
  const m = new Map();
  return {
    on: (ev, fn) => { if (!m.has(ev)) m.set(ev, new Set()); m.get(ev).add(fn); },
    emit: (ev, arg) => m.get(ev)?.forEach(fn => { try { fn(arg); } catch (e) { console.warn(e); } }),
  };
})();

// ---- interactive viz handles (static/viz) ---------------------------------------------------
const V = {};
const VIZ_TAB = { glyph: 'token', heat: 'token', sim: 'token', graph: 'token', chips: 'analyze', river: 'analyze', abl: 'analyze',
  ridge: 'stats', hex: 'stats', slope: 'compare' };
const stale = new Set();
function killViz(...keys) {
  for (const k of keys.length ? keys : Object.keys(V)) {
    try { V[k]?.destroy?.(); } catch (e) { console.warn(e); }
    delete V[k];
    if (!keys.length && VIZ_TAB[k]) stale.add(VIZ_TAB[k]);
  }
}
window.pleExplorer = { state, bus, viz: V };

// ---- API + feedback ------------------------------------------------------------------
let pending = 0;
function busy(on) {
  const p = $('progress');
  if (on) { pending++; $('loading').classList.remove('hidden'); p.classList.remove('done'); p.classList.add('on'); return; }
  if (--pending > 0) return;
  $('loading').classList.add('hidden');
  p.classList.add('done');
  setTimeout(() => { if (!pending) p.classList.remove('on', 'done'); }, 250);
}
async function api(path, body) {
  busy(true);
  try {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    return data;
  } finally { busy(false); }
}
const q = params => new URLSearchParams({ run: state.run?.name ?? '', ...params }).toString();

/** Same semantics as the old banner: banner(msg) = error, banner(msg,'warn') = warning, banner(null) clears. */
function banner(msg, kind = 'error') {
  if (!msg) { clearToasts('error'); clearToasts('warn'); return; }
  toast(msg, kind === 'warn' ? 'warn' : 'error');
}
async function guarded(fn) {
  try { await fn(); } catch (e) { banner(e.message); }
}

const L = () => state.run?.config.n_layer ?? 0;
const hasPle = () => !!state.run?.has_ple;
const layerDot = l => `<span class="ldot" style="background:var(--l${l % 6})"></span>`;

function fillLayerSelect(sel, withFull, current) {
  const opts = [];
  if (withFull) opts.push(`<option value="-1" title="all layers (full row)">full</option>`);
  for (let l = 0; l < L(); l++) opts.push(`<option value="${l}" title="layer ${l}">L${l}</option>`);
  sel.innerHTML = opts.join('');
  if (current != null && [...sel.options].some(o => o.value === String(current))) sel.value = String(current);
}
const segFmt = o => (o.value === '-1' ? 'full' : /^\d+$/.test(o.value) ? `${layerDot(+o.value)}L${o.value}` : esc(o.text));

function emptyState(el, title, text) {
  const btns = SAMPLE_TOKENS.map(s => `<button type="button" class="sample tok" data-token="${esc(s)}">${tokHTML(s)}</button>`).join('');
  el.innerHTML = `<div class="empty"><h4>${esc(title)}</h4><div>${esc(text)}</div><div class="samples">${btns}</div></div>`;
}

// ---- runs ------------------------------------------------------------------------------
async function loadRuns() {
  const data = await api('/api/runs');
  state.runs = data.runs.filter(r => !r.error);
  const sel = $('run');
  sel.innerHTML = state.runs.map(r => `<option value="${esc(r.name)}">${esc(r.name)}${r.has_ple ? '' : ' (baseline)'}</option>`).join('');
  if (!state.runs.length) { banner('No runs found. Create one with: python -m explorer.make_demo_run'); return; }
  const route = parseHash();
  let saved = null; try { saved = localStorage.getItem('ple.run'); } catch { /* ignore */ }
  const pick = state.runs.find(r => r.name === route.run) || state.runs.find(r => r.name === saved)
    || state.runs.find(r => r.name === 'ple-small') || state.runs.find(r => r.has_ple) || state.runs[0];
  sel.value = pick.name;
  const bases = state.runs.filter(r => !r.has_ple), ples = state.runs.filter(r => r.has_ple);
  $('cmp-base').innerHTML = bases.map(r => `<option>${esc(r.name)}</option>`).join('');
  $('cmp-ple').innerHTML = ples.map(r => `<option>${esc(r.name)}</option>`).join('');
  if (ples.some(r => r.name === 'ple-small')) $('cmp-ple').value = 'ple-small';
  selectRun(pick.name, { route: true });
}

function renderKpis() {
  const r = state.run, c = r.config, p = r.params;
  const M = n => n / 1e6;
  countUp($('kpi-table'), M(p.ple_table), { decimals: 1, suffix: 'M' });
  countUp($('kpi-L'), c.n_layer); countUp($('kpi-p'), c.ple_dim);
  countUp($('kpi-d'), c.d_model);
  countUp($('kpi-dense'), M(p.dense), { decimals: 1, suffix: 'M' });
  const ts = r.extra?.tokens_seen;
  countUp($('kpi-tokens'), ts != null ? ts / 1e6 : null, { decimals: 1, suffix: 'M' });
  countUp($('kpi-loss'), r.val_loss != null ? +r.val_loss : null, { decimals: 3 });
  $('run-info').textContent = `L${c.n_layer} · d${c.d_model} · p${c.ple_dim}`;
  document.querySelector('.run-pill').classList.toggle('baseline', !r.has_ple);
}

function selectRun(name, { route = false } = {}) {
  state.run = state.runs.find(r => r.name === name);
  if (!state.run) return;
  try { localStorage.setItem('ple.run', name); } catch { /* ignore */ }
  $('run').value = name;
  renderKpis();
  fillLayerSelect($('nb-layer'), true, $('nb-layer').value || -1);
  fillLayerSelect($('map-layer'), true, $('map-layer').value || 0);
  fillLayerSelect($('an-layer'), false, +$('an-layer').value < L() ? $('an-layer').value : 0);
  $('an-layer-val').textContent = $('an-layer').value;
  $('an-layers').innerHTML = Array.from({ length: L() }, (_, l) =>
    `<label title="ablate layer ${l}"><input type="checkbox" data-layer="${l}">${layerDot(l)}${l}</label>`).join('');
  state.analysis = null; state.graph = null; state.normDist = null;
  killViz();
  state.dirty = new Set(['token', 'map', 'stats', 'analyze']);
  about.refresh();
  if (map3d.handle && hasPle()) map3d.handle.setRun(name).catch(e => toast(e.message, 'error'));
  bus.emit('run', state.run);
  syncSegments();
  if (route) applyRoute(); else { showTab(state.tab, { push: false }); writeHash(true); }
}

// ---- routing -----------------------------------------------------------------------------
// #<tab>[/<arg>][?run=<name>]   e.g.  #token/%20dog?run=ple-small   #map   #analyze
function parseHash() {
  const h = location.hash.slice(1);
  const [path, query = ''] = h.split('?');
  const params = new URLSearchParams(query);
  if (path.startsWith('run=')) return { tab: null, arg: null, run: decodeURIComponent(path.slice(4)) };
  const i = path.indexOf('/');
  const tab = (i < 0 ? path : path.slice(0, i)) || null;
  const arg = i < 0 ? null : decodeURIComponent(path.slice(i + 1));
  return { tab: TABS.includes(tab) ? tab : null, arg, run: params.get('run') };
}
function hashFor(tab) {
  let h = `#${tab}`;
  if (tab === 'token' && state.tokenQuery) h += `/${encodeURIComponent(state.tokenQuery)}`;
  if (state.run) h += `?run=${encodeURIComponent(state.run.name)}`;
  return h;
}
let routing = false;
function writeHash(replace = false) {
  if (routing) return;
  const h = hashFor(state.tab);
  if (h === location.hash) return;
  if (replace) history.replaceState(null, '', h); else history.pushState(null, '', h);
}
function applyRoute() {
  const r = parseHash();
  if (r.run && state.run && r.run !== state.run.name && state.runs.some(x => x.name === r.run)) {
    selectRun(r.run, { route: true }); return;
  }
  routing = true;
  try {
    const tab = r.tab || state.tab;
    if (tab === 'token' && r.arg != null && r.arg !== state.tokenQuery) {
      state.tokenQuery = r.arg; state.tokenId = null; $('token-q').value = r.arg; state.dirty.add('token');
    }
    showTab(tab);
  } finally { routing = false; }
  writeHash(true);
}
window.addEventListener('popstate', () => { if (state.runs.length) applyRoute(); });

// ---- tabs --------------------------------------------------------------------------------
function moveInk() {
  const b = document.querySelector(`#tabs button[data-tab="${state.tab}"]`), nav = $('tabs');
  if (!b) return;
  nav.style.setProperty('--ink-x', `${b.offsetLeft}px`);
  nav.style.setProperty('--ink-w', `${b.offsetWidth}px`);
  nav.querySelector('.tab-ink').style.width = `${b.offsetWidth}px`;
}

function showTab(tab, { push = true } = {}) {
  const prev = state.tab;
  state.tab = tab;
  document.querySelectorAll('#tabs button').forEach(b => {
    const on = b.dataset.tab === tab;
    b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll('.tab').forEach(s => s.classList.toggle('active', s.id === `tab-${tab}`));
  moveInk();
  tooltip.hide();
  popovers?.close();
  if (prev === 'map' && tab !== 'map') map3d.leave();
  if (push) writeHash();
  bus.emit('tab', tab);
  if (tab === 'about') { about.refresh(); return; }
  if (['token', 'map', 'stats', 'analyze'].includes(tab) && state.run && !hasPle()) {
    banner(`'${state.run.name}' is a baseline run with no PLE table; pick a PLE run for this view.`, 'warn');
    if (tab === 'map') map3d.leave();
    return;
  }
  if (tab === 'map') map3d.enter();
  if (tab === 'compare' && !state.compare) guarded(runCompare);
  if (stale.has(tab)) { stale.delete(tab); if (!state.dirty.has(tab)) rerender(); }
  if (!state.dirty.has(tab)) return;
  state.dirty.delete(tab);
  if (tab === 'token') guarded(() => loadToken(state.tokenId != null ? `#${state.tokenId}` : $('token-q').value));
  if (tab === 'map') guarded(loadMap);
  if (tab === 'stats') guarded(loadStats);
  if (tab === 'analyze') guarded(runAnalyze);
}

// ---- Token tab ---------------------------------------------------------------------------
async function loadToken(query) {
  if (!query || !query.length) { emptyState($('token-head'), 'Look up a token', 'Type a token (note the leading space for words) or #id, or try one of these:'); return; }
  skeleton($('token-norms'), 190); skeleton($('nb-table'), 0, 'table');
  if (!V.heat) skeleton($('token-heat'), 160);
  if (!V.sim) skeleton($('token-sim'), 260);
  if (!V.glyph) skeleton($('token-glyph'), 300);
  if (!V.graph) skeleton($('token-graph'), 300);
  let t;
  try { t = await api(`/api/token?${q({ q: query })}`); } catch (e) {
    emptyState($('token-head'), 'Nothing to show', `${e.message} Try one of these:`);
    killViz('glyph', 'heat', 'sim', 'graph');
    for (const id of ['token-norms', 'token-heat', 'token-sim', 'token-glyph', 'token-graph', 'nb-table']) $(id).innerHTML = '';
    throw e;
  }
  state.tokenId = t.id;
  state.tokenQuery = query;
  $('token-q').value = query;
  writeHash(true);
  const sim = await api(`/api/layer_similarity?${q({ id: t.id })}`);
  renderTokenTab(t, sim);
  bus.emit('focus-token', t.id);
  await loadNeighbors();
}

function renderTokenTab(t, sim) {
  state.token = t; state.sim = sim;
  $('token-pieces').innerHTML = t.pieces.length > 1
    ? `query is ${t.pieces.length} tokens: ${t.pieces.map(p => `<span class="tok">${tokHTML(p.piece)}</span>`).join(' · ')} (showing the first)` : '';
  const ratio = t.norms.reduce((a, b) => a + b, 0) / t.norms.length / t.init_norm;
  $('token-head').innerHTML = `<span class="big tok" title="click to copy" tabindex="0" role="button" aria-label="Copy token">${tokHTML(t.piece)}</span>
    <span class="meta"><span class="badge">id <b>${t.id}</b></span>
    ${t.frequency != null ? `<span class="badge"><b>${t.frequency.toLocaleString()}</b> in train.bin</span>` : ''}
    <span class="badge">L <b>${t.n_layer}</b> × p <b>${t.ple_dim}</b></span>
    <span class="badge">mean norm <b>${ratio.toFixed(2)}×</b> init</span>
    ${t.downsampled ? '<span class="badge">heatmap pooled</span>' : ''}</span>`;
  const labels = t.norms.map((_, l) => `L${l}`);
  barChart($('token-norms'), t.norms, {
    labels, height: 190, refLine: { value: t.init_norm, label: 'init' }, color: (_, i) => cssv(`--l${i % 6}`) || '#22d3ee',
    yLabel: 'norm', tip: i => `layer ${i}: ‖slice‖ = ${fmt(t.norms[i])} (${fmt(t.norms[i] / t.init_norm, 2)}× init)`,
    onClick: i => { $('nb-layer').value = i; syncSegments(); guarded(loadNeighbors); },
  });
  const hl = (k, h) => V[k]?.highlight(h?.layer ?? null, h?.dim ?? null);
  if (V.glyph) V.glyph.update(t);
  else V.glyph = viz.glyph($('token-glyph'), t, { size: 340, onHover: h => hl('heat', h) });
  if (V.heat) V.heat.update(t);
  else V.heat = viz.heatmap2($('token-heat'), t, { cellH: Math.max(14, Math.min(28, 200 / t.n_layer)), onHover: h => hl('glyph', h) });
  if (V.sim) V.sim.update(sim);
  else V.sim = viz.simMatrix($('token-sim'), sim, { size: 300 });
  $('global-sim-title').textContent = `Toggle between this token and the average over the top ${sim.global_n} tokens (${sim.global_source}).`;
  if (state.graph && state.graph.id === t.id) renderGraph(state.graph);
}

function renderGraph(g) {
  state.graph = g;
  if (V.graph) V.graph.update(g);
  else V.graph = viz.forceGraph($('token-graph'), g, { height: 340, onNodeClick: n => { if (n.id !== state.tokenId) openToken(n.id); } });
}

async function loadNeighbors() {
  if (state.tokenId == null) return;
  const layer = $('nb-layer').value, k = $('nb-k').value || 20;
  const id = state.tokenId;
  const [nb, g] = await Promise.all([
    api(`/api/neighbors?${q({ id, layer, k })}`),
    api(`/api/neighbor_graph?${q({ id, layer, k: 12 })}`),
  ]);
  if (id === state.tokenId) renderGraph(g);
  $('nb-table').innerHTML = `<tr><th>#</th><th>token</th><th>id</th><th>cosine</th><th></th></tr>` +
    nb.neighbors.map((n, i) => `<tr class="link" tabindex="0" data-id="${n.id}" aria-label="open token ${esc(visTok(n.piece))}"><td class="num">${i + 1}</td><td class="tok">${tokHTML(n.piece)}</td>
      <td class="num">${n.id}</td><td class="num">${n.sim.toFixed(3)}</td>
      <td><span class="simbar" style="width:${Math.max(0, n.sim) * 140}px"></span></td></tr>`).join('');
}

function openToken(id) {
  state.tokenId = id;
  state.tokenQuery = `#${id}`;
  $('token-q').value = `#${id}`;
  state.dirty.add('token');
  showTab('token');
}
function inspectToken(text) {
  state.tokenId = null; $('token-q').value = text;
  if (state.tab !== 'token') { state.dirty.add('token'); showTab('token'); } else guarded(() => loadToken(text));
}

// ---- Map tab -----------------------------------------------------------------------------
let mapData = null;
async function loadMap() {
  const layer = $('map-layer').value, n = $('map-n').value;
  if (!mapData) skeleton($('map-plot'), 420);
  mapData = await api(`/api/projection?${q({ layer, n })}`);
  renderMap();
}
function renderMap() {
  if (!mapData || map3d.active) return;
  const pts = mapData.points, mode = $('map-color').value;
  const norms = pts.map(p => p.norm), nmin = Math.min(...norms), nmax = Math.max(...norms);
  const find = $('map-find').value;
  const hi = find ? pts.findIndex(p => p.piece === find || p.piece === ` ${find}` || String(p.id) === find.replace('#', '')) : -1;
  $('map-info').textContent = `${mapData.n} tokens (${mapData.source}) · PC1 ${(mapData.explained_variance[0] * 100).toFixed(1)}% · PC2 ${(mapData.explained_variance[1] * 100).toFixed(1)}% of variance`
    + (find && hi < 0 ? ' · token not in this set' : '');
  scatter($('map-plot'), pts.map((p, i) => ({
    x: p.x, y: p.y, label: visTok(p.piece), r: 3,
    color: mode === 'rank' ? seq(0.2 + 0.8 * (1 - Math.log(i + 1) / Math.log(pts.length))) : seq(0.2 + 0.8 * (p.norm - nmin) / (nmax - nmin || 1)),
  })), {
    height: Math.max(420, window.innerHeight - 330), labelTop: 40, xLabel: 'PC1', yLabel: 'PC2',
    highlight: hi >= 0 ? hi : null,
    tip: i => `<b class="tok">${tokHTML(pts[i].piece)}</b> <span class="muted">#${pts[i].id}</span><br>rank ${i + 1}`
      + `${pts[i].freq != null ? ` · freq ${pts[i].freq.toLocaleString()}` : ''}<br>norm ${fmt(pts[i].norm)}`,
    onClick: i => openToken(pts[i].id),
  });
}

/** 2D | 3D map controller. map3d.js is owned by another module and loaded lazily. */
const map3d = {
  handle: null, active: false, loading: null,
  webgl() {
    try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
  },
  set2D(note) {
    this.active = false;
    $('map3d').classList.add('hidden');
    for (const id of ['map-plot', 'map-hint']) $(id).classList.remove('hidden');
    for (const id of ['map-n', 'map-color']) $(id).closest('label').classList.remove('hidden');
    if (note) { $('map3d-note').textContent = note; $('map3d-note').classList.remove('hidden'); }
    if ($('map-mode').value !== '2d') { $('map-mode').value = '2d'; syncSegments(); }
    if (state.tab === 'map' && hasPle()) { if (mapData) renderMap(); else guarded(loadMap); }
  },
  destroy() {
    try { this.handle?.destroy(); } catch (e) { console.warn(e); }
    this.handle = null;
    $('map3d').innerHTML = '';
  },
  async enter() {
    if ($('map-mode').value !== '3d') { this.active = false; return; }
    this.active = true;
    $('map3d-note').classList.add('hidden');
    $('map3d').classList.remove('hidden');
    for (const id of ['map-plot', 'map-hint']) $(id).classList.add('hidden');
    for (const id of ['map-n', 'map-color']) $(id).closest('label').classList.add('hidden');
    if (this.handle) { requestAnimationFrame(() => this.handle?.resize()); return; }
    if (this.loading) return;
    this.loading = (async () => {
      try {
        const m = await import('/static/map3d.js');
        if (typeof m.mountMap3D !== 'function') throw new Error('map3d.js has no mountMap3D export');
        const handle = await m.mountMap3D($('map3d'), {
          run: state.run.name, api, onTokenClick: id => openToken(id),
          theme: { bg: cssv('--plate-bg'), fg: cssv('--plate-fg'), accent: cssv('--plate-accent'), accent2: cssv('--plate-accent2'), muted: cssv('--plate-muted') },
        });
        if (!this.active || state.tab !== 'map') { try { handle.destroy(); } catch { /* ignore */ } $('map3d').innerHTML = ''; return; }
        this.handle = handle;
        const l = $('map-layer').value;
        if (l !== '' && l !== '0') handle.setLayer(l === '-1' ? 'full' : +l);
        if ($('map-find').value) handle.highlight($('map-find').value);
      } catch (e) {
        console.warn('3D map unavailable:', e);
        this.destroy();
        this.set2D(`3D map unavailable (${e.message || e}). Showing the 2D map instead.`);
      } finally { this.loading = null; }
    })();
  },
  leave() { if (this.handle) this.destroy(); },
};

// ---- Analyze tab -------------------------------------------------------------------------
function ablationMask() {
  if (!$('an-ablate').checked) return null;
  if ($('an-all').checked) return 'all';
  const ls = [...document.querySelectorAll('#an-layers input:checked')].map(i => +i.dataset.layer);
  return ls.length ? ls : null;
}

async function runAnalyze() {
  const mask = ablationMask();
  if (!$('an-text').value.trim()) { $('an-chips').innerHTML = '<div class="empty">Type some text or pick a sample above.</div>'; return; }
  if (!state.analysis) skeleton($('an-chips'), 0, 'chips');
  state.analysis = await api('/api/analyze', { run: state.run.name, text: $('an-text').value, mask });
  if (state.analysis.warnings.length) banner(state.analysis.warnings.join('; '), 'warn');
  if (mask && $('an-color').value === 'gate') $('an-color').value = 'delta';
  if (!mask && $('an-color').value === 'delta') $('an-color').value = 'gate';
  syncSegments();
  state.selected = null;
  renderAnalyze();
}

function colorInfo(a) {
  const mode = $('an-color').value, l = +$('an-layer').value;
  if (mode === 'gate') {
    const vals = a.gate.map(g => g[l]), mx = Math.max(...vals), mn = Math.min(...vals);
    return { vals, color: v => seq(0.15 + 0.85 * (v - mn) / (mx - mn || 1)), legend: [`gate L${l} ${fmt(mn)}`, fmt(mx)], ramp: 'seq' };
  }
  if (mode === 'loss') {
    const vals = a.loss, mx = Math.max(...vals.filter(v => v != null), 1e-9);
    return { vals, color: v => (v == null ? cssv('--zero') : seq(0.15 + 0.85 * v / mx)), legend: ['loss 0', fmt(mx)], ramp: 'seq' };
  }
  if (!a.ablation) return { vals: a.loss.map(() => null), color: () => cssv('--zero'), legend: ['enable "Ablate PLE" to colour by Δloss', ''], ramp: null };
  const vals = a.ablation.delta, mx = Math.max(...vals.filter(v => v != null).map(Math.abs), 1e-9);
  return { vals, color: v => (v == null ? cssv('--zero') : div(v / mx)), legend: [`Δloss −${fmt(mx)}`, `+${fmt(mx)}`], ramp: 'div' };
}

function renderAnalyze() {
  const a = state.analysis;
  if (!a) return;
  const ci = colorInfo(a);
  $('an-layer-wrap').style.display = $('an-color').value === 'gate' ? '' : 'none';
  renderAnalyzeViz(a);
  const ramp = ci.ramp === 'seq' ? `linear-gradient(90deg,${[0.15, 0.35, 0.55, 0.75, 1].map(seq).join(',')})`
    : ci.ramp === 'div' ? `linear-gradient(90deg,${div(-1)},${div(0)},${div(1)})` : 'none';
  $('an-legend').innerHTML = `<span>${esc(ci.legend[0])}</span>${ci.ramp ? `<span class="ramp" style="background:${ramp}"></span>` : ''}<span>${esc(ci.legend[1])}</span>`;
  let s = `<b>${a.tokens.length}</b> tokens · mean loss <b>${fmt(a.mean_loss)}</b>`;
  if (a.ablation) {
    const d = a.ablation.mean_delta;
    s += ` · ablated [${a.ablation.layers.join(',')}] → <b>${fmt(a.ablation.mean_loss)}</b> <span class="${d >= 0 ? 'up' : 'down'}">(Δ ${d >= 0 ? '▲ +' : '▼ '}${fmt(d)})</span>`;
  }
  $('an-summary').innerHTML = s;
  const labels = Array.from({ length: a.n_layer }, (_, l) => `L${l}`);
  barChart($('an-contrib'), a.contrib_ratio_mean, { labels, height: 170, color: () => cssv('--gate'),
    tip: i => `layer ${i}: PLE update is ${fmt(a.contrib_ratio_mean[i] * 100, 1)}% of ‖h‖` });
  const gm = labels.map((_, l) => a.gate.reduce((s2, g) => s2 + g[l], 0) / a.gate.length);
  barChart($('an-gate-mean'), gm, { labels, height: 150, color: (_, i) => (i === +$('an-layer').value ? cssv('--ple') : cssv('--accent-dim')),
    tip: i => `layer ${i}: mean gate ${fmt(gm[i])}`,
    onClick: i => { $('an-layer').value = i; $('an-color').value = 'gate'; $('an-layer-val').textContent = i; syncSegments(); renderAnalyze(); } });
  if (a.ablation) {
    const worst = a.ablation.delta.map((d, i) => [d, i]).filter(x => x[0] != null).sort((x, y) => y[0] - x[0]).slice(0, 8);
    $('an-ablation-summary').innerHTML = `<div class="card-head" style="margin-top:14px"><h3>Most PLE-dependent tokens <small>largest Δloss</small></h3></div>
      <table class="table">${worst.map(([d, i]) => `<tr class="link" tabindex="0" data-i="${i}"><td class="tok">${tokHTML(a.tokens[i].piece)}</td><td class="num">${d >= 0 ? '▲ +' : '▼ '}${fmt(d)}</td></tr>`).join('')}</table>`;
  } else {
    $('an-ablation-summary').innerHTML = '<p class="hint">Switch on “Ablate PLE” to see which tokens rely on the PLE token part.</p>';
  }
  drawSelected(state.selected ?? 0);
}

function renderAnalyzeViz(a) {
  const link = (src, i) => {
    for (const k of ['chips', 'river', 'abl']) if (k !== src) V[k]?.highlight(i);
    drawSelected(i ?? state.selected ?? 0);
  };
  killViz('chips');
  V.chips = viz.sparkChips($('an-chips'), a, { color: $('an-color').value, layer: +$('an-layer').value, selected: state.selected,
    onHover: i => link('chips', i), onClick: selectChip });
  if (!V.river) V.river = viz.river($('an-river'), a, { height: 240, onHover: i => link('river', i), onClick: selectChip });
  else if (V.riverData !== a) V.river.update(a);
  V.riverData = a;
  $('an-abl-card').classList.toggle('hidden', !a.ablation);
  if (!a.ablation) killViz('abl', 'ablData');
  else if (!V.abl || V.ablData !== a) {
    killViz('abl');
    V.abl = viz.ablation($('an-abl'), a, { height: 280, onHover: i => link('abl', i), onClick: selectChip });
    V.ablData = a;
  }
}

function selectChip(i) {
  if (!state.analysis || i == null || !state.analysis.tokens[i]) return;
  state.selected = i;
  V.chips?.select(i);
  drawSelected(i);
  bus.emit('focus-token', state.analysis.tokens[i].id);
}

function drawSelected(i) {
  const a = state.analysis;
  if (!a || i == null || !a.tokens[i]) return;
  const t = a.tokens[i];
  let title = `<span class="tok">${tokHTML(t.piece)}</span> #${t.id} at position ${i}`;
  if (a.loss[i] != null) title += ` · loss ${fmt(a.loss[i])}`;
  if (a.ablation && a.ablation.loss[i] != null) title += ` → ablated ${fmt(a.ablation.loss[i])}`;
  $('an-sel-title').innerHTML = title;
  barChart($('an-sel-gate'), a.gate[i], { labels: a.gate[i].map((_, l) => `L${l}`), height: 150,
    color: (_, l) => (l === +$('an-layer').value ? cssv('--ple') : cssv('--accent-dim')),
    tip: l => `layer ${l}: gate ${fmt(a.gate[i][l])} · contrib ${fmt(a.contrib_ratio[i][l] * 100, 1)}% of ‖h‖` });
  const next = a.tokens[i + 1];
  $('an-sel-top5').innerHTML = `<tr><th>prediction</th><th>p</th></tr>` + a.top5[i].map(p =>
    `<tr${next && next.id === p.id ? ' class="hit"' : ''}><td class="tok">${tokHTML(p.piece)}</td><td class="num">${p.p.toFixed(3)}</td></tr>`).join('')
    + (next ? `<tr><td colspan="2" class="muted">actual next: <span class="tok">${tokHTML(next.piece)}</span></td></tr>` : '');
}

// ---- Stats tab -----------------------------------------------------------------------------
async function loadStats() {
  killViz('ridge', 'hex');
  skeleton($('stats-hist'), 360); skeleton($('stats-freq'), 340); skeleton($('stats-quant'), 220);
  const [s, nd] = await Promise.all([api(`/api/stats?${q({})}`), api(`/api/norm_dist?${q({})}`)]);
  state.normDist = nd;
  renderStats(s);
}

function renderStats(s) {
  state.stats = s;
  const nd = state.normDist, nf = s.norm_vs_freq;
  if (nd) {
    if (V.ridge) V.ridge.update(nd); else V.ridge = viz.ridgeline($('stats-hist'), nd, { height: 360 });
  } else renderHistHeatmap(s);
  if (nd?.hexbin && nf.freq) {
    if (V.hex) V.hex.update(nd);
    else V.hex = viz.hexbin($('stats-freq'), nd, { height: 340, onClick: b => { if (b?.examples?.length) openToken(b.examples[0].id); } });
    $('stats-freq-note').textContent = `All ${(nd.n_seen + nd.n_unseen).toLocaleString()} tokens, hex-binned (${nd.n_unseen.toLocaleString()} never seen in train.bin). `
      + 'Rare tokens get few gradient updates, so their rows stay near init. Click a hex to open one of its tokens.';
  } else renderFreqScatter(s);
  renderQuant(s);
}

function renderHistHeatmap(s) {
  const centers = s.hist.edges.slice(0, -1).map((e, i) => (e + s.hist.edges[i + 1]) / 2);
  const matrix = s.hist.counts.map(row => row.map(c => Math.log10(c + 1)));
  heatmap($('stats-hist'), matrix, {
    cellH: Math.max(16, Math.min(32, 240 / s.n_layer)), rowLabels: matrix.map((_, l) => `L${l}`),
    colLabels: centers.map(c => fmt(c, 2)),
    tip: (r, c) => `layer ${r}: ${s.hist.counts[r][c].toLocaleString()} tokens with norm ${fmt(s.hist.edges[c], 3)}–${fmt(s.hist.edges[c + 1], 3)}`
      + `<br>layer mean ${fmt(s.layer_mean_norm[r])} · init ≈ ${fmt(s.init_norm)}`,
  });
}

function renderFreqScatter(s) {
  const nf = s.norm_vs_freq;
  if (nf.freq) {
    scatter($('stats-freq'), nf.ids.map((id, i) => ({ x: nf.freq[i], y: nf.norm[i], label: visTok(nf.pieces[i]), r: 2, color: cssv('--accent') })), {
      height: 320, logX: true, alpha: 0.5, xLabel: 'training count (log₁₀(1+n))', yLabel: 'full-row norm',
      line: s.trend ? { x: s.trend.map(t => 10 ** t.log10_freq - 1), y: s.trend.map(t => t.mean_norm) } : null,
      refY: { value: s.init_full_norm, label: 'init' },
      tip: i => `<b class="tok">${tokHTML(nf.pieces[i])}</b> #${nf.ids[i]}<br>count ${nf.freq[i].toLocaleString()} · norm ${fmt(nf.norm[i])}`,
      onClick: i => openToken(nf.ids[i]),
    });
    $('stats-freq-note').textContent = `${nf.ids.length} random tokens; orange = mean norm per frequency bucket, dashed = init norm. Rare tokens get few gradient updates, so their rows stay near init.`;
  } else {
    scatter($('stats-freq'), nf.ids.map((id, i) => ({ x: id, y: nf.norm[i], label: visTok(nf.pieces[i]), r: 2, color: cssv('--accent') })), {
      height: 320, alpha: 0.5, xLabel: 'token id (no train.bin, so no frequencies)', yLabel: 'full-row norm',
      refY: { value: s.init_full_norm, label: 'init' },
      tip: i => `<b class="tok">${tokHTML(nf.pieces[i])}</b> #${nf.ids[i]} · norm ${fmt(nf.norm[i])}`, onClick: i => openToken(nf.ids[i]),
    });
    $('stats-freq-note').textContent = 'data/tinystories/train.bin not found: plotted against token id instead of frequency (GPT-2 ids are roughly ordered by BPE merge frequency).';
  }
}

function renderQuant(s) {
  if (s.quant) {
    barChart($('stats-quant'), s.quant.map(x => x.rel_error * 100), {
      labels: s.quant.map(x => `L${x.layer}`), height: 220, color: (_, i) => cssv(`--l${i % 6}`), yLabel: 'relative error %',
      tip: i => `layer ${i}: rel. error ${fmt(s.quant[i].rel_error * 100, 3)}% · mean max |err| ${fmt(s.quant[i].mean_max_abs, 5)}`,
    });
    $('stats-quant-note').textContent = 'Per-row symmetric int8 (scale = max|row|/127). Layers whose slices are much smaller than the row maximum lose relatively more precision.';
  } else {
    $('stats-quant').innerHTML = '<div class="empty">no ple_table.i8.bin in this run</div>';
  }
}

// ---- Compare tab ---------------------------------------------------------------------------
async function runCompare() {
  const base = $('cmp-base').value, ple = $('cmp-ple').value;
  if (!base || !ple) {
    $('cmp-summary').textContent = '';
    $('cmp-chips').innerHTML = `<div class="empty">Compare needs a baseline run (ple_dim=0) and a PLE run, e.g. runs/baseline-small and runs/ple-small. They will appear here once training writes them.</div>`;
    return;
  }
  skeleton($('cmp-chips'), 0, 'chips');
  const c = await api('/api/compare', { text: $('cmp-text').value, baseline: base, ple });
  if (c.warnings.length) banner(c.warnings.join('; '), 'warn');
  renderCompare(c);
}

function renderCompare(c) {
  state.compare = c;
  const mx = Math.max(...c.diff.filter(v => v != null).map(Math.abs), 1e-9);
  $('cmp-chips').innerHTML = c.tokens.map((t, i) => {
    const bg = c.diff[i] == null ? cssv('--zero') : div(c.diff[i] / mx);
    return `<span class="chip" tabindex="0" data-i="${i}" style="background:${bg};color:${inkFor(bg) || '#fff'}">${tokHTML(t.piece)}</span>`;
  }).join('');
  $('cmp-legend').innerHTML = `<span>baseline better −${fmt(mx)}</span><span class="ramp" style="background:linear-gradient(90deg,${div(-1)},${div(0)},${div(1)})"></span><span>+${fmt(mx)} PLE better</span>`;
  const d = c.mean_diff;
  $('cmp-summary').innerHTML = `mean loss: ${esc(c.baseline.run)} <b>${fmt(c.baseline.mean_loss)}</b> vs ${esc(c.ple.run)} <b>${fmt(c.ple.mean_loss)}</b> <span class="${d >= 0 ? 'up' : 'down'}">(Δ ${d >= 0 ? '▲ +' : '▼ '}${fmt(d)})</span>`;
  const tip = i => `<b class="tok">${tokHTML(c.tokens[i].piece)}</b><br>baseline ${fmt(c.baseline.loss[i])} · PLE ${fmt(c.ple.loss[i])}<br>Δ ${fmt(c.diff[i])}`;
  $('cmp-chips').querySelectorAll('.chip').forEach(el => {
    el.addEventListener('mousemove', ev => tooltip.show(tip(+el.dataset.i), ev));
    el.addEventListener('mouseleave', () => tooltip.hide());
  });
  const chips = [...$('cmp-chips').querySelectorAll('.chip')];
  chips.forEach(el => {
    el.addEventListener('mouseenter', () => V.slope?.highlight(+el.dataset.i));
    el.addEventListener('mouseleave', () => V.slope?.highlight(null));
  });
  const hot = i => chips.forEach((el, j) => el.classList.toggle('sel', j === i));
  if (V.slope) V.slope.update(c);
  else V.slope = viz.slope($('cmp-bars'), c, { height: 380, onHover: hot });
}

// ---- re-render the active tab from cached data (resize, theme) ------------------------------------
function rerender() {
  if (state.tab === 'map') renderMap();
  else if (state.tab === 'analyze') renderAnalyze();
  else if (state.tab === 'token' && state.token) renderTokenTab(state.token, state.sim);
  else if (state.tab === 'stats' && state.stats) renderStats(state.stats);
  else if (state.tab === 'compare' && state.compare) renderCompare(state.compare);
}

// ---- shell: palette, shortcuts, theme --------------------------------------------------------
let popovers = null;
const about = initAbout({ getRun: () => state.run, api });
const SAMPLE_TEXTS = () => [...document.querySelectorAll('#tab-analyze .sample[data-text]')];

function ablateAll() {
  $('an-ablate').checked = true; $('an-all').checked = true;
  if (state.tab !== 'analyze') { state.dirty.add('analyze'); showTab('analyze'); } else guarded(runAnalyze);
}
function analyzeText(text) {
  $('an-text').value = text;
  if (state.tab !== 'analyze') { state.dirty.add('analyze'); showTab('analyze'); } else guarded(runAnalyze);
}

const palette = createPalette({
  commands: () => [
    ...TABS.map((t, i) => ({ id: `go-${t}`, group: 'Go to', title: `Go to ${t[0].toUpperCase()}${t.slice(1)}`, keys: [String(i + 1)], icon: '→', run: () => showTab(t) })),
    { id: 'theme', group: 'Actions', title: `Switch to ${currentTheme() === 'dark' ? 'light' : 'dark'} theme`, keys: ['T'], icon: '◐', run: toggleTheme },
    { id: 'ablate', group: 'Actions', title: 'Analyze: ablate all layers', icon: '⊘', keywords: 'ablation mask zero', run: ablateAll },
    { id: 'map2d', group: 'Actions', title: 'Map: 2D view', icon: '◫', run: () => { $('map-mode').value = '2d'; $('map-mode').dispatchEvent(new Event('change')); showTab('map'); } },
    { id: 'map3d', group: 'Actions', title: 'Map: 3D view', icon: '◈', run: () => { $('map-mode').value = '3d'; $('map-mode').dispatchEvent(new Event('change')); showTab('map'); } },
    { id: 'help', group: 'Actions', title: 'Keyboard shortcuts', keys: ['?'], icon: '⌨', run: showShortcuts },
    ...SAMPLE_TEXTS().map(b => ({ id: `s-${b.textContent}`, group: 'Analyze sample', title: `Analyze: ${b.textContent}`, icon: '¶', run: () => analyzeText(b.dataset.text) })),
    ...SAMPLE_TOKENS.map(t => ({ id: `t-${t}`, group: 'Inspect token', title: `Inspect token ${visTok(t)}`, icon: '#', run: () => inspectToken(t) })),
    ...state.runs.map(r => ({ id: `run-${r.name}`, group: 'Switch run', title: `Run: ${r.name}${r.has_ple ? '' : ' (baseline)'}`, icon: '●',
      hint: r.name === state.run?.name ? 'current' : '', run: () => selectRun(r.name) })),
  ],
  dynamic: text => {
    const t = text.replace(/^(token|inspect)\s*:?\s?/i, '');
    const out = [{ id: 'dyn-tok', title: `Inspect token “${visTok(t)}”`, icon: '#', hint: 'Token tab', run: () => inspectToken(t) }];
    if (!t.startsWith(' ') && !t.startsWith('#')) out.push({ id: 'dyn-tok2', title: `Inspect token “${visTok(` ${t}`)}”`, icon: '#', hint: 'with leading space', run: () => inspectToken(` ${t}`) });
    if (text.trim().split(/\s+/).length > 1) out.push({ id: 'dyn-an', title: `Analyze text “${text.slice(0, 40)}”`, icon: '¶', run: () => analyzeText(text) });
    return out;
  },
});

function showShortcuts() { $('shortcuts').hidden = false; $('close-shortcuts').focus(); }
function hideShortcuts() { $('shortcuts').hidden = true; }

function wireShell() {
  popovers = initPopovers(INFO);
  $('open-palette').addEventListener('click', () => palette.open());
  $('theme-toggle').addEventListener('click', toggleTheme);
  $('open-help').addEventListener('click', showShortcuts);
  $('close-shortcuts').addEventListener('click', hideShortcuts);
  $('shortcuts').addEventListener('mousedown', ev => { if (ev.target.id === 'shortcuts') hideShortcuts(); });
  onTheme(t => {
    $('theme-toggle').setAttribute('aria-pressed', String(t === 'light'));
    killViz();
    stale.delete(state.tab);
    rerender();
    if (map3d.handle) { map3d.destroy(); map3d.enter(); }
    bus.emit('theme', t);
  });
  document.addEventListener('keydown', ev => {
    const typing = ev.target.matches?.('input, textarea, select, [contenteditable]');
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'k') { ev.preventDefault(); palette.toggle(); return; }
    if (ev.key === 'Escape') { if (!$('shortcuts').hidden) hideShortcuts(); if (palette.isOpen()) palette.close(); tooltip.hide(); return; }
    if (typing || ev.ctrlKey || ev.metaKey || ev.altKey || palette.isOpen()) return;
    if (/^[1-6]$/.test(ev.key)) { ev.preventDefault(); showTab(TABS[+ev.key - 1]); }
    else if (ev.key === '/') {
      ev.preventDefault();
      if (state.tab === 'map') $('map-find').focus(); else { if (state.tab !== 'token') showTab('token'); $('token-q').focus(); $('token-q').select(); }
    } else if (ev.key === '?') { ev.preventDefault(); $('shortcuts').hidden ? showShortcuts() : hideShortcuts(); }
    else if (ev.key.toLowerCase() === 't') toggleTheme();
  });
  // tab keyboard navigation (arrow keys inside the tablist)
  $('tabs').addEventListener('keydown', ev => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
    if (!d) return;
    const i = TABS.indexOf(state.tab), t = TABS[(i + d + TABS.length) % TABS.length];
    showTab(t); document.querySelector(`#tabs button[data-tab="${t}"]`).focus();
  });
  // samples
  document.addEventListener('click', ev => {
    const s = ev.target.closest('.sample[data-token]');
    if (s) inspectToken(s.dataset.token);
    const st = ev.target.closest('.sample[data-text]');
    if (st) analyzeText(st.dataset.text);
  });
  // copy token
  $('token-head').addEventListener('click', ev => {
    const big = ev.target.closest('.big');
    if (!big || !state.token) return;
    navigator.clipboard?.writeText(state.token.piece).then(() => toast(`copied ${visTok(state.token.piece)}`, 'ok'), () => {});
  });
  segmented($('map-mode'), { label: 'Map dimension' });
  segmented($('an-color'), { label: 'Colour tokens by' });
  segmented($('an-layer'), { label: 'Gate layer', format: segFmt });
  segmented($('map-layer'), { label: 'Map layer', format: segFmt });
  segmented($('nb-layer'), { label: 'Neighbour layer', format: segFmt });
  window.addEventListener('resize', moveInk);
  document.fonts?.ready.then(() => { moveInk(); syncSegments(); });
}

// ---- wiring --------------------------------------------------------------------------------
function wire() {
  $('tabs').addEventListener('click', ev => { const t = ev.target.closest('button[data-tab]')?.dataset.tab; if (t) showTab(t); });
  $('run').addEventListener('change', ev => selectRun(ev.target.value));
  $('token-form').addEventListener('submit', ev => { ev.preventDefault(); state.tokenId = null; guarded(() => loadToken($('token-q').value)); });
  $('nb-layer').addEventListener('change', () => guarded(loadNeighbors));
  $('nb-k').addEventListener('change', () => guarded(loadNeighbors));
  const openRow = ev => { const tr = ev.target.closest('tr[data-id]'); if (tr) { state.tokenId = +tr.dataset.id; guarded(() => loadToken(`#${tr.dataset.id}`)); } };
  $('nb-table').addEventListener('click', openRow);
  $('nb-table').addEventListener('keydown', ev => { if (ev.key === 'Enter') openRow(ev); });
  $('map-layer').addEventListener('change', () => {
    if (map3d.handle) { const l = $('map-layer').value; map3d.handle.setLayer(l === '-1' ? 'full' : +l); }
    if (!map3d.active) guarded(loadMap); else mapData = null;
  });
  $('map-n').addEventListener('change', () => guarded(loadMap));
  $('map-color').addEventListener('change', renderMap);
  $('map-find').addEventListener('input', () => { if (map3d.active) map3d.handle?.highlight($('map-find').value); else renderMap(); });
  $('map-mode').addEventListener('change', () => {
    try { localStorage.setItem('ple.mapmode', $('map-mode').value); } catch { /* ignore */ }
    if ($('map-mode').value === '2d') { map3d.destroy(); map3d.set2D(); $('map3d-note').classList.add('hidden'); }
    else if (state.tab === 'map' && hasPle()) map3d.enter();
  });
  $('an-run').addEventListener('click', () => guarded(runAnalyze));
  $('an-ablate').addEventListener('change', () => guarded(runAnalyze));
  $('an-all').addEventListener('change', () => { if ($('an-ablate').checked) guarded(runAnalyze); });
  $('an-layers').addEventListener('change', () => { $('an-all').checked = false; if ($('an-ablate').checked) guarded(runAnalyze); });
  $('an-layer').addEventListener('change', () => { $('an-layer-val').textContent = $('an-layer').value; renderAnalyze(); });
  $('an-color').addEventListener('change', renderAnalyze);
  $('an-text').addEventListener('keydown', ev => { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) guarded(runAnalyze); });
  const chipIndex = ev => { const el = ev.target.closest('[data-i]'); return el ? +el.dataset.i : null; };
  // chip hover, tooltip, click and arrow keys are handled by viz.sparkChips (see renderAnalyzeViz)
  const select = ev => selectChip(chipIndex(ev));
  $('an-chips').addEventListener('focusin', ev => { const i = chipIndex(ev); if (i != null && state.analysis) drawSelected(i); });
  $('an-ablation-summary').addEventListener('click', select);
  $('cmp-run').addEventListener('click', () => guarded(runCompare));
  let resize;
  window.addEventListener('resize', () => {
    clearTimeout(resize);
    resize = setTimeout(() => { rerender(); map3d.handle?.resize(); }, 200);
  });
}

// default map mode: saved choice, else 3D when WebGL is available
let savedMode = null; try { savedMode = localStorage.getItem('ple.mapmode'); } catch { /* ignore */ }
$('map-mode').value = savedMode === '2d' || savedMode === '3d' ? savedMode : (map3d.webgl() ? '3d' : '2d');
// page-load choreography: stagger index per card (style.css reads --i), then switch to tab ink-wipes
document.querySelectorAll('.tab').forEach(t => t.querySelectorAll('.searchbar, .about-hero, .card').forEach((el, i) => el.style.setProperty('--i', String(Math.min(i, 12)))));
setTimeout(() => document.documentElement.classList.add('settled'), reducedMotion() ? 0 : 1800);
$('theme-toggle').setAttribute('aria-pressed', String(currentTheme() === 'light'));
wireShell();
wire();
guarded(loadRuns);
