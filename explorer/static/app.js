import { barChart, heatmap, scatter, tooltip, esc, vis, fmt, seq, div } from './charts.js';

const $ = id => document.getElementById(id);
const state = {
  runs: [], run: null, tab: 'token', tokenId: null, analysis: null, selected: null,
  dirty: new Set(['token', 'map', 'stats']),
};

// ---- API ---------------------------------------------------------------------
let pending = 0;
async function api(path, body) {
  pending++; $('loading').classList.remove('hidden');
  try {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    return data;
  } finally {
    if (--pending === 0) $('loading').classList.add('hidden');
  }
}
const q = params => new URLSearchParams({ run: state.run?.name ?? '', ...params }).toString();

function banner(msg, kind = 'error') {
  const b = $('banner');
  if (!msg) { b.classList.add('hidden'); return; }
  b.textContent = msg; b.className = `banner ${kind === 'warn' ? 'warn' : ''}`;
}
async function guarded(fn) {
  try { banner(null); await fn(); } catch (e) { banner(e.message); }
}

const L = () => state.run?.config.n_layer ?? 0;
const hasPle = () => !!state.run?.has_ple;

function fillLayerSelect(sel, withFull, current) {
  const opts = [];
  if (withFull) opts.push(`<option value="-1">all (full row)</option>`);
  for (let l = 0; l < L(); l++) opts.push(`<option value="${l}">${l}</option>`);
  sel.innerHTML = opts.join('');
  if (current != null && [...sel.options].some(o => o.value === String(current))) sel.value = String(current);
}

// ---- runs / tabs ---------------------------------------------------------------
async function loadRuns() {
  const data = await api('/api/runs');
  state.runs = data.runs.filter(r => !r.error);
  const sel = $('run');
  const prev = state.run?.name ?? new URLSearchParams(location.hash.slice(1)).get('run');
  sel.innerHTML = state.runs.map(r => `<option value="${esc(r.name)}">${esc(r.name)}${r.has_ple ? '' : ' (baseline)'}</option>`).join('');
  if (!state.runs.length) { banner('No runs found. Create one with: python -m explorer.make_demo_run'); return; }
  const pick = state.runs.find(r => r.name === prev) || state.runs.find(r => r.name === 'ple-small')
    || state.runs.find(r => r.has_ple) || state.runs[0];
  sel.value = pick.name;
  const bases = state.runs.filter(r => !r.has_ple), ples = state.runs.filter(r => r.has_ple);
  $('cmp-base').innerHTML = bases.map(r => `<option>${esc(r.name)}</option>`).join('');
  $('cmp-ple').innerHTML = ples.map(r => `<option>${esc(r.name)}</option>`).join('');
  if (ples.some(r => r.name === 'ple-small')) $('cmp-ple').value = 'ple-small';
  selectRun(pick.name);
}

function selectRun(name) {
  state.run = state.runs.find(r => r.name === name);
  const r = state.run, c = r.config, p = r.params;
  const M = n => `${(n / 1e6).toFixed(1)}M`;
  $('run-info').textContent = `L=${c.n_layer} d=${c.d_model} p=${c.ple_dim} · ${M(p.dense)} dense + ${M(p.ple_table)} PLE`
    + (r.val_loss != null ? ` · val ${(+r.val_loss).toFixed(3)}` : '');
  fillLayerSelect($('nb-layer'), true, $('nb-layer').value || -1);
  fillLayerSelect($('map-layer'), true, $('map-layer').value || 0);
  $('an-layer').max = Math.max(0, c.n_layer - 1);
  if (+$('an-layer').value >= c.n_layer) $('an-layer').value = 0;
  $('an-layer-val').textContent = $('an-layer').value;
  $('an-layers').innerHTML = Array.from({ length: c.n_layer }, (_, l) =>
    `<label><input type="checkbox" data-layer="${l}">${l}</label>`).join('');
  state.analysis = null;
  state.dirty = new Set(['token', 'map', 'stats', 'analyze']);
  history.replaceState(null, '', `#run=${encodeURIComponent(name)}`);
  showTab(state.tab);
}

function showTab(tab) {
  state.tab = tab;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab').forEach(s => s.classList.toggle('active', s.id === `tab-${tab}`));
  tooltip.hide();
  if (['token', 'map', 'stats', 'analyze'].includes(tab) && state.run && !hasPle()) {
    banner(`'${state.run.name}' is a baseline run with no PLE table; pick a PLE run for this view.`, 'warn');
    return;
  }
  banner(null);
  if (!state.dirty.has(tab)) return;
  state.dirty.delete(tab);
  if (tab === 'token') guarded(() => loadToken(state.tokenId != null ? `#${state.tokenId}` : $('token-q').value));
  if (tab === 'map') guarded(loadMap);
  if (tab === 'stats') guarded(loadStats);
  if (tab === 'analyze') guarded(runAnalyze);
}

// ---- Token tab -----------------------------------------------------------------
async function loadToken(query) {
  const t = await api(`/api/token?${q({ q: query })}`);
  state.tokenId = t.id;
  $('token-q').value = query;
  $('token-pieces').innerHTML = t.pieces.length > 1
    ? `query is ${t.pieces.length} tokens: ${t.pieces.map(p => `<span class="tok">${esc(vis(p.piece))}</span>`).join(' · ')} (showing the first)`
    : '';
  $('token-head').innerHTML = `<span class="big tok">${esc(vis(t.piece))}</span>
    <span class="muted">id ${t.id}${t.frequency != null ? ` · ${t.frequency.toLocaleString()} occurrences in train.bin` : ''}
    · L=${t.n_layer}, p=${t.ple_dim}${t.downsampled ? ' (heatmap downsampled)' : ''}</span>`;
  const labels = t.norms.map((_, l) => `L${l}`);
  barChart($('token-norms'), t.norms, {
    labels, height: 190, refLine: { value: t.init_norm, label: 'init' }, color: () => '#60a5fa',
    yLabel: 'norm', tip: i => `layer ${i}: ‖slice‖ = ${fmt(t.norms[i])}`,
    onClick: i => { $('nb-layer').value = i; guarded(loadNeighbors); },
  });
  heatmap($('token-heat'), t.vectors, {
    diverging: true, cellH: Math.max(14, Math.min(28, 200 / t.n_layer)), rowLabels: labels,
    tip: (r, c, v) => `layer ${r}, dim ${c}${t.downsampled ? ' (pooled)' : ''}: ${fmt(v, 4)}`,
  });
  const sim = await api(`/api/layer_similarity?${q({ id: t.id })}`);
  const simOpts = { diverging: true, vmin: -1, vmax: 1, square: true, showValues: L() <= 8, rowLabels: labels, colLabels: labels,
    tip: (r, c, v) => `cos(L${r}, L${c}) = ${fmt(v, 3)}` };
  heatmap($('token-sim'), sim.matrix, simOpts);
  $('global-sim-title').textContent = `average over top ${sim.global_n} tokens (${sim.global_source})`;
  heatmap($('global-sim'), sim.global, simOpts);
  await loadNeighbors();
}

async function loadNeighbors() {
  if (state.tokenId == null) return;
  const layer = $('nb-layer').value, k = $('nb-k').value || 20;
  const nb = await api(`/api/neighbors?${q({ id: state.tokenId, layer, k })}`);
  $('nb-table').innerHTML = `<tr><th>#</th><th>token</th><th>id</th><th>cosine</th><th></th></tr>` +
    nb.neighbors.map((n, i) => `<tr class="link" data-id="${n.id}"><td>${i + 1}</td><td class="tok">${esc(vis(n.piece))}</td>
      <td class="num">${n.id}</td><td class="num">${n.sim.toFixed(3)}</td>
      <td><span class="simbar" style="width:${Math.max(0, n.sim) * 120}px"></span></td></tr>`).join('');
}

function openToken(id) {
  state.tokenId = id;
  $('token-q').value = `#${id}`;
  state.dirty.add('token');
  showTab('token');
}

// ---- Map tab -------------------------------------------------------------------
let mapData = null;
async function loadMap() {
  const layer = $('map-layer').value, n = $('map-n').value;
  mapData = await api(`/api/projection?${q({ layer, n })}`);
  drawMap();
}
function drawMap() {
  if (!mapData) return;
  const pts = mapData.points, mode = $('map-color').value;
  const norms = pts.map(p => p.norm), nmin = Math.min(...norms), nmax = Math.max(...norms);
  const find = $('map-find').value;
  const hi = find ? pts.findIndex(p => p.piece === find || p.piece === ` ${find}` || String(p.id) === find.replace('#', '')) : -1;
  $('map-info').textContent = `${mapData.n} tokens (${mapData.source}) · PC1 ${(mapData.explained_variance[0] * 100).toFixed(1)}%, PC2 ${(mapData.explained_variance[1] * 100).toFixed(1)}% of variance`
    + (find && hi < 0 ? ' · token not in this set' : '');
  scatter($('map-plot'), pts.map((p, i) => ({
    x: p.x, y: p.y, label: vis(p.piece), r: 3,
    color: mode === 'rank' ? seq(1 - Math.log(i + 1) / Math.log(pts.length)) : seq((p.norm - nmin) / (nmax - nmin || 1)),
  })), {
    height: Math.max(420, window.innerHeight - 250), labelTop: 40, xLabel: 'PC1', yLabel: 'PC2',
    highlight: hi >= 0 ? hi : null,
    tip: i => `<b class="tok">${esc(vis(pts[i].piece))}</b> <span class="muted">#${pts[i].id}</span><br>rank ${i + 1}`
      + `${pts[i].freq != null ? ` · freq ${pts[i].freq.toLocaleString()}` : ''}<br>norm ${fmt(pts[i].norm)}`,
    onClick: i => openToken(pts[i].id),
  });
}

// ---- Analyze tab -----------------------------------------------------------------
function ablationMask() {
  if (!$('an-ablate').checked) return null;
  if ($('an-all').checked) return 'all';
  const ls = [...document.querySelectorAll('#an-layers input:checked')].map(i => +i.dataset.layer);
  return ls.length ? ls : null;
}

async function runAnalyze() {
  const mask = ablationMask();
  state.analysis = await api('/api/analyze', { run: state.run.name, text: $('an-text').value, mask });
  if (state.analysis.warnings.length) banner(state.analysis.warnings.join('; '), 'warn');
  if (mask && $('an-color').value === 'gate') $('an-color').value = 'delta';
  if (!mask && $('an-color').value === 'delta') $('an-color').value = 'gate';
  state.selected = null;
  drawAnalysis();
}

function colorInfo(a) {
  const mode = $('an-color').value, l = +$('an-layer').value;
  if (mode === 'gate') {
    const vals = a.gate.map(g => g[l]), mx = Math.max(...vals), mn = Math.min(...vals);
    return { vals, color: v => seq((v - mn) / (mx - mn || 1)), legend: [`gate L${l}: ${fmt(mn)}`, fmt(mx)], ramp: 'seq' };
  }
  if (mode === 'loss') {
    const vals = a.loss, mx = Math.max(...vals.filter(v => v != null), 1e-9);
    return { vals, color: v => (v == null ? '#333' : seq(v / mx)), legend: ['loss 0', fmt(mx)], ramp: 'seq' };
  }
  if (!a.ablation) return { vals: a.loss.map(() => null), color: () => '#333', legend: ['enable "Ablate PLE" and re-run', ''], ramp: null };
  const vals = a.ablation.delta, mx = Math.max(...vals.filter(v => v != null).map(Math.abs), 1e-9);
  return { vals, color: v => (v == null ? '#333' : div(v / mx)), legend: [`Δloss −${fmt(mx)}`, `+${fmt(mx)}`], ramp: 'div' };
}

function drawAnalysis() {
  const a = state.analysis;
  if (!a) return;
  const ci = colorInfo(a);
  $('an-layer-wrap').style.display = $('an-color').value === 'gate' ? '' : 'none';
  $('an-chips').innerHTML = a.tokens.map((t, i) =>
    `<span class="chip${state.selected === i ? ' sel' : ''}" data-i="${i}" style="background:${ci.color(ci.vals[i])}">${esc(vis(t.piece))}</span>`).join('');
  const ramp = ci.ramp === 'seq' ? 'linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)'
    : ci.ramp === 'div' ? 'linear-gradient(90deg,#3b82f6,#282c38,#ef4444)' : 'none';
  $('an-legend').innerHTML = `<span>${ci.legend[0]}</span>${ci.ramp ? `<span class="ramp" style="background:${ramp}"></span>` : ''}<span>${ci.legend[1]}</span>`;
  let s = `${a.tokens.length} tokens · mean loss ${fmt(a.mean_loss)}`;
  if (a.ablation) s += ` · ablated [${a.ablation.layers.join(',')}] → ${fmt(a.ablation.mean_loss)} (Δ ${a.ablation.mean_delta >= 0 ? '+' : ''}${fmt(a.ablation.mean_delta)})`;
  $('an-summary').textContent = s;
  const labels = Array.from({ length: a.n_layer }, (_, l) => `L${l}`);
  barChart($('an-contrib'), a.contrib_ratio_mean, { labels, height: 170, color: () => '#a78bfa',
    tip: i => `layer ${i}: PLE update is ${fmt(a.contrib_ratio_mean[i] * 100, 1)}% of ‖h‖` });
  const gm = labels.map((_, l) => a.gate.reduce((s, g) => s + g[l], 0) / a.gate.length);
  barChart($('an-gate-mean'), gm, { labels, height: 150, color: (_, i) => (i === +$('an-layer').value ? '#fde047' : '#60a5fa'),
    tip: i => `layer ${i}: mean gate ${fmt(gm[i])}`, onClick: i => { $('an-layer').value = i; $('an-color').value = 'gate'; $('an-layer-val').textContent = i; drawAnalysis(); } });
  if (a.ablation) {
    const worst = a.ablation.delta.map((d, i) => [d, i]).filter(x => x[0] != null).sort((x, y) => y[0] - x[0]).slice(0, 8);
    $('an-ablation-summary').innerHTML = `<h3 style="margin-top:14px">Most PLE-dependent tokens <small>largest Δloss</small></h3>
      <table class="table">${worst.map(([d, i]) => `<tr class="link" data-i="${i}"><td class="tok">${esc(vis(a.tokens[i].piece))}</td><td class="num">${d >= 0 ? '+' : ''}${fmt(d)}</td></tr>`).join('')}</table>`;
  } else {
    $('an-ablation-summary').innerHTML = '<p class="hint">Tick "Ablate PLE" and re-run to see which tokens rely on the PLE token part.</p>';
  }
  drawSelected(state.selected ?? 0);
}

function drawSelected(i) {
  const a = state.analysis;
  if (!a || i == null || !a.tokens[i]) return;
  const t = a.tokens[i];
  let title = `<span class="tok">${esc(vis(t.piece))}</span> #${t.id} at position ${i}`;
  if (a.loss[i] != null) title += ` · loss ${fmt(a.loss[i])}`;
  if (a.ablation && a.ablation.loss[i] != null) title += ` → ablated ${fmt(a.ablation.loss[i])}`;
  $('an-sel-title').innerHTML = title;
  barChart($('an-sel-gate'), a.gate[i], { labels: a.gate[i].map((_, l) => `L${l}`), height: 150,
    color: (_, l) => (l === +$('an-layer').value ? '#fde047' : '#60a5fa'),
    tip: l => `layer ${l}: gate ${fmt(a.gate[i][l])} · contrib ${fmt(a.contrib_ratio[i][l] * 100, 1)}% of ‖h‖` });
  const next = a.tokens[i + 1];
  $('an-sel-top5').innerHTML = `<tr><th>prediction</th><th>p</th></tr>` + a.top5[i].map(p =>
    `<tr${next && next.id === p.id ? ' style="color:#4ade80"' : ''}><td class="tok">${esc(vis(p.piece))}</td><td class="num">${p.p.toFixed(3)}</td></tr>`).join('')
    + (next ? `<tr><td colspan="2" class="muted">actual next: <span class="tok">${esc(vis(next.piece))}</span></td></tr>` : '');
}

function chipTip(i) {
  const a = state.analysis, t = a.tokens[i];
  const bars = a.gate[i].map((g, l) => {
    const mx = Math.max(...a.gate.flat(), 1e-9);
    return `<div>L${l} <span class="simbar" style="width:${g / mx * 120}px;background:#60a5fa"></span> ${fmt(g)}</div>`;
  }).join('');
  const top = a.top5[i].slice(0, 5).map(p => `<span class="tok">${esc(vis(p.piece))}</span> ${(p.p * 100).toFixed(1)}%`).join('<br>');
  let extra = a.loss[i] != null ? `loss ${fmt(a.loss[i])}` : 'first token (no loss)';
  if (a.ablation && a.ablation.delta[i] != null) extra += ` · Δ ${a.ablation.delta[i] >= 0 ? '+' : ''}${fmt(a.ablation.delta[i])}`;
  return `<b class="tok">${esc(vis(t.piece))}</b> <span class="muted">#${t.id}</span><br>${extra}<hr style="border-color:#2a2f3d">gate per layer${bars}<hr style="border-color:#2a2f3d">next-token top-5<br>${top}`;
}

// ---- Stats tab -------------------------------------------------------------------
async function loadStats() {
  const s = await api(`/api/stats?${q({})}`);
  const centers = s.hist.edges.slice(0, -1).map((e, i) => (e + s.hist.edges[i + 1]) / 2);
  const matrix = s.hist.counts.map(row => row.map(c => Math.log10(c + 1)));
  heatmap($('stats-hist'), matrix, {
    cellH: Math.max(16, Math.min(32, 240 / s.n_layer)), rowLabels: matrix.map((_, l) => `L${l}`),
    colLabels: centers.map(c => fmt(c, 2)),
    tip: (r, c) => `layer ${r}: ${s.hist.counts[r][c].toLocaleString()} tokens with norm ${fmt(s.hist.edges[c], 3)}–${fmt(s.hist.edges[c + 1], 3)}`
      + `<br>layer mean ${fmt(s.layer_mean_norm[r])} · init ≈ ${fmt(s.init_norm)}`,
  });
  const nf = s.norm_vs_freq;
  if (nf.freq) {
    scatter($('stats-freq'), nf.ids.map((id, i) => ({ x: nf.freq[i], y: nf.norm[i], label: vis(nf.pieces[i]), r: 2, color: '#60a5fa' })), {
      height: 320, logX: true, alpha: 0.5, xLabel: 'training count (log₁₀(1+n))', yLabel: 'full-row norm',
      line: s.trend ? { x: s.trend.map(t => 10 ** t.log10_freq - 1), y: s.trend.map(t => t.mean_norm) } : null,
      refY: { value: s.init_full_norm, label: 'init' },
      tip: i => `<b class="tok">${esc(vis(nf.pieces[i]))}</b> #${nf.ids[i]}<br>count ${nf.freq[i].toLocaleString()} · norm ${fmt(nf.norm[i])}`,
      onClick: i => openToken(nf.ids[i]),
    });
    $('stats-freq-note').textContent = `${nf.ids.length} random tokens; orange = mean norm per frequency bucket. Rare tokens get few gradient updates, so their rows stay near the init norm (purple).`;
  } else {
    scatter($('stats-freq'), nf.ids.map((id, i) => ({ x: id, y: nf.norm[i], label: vis(nf.pieces[i]), r: 2 })), {
      height: 320, alpha: 0.5, xLabel: 'token id (no train.bin, so no frequencies)', yLabel: 'full-row norm',
      refY: { value: s.init_full_norm, label: 'init' },
      tip: i => `<b class="tok">${esc(vis(nf.pieces[i]))}</b> #${nf.ids[i]} · norm ${fmt(nf.norm[i])}`, onClick: i => openToken(nf.ids[i]),
    });
    $('stats-freq-note').textContent = 'data/tinystories/train.bin not found: plotted against token id instead of frequency (GPT-2 ids are roughly ordered by BPE merge frequency).';
  }
  if (s.quant) {
    barChart($('stats-quant'), s.quant.map(x => x.rel_error * 100), {
      labels: s.quant.map(x => `L${x.layer}`), height: 220, color: () => '#f59e0b', yLabel: 'relative error %',
      tip: i => `layer ${i}: rel. error ${fmt(s.quant[i].rel_error * 100, 3)}% · mean max |err| ${fmt(s.quant[i].mean_max_abs, 5)}`,
    });
    $('stats-quant-note').textContent = 'Per-row symmetric int8 (scale = max|row|/127). Layers whose slices are much smaller than the row maximum lose relatively more precision.';
  } else {
    $('stats-quant').innerHTML = '<div class="empty">no ple_table.i8.bin in this run</div>';
  }
}

// ---- Compare tab -----------------------------------------------------------------
async function runCompare() {
  const base = $('cmp-base').value, ple = $('cmp-ple').value;
  if (!base || !ple) {
    $('cmp-summary').textContent = '';
    $('cmp-chips').innerHTML = `<div class="empty">Compare needs a baseline run (ple_dim=0) and a PLE run, e.g. runs/baseline-small and runs/ple-small. They will appear here once training writes them.</div>`;
    return;
  }
  const c = await api('/api/compare', { text: $('cmp-text').value, baseline: base, ple });
  if (c.warnings.length) banner(c.warnings.join('; '), 'warn');
  const mx = Math.max(...c.diff.filter(v => v != null).map(Math.abs), 1e-9);
  $('cmp-chips').innerHTML = c.tokens.map((t, i) => `<span class="chip" data-i="${i}" style="background:${c.diff[i] == null ? '#333' : div(c.diff[i] / mx)}">${esc(vis(t.piece))}</span>`).join('');
  $('cmp-legend').innerHTML = `<span>baseline better −${fmt(mx)}</span><span class="ramp" style="background:linear-gradient(90deg,#3b82f6,#282c38,#ef4444)"></span><span>+${fmt(mx)} PLE better</span>`;
  $('cmp-summary').textContent = `mean loss: ${c.baseline.run} ${fmt(c.baseline.mean_loss)} vs ${c.ple.run} ${fmt(c.ple.mean_loss)} (Δ ${fmt(c.mean_diff)})`;
  const tip = i => `<b class="tok">${esc(vis(c.tokens[i].piece))}</b><br>baseline ${fmt(c.baseline.loss[i])} · PLE ${fmt(c.ple.loss[i])}<br>Δ ${fmt(c.diff[i])}`;
  $('cmp-chips').querySelectorAll('.chip').forEach(el => {
    el.addEventListener('mousemove', ev => tooltip.show(tip(+el.dataset.i), ev));
    el.addEventListener('mouseleave', () => tooltip.hide());
  });
  barChart($('cmp-bars'), c.diff.map(v => v ?? 0), {
    labels: c.tokens.map(t => vis(t.piece)), height: 220, min: Math.min(0, ...c.diff.filter(v => v != null)),
    color: v => (v >= 0 ? '#ef4444' : '#3b82f6'), tip,
  });
}

// ---- wiring --------------------------------------------------------------------
function wire() {
  $('tabs').addEventListener('click', ev => { const t = ev.target.dataset?.tab; if (t) { if (t === 'compare') guarded(runCompare); showTab(t); } });
  $('run').addEventListener('change', ev => selectRun(ev.target.value));
  $('token-form').addEventListener('submit', ev => { ev.preventDefault(); guarded(() => loadToken($('token-q').value)); });
  $('nb-layer').addEventListener('change', () => guarded(loadNeighbors));
  $('nb-k').addEventListener('change', () => guarded(loadNeighbors));
  $('nb-table').addEventListener('click', ev => { const tr = ev.target.closest('tr[data-id]'); if (tr) guarded(() => loadToken(`#${tr.dataset.id}`)); });
  for (const id of ['map-layer', 'map-n']) $(id).addEventListener('change', () => guarded(loadMap));
  $('map-color').addEventListener('change', drawMap);
  $('map-find').addEventListener('input', drawMap);
  $('an-run').addEventListener('click', () => guarded(runAnalyze));
  $('an-ablate').addEventListener('change', () => guarded(runAnalyze));
  $('an-all').addEventListener('change', () => { if ($('an-ablate').checked) guarded(runAnalyze); });
  $('an-layers').addEventListener('change', () => { $('an-all').checked = false; if ($('an-ablate').checked) guarded(runAnalyze); });
  $('an-layer').addEventListener('input', () => { $('an-layer-val').textContent = $('an-layer').value; drawAnalysis(); });
  $('an-color').addEventListener('change', drawAnalysis);
  $('an-text').addEventListener('keydown', ev => { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) guarded(runAnalyze); });
  const chipIndex = ev => { const el = ev.target.closest('[data-i]'); return el ? +el.dataset.i : null; };
  $('an-chips').addEventListener('mousemove', ev => { const i = chipIndex(ev); if (i == null) return tooltip.hide(); tooltip.show(chipTip(i), ev); });
  $('an-chips').addEventListener('mouseleave', () => tooltip.hide());
  const select = ev => { const i = chipIndex(ev); if (i != null) { state.selected = i; drawAnalysis(); } };
  $('an-chips').addEventListener('click', select);
  $('an-ablation-summary').addEventListener('click', select);
  $('cmp-run').addEventListener('click', () => guarded(runCompare));
  let resize;
  window.addEventListener('resize', () => {
    clearTimeout(resize);
    resize = setTimeout(() => {
      if (state.tab === 'map') drawMap();
      else if (state.tab === 'analyze') drawAnalysis();
      else if (['token', 'stats'].includes(state.tab)) { state.dirty.add(state.tab); showTab(state.tab); }
    }, 200);
  });
}

wire();
guarded(loadRuns);
