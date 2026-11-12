// Animated L×p heatmap (DESIGN_IDEAS #16) and interactive L×L layer-similarity matrix (#18-lite).
import { d3, theme, layerColor, diverging, tween, tip, tipHtml, esc, vis, fmt, mount, hidpiCanvas, widthOf, onResize, segmented, alpha, dur } from './core.js';

/**
 * Token heatmap: rows = layers, columns = dims, diverging colour; values cross-fade on update();
 * crosshair hover with row/column highlight; optional "sort dims by |value|" (columns slide).
 * data: /api/token response {id, piece, vectors[L][p], norms[L], downsampled?}
 * opts: {cellH?: 26, vmax?: fixed scale, controls?: true (show sort toggle),
 *        onHover?({layer, dim, value}|null), onClick?({layer, dim, value})}
 * returns {update(data), highlight(layer|null, dim?), setSort(bool), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-heat2', 'Per-layer vector heatmap');
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  let sorted = false;
  if (opts.controls !== false) {
    bar.append(segmented([{ value: 'dim', label: 'dim order' }, { value: 'abs', label: 'sort by |v|' }], 'dim', v => setSort(v === 'abs')));
  }
  bar.append(readout);
  root.append(bar);
  const canvas = document.createElement('canvas');
  root.append(canvas);
  let th = theme(), div = diverging(th);
  let w = 0, h = 0, ctx, box, cur = null, shown = null, colPos = null, colFrom = null, kc = 1, hot = null, ext = null;
  let cancelV = () => {}, cancelC = () => {};
  const cellH = opts.cellH || 26;

  function layout() {
    if (!cur) return;
    const L = cur.vectors.length;
    w = widthOf(el); h = L * cellH + 36;
    ctx = hidpiCanvas(canvas, w, h);
    box = { l: 40, r: w - 8, t: 4, b: 4 + L * cellH };
  }

  function order(d) {
    const p = d.vectors[0].length;
    const idx = d3.range(p);
    if (!sorted) return idx.map((_, j) => j);
    const energy = idx.map(j => d.vectors.reduce((s, r) => s + Math.abs(r[j]), 0));
    const rank = idx.slice().sort((a, b) => energy[b] - energy[a]);
    const pos = new Array(p);
    rank.forEach((j, r) => { pos[j] = r; });
    return pos;
  }

  function vmaxOf(d) {
    const flat = d.vectors.flat().map(Math.abs).sort((a, b) => a - b);
    return opts.vmax || flat[Math.floor(flat.length * 0.99)] || 1;
  }

  function draw() {
    if (!shown) return;
    const L = shown.length, p = shown[0].length, cw = (box.r - box.l) / p;
    ctx.clearRect(0, 0, w, h);
    const focus = hot ?? ext;
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < p; j++) {
        const x = box.l + colAt(j) * cw, y = box.t + l * cellH;
        ctx.fillStyle = div(shown[l][j]);
        ctx.fillRect(x + 0.5, y + 0.5, Math.max(1, cw - 1), cellH - 1);
      }
      ctx.fillStyle = focus && focus.layer === l ? th.text : layerColor(th, l);
      ctx.font = `600 11px ${th.fontMono}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(`L${l}`, box.l - 8, box.t + (l + 0.5) * cellH);
      ctx.fillStyle = layerColor(th, l);
      ctx.fillRect(box.l - 4, box.t + l * cellH + 3, 2, cellH - 6);
    }
    if (focus) {
      const { layer, dim } = focus;
      ctx.fillStyle = alpha(th.bg, 0.5);
      for (let l = 0; l < L; l++) if (l !== layer) ctx.fillRect(box.l, box.t + l * cellH, box.r - box.l, cellH);
      ctx.strokeStyle = th.accent; ctx.lineWidth = 1;
      ctx.strokeRect(box.l + 0.5, box.t + layer * cellH + 0.5, box.r - box.l - 1, cellH - 1);
      if (dim != null) {
        const x = box.l + colAt(dim) * cw;
        ctx.strokeStyle = alpha(th.accent, 0.55);
        ctx.strokeRect(x + 0.5, box.t + 0.5, cw - 1, L * cellH - 1);
        ctx.lineWidth = 2; ctx.strokeStyle = th.accent;
        ctx.strokeRect(x, box.t + layer * cellH, cw, cellH);
        ctx.lineWidth = 1;
      }
    }
    // dim ticks
    ctx.fillStyle = th.muted; ctx.font = `10px ${th.fontMono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const every = Math.max(1, Math.ceil(p / Math.floor((box.r - box.l) / 30)));
    if (!sorted) for (let j = 0; j < p; j += every) ctx.fillText(String(j), box.l + (j + 0.5) * cw, box.b + 5);
    else ctx.fillText('dims sorted by Σ|v| across layers →', (box.l + box.r) / 2, box.b + 5);
    // legend
    const lw = 120, lx = box.r - lw, ly = box.b + 22;
    for (let i = 0; i < lw; i++) { ctx.fillStyle = div((i / lw) * 2 - 1); ctx.fillRect(lx + i, ly, 1, 6); }
    ctx.fillStyle = th.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`−${fmt(cur._vmax, 2)}`, lx - 4, ly + 3);
    ctx.textAlign = 'left'; ctx.fillText(`+${fmt(cur._vmax, 2)}`, box.r + 2 > w - 30 ? lx + lw + 2 : box.r + 2, ly + 3);
  }

  function colAt(j) {
    if (!colFrom || kc >= 1) return colPos[j];
    return colFrom[j] + (colPos[j] - colFrom[j]) * kc;
  }

  function setSort(s) {
    sorted = s;
    colFrom = colPos.slice();
    colPos = order(cur);
    kc = 0;
    cancelC();
    cancelC = tween(650, t => { kc = t; draw(); }, d3.easeCubicInOut);
  }

  function hitTest(ev) {
    if (!shown) return null;
    const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    const L = shown.length, p = shown[0].length, cw = (box.r - box.l) / p;
    if (x < box.l || x > box.r || y < box.t || y > box.b) return null;
    const c = Math.min(p - 1, Math.floor((x - box.l) / cw)), l = Math.min(L - 1, Math.floor((y - box.t) / cellH));
    const j = colPos.indexOf(c);
    return { layer: l, dim: j, value: cur.vectors[l][j] };
  }

  const onMove = ev => {
    const hh = hitTest(ev);
    if (hh?.layer !== hot?.layer || hh?.dim !== hot?.dim) { hot = hh; draw(); opts.onHover?.(hh); }
    if (!hh) { tip.hide(); readout.innerHTML = ''; return; }
    const d = cur;
    const rowVals = d.vectors[hh.layer], rank = rowVals.map(Math.abs).filter(v => v > Math.abs(hh.value)).length + 1;
    readout.innerHTML = `L${hh.layer} · dim ${hh.dim} · <b>${hh.value >= 0 ? '+' : ''}${fmt(hh.value, 4)}</b>`;
    tip.show(tipHtml(`${esc(vis(d.piece))}<small>#${d.id}</small>`, [
      ['layer', `L${hh.layer}`, layerColor(th, hh.layer)],
      ['dim', `${hh.dim}${d.downsampled ? ' (pooled)' : ''}`],
      ['value', `${hh.value >= 0 ? '+' : ''}${fmt(hh.value, 4)}`, div(hh.value / d._vmax)],
      ['|v| rank in slice', `${rank} / ${rowVals.length}`],
      ['‖slice‖', fmt(d.norms[hh.layer], 3)],
    ]), ev);
  };
  const onLeave = () => { hot = null; tip.hide(); readout.innerHTML = ''; draw(); opts.onHover?.(null); };
  const onClick = ev => { const hh = hitTest(ev); if (hh) opts.onClick?.(hh); };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', onLeave);
  canvas.addEventListener('click', onClick);
  canvas.style.cursor = 'crosshair';

  function update(d) {
    th = theme(); div = diverging(th);
    const vmax = vmaxOf(d);
    d = { ...d, _vmax: vmax };
    const target = d.vectors.map(r => r.map(v => v / vmax));
    const start = shown && shown.length === target.length && shown[0].length === target[0].length
      ? shown.map(r => r.slice()) : target.map(r => r.map(() => 0));
    const shapeChanged = !cur || cur.vectors.length !== d.vectors.length;
    cur = d;
    if (shapeChanged) layout();
    colFrom = colPos; colPos = order(d);
    if (!colFrom || colFrom.length !== colPos.length) colFrom = colPos.slice();
    root.setAttribute('aria-label', `Heatmap of ${d.piece}: ${d.vectors.length} layers × ${d.vectors[0].length} dims`);
    cancelV();
    cancelV = tween(600, t => {
      kc = t;
      shown = target.map((r, l) => r.map((v, j) => start[l][j] + (v - start[l][j]) * t));
      draw();
    }, d3.easeCubicOut);
  }

  update(data);
  const off = onResize(el, () => { layout(); draw(); });
  return {
    update,
    setSort,
    highlight(layer, dim = null) { ext = layer == null ? null : { layer, dim }; draw(); },
    destroy() { cancelV(); cancelC(); off(); tip.hide(); el.innerHTML = ''; },
  };
}

/**
 * Layer-similarity matrix (L×L cosine), SVG. Toggle between this token and the global average;
 * cells tween colour/size; hovering a cell highlights its row and column and both layer labels.
 * data: /api/layer_similarity response {matrix?[L][L], global[L][L], global_n, piece?, id?}
 * opts: {size?: px (default min(width, 360)), mode?: 'token'|'global'|'diff', onHover?({a, b, value}|null)}
 * returns {update(data), setMode(mode), destroy()}
 */
export function similarity(el, data, opts = {}) {
  const root = mount(el, 'viz-sim', 'Layer similarity matrix');
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  let mode = opts.mode || (data.matrix ? 'token' : 'global');
  const seg = segmented([
    { value: 'token', label: 'token', title: 'cosine between this token\'s slices' },
    { value: 'global', label: 'global', title: 'average over the most frequent tokens' },
    { value: 'diff', label: 'Δ', title: 'token − global' },
  ], mode, v => setMode(v));
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  bar.append(seg, readout);
  root.append(bar);
  const svg = d3.select(root).append('svg');
  const g = svg.append('g');
  let th = theme(), div = diverging(th), cur = data;

  function matrixOf() {
    if (mode === 'token' && cur.matrix) return cur.matrix;
    if (mode === 'diff' && cur.matrix) return cur.matrix.map((r, i) => r.map((v, j) => v - cur.global[i][j]));
    return cur.global;
  }

  function draw(animate = true) {
    th = theme(); div = diverging(th);
    const m = matrixOf(), L = m.length;
    const W = Math.min(opts.size || 360, widthOf(el)), pad = 30, cs = (W - pad - 4) / L;
    svg.attr('width', W).attr('height', pad + cs * L + 4);
    const cells = m.flatMap((r, i) => r.map((v, j) => ({ i, j, v })));
    const t = svg.transition().duration(animate ? dur(550) : 0).ease(d3.easeCubicOut);
    const sel = g.selectAll('g.cell').data(cells, d => `${d.i}-${d.j}`).join(enter => {
      const c = enter.append('g').attr('class', 'cell');
      c.append('rect').attr('rx', 3).attr('width', 0).attr('height', 0);
      c.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'central').attr('font-size', 10);
      return c;
    });
    sel.attr('transform', d => `translate(${pad + d.j * cs},${pad + d.i * cs})`);
    sel.select('rect').transition(t)
      .attr('x', d => (cs - 2) * (1 - sz(d)) / 2 + 1).attr('y', d => (cs - 2) * (1 - sz(d)) / 2 + 1)
      .attr('width', d => (cs - 2) * sz(d)).attr('height', d => (cs - 2) * sz(d))
      .attr('fill', d => (d.i === d.j && mode !== 'diff' ? th.surface3 : div(d.v)));
    sel.select('text').attr('x', cs / 2).attr('y', cs / 2)
      .attr('fill', d => (Math.abs(d.v) > 0.55 && d.i !== d.j ? '#0b0f17' : th.muted))
      .text(d => (cs > 34 ? (d.i === d.j && mode !== 'diff' ? '' : fmtS(d.v)) : ''));
    const labs = (cls, xy) => g.selectAll(`text.${cls}`).data(d3.range(L)).join('text').attr('class', cls)
      .attr('font-size', 11).attr('font-weight', 600).attr('fill', l => layerColor(th, l)).text(l => `L${l}`).each(xy);
    labs('rl', function (l) { d3.select(this).attr('x', pad - 6).attr('y', pad + (l + 0.5) * cs).attr('text-anchor', 'end').attr('dominant-baseline', 'central'); });
    labs('cl', function (l) { d3.select(this).attr('x', pad + (l + 0.5) * cs).attr('y', pad - 8).attr('text-anchor', 'middle'); });
    sel.on('pointerenter', (ev, d) => focus(d)).on('pointermove', (ev, d) => showTip(ev, d)).on('pointerleave', () => focus(null));
    function sz(d) { return d.i === d.j && mode !== 'diff' ? 0.35 : 0.45 + 0.55 * Math.min(1, Math.abs(d.v)); }
  }

  function focus(d) {
    g.selectAll('g.cell').attr('opacity', c => (!d || c.i === d.i || c.j === d.j ? 1 : 0.25));
    g.selectAll('text.rl').attr('opacity', l => (!d || l === d.i ? 1 : 0.35));
    g.selectAll('text.cl').attr('opacity', l => (!d || l === d.j ? 1 : 0.35));
    if (!d) { tip.hide(); readout.textContent = ''; }
    else readout.innerHTML = `cos(L${d.i}, L${d.j}) = <b>${fmtS(d.v)}</b>`;
    opts.onHover?.(d ? { a: d.i, b: d.j, value: d.v } : null);
  }
  function showTip(ev, d) {
    const name = cur.piece != null ? `${esc(vis(cur.piece))}<small>${mode}</small>` : 'global';
    const rows = [[`cos(L${d.i}, L${d.j})`, fmtS(d.v), div(d.v)]];
    if (cur.matrix) rows.push(['this token', fmtS(cur.matrix[d.i][d.j])], [`global (top ${cur.global_n})`, fmtS(cur.global[d.i][d.j])]);
    tip.show(tipHtml(name, rows, 'cosine between the two layer slices; ≈0 = unrelated subspaces'), ev);
  }

  function setMode(m) { mode = m; seg.set(m); draw(true); }
  draw(true);
  const off = onResize(el, () => draw(false));
  return {
    update(d) { cur = d; if (!d.matrix && mode !== 'global') mode = 'global'; seg.set(mode); draw(true); },
    setMode,
    destroy() { off(); tip.hide(); el.innerHTML = ''; },
  };
}

const fmtS = v => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
