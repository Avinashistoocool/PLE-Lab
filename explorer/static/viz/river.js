// Gate river (DESIGN_IDEAS #5): streamgraph of per-layer gate magnitude along the token sequence. SVG.
import { d3, theme, layerColor, tip, tipHtml, esc, vis, fmt, mount, widthOf, onResize, segmented, dur, alpha } from './core.js';

const MODES = {
  gate: { label: 'gate', get: (a, i, l) => a.gate[i][l], fmt: v => fmt(v, 3), what: 'mean |gate|' },
  ratio: { label: 'contrib', get: (a, i, l) => a.contrib_ratio[i][l], fmt: v => `${fmt(v * 100, 1)}%`, what: '‖PLE update‖ / ‖h‖' },
  effective: { label: 'gate × contrib', get: (a, i, l) => a.gate[i][l] * a.contrib_ratio[i][l], fmt: v => fmt(v, 4), what: 'gate × ratio' },
};

/**
 * data: /api/analyze response {tokens[{id,piece}], gate[T][L], contrib_ratio[T][L], loss[T], ablation?}
 * opts: {height?: 260, mode?: 'gate'|'ratio'|'effective', offset?: 'wiggle'|'zero', controls?: true,
 *        onHover?(i|null), onClick?(i)}
 * returns {update(data), highlight(i|null), setMode(m), setOffset(o), destroy()}
 * Link to chips: river.onHover → chips.highlight(i); chips.onHover → river.highlight(i).
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-river', 'Gate river: per-layer gate along the text');
  let mode = opts.mode || 'gate', offset = opts.offset || 'wiggle';
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  const segM = segmented(Object.entries(MODES).map(([k, m]) => ({ value: k, label: m.label, title: m.what })), mode, setMode);
  const segO = segmented([{ value: 'wiggle', label: 'stream' }, { value: 'zero', label: 'stacked' }], offset, setOffset);
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  const legend = document.createElement('div'); legend.className = 'viz-legend';
  if (opts.controls !== false) bar.append(segM, segO);
  bar.append(readout, legend);
  root.append(bar);
  const H = opts.height || 260, mB = 54, mT = 8;
  const svg = d3.select(root).append('svg').attr('height', H);
  const clipId = `vz-clip-${Math.random().toString(36).slice(2)}`;
  const clipRect = svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('height', H).attr('width', 0);
  const gBands = svg.append('g').attr('clip-path', `url(#${clipId})`);
  const gAxis = svg.append('g');
  const cursor = svg.append('g').attr('pointer-events', 'none').attr('opacity', 0);
  cursor.append('line').attr('class', 'cur').attr('y1', mT).attr('y2', H - mB + 4);
  const dots = cursor.append('g');
  const overlay = svg.append('rect').attr('fill', 'transparent').attr('y', mT).attr('height', H - mB - mT);
  let th = theme(), a = data, W = 0, x = null, series = null, solo = null, hot = null, first = true;

  function stackData() {
    const L = a.n_layer ?? a.gate[0].length, T = a.tokens.length, m = MODES[mode];
    const rows = d3.range(T).map(i => Object.fromEntries(d3.range(L).map(l => [l, m.get(a, i, l)])));
    return d3.stack().keys(d3.range(L))
      .offset(offset === 'wiggle' ? d3.stackOffsetWiggle : d3.stackOffsetNone)
      .order(offset === 'wiggle' ? d3.stackOrderInsideOut : d3.stackOrderNone)(rows);
  }

  function draw(animate) {
    th = theme(); W = widthOf(el);
    svg.attr('width', W);
    const T = a.tokens.length, L = a.gate[0].length, mL = 8, mR = 8;
    x = d3.scalePoint().domain(d3.range(T)).range([mL + 6, W - mR - 6]);
    series = stackData();
    const lo = d3.min(series, s => d3.min(s, d => d[0])), hi = d3.max(series, s => d3.max(s, d => d[1]));
    const y = d3.scaleLinear().domain([lo, hi]).range([H - mB, mT + 4]);
    const area = d3.area().x((d, i) => x(i)).y0(d => y(d[0])).y1(d => y(d[1])).curve(d3.curveCatmullRom.alpha(0.5));
    const t = svg.transition().duration(animate ? dur(700) : 0).ease(d3.easeCubicInOut);
    gBands.selectAll('path.band').data(series, s => s.key).join(
      e => e.append('path').attr('class', 'band').attr('d', area),
    ).attr('fill', s => layerColor(th, s.key)).attr('fill-opacity', 0.82)
      .attr('stroke', s => alpha(layerColor(th, s.key), 1)).attr('stroke-width', 0.6)
      .classed('on', s => solo === s.key)
      .transition(t).attr('d', area);
    if (first) {  // reveal left→right
      clipRect.attr('width', 0).transition().duration(dur(1100)).ease(d3.easeCubicOut).attr('width', W);
      first = false;
    } else clipRect.attr('width', W);
    // token labels (rotated if crowded)
    const step = (x.range()[1] - x.range()[0]) / Math.max(1, T - 1);
    const rot = step < 34;
    const every = Math.max(1, Math.ceil(11 / step));
    gAxis.selectAll('text.tk').data(a.tokens.map((tk, i) => ({ tk, i })), d => d.i).join('text').attr('class', 'tk')
      .attr('font-size', 11).attr('fill', th.muted)
      .attr('transform', d => `translate(${x(d.i)},${H - mB + 10}) rotate(${rot ? -50 : 0})`)
      .attr('text-anchor', rot ? 'end' : 'middle').attr('dominant-baseline', rot ? 'central' : 'hanging')
      .text(d => (d.i % every === 0 ? clip(vis(d.tk.piece), 9) : ''))
      .style('cursor', 'pointer');
    gAxis.selectAll('line.base').data([0]).join('line').attr('class', 'base')
      .attr('x1', x.range()[0]).attr('x2', x.range()[1]).attr('y1', H - mB + 4).attr('y2', H - mB + 4).attr('stroke', th.hairline);
    cursor.select('line.cur').attr('stroke', th.accent).attr('stroke-width', 1).attr('stroke-dasharray', '2 3');
    overlay.attr('x', 0).attr('width', W);
    legend.innerHTML = d3.range(L).map(l => `<span data-l="${l}" class="${solo != null && solo !== l ? 'off' : ''}"><i style="background:${layerColor(th, l)}"></i>L${l}</span>`).join('');
    legend.querySelectorAll('[data-l]').forEach(s => s.addEventListener('click', () => { const l = +s.dataset.l; solo = solo === l ? null : l; root.classList.toggle('solo', solo != null); draw(false); }));
    root.setAttribute('aria-label', `Gate river over ${T} tokens and ${L} layers, mode ${MODES[mode].label}`);
    placeCursor();
    draw.y = y;
  }

  function nearest(px) {
    const dom = x.domain(), step = x.step();
    return Math.max(0, Math.min(dom.length - 1, Math.round((px - x.range()[0]) / (step || 1))));
  }

  function placeCursor() {
    if (hot == null || !x || !series) { cursor.attr('opacity', 0); gAxis.selectAll('text.tk').attr('fill', th.muted).attr('font-weight', null); return; }
    const y = draw.y, cx = x(hot);
    cursor.attr('opacity', 1).select('line.cur').attr('x1', cx).attr('x2', cx);
    dots.selectAll('circle').data(series, s => s.key).join('circle').attr('r', 3).attr('cx', cx)
      .attr('cy', s => y((s[hot][0] + s[hot][1]) / 2)).attr('fill', s => layerColor(th, s.key)).attr('stroke', th.bg);
    gAxis.selectAll('text.tk').attr('fill', d => (d.i === hot ? th.accent : th.muted)).attr('font-weight', d => (d.i === hot ? 700 : null))
      .text(d => {
        const step = x.step(), every = Math.max(1, Math.ceil(11 / step));
        return d.i % every === 0 || d.i === hot ? clip(vis(d.tk.piece), 9) : '';
      });
    const m = MODES[mode];
    readout.innerHTML = `<b>${esc(vis(a.tokens[hot].piece))}</b> @${hot} · ` + d3.range(a.gate[0].length).map(l => `<span style="color:${layerColor(th, l)}">L${l}</span> ${m.fmt(m.get(a, hot, l))}`).join(' ');
  }

  function tipFor(ev, i) {
    const m = MODES[mode], L = a.gate[0].length;
    const vals = d3.range(L).map(l => m.get(a, i, l));
    const best = d3.maxIndex(vals);
    const rows = d3.range(L).map(l => [`L${l}${l === best ? ' ◀' : ''}`, m.fmt(vals[l]), layerColor(th, l)]);
    let foot = `${m.what}`;
    if (a.loss[i] != null) foot += ` · loss ${fmt(a.loss[i])}`;
    if (a.ablation && a.ablation.delta[i] != null) foot += ` · Δablate ${a.ablation.delta[i] >= 0 ? '+' : ''}${fmt(a.ablation.delta[i])}`;
    tip.show(tipHtml(`${esc(vis(a.tokens[i].piece))}<small>pos ${i} · #${a.tokens[i].id}</small>`, rows, foot), ev);
  }

  overlay.on('pointermove', ev => {
    const [px] = d3.pointer(ev);
    const i = nearest(px);
    if (i !== hot) { hot = i; placeCursor(); opts.onHover?.(i); }
    tipFor(ev, i);
  }).on('pointerleave', () => { hot = null; placeCursor(); tip.hide(); readout.textContent = ''; opts.onHover?.(null); })
    .on('click', ev => { const i = nearest(d3.pointer(ev)[0]); opts.onClick?.(i); })
    .style('cursor', 'crosshair');

  function setMode(m) { mode = m; segM.set(m); draw(true); }
  function setOffset(o) { offset = o; segO.set(o); draw(true); }

  draw(true);
  const off = onResize(el, () => draw(false));
  return {
    update(d) { a = d; hot = null; draw(true); },
    highlight(i) { hot = i; placeCursor(); },
    setMode, setOffset,
    destroy() { off(); tip.hide(); el.innerHTML = ''; },
  };
}

const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
