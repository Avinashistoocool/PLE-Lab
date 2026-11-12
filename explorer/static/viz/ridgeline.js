// Ridgeline histograms of slice norms per layer (DESIGN_IDEAS #7). SVG.
// Each ridge: tokens seen in training (filled, layer colour) stacked over unseen (outline/hatched grey).
import { d3, theme, layerColor, tip, tipHtml, fmt, mount, widthOf, onResize, segmented, dur, alpha } from './core.js';

/**
 * data: /api/norm_dist response {edges[B+1], layers:[{layer, seen[B], unseen[B], mean, mean_seen, median, p99,
 *        frac_gt_2x_init}], init_norm, n_seen, n_unseen, has_freq}
 * opts: {height?: 360, scale?: 'log'|'sqrt'|'linear', overlap?: 1.8, onHover?({layer, bin}|null)}
 * returns {update(data), setScale(s), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-ridge', 'Ridgeline of slice norms per layer');
  let scale = opts.scale || 'log';
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  const seg = segmented([{ value: 'log', label: 'log' }, { value: 'sqrt', label: '√' }, { value: 'linear', label: 'linear' }], scale, setScale);
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  const legend = document.createElement('div'); legend.className = 'viz-legend';
  bar.append(seg, readout, legend);
  root.append(bar);
  const H = opts.height || 360;
  const svg = d3.select(root).append('svg').attr('height', H);
  const hatchId = `vz-hatch-${Math.random().toString(36).slice(2)}`;
  const pat = svg.append('defs').append('pattern').attr('id', hatchId).attr('width', 5).attr('height', 5).attr('patternUnits', 'userSpaceOnUse').attr('patternTransform', 'rotate(45)');
  pat.append('line').attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 5).attr('stroke-width', 1.4);
  const gGrid = svg.append('g'), gRidges = svg.append('g'), gRef = svg.append('g');
  const cursor = svg.append('line').attr('pointer-events', 'none').attr('opacity', 0);
  const overlay = svg.append('rect').attr('fill', 'transparent');
  let th = theme(), d = data, first = true, x, y, h, W;

  function tf(c) { return scale === 'log' ? Math.log1p(c) : scale === 'sqrt' ? Math.sqrt(c) : c; }

  function draw(animate) {
    th = theme(); W = widthOf(el); svg.attr('width', W);
    pat.select('line').attr('stroke', alpha(th.muted, 0.55));
    const L = d.layers.length, mL = 44, mR = 14, mT = 22, mB = 30;
    const edges = d.edges, B = edges.length - 1, centers = d3.range(B).map(i => (edges[i] + edges[i + 1]) / 2);
    const xmax = Math.max(edges[B], d.init_norm * 1.2);
    x = d3.scaleLinear().domain([0, xmax]).range([mL, W - mR]);
    y = d3.scaleBand().domain(d3.range(L)).range([mT, H - mB]).paddingInner(0);
    const band = y.bandwidth(), overlap = opts.overlap || 1.8;
    const peak = d3.max(d.layers, r => d3.max(r.seen.map((s, i) => tf(s + r.unseen[i])))) || 1;
    h = d3.scaleLinear().domain([0, peak]).range([0, band * overlap]);
    const t = svg.transition().duration(animate ? dur(750) : 0).ease(d3.easeCubicOut);
    // grid
    gGrid.selectAll('*').remove();
    x.ticks(8).forEach(v => {
      gGrid.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', mT - 10).attr('y2', H - mB).attr('stroke', th.hairline);
      gGrid.append('text').attr('x', x(v)).attr('y', H - mB + 14).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', th.muted).text(v);
    });
    gGrid.append('text').attr('x', W - mR).attr('y', H - 2).attr('text-anchor', 'end').attr('font-size', 10).attr('fill', th.muted).text('slice norm ‖E[t]ₗ·√p‖');
    // ridges: draw from the back (L0 on top row) so lower rows overlap upper ones
    const pts = (r, which) => centers.map((c, i) => [c, which === 'all' ? r.seen[i] + r.unseen[i] : r.seen[i]]);
    const area = base => d3.area().x(p => x(p[0])).y0(base).y1(p => base - h(tf(p[1]))).curve(d3.curveMonotoneX);
    const zero = base => d3.area().x(p => x(p[0])).y0(base).y1(base).curve(d3.curveMonotoneX);
    const g = gRidges.selectAll('g.ridge').data(d.layers, r => r.layer).join(e => {
      const gg = e.append('g').attr('class', 'ridge');
      gg.append('path').attr('class', 'all');
      gg.append('path').attr('class', 'seen');
      gg.append('line').attr('class', 'base');
      gg.append('text').attr('class', 'lab');
      gg.append('line').attr('class', 'mean');
      return gg;
    });
    g.each(function (r) {
      const base = y(r.layer) + band, gg = d3.select(this), c = layerColor(th, r.layer);
      const pa = gg.select('path.all').attr('fill', `url(#${hatchId})`).attr('stroke', alpha(th.muted, 0.7)).attr('stroke-width', 1);
      const ps = gg.select('path.seen').attr('fill', alpha(c, 0.55)).attr('stroke', c).attr('stroke-width', 1.5);
      if (first) { pa.attr('d', zero(base)(pts(r, 'all'))); ps.attr('d', zero(base)(pts(r, 'seen'))); }
      pa.transition(t).delay(animate ? dur(r.layer * 90) : 0).attr('d', area(base)(pts(r, 'all')));
      ps.transition(t).delay(animate ? dur(r.layer * 90) : 0).attr('d', area(base)(pts(r, 'seen')));
      gg.select('line.base').attr('x1', mL).attr('x2', W - mR).attr('y1', base).attr('y2', base).attr('stroke', alpha(c, 0.35));
      gg.select('text.lab').attr('x', mL - 8).attr('y', base - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('font-weight', 600).attr('fill', c).text(`L${r.layer}`);
      const ms = r.mean_seen ?? r.mean;
      gg.select('line.mean').attr('x1', x(ms)).attr('x2', x(ms)).attr('y1', base).attr('y2', base - band * 0.8)
        .attr('stroke', th.text).attr('stroke-width', 1.5).attr('stroke-linecap', 'round');
    });
    first = false;
    // init reference
    gRef.selectAll('*').remove();
    gRef.append('line').attr('x1', x(d.init_norm)).attr('x2', x(d.init_norm)).attr('y1', mT - 12).attr('y2', H - mB)
      .attr('stroke', th.ple).attr('stroke-dasharray', '4 4').attr('stroke-width', 1.2);
    gRef.append('text').attr('x', x(d.init_norm) + 4).attr('y', mT - 6).attr('font-size', 10).attr('fill', th.ple).text(`init ≈ ${fmt(d.init_norm, 2)}`);
    cursor.attr('y1', mT - 10).attr('y2', H - mB).attr('stroke', th.accent).attr('stroke-dasharray', '2 3');
    overlay.attr('x', mL).attr('y', mT).attr('width', W - mL - mR).attr('height', H - mT - mB);
    legend.innerHTML = `<span><i style="background:${alpha(th.layers[1], 0.6)};border:1px solid ${th.layers[1]}"></i>seen in train (${d.n_seen.toLocaleString()})</span>`
      + `<span><i style="background:repeating-linear-gradient(45deg,${th.muted} 0 1px,transparent 1px 4px);border:1px solid ${th.muted}"></i>never seen (${d.n_unseen.toLocaleString()})</span>`
      + `<span><i style="width:2px;background:${th.text}"></i>mean (seen)</span>`;
    root.setAttribute('aria-label', `Slice norm distributions per layer; init norm ${fmt(d.init_norm, 2)}; ${d.n_unseen} tokens never seen in training`);
  }

  overlay.on('pointermove', ev => {
    const [px, py] = d3.pointer(ev);
    const B = d.edges.length - 1, v = x.invert(px);
    const bin = Math.max(0, Math.min(B - 1, d3.bisectRight(d.edges, v) - 1));
    // which ridge: the row whose baseline is the first one below the pointer
    const L = d.layers.length;
    let l = Math.floor((py - y.range()[0]) / y.bandwidth());
    l = Math.max(0, Math.min(L - 1, l));
    const r = d.layers[l];
    cursor.attr('opacity', 1).attr('x1', px).attr('x2', px);
    gRidges.selectAll('g.ridge').attr('opacity', q => (q.layer === l ? 1 : 0.45));
    readout.innerHTML = `L${l} · norm ${fmt(d.edges[bin], 2)}–${fmt(d.edges[bin + 1], 2)} · <b>${(r.seen[bin] + r.unseen[bin]).toLocaleString()}</b> tokens`;
    tip.show(tipHtml(`L${l}<small>‖slice‖ ${fmt(d.edges[bin], 2)}–${fmt(d.edges[bin + 1], 2)}</small>`, [
      ['seen in train', r.seen[bin].toLocaleString(), layerColor(th, l)],
      ['never seen', r.unseen[bin].toLocaleString(), th.muted],
      ['layer mean (all / seen)', `${fmt(r.mean, 3)} / ${fmt(r.mean_seen, 3)}`],
      ['median · p99', `${fmt(r.median, 3)} · ${fmt(r.p99, 3)}`],
      ['> 2× init', `${fmt(r.frac_gt_2x_init * 100, 2)}%`],
    ], `init norm ≈ ${fmt(d.init_norm, 3)} (dashed)`), ev);
    opts.onHover?.({ layer: l, bin });
  }).on('pointerleave', () => {
    cursor.attr('opacity', 0); tip.hide(); readout.textContent = '';
    gRidges.selectAll('g.ridge').attr('opacity', 1); opts.onHover?.(null);
  });

  function setScale(s) { scale = s; seg.set(s); draw(true); }
  draw(true);
  const off = onResize(el, () => draw(false));
  return { update(nd) { d = nd; draw(true); }, setScale, destroy() { off(); tip.hide(); el.innerHTML = ''; } };
}
