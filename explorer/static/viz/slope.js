// Compare: slope chart (baseline → PLE loss per token) + diverging Δ bar strip, linked hover (DESIGN_IDEAS #8). SVG.
import { d3, theme, diverging, tip, tipHtml, esc, vis, fmt, mount, widthOf, onResize, segmented, dur, alpha } from './core.js';

/**
 * data: /api/compare response {tokens, baseline:{run, loss[T], mean_loss}, ple:{run, loss[T], mean_loss}, diff[T], mean_diff}
 *       diff = baseline − PLE (positive = PLE better).
 * opts: {height?: 380, order?: 'seq'|'sorted', onHover?(i|null), onClick?(i)}
 * returns {update(data), highlight(i|null), setOrder(o), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-slope', 'Baseline versus PLE loss per token');
  let order = opts.order || 'sorted';
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  const seg = segmented([{ value: 'seq', label: 'text order' }, { value: 'sorted', label: 'sorted by Δ' }], order, setOrder);
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  const legend = document.createElement('div'); legend.className = 'viz-legend';
  bar.append(seg, readout, legend);
  root.append(bar);
  const H = opts.height || 380;
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;gap:14px;align-items:flex-start';
  root.append(wrap);
  const svgS = d3.select(wrap).append('svg').attr('height', H);
  const svgB = d3.select(wrap).append('svg').attr('height', H);
  const gS = svgS.append('g'), gSl = svgS.append('g'), gSd = svgS.append('g'), gSt = svgS.append('g');
  const gB = svgB.append('g'), gBb = svgB.append('g'), gBl = svgB.append('g'), gBx = svgB.append('g');
  let th = theme(), div = diverging(th), c = data, rows = [], hot = null, dmax = 1;

  function prep() {
    rows = c.tokens.map((t, i) => ({ i, t, b: c.baseline.loss[i], p: c.ple.loss[i], d: c.diff[i] })).filter(r => r.b != null && r.p != null);
    dmax = d3.max(rows, r => Math.abs(r.d)) || 1e-6;
  }

  function draw(animate) {
    th = theme(); div = diverging(th);
    const W = widthOf(el), wS = Math.max(240, Math.min(420, Math.floor(W * 0.38))), wB = Math.max(200, W - wS - 14);
    svgS.attr('width', wS); svgB.attr('width', wB);
    const t = d3.transition().duration(animate ? dur(650) : 0).ease(d3.easeCubicInOut);
    // ---- slope
    const mT = 26, mB = 18, xL = 58, xR = wS - 58;
    const lmax = d3.max(rows, r => Math.max(r.b, r.p)) || 1;
    const y = d3.scaleLinear().domain([0, lmax * 1.04]).nice().range([H - mB, mT]);
    gS.selectAll('*').remove();
    for (const [xx, name, mean] of [[xL, c.baseline.run, c.baseline.mean_loss], [xR, c.ple.run, c.ple.mean_loss]]) {
      gS.append('line').attr('x1', xx).attr('x2', xx).attr('y1', mT).attr('y2', H - mB).attr('stroke', th.hairlineStrong);
      gS.append('text').attr('x', xx).attr('y', 10).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600).attr('fill', xx === xR ? th.ple : th.ctx).text(name);
      gS.append('path').attr('d', `M${xx - 6},${y(mean)}h12`).attr('stroke', th.text).attr('stroke-width', 2);
    }
    y.ticks(5).forEach(v => {
      gS.append('text').attr('x', xL - 8).attr('y', y(v)).attr('text-anchor', 'end').attr('dominant-baseline', 'central').attr('font-size', 10).attr('fill', th.muted).text(v);
      gS.append('line').attr('x1', xL - 3).attr('x2', xL).attr('y1', y(v)).attr('y2', y(v)).attr('stroke', th.muted);
    });
    gS.append('line').attr('x1', xL).attr('x2', xR).attr('y1', y(c.baseline.mean_loss)).attr('y2', y(c.ple.mean_loss))
      .attr('stroke', th.text).attr('stroke-width', 2.5).attr('stroke-dasharray', '6 3').attr('opacity', 0.9);
    gS.append('text').attr('x', (xL + xR) / 2).attr('y', H - 2).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', th.muted).text('per-token loss (nats) · dashed = mean');
    const sorted = rows.slice().sort((a, b) => Math.abs(a.d) - Math.abs(b.d));  // biggest swings on top
    gSl.selectAll('line').data(sorted, r => r.i).join(e => e.append('line').attr('x1', xL).attr('x2', xL).attr('y1', r => y(r.b)).attr('y2', r => y(r.b)))
      .attr('stroke', r => div(r.d / dmax)).attr('stroke-width', r => 1 + 1.5 * Math.abs(r.d) / dmax).attr('stroke-linecap', 'round')
      .attr('stroke-opacity', r => 0.25 + 0.75 * Math.abs(r.d) / dmax)
      .transition(t).delay(r => (animate ? dur(r.i * 6) : 0))
      .attr('x1', xL).attr('y1', r => y(r.b)).attr('x2', xR).attr('y2', r => y(r.p));
    gSd.selectAll('circle').data(rows.flatMap(r => [{ r, s: 'b' }, { r, s: 'p' }]), q => `${q.r.i}${q.s}`).join('circle')
      .attr('r', 2.2).attr('cx', q => (q.s === 'b' ? xL : xR)).attr('cy', q => y(q.s === 'b' ? q.r.b : q.r.p))
      .attr('fill', q => div(q.r.d / dmax)).attr('opacity', 0.9);
    gSt.selectAll('*').remove();
    // hit lines (wide, invisible)
    gSt.selectAll('line').data(rows, r => r.i).join('line').attr('x1', xL).attr('x2', xR).attr('y1', r => y(r.b)).attr('y2', r => y(r.p))
      .attr('stroke', 'transparent').attr('stroke-width', 8).style('cursor', 'pointer')
      .on('pointerenter', (ev, r) => setHot(r.i, true)).on('pointermove', (ev, r) => showTip(ev, r.i))
      .on('pointerleave', () => setHot(null, true)).on('click', (ev, r) => opts.onClick?.(r.i));
    draw.y = y; draw.xL = xL; draw.xR = xR;

    // ---- diverging bars
    const bm = { l: 74, r: 30, t: 8, b: 24 };
    const ord = order === 'sorted' ? rows.slice().sort((a, b) => b.d - a.d) : rows.slice();
    const yb = d3.scaleBand().domain(ord.map(r => r.i)).range([bm.t, H - bm.b]).paddingInner(0.18);
    const xb = d3.scaleLinear().domain([-dmax, dmax]).range([bm.l, wB - bm.r]);
    gB.selectAll('*').remove();
    xb.ticks(5).forEach(v => {
      gB.append('line').attr('x1', xb(v)).attr('x2', xb(v)).attr('y1', bm.t).attr('y2', H - bm.b).attr('stroke', v === 0 ? th.hairlineStrong : th.hairline);
      gB.append('text').attr('x', xb(v)).attr('y', H - bm.b + 12).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', th.muted).text(v > 0 ? `+${v}` : v);
    });
    gB.append('text').attr('x', bm.l).attr('y', H - 1).attr('font-size', 10).attr('fill', th.neg).text('◀ baseline better');
    gB.append('text').attr('x', wB - bm.r).attr('y', H - 1).attr('text-anchor', 'end').attr('font-size', 10).attr('fill', th.pos).text('PLE better ▶');
    const bh = yb.bandwidth();
    gBb.selectAll('rect').data(rows, r => r.i).join(e => e.append('rect').attr('x', xb(0)).attr('width', 0).attr('y', r => yb(r.i)).attr('rx', 1.5))
      .attr('height', bh).attr('fill', r => div(r.d / dmax))
      .transition(t)
      .attr('y', r => yb(r.i)).attr('x', r => Math.min(xb(0), xb(r.d))).attr('width', r => Math.abs(xb(r.d) - xb(0)));
    const showLab = bh >= 7;
    gBl.selectAll('text').data(rows, r => r.i).join('text')
      .attr('x', bm.l - 6).attr('text-anchor', 'end').attr('dominant-baseline', 'central').attr('font-size', Math.min(11, bh)).attr('fill', th.muted)
      .text(r => (showLab ? clip(vis(r.t.piece), 9) : ''))
      .transition(t).attr('y', r => yb(r.i) + bh / 2);
    // running mean line in sorted order
    gBx.selectAll('*').remove();
    if (order === 'seq') {
      let s = 0;
      const pts = ord.map((r, k) => { s += r.d; return [xb(s / (k + 1)), yb(r.i) + bh / 2]; });
      gBx.append('path').attr('d', d3.line().curve(d3.curveMonotoneY)(pts)).attr('fill', 'none').attr('stroke', th.text).attr('stroke-width', 1.5).attr('opacity', 0.8);
      gBx.append('text').attr('x', pts[pts.length - 1][0] + 4).attr('y', pts[pts.length - 1][1]).attr('font-size', 10).attr('fill', th.text).attr('dominant-baseline', 'central').text('running mean');
    }
    gBx.append('rect').attr('x', 0).attr('y', bm.t).attr('width', wB).attr('height', H - bm.t - bm.b).attr('fill', 'transparent')
      .on('pointermove', ev => {
        const [, py] = d3.pointer(ev);
        const k = Math.max(0, Math.min(ord.length - 1, Math.floor((py - bm.t) / yb.step())));
        const i = ord[k].i;
        if (i !== hot) setHot(i, true);
        showTip(ev, i);
      }).on('pointerleave', () => setHot(null, true)).on('click', () => hot != null && opts.onClick?.(hot));
    draw.yb = yb; draw.xb = xb;
    const better = rows.filter(r => r.d > 0).length;
    legend.innerHTML = `<span>baseline better</span><span class="ramp" style="background:linear-gradient(90deg,${th.neg},${th.zero},${th.pos})"></span><span>PLE better</span>`;
    readout.innerHTML = `mean Δ <b>${c.mean_diff >= 0 ? '+' : ''}${fmt(c.mean_diff, 4)}</b> · PLE better on <b>${better}</b>/${rows.length} tokens`;
    root.setAttribute('aria-label', `Per-token loss: ${c.baseline.run} ${fmt(c.baseline.mean_loss)} vs ${c.ple.run} ${fmt(c.ple.mean_loss)}; PLE better on ${better} of ${rows.length} tokens`);
    applyHot();
  }

  function applyHot() {
    gSl.selectAll('line').attr('opacity', r => (hot == null || r.i === hot ? 1 : 0.12))
      .attr('stroke-width', r => (r.i === hot ? 3.5 : 1 + 1.5 * Math.abs(r.d) / dmax));
    gSd.selectAll('circle').attr('opacity', q => (hot == null || q.r.i === hot ? 0.9 : 0.1)).attr('r', q => (q.r.i === hot ? 4 : 2.2));
    gBb.selectAll('rect').attr('opacity', r => (hot == null || r.i === hot ? 1 : 0.3)).attr('stroke', r => (r.i === hot ? th.accent : null));
    gBl.selectAll('text').attr('fill', r => (r.i === hot ? th.accent : th.muted)).attr('font-weight', r => (r.i === hot ? 700 : null))
      .text(r => (draw.yb.bandwidth() >= 7 || r.i === hot ? clip(vis(r.t.piece), 9) : ''));
    gS.selectAll('text.hl').remove();
    if (hot != null) {
      const r = rows.find(q => q.i === hot);
      if (r) {
        const y = draw.y;
        gS.append('text').attr('class', 'hl').attr('x', draw.xL - 8).attr('y', y(r.b)).attr('text-anchor', 'end').attr('dominant-baseline', 'central')
          .attr('font-size', 11).attr('font-weight', 700).attr('fill', th.accent).attr('paint-order', 'stroke').attr('stroke', th.bg).attr('stroke-width', 3).text(fmt(r.b, 2));
        gS.append('text').attr('class', 'hl').attr('x', draw.xR + 8).attr('y', y(r.p)).attr('dominant-baseline', 'central')
          .attr('font-size', 11).attr('font-weight', 700).attr('fill', th.accent).attr('paint-order', 'stroke').attr('stroke', th.bg).attr('stroke-width', 3).text(`${fmt(r.p, 2)} ${clip(vis(r.t.piece), 8)}`);
      }
    }
  }

  function setHot(i, fire) {
    if (i === hot) return;
    hot = i; applyHot();
    if (i == null) tip.hide();
    if (fire) opts.onHover?.(i);
  }

  function showTip(ev, i) {
    const d = c.diff[i];
    tip.show(tipHtml(`${esc(vis(c.tokens[i].piece))}<small>pos ${i}</small>`, [
      [c.baseline.run, fmt(c.baseline.loss[i]), th.ctx],
      [c.ple.run, fmt(c.ple.loss[i]), th.ple],
      ['Δ (base − PLE)', `${d >= 0 ? '▲ +' : '▼ '}${fmt(d, 4)}`, div(d / dmax)],
    ], d >= 0 ? 'PLE predicts this token better' : 'baseline predicts this token better'), ev);
  }

  function setOrder(o) { order = o; seg.set(o); draw(true); }
  prep(); draw(true);
  const off = onResize(el, () => draw(false));
  return {
    update(nd) { c = nd; hot = null; prep(); gSl.selectAll('*').remove(); gBb.selectAll('*').remove(); draw(true); },
    highlight(i) { setHot(i, false); },
    setOrder,
    destroy() { off(); tip.hide(); el.innerHTML = ''; },
  };
}

const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
