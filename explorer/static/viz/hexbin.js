// Hexbin density of full-row norm vs log training frequency over ALL tokens (DESIGN_IDEAS #9). Canvas.
// Bins are computed server-side (/api/norm_dist → hexbin) in a normalised space; drawing is an affine map,
// so the hexagons still tile exactly at any aspect ratio.
import { d3, theme, viridis, tip, tipHtml, esc, vis, fmt, mount, hidpiCanvas, widthOf, onResize, tween, alpha } from './core.js';

/**
 * data: /api/norm_dist response {hexbin:{radius, aspect, x_domain, y_domain, bins:[{x, y, n, mean_norm, mean_logf,
 *        examples:[{id, piece, freq, norm}]}]}, trend:[{log10_freq, mean_norm, count}], init_full_norm}
 * opts: {height?: 340, onClick?(bin) (e.g. open bin.examples[0].id), onHover?(bin|null)}
 * returns {update(data), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-hexbin', 'Hexbin of norm versus frequency');
  const canvas = document.createElement('canvas');
  root.append(canvas);
  const legend = document.createElement('div'); legend.className = 'viz-legend';
  legend.style.marginTop = '4px';
  root.append(legend);
  const H = opts.height || 340, m = { l: 50, r: 14, t: 10, b: 34 };
  let th = theme(), d = data, W, ctx, sx, sy, cmax = 1, hot = null, grow = 1, cancel = () => {};

  function layout() {
    W = widthOf(el); ctx = hidpiCanvas(canvas, W, H);
    const hb = d.hexbin;
    sx = v => m.l + v * (W - m.l - m.r);
    sy = v => H - m.b - (v / hb.aspect) * (H - m.t - m.b);
    cmax = d3.max(hb.bins, b => b.n) || 1;
  }

  function hexPath(cx, cy, k = 1) {
    const hb = d.hexbin, rx = hb.radius * (W - m.l - m.r) * k, ry = (hb.radius / hb.aspect) * (H - m.t - m.b) * k;
    ctx.moveTo(cx, cy - ry);
    for (let i = 1; i < 6; i++) { const a = (i * Math.PI) / 3; ctx.lineTo(cx + Math.sin(a) * rx, cy - Math.cos(a) * ry); }
    ctx.closePath();
  }

  const xv = lf => (lf - d.hexbin.x_domain[0]) / (d.hexbin.x_domain[1] - d.hexbin.x_domain[0]);
  const yv = n => ((n - d.hexbin.y_domain[0]) / (d.hexbin.y_domain[1] - d.hexbin.y_domain[0])) * d.hexbin.aspect;

  function draw() {
    th = theme();
    const hb = d.hexbin;
    ctx.clearRect(0, 0, W, H);
    ctx.font = `10px ${th.fontMono}`;
    // grid + axes
    const xs = d3.scaleLinear().domain(hb.x_domain).range([sx(0), sx(1)]);
    const ys = d3.scaleLinear().domain(hb.y_domain).range([sy(0), sy(hb.aspect)]);
    ctx.strokeStyle = th.hairline; ctx.lineWidth = 1; ctx.fillStyle = th.muted;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of xs.ticks(7)) { ctx.beginPath(); ctx.moveTo(xs(t), m.t); ctx.lineTo(xs(t), H - m.b); ctx.stroke(); ctx.fillText(t === 0 ? '0' : `1e${t}`, xs(t), H - m.b + 5); }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of ys.ticks(5)) { ctx.beginPath(); ctx.moveTo(m.l, ys(t)); ctx.lineTo(W - m.r, ys(t)); ctx.stroke(); ctx.fillText(String(t), m.l - 6, ys(t)); }
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('train count (1 + n, log)', W - m.r, H - 2);
    ctx.save(); ctx.translate(12, (m.t + H - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('full-row norm', 0, 0); ctx.restore();
    // hexes, batched by colour bucket (log count)
    const lc = n => Math.log1p(n) / Math.log1p(cmax);
    const buckets = new Map();
    for (const b of hb.bins) {
      const q = Math.round(lc(b.n) * 48);
      if (!buckets.has(q)) buckets.set(q, []);
      buckets.get(q).push(b);
    }
    for (const [q, bs] of buckets) {
      ctx.fillStyle = viridis(q / 48);
      ctx.beginPath();
      for (const b of bs) hexPath(sx(b.x), sy(b.y), grow * (0.55 + 0.45 * Math.min(1, q / 10)));
      ctx.fill();
    }
    // brightest hexes get a soft glow (glow = data)
    ctx.save(); ctx.shadowColor = alpha(viridis(1), 0.8); ctx.shadowBlur = 10; ctx.fillStyle = viridis(1);
    ctx.beginPath(); for (const b of hb.bins) if (b.n > cmax * 0.25) hexPath(sx(b.x), sy(b.y), grow); ctx.fill(); ctx.restore();
    // init line
    ctx.setLineDash([4, 4]); ctx.strokeStyle = th.ple; ctx.lineWidth = 1.2;
    const yi = sy(yv(d.init_full_norm));
    ctx.beginPath(); ctx.moveTo(m.l, yi); ctx.lineTo(W - m.r, yi); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = th.ple; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(`init ≈ ${fmt(d.init_full_norm, 2)}`, m.l + 4, yi - 3);
    // trend
    if (d.trend && d.trend.length > 1) {
      const pts = d.trend.filter(t => t.count >= 3);
      ctx.strokeStyle = alpha(th.bg, 0.8); ctx.lineWidth = 4.5; ctx.beginPath();
      pts.forEach((t, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(xv(t.log10_freq + 0.125)), sy(yv(t.mean_norm)))); ctx.stroke();
      ctx.strokeStyle = th.accent; ctx.lineWidth = 2; ctx.stroke();
    }
    if (hot) {
      ctx.strokeStyle = th.text; ctx.lineWidth = 2; ctx.beginPath(); hexPath(sx(hot.x), sy(hot.y), 1.15); ctx.stroke();
    }
    legend.innerHTML = `<span>1</span><span class="ramp" style="background:linear-gradient(90deg,${[0, .25, .5, .75, 1].map(viridis).join(',')})"></span><span>${cmax.toLocaleString()} tokens / hex (log)</span>`
      + `<span><i style="height:2px;background:${th.accent}"></i>mean norm per ¼-decade</span><span><i style="height:0;border-top:1px dashed ${th.ple}"></i>init norm</span>`
      + `<span>${d3.sum(hb.bins, b => b.n).toLocaleString()} tokens</span>`;
  }

  function hitTest(ev) {
    const r = canvas.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    const hb = d.hexbin, rx = hb.radius * (W - m.l - m.r), ry = (hb.radius / hb.aspect) * (H - m.t - m.b);
    let best = null, bd = Infinity;
    for (const b of hb.bins) {
      const dx = (sx(b.x) - px) / rx, dy = (sy(b.y) - py) / ry, dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = b; }
    }
    return bd <= 1.1 ? best : null;
  }
  const onMove = ev => {
    const b = hitTest(ev);
    if (b !== hot) { hot = b; draw(); opts.onHover?.(b); }
    canvas.style.cursor = b && opts.onClick ? 'pointer' : 'crosshair';
    if (!b) { tip.hide(); return; }
    const hb = d.hexbin;
    const fx = hb.x_domain[0] + b.x * (hb.x_domain[1] - hb.x_domain[0]);
    const ny = hb.y_domain[0] + (b.y / hb.aspect) * (hb.y_domain[1] - hb.y_domain[0]);
    tip.show(tipHtml(`${b.n.toLocaleString()} tokens<small>hex</small>`, [
      ['train count ≈', `${Math.round(10 ** fx - 1).toLocaleString()}`],
      ['norm ≈', fmt(ny, 2)],
      ['mean norm', fmt(b.mean_norm, 3)],
      ...b.examples.map(e => [`<span class="tok">${esc(vis(e.piece))}</span>`, `${e.freq.toLocaleString()} · ${fmt(e.norm, 2)}`]),
    ], `examples = most frequent in this hex${opts.onClick ? ' · click to open' : ''}`), ev);
  };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', () => { hot = null; tip.hide(); draw(); opts.onHover?.(null); });
  canvas.addEventListener('click', ev => { const b = hitTest(ev); if (b) opts.onClick?.(b); });

  function full(nd) {
    d = nd;
    if (!d.hexbin) { root.innerHTML = '<div class="viz-readout">no training-frequency data (train.bin) on the server</div>'; return; }
    layout(); cancel();
    cancel = tween(700, k => { grow = k; draw(); }, d3.easeBackOut.overshoot(1.4));
  }
  full(data);
  const off = onResize(el, () => { layout(); draw(); });
  return { update: full, destroy() { cancel(); off(); tip.hide(); el.innerHTML = ''; } };
}
