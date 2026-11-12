// Minimal canvas charts for the PLE Explorer (no external dependencies).

const DPR = window.devicePixelRatio || 1;
const FONT = '11px "Segoe UI", system-ui, sans-serif';
const AXIS = '#8b93a7';
const GRID = '#262b38';
const TEXT = '#d7dbe6';

export const tooltip = (() => {
  const el = document.createElement('div');
  el.className = 'tooltip';
  document.body.appendChild(el);
  return {
    show(html, ev) {
      el.innerHTML = html;
      el.style.display = 'block';
      const pad = 14, r = el.getBoundingClientRect();
      let x = ev.clientX + pad, y = ev.clientY + pad;
      if (x + r.width > window.innerWidth) x = ev.clientX - r.width - pad;
      if (y + r.height > window.innerHeight) y = ev.clientY - r.height - pad;
      el.style.left = `${Math.max(0, x)}px`;
      el.style.top = `${Math.max(0, y)}px`;
    },
    hide() { el.style.display = 'none'; },
  };
})();

export function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Visible token text: show leading spaces as '␣' and newlines as '⏎'. */
export function vis(piece) {
  return piece.replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
}

// ---- colour scales -------------------------------------------------------
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const DIVERGING = [[59, 130, 246], [40, 44, 56], [239, 68, 68]];

function interp(stops, t) {
  t = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const s = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(s)), f = s - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
}
export const seq = t => interp(VIRIDIS, t);
/** t in [-1, 1]: blue (negative) → dark → red (positive). */
export const div = t => interp(DIVERGING, (t + 1) / 2);

// ---- helpers -------------------------------------------------------------
function makeCanvas(container, height) {
  container.innerHTML = '';
  const canvas = document.createElement('canvas');
  const w = Math.max(160, container.clientWidth || 400);
  canvas.width = w * DPR;
  canvas.height = height * DPR;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${height}px`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(DPR, DPR);
  ctx.font = FONT;
  return { canvas, ctx, w, h: height };
}

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}

export function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  return (+v.toFixed(d)).toString();
}

function hover(canvas, find, render, onClick) {
  canvas.addEventListener('mousemove', ev => {
    const r = canvas.getBoundingClientRect();
    const hit = find(ev.clientX - r.left, ev.clientY - r.top);
    canvas.style.cursor = hit != null && onClick ? 'pointer' : 'default';
    if (hit == null) tooltip.hide(); else tooltip.show(render(hit), ev);
  });
  canvas.addEventListener('mouseleave', () => tooltip.hide());
  if (onClick) {
    canvas.addEventListener('click', ev => {
      const r = canvas.getBoundingClientRect();
      const hit = find(ev.clientX - r.left, ev.clientY - r.top);
      if (hit != null) { tooltip.hide(); onClick(hit); }
    });
  }
}

function axes(ctx, box, xTicks, yTicks, sx, sy, xFmt = fmt, yFmt = fmt) {
  ctx.strokeStyle = GRID; ctx.fillStyle = AXIS; ctx.lineWidth = 1;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const t of yTicks) {
    const y = sy(t);
    ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.r, y); ctx.stroke();
    ctx.fillText(yFmt(t), box.l - 5, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const t of xTicks || []) ctx.fillText(xFmt(t), sx(t), box.b + 5);
  ctx.strokeStyle = AXIS;
  ctx.beginPath(); ctx.moveTo(box.l, box.t); ctx.lineTo(box.l, box.b); ctx.lineTo(box.r, box.b); ctx.stroke();
}

function axisLabels(ctx, box, w, h, xLabel, yLabel) {
  ctx.fillStyle = AXIS; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  if (xLabel) ctx.fillText(xLabel, (box.l + box.r) / 2, h - 2);
  if (yLabel) {
    ctx.save(); ctx.translate(11, (box.t + box.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle'; ctx.fillText(yLabel, 0, 0); ctx.restore();
  }
}

// ---- bar chart -------------------------------------------------------------
/**
 * values: number[]; opts: {labels, height, color(v,i), refLine:{value,label}, yLabel, xLabel, tip(i), onClick(i), min}
 */
export function barChart(container, values, opts = {}) {
  const { ctx, canvas, w, h } = makeCanvas(container, opts.height || 180);
  const box = { l: 48, r: w - 10, t: 10, b: h - (opts.xLabel ? 34 : 22) };
  const vmax = Math.max(...values, opts.refLine ? opts.refLine.value : -Infinity, 1e-9);
  const vmin = Math.min(0, opts.min ?? 0, ...values);
  const sy = v => box.b - (v - vmin) / (vmax - vmin || 1) * (box.b - box.t);
  axes(ctx, box, null, niceTicks(vmin, vmax, 4), null, sy);
  const n = values.length, bw = (box.r - box.l) / n;
  values.forEach((v, i) => {
    ctx.fillStyle = opts.color ? opts.color(v, i) : '#60a5fa';
    const y0 = sy(Math.max(0, vmin)), y1 = sy(v);
    ctx.fillRect(box.l + i * bw + bw * 0.12, Math.min(y0, y1), Math.max(1, bw * 0.76), Math.abs(y1 - y0));
  });
  ctx.fillStyle = AXIS; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const labels = opts.labels || values.map((_, i) => String(i));
  const every = Math.ceil(n / Math.max(1, Math.floor((box.r - box.l) / 34)));
  labels.forEach((lab, i) => { if (i % every === 0) ctx.fillText(lab, box.l + (i + 0.5) * bw, box.b + 4); });
  if (opts.refLine) {
    const y = sy(opts.refLine.value);
    ctx.strokeStyle = '#f59e0b'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.r, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText(opts.refLine.label, box.r, y - 2);
  }
  axisLabels(ctx, box, w, h, opts.xLabel, opts.yLabel);
  hover(canvas, (x, y) => {
    if (x < box.l || x > box.r || y < box.t || y > box.b + 16) return null;
    return Math.min(n - 1, Math.floor((x - box.l) / bw));
  }, i => (opts.tip ? opts.tip(i) : `<b>${esc(labels[i])}</b>: ${fmt(values[i])}`), opts.onClick);
}

// ---- heatmap ---------------------------------------------------------------
/**
 * matrix: number[][] (rows × cols). opts: {height, cellH, diverging, vmin, vmax, rowLabels, colLabels,
 *   showValues, tip(r,c,v)}
 */
export function heatmap(container, matrix, opts = {}) {
  const rows = matrix.length, cols = matrix[0].length;
  const left = opts.rowLabels ? 46 : 8, top = opts.colLabels ? 18 : 4;
  const width = Math.max(160, container.clientWidth || 400);
  const square = opts.square;
  const cw = square ? Math.min((width - left - 8) / cols, 64) : (width - left - 8) / cols;
  const ch = square ? cw : (opts.cellH || 22);
  const legend = 26;
  const { ctx, canvas } = makeCanvas(container, top + rows * ch + legend);
  const flat = matrix.flat();
  let vmax = opts.vmax ?? Math.max(...flat.map(Math.abs));
  let vmin = opts.vmin ?? (opts.diverging ? -vmax : Math.min(...flat));
  if (!(vmax > vmin)) { vmax = vmin + 1; }
  const color = v => (opts.diverging ? div(v / Math.max(Math.abs(vmin), Math.abs(vmax))) : seq((v - vmin) / (vmax - vmin)));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = matrix[r][c];
      ctx.fillStyle = color(v);
      ctx.fillRect(left + c * cw, top + r * ch, Math.ceil(cw), Math.ceil(ch));
      if (opts.showValues && cw > 30) {
        ctx.fillStyle = Math.abs((v - vmin) / (vmax - vmin)) > 0.6 && !opts.diverging ? '#111' : TEXT;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(v.toFixed(2), left + (c + 0.5) * cw, top + (r + 0.5) * ch);
      }
    }
  }
  ctx.fillStyle = AXIS;
  if (opts.rowLabels) {
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    opts.rowLabels.forEach((l, r) => ctx.fillText(l, left - 5, top + (r + 0.5) * ch));
  }
  if (opts.colLabels) {
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    const every = Math.ceil(cols / Math.max(1, Math.floor(cols * cw / 28)));
    opts.colLabels.forEach((l, c) => { if (c % every === 0) ctx.fillText(l, left + (c + 0.5) * cw, top - 3); });
  }
  // legend
  const ly = top + rows * ch + 8, lw = Math.min(200, cols * cw);
  for (let i = 0; i < lw; i++) {
    ctx.fillStyle = color(vmin + (vmax - vmin) * i / lw);
    ctx.fillRect(left + i, ly, 1, 8);
  }
  ctx.fillStyle = AXIS; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.fillText(`${fmt(vmin, 2)} … ${fmt(vmax, 2)}`, left + lw + 8, ly + 4);
  hover(canvas, (x, y) => {
    const c = Math.floor((x - left) / cw), r = Math.floor((y - top) / ch);
    return r >= 0 && r < rows && c >= 0 && c < cols ? [r, c] : null;
  }, ([r, c]) => (opts.tip ? opts.tip(r, c, matrix[r][c]) : `[${r}, ${c}] = ${fmt(matrix[r][c], 4)}`));
}

// ---- scatter ---------------------------------------------------------------
/**
 * points: {x, y, color?, r?, label?}[]; opts: {height, logX, xLabel, yLabel, tip(i), onClick(i),
 *   labelTop:n (draw label text for first n points), line:{x[],y[]}, refY:{value,label}, highlight:i}
 */
export function scatter(container, points, opts = {}) {
  const { ctx, canvas, w, h } = makeCanvas(container, opts.height || 420);
  const box = { l: 56, r: w - 12, t: 12, b: h - 36 };
  const tx = opts.logX ? v => Math.log10(Math.max(v, 0) + 1) : v => v;
  const xs = points.map(p => tx(p.x)), ys = points.map(p => p.y);
  if (opts.line) { xs.push(...opts.line.x.map(tx)); ys.push(...opts.line.y); }
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (opts.refY) { y0 = Math.min(y0, opts.refY.value); y1 = Math.max(y1, opts.refY.value); }
  const px = (x1 - x0) * 0.04 || 1, py = (y1 - y0) * 0.06 || 1;
  x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = v => box.l + (v - x0) / (x1 - x0) * (box.r - box.l);
  const sy = v => box.b - (v - y0) / (y1 - y0) * (box.b - box.t);
  const xt = opts.logX ? niceTicks(x0, x1, 6).filter(t => Number.isInteger(t)) : niceTicks(x0, x1, 6);
  axes(ctx, box, xt, niceTicks(y0, y1, 5), sx, sy, opts.logX ? t => (t === 0 ? '0' : `1e${t}`) : fmt);
  const sxp = points.map(p => sx(tx(p.x))), syp = points.map(p => sy(p.y));
  ctx.globalAlpha = opts.alpha ?? 0.75;
  points.forEach((p, i) => {
    ctx.fillStyle = p.color || '#60a5fa';
    ctx.beginPath(); ctx.arc(sxp[i], syp[i], p.r || 2.5, 0, Math.PI * 2); ctx.fill();
  });
  ctx.globalAlpha = 1;
  if (opts.line) {
    ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.beginPath();
    opts.line.x.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(tx(x)), sy(opts.line.y[i])));
    ctx.stroke(); ctx.lineWidth = 1;
  }
  if (opts.refY) {
    const y = sy(opts.refY.value);
    ctx.strokeStyle = '#a78bfa'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.r, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#a78bfa'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText(opts.refY.label, box.r, y - 2);
  }
  if (opts.labelTop) {
    ctx.fillStyle = TEXT; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    points.slice(0, opts.labelTop).forEach((p, i) => p.label && ctx.fillText(p.label, sxp[i] + 4, syp[i]));
  }
  if (opts.highlight != null && points[opts.highlight]) {
    const i = opts.highlight;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sxp[i], syp[i], 7, 0, Math.PI * 2); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.fillText(points[i].label || '', sxp[i] + 9, syp[i]);
  }
  axisLabels(ctx, box, w, h, opts.xLabel, opts.yLabel);
  hover(canvas, (x, y) => {
    let best = null, bd = 64;
    for (let i = 0; i < points.length; i++) {
      const d = (sxp[i] - x) ** 2 + (syp[i] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }, i => (opts.tip ? opts.tip(i) : esc(points[i].label ?? i)), opts.onClick);
}
