// Layer "fingerprint" radial glyph (DESIGN_IDEAS #4): 6 petals (layers) × p spokes (dims).
// Canvas; spoke length = |value|, colour = sign (blue ↔ orange). Morphs spoke by spoke on update.
import { theme, layerColor, diverging, tween, tip, tipHtml, esc, vis, fmt, mount, hidpiCanvas, widthOf, onResize, alpha } from './core.js';

/**
 * render(el, data, opts)
 * data: /api/token response: {id, piece, vectors[L][p], norms[L], init_norm, ple_dim, n_layer, downsampled?}
 * opts: {size?: px (default: min(container width, 440)), vmax?: fixed |value| scale (default per token, 99th pct),
 *        onHover?({layer, dim, value}|null), onClick?({layer, dim, value}), showLabels?: true}
 * returns {update(data), highlight(layer|null, dim?), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-glyph', 'Layer fingerprint glyph');
  const canvas = document.createElement('canvas');
  root.appendChild(canvas);
  let th = theme(), div = diverging(th);
  let size = 0, ctx = null, cur = null, from = null, to = null, progress = 1, hot = null, ext = null, cancel = () => {};
  let geom = null;

  function layout() {
    size = Math.min(opts.size || 440, widthOf(el));
    ctx = hidpiCanvas(canvas, size, size);
    const R = size / 2;
    geom = { cx: R, cy: R, R0: R * 0.27, R1: R * 0.80, Rl: R * 0.90 };
  }

  function prep(d) {
    const L = d.vectors.length, p = d.vectors[0].length;
    const flat = d.vectors.flat().map(Math.abs).sort((a, b) => a - b);
    const vmax = opts.vmax || flat[Math.floor(flat.length * 0.99)] || flat[flat.length - 1] || 1;
    return { d, L, p, vmax, vals: d.vectors.map(r => r.slice()) };
  }

  // per-spoke angle; petals separated by a small gap
  function angleOf(L, p, l, j) {
    const span = (2 * Math.PI) / L, gap = span * 0.07;
    return -Math.PI / 2 + l * span + gap / 2 + (j + 0.5) * (span - gap) / p;
  }

  function spokeValue(l, j, k) {
    const a = from && from.L === to.L && from.p === to.p ? from.vals[l][j] / from.vmax : 0;
    const b = to.vals[l][j] / to.vmax;
    return a + (b - a) * k;
  }

  function draw() {
    if (!to) return;
    const { cx, cy, R0, R1, Rl } = geom, { L, p, d } = to;
    ctx.clearRect(0, 0, size, size);
    const span = (2 * Math.PI) / L, gap = span * 0.07;
    const focus = hot ?? ext;
    // petal backgrounds + reference rings
    for (let l = 0; l < L; l++) {
      const a0 = -Math.PI / 2 + l * span + gap / 2, a1 = a0 + span - gap;
      const dim = focus && focus.layer !== l ? 0.35 : 1;
      ctx.globalAlpha = dim;
      ctx.beginPath(); ctx.arc(cx, cy, R1 + 4, a0, a1); ctx.arc(cx, cy, R0 - 3, a1, a0, true); ctx.closePath();
      ctx.fillStyle = hexA(layerColor(th, l), 0.06); ctx.fill();
      ctx.strokeStyle = hexA(layerColor(th, l), 0.25); ctx.lineWidth = 1; ctx.stroke();
      // RMS ring for this slice (same scale as spokes): ‖slice‖/√p
      const rms = (d.norms[l] / Math.sqrt(p)) / to.vmax;
      const rr = R0 + (R1 - R0) * Math.min(1.05, rms) * ease(progress);
      ctx.beginPath(); ctx.arc(cx, cy, rr, a0, a1);
      ctx.strokeStyle = layerColor(th, l); ctx.lineWidth = 2; ctx.stroke();
      // labels
      if (opts.showLabels !== false) {
        const am = (a0 + a1) / 2, lx = cx + Math.cos(am) * Rl, ly = cy + Math.sin(am) * Rl;
        ctx.fillStyle = layerColor(th, l);
        ctx.font = `600 ${Math.max(10, size / 36)}px ${th.fontMono}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`L${l}`, lx, ly - 6);
        ctx.fillStyle = th.muted; ctx.font = `${Math.max(9, size / 46)}px ${th.fontMono}`;
        ctx.fillText(fmt(d.norms[l], 2), lx, ly + 7);
      }
    }
    ctx.globalAlpha = 1;
    // init RMS ring (dashed amber): init slice norm / √p
    const ri = R0 + (R1 - R0) * Math.min(1.05, (d.init_norm / Math.sqrt(p)) / to.vmax);
    ctx.setLineDash([3, 4]); ctx.strokeStyle = th.ple; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, ri, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(cx, cy, R0 - 3, 0, 2 * Math.PI); ctx.strokeStyle = th.hairlineStrong; ctx.stroke();
    // spokes (staggered morph)
    const n = L * p, stagger = 0.55;
    ctx.lineCap = 'round';
    const lw = Math.max(1, Math.min(3.2, (2 * Math.PI * R0) / n * 0.8 * 1.6));
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < p; j++) {
        const i = l * p + j;
        const k = ease(Math.min(1, Math.max(0, (progress - (i / n) * stagger) / (1 - stagger))));
        const v = spokeValue(l, j, k);
        const a = angleOf(L, p, l, j), r = R0 + (R1 - R0) * Math.min(1.12, Math.abs(v));
        const isHot = focus && focus.layer === l && focus.dim === j;
        ctx.globalAlpha = focus && focus.layer !== l ? 0.3 : 1;
        ctx.strokeStyle = isHot ? th.accent : div(v);
        ctx.lineWidth = isHot ? lw + 2 : lw;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * R0, cy + Math.sin(a) * R0);
        ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
        ctx.stroke();
        if (Math.abs(v) > 0.8 && !isHot) {  // glow on strong dims (glow only on data)
          ctx.globalAlpha *= 0.25; ctx.lineWidth = lw + 3; ctx.stroke(); ctx.lineWidth = lw;
        }
      }
    }
    ctx.globalAlpha = 1;
    // centre readout
    const full = Math.sqrt(d.norms.reduce((s, x) => s + x * x, 0));
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = th.text; ctx.font = `600 ${Math.max(12, size / 22)}px ${th.fontMono}`;
    ctx.fillText(clip(vis(d.piece), 10), cx, cy - size / 40);
    ctx.fillStyle = th.muted; ctx.font = `${Math.max(9, size / 44)}px ${th.fontMono}`;
    ctx.fillText(`‖row‖ ${fmt(full, 2)}`, cx, cy + size / 32);
    ctx.fillText(`×${fmt(full / (d.init_norm * Math.sqrt(L)), 2)} init`, cx, cy + size / 32 + size / 36);
  }

  function hit(ev) {
    if (!to) return null;
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left - geom.cx, y = ev.clientY - r.top - geom.cy;
    const rad = Math.hypot(x, y);
    if (rad < geom.R0 - 6 || rad > geom.R1 * 1.15) return null;
    const { L, p } = to, span = (2 * Math.PI) / L, gap = span * 0.07;
    let a = Math.atan2(y, x) + Math.PI / 2;
    a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const l = Math.floor(a / span);
    const local = a - l * span - gap / 2;
    if (local < 0 || local > span - gap) return null;
    const j = Math.min(p - 1, Math.floor(local / ((span - gap) / p)));
    return { layer: l, dim: j, value: to.d.vectors[l][j] };
  }

  const onMove = ev => {
    const h = hit(ev);
    if (h?.layer !== hot?.layer || h?.dim !== hot?.dim) { hot = h; if (progress >= 1) draw(); opts.onHover?.(h); }
    if (!h) { tip.hide(); canvas.style.cursor = 'default'; return; }
    canvas.style.cursor = opts.onClick ? 'pointer' : 'crosshair';
    const d = to.d, p = to.p;
    tip.show(tipHtml(`${esc(vis(d.piece))}<small>#${d.id}</small>`, [
      ['layer', `L${h.layer}`, layerColor(th, h.layer)],
      ['dim', `${h.dim}${d.downsampled ? ' (pooled)' : ''}`],
      ['value', `${h.value >= 0 ? '+' : ''}${fmt(h.value, 4)}`, div(h.value / to.vmax)],
      ['‖slice‖', `${fmt(d.norms[h.layer], 3)} · ×${fmt(d.norms[h.layer] / d.init_norm, 2)} init`],
      ['slice rms', fmt(d.norms[h.layer] / Math.sqrt(p), 4)],
    ]), ev);
  };
  const onLeave = () => { hot = null; tip.hide(); draw(); opts.onHover?.(null); };
  const onClick = ev => { const h = hit(ev); if (h) opts.onClick?.(h); };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', onLeave);
  canvas.addEventListener('click', onClick);

  function kAt(l, j) {
    const n = to.L * to.p;
    return ease(Math.min(1, Math.max(0, (progress - ((l * to.p + j) / n) * 0.55) / 0.45)));
  }

  function update(d) {
    th = theme(); div = diverging(th);
    root.setAttribute('aria-label', `Layer fingerprint of ${d.piece}: slice norms ${d.norms.map(n => fmt(n, 2)).join(', ')}`);
    // morph from whatever is on screen right now (normalised values, so vmax = 1)
    from = to ? { L: to.L, p: to.p, vmax: 1, vals: to.vals.map((row, l) => row.map((_, j) => spokeValue(l, j, kAt(l, j)))) } : null;
    to = prep(d);
    cancel();
    progress = 0;
    cancel = tween(1100, k => { progress = k; draw(); }, t => t);
  }

  layout();
  const off = onResize(el, () => { layout(); draw(); });
  update(data);

  return {
    update,
    highlight(layer, dim = null) { ext = layer == null ? null : { layer, dim }; if (progress >= 1) draw(); },
    destroy() { cancel(); off(); tip.hide(); canvas.removeEventListener('pointermove', onMove); el.innerHTML = ''; },
  };
}

const ease = t => 1 - Math.pow(1 - t, 3);
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const hexA = alpha;
