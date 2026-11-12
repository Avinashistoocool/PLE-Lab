// Sparkline chips (DESIGN_IDEAS #10): token chips with a tiny per-layer bar sparkline under each piece.
import { theme, layerColor, diverging, viridis, tip, tipHtml, esc, vis, fmt, mount } from './core.js';

/** SVG string for one sparkline. values: number[L]; vmax: shared scale. */
export function sparkSvg(values, vmax, th = theme()) {
  const L = values.length, bw = 3, gap = 1, W = L * (bw + gap) - gap, H = 10;
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${values.map((v, l) => {
    const h = Math.max(0.8, (H * Math.max(0, v)) / (vmax || 1));
    return `<rect x="${l * (bw + gap)}" y="${H - h}" width="${bw}" height="${h}" fill="${layerColor(th, l)}"/>`;
  }).join('')}</svg>`;
}

/**
 * data: /api/analyze response {tokens, gate[T][L], contrib_ratio[T][L], loss[T], ablation?}
 * opts: {color?: 'loss'|'delta'|'gate'|'none' (chip background; default 'delta' if ablation else 'loss'),
 *        spark?: 'gate'|'ratio', selected?: i, onHover?(i|null), onClick?(i)}
 * returns {update(data, opts?), highlight(i|null), select(i|null), setColor(mode), destroy()}
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-chips', 'Token chips with per-layer sparklines');
  root.setAttribute('role', 'list');
  let a = data, o = { ...opts }, th = theme(), hot = null;

  function colorFor(i) {
    const mode = o.color || (a.ablation ? 'delta' : 'loss');
    if (mode === 'none') return th.surface2;
    if (mode === 'gate') {
      const l = o.layer ?? 0, vals = a.gate.map(g => g[l]), mx = Math.max(...vals), mn = Math.min(...vals);
      return viridis((a.gate[i][l] - mn) / (mx - mn || 1));
    }
    if (mode === 'delta' && a.ablation) {
      const v = a.ablation.delta[i], mx = Math.max(...a.ablation.delta.filter(x => x != null).map(Math.abs), 1e-9);
      return v == null ? th.surface2 : diverging(th)(v / mx);
    }
    const mx = Math.max(...a.loss.filter(v => v != null), 1e-9);
    return a.loss[i] == null ? th.surface2 : viridis(a.loss[i] / mx);
  }

  function draw() {
    th = theme();
    const key = o.spark === 'ratio' ? 'contrib_ratio' : 'gate';
    const vmax = Math.max(...a[key].flat(), 1e-9);
    root.innerHTML = a.tokens.map((t, i) => `<span class="viz-chip${o.selected === i ? ' sel' : ''}" role="listitem" tabindex="0" data-i="${i}" style="background-color:${colorFor(i)}">`
      + `<span>${esc(vis(t.piece))}</span>${sparkSvg(a[key][i], vmax, th)}</span>`).join('');
  }

  const chipAt = ev => ev.target.closest?.('.viz-chip');
  const onMove = ev => {
    const c = chipAt(ev);
    if (!c) return;
    const i = +c.dataset.i;
    if (i !== hot) { setHot(i); o.onHover?.(i); }
    const L = a.gate[0].length;
    const rows = Array.from({ length: L }, (_, l) => [`L${l} gate`, `${fmt(a.gate[i][l])} · ${fmt(a.contrib_ratio[i][l] * 100, 1)}%`, layerColor(th, l)]);
    let foot = a.loss[i] != null ? `loss ${fmt(a.loss[i])}` : 'first token (no loss)';
    if (a.ablation && a.ablation.delta[i] != null) foot += ` · Δ ${a.ablation.delta[i] >= 0 ? '+' : ''}${fmt(a.ablation.delta[i])}`;
    if (a.top5) foot += `<br>next: ${a.top5[i].slice(0, 3).map(p => `<span class="tok">${esc(vis(p.piece))}</span> ${(p.p * 100).toFixed(0)}%`).join(' · ')}`;
    tip.show(tipHtml(`${esc(vis(a.tokens[i].piece))}<small>pos ${i} · #${a.tokens[i].id}</small>`, rows, foot), ev);
  };
  const onOut = ev => { if (!root.contains(ev.relatedTarget)) { setHot(null); tip.hide(); o.onHover?.(null); } };
  const onClick = ev => { const c = chipAt(ev); if (c) { select(+c.dataset.i); o.onClick?.(+c.dataset.i); } };
  const onKey = ev => {
    const c = chipAt(ev); if (!c) return;
    const i = +c.dataset.i;
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(i); o.onClick?.(i); }
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
      const n = root.querySelector(`[data-i="${i + (ev.key === 'ArrowRight' ? 1 : -1)}"]`);
      if (n) { n.focus(); setHot(+n.dataset.i); o.onHover?.(+n.dataset.i); }
    }
  };
  root.addEventListener('pointermove', onMove);
  root.addEventListener('pointerout', onOut);
  root.addEventListener('click', onClick);
  root.addEventListener('keydown', onKey);

  function setHot(i) {
    if (hot != null) root.querySelector(`[data-i="${hot}"]`)?.classList.remove('hot');
    hot = i;
    if (i != null) root.querySelector(`[data-i="${i}"]`)?.classList.add('hot');
  }
  function select(i) {
    o.selected = i;
    root.querySelectorAll('.viz-chip.sel').forEach(c => c.classList.remove('sel'));
    if (i != null) root.querySelector(`[data-i="${i}"]`)?.classList.add('sel');
  }

  draw();
  return {
    update(d, more = {}) { a = d; Object.assign(o, more); hot = null; draw(); },
    highlight: setHot,
    select,
    /** Re-colours chips in place (CSS transition = colour morph, e.g. 'loss' → 'delta' after ablation). */
    setColor(mode, layer) {
      o.color = mode; if (layer != null) o.layer = layer;
      root.querySelectorAll('.viz-chip').forEach((c, i) => { c.style.transitionDelay = `${Math.min(12 * i, 500)}ms`; c.style.backgroundColor = colorFor(i); });
    },
    destroy() { tip.hide(); el.innerHTML = ''; },
  };
}
