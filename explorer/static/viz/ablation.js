// Ablation before → after morph (DESIGN_IDEAS #6). SVG.
// Top: per-token loss bars grow from the normal loss to the ablated loss (ghost tick = normal loss).
// Bottom: diverging Δloss strip (orange = harder without PLE, blue = easier). Mean-loss counter ticks.
import { d3, theme, diverging, viridis, tip, tipHtml, esc, vis, fmt, mount, widthOf, onResize, dur, countUp, alpha } from './core.js';

/**
 * data: /api/analyze response WITH ablation: {tokens, loss[T], mean_loss, ablation:{layers, loss[T], mean_loss, delta[T], mean_delta}}
 *       (without `ablation` it shows the normal loss only and a hint)
 * opts: {height?: 300, onHover?(i|null), onClick?(i), autoplay?: true}
 * returns {update(data), highlight(i|null), play(), show('before'|'after'), destroy()}
 * Keyboard: hold Space while the pointer is over the chart (or the chart has focus) to see the un-ablated state.
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-ablation', 'Ablation before and after');
  root.tabIndex = 0;
  const bar = document.createElement('div'); bar.className = 'viz-bar';
  const counter = document.createElement('span'); counter.className = 'viz-readout';
  const readout = document.createElement('span'); readout.className = 'viz-readout viz-grow';
  const btn = document.createElement('button'); btn.className = 'viz-btn'; btn.type = 'button'; btn.textContent = '↻ replay';
  const hold = document.createElement('button'); hold.className = 'viz-btn'; hold.type = 'button'; hold.textContent = 'hold: before';
  hold.title = 'press and hold (or hold Space) to show the un-ablated loss';
  bar.append(counter, readout, hold, btn);
  root.append(bar);
  const H = opts.height || 300;
  const svg = d3.select(root).append('svg').attr('height', H);
  const gGrid = svg.append('g'), gBars = svg.append('g'), gGhost = svg.append('g'), gDelta = svg.append('g'), gAxis = svg.append('g');
  const hl = svg.append('rect').attr('pointer-events', 'none').attr('opacity', 0);
  const overlay = svg.append('rect').attr('fill', 'transparent');
  let th = theme(), div = diverging(th), a = data, W = 0, x, yL, yD, state = 'after', hot = null, cancelCount = () => {};
  const mL = 40, mR = 10, mT = 8, deltaH = 74, labelH = 46;

  function geometry() {
    W = widthOf(el);
    svg.attr('width', W);
    const T = a.tokens.length;
    x = d3.scaleBand().domain(d3.range(T)).range([mL, W - mR]).paddingInner(0.18);
    const hasAbl = !!a.ablation;
    const lossMax = d3.max([...a.loss, ...(hasAbl ? a.ablation.loss : [])].filter(v => v != null)) || 1;
    const topB = H - labelH - (hasAbl ? deltaH + 8 : 0);
    yL = d3.scaleLinear().domain([0, lossMax * 1.05]).nice().range([topB, mT]);
    if (hasAbl) {
      const dm = d3.max(a.ablation.delta.filter(v => v != null).map(Math.abs)) || 1e-6;
      const dTop = topB + 8;
      yD = d3.scaleLinear().domain([-dm, dm]).range([dTop + deltaH, dTop]);
    } else yD = null;
  }

  function staticParts() {
    th = theme(); div = diverging(th);
    const T = a.tokens.length;
    gGrid.selectAll('*').remove();
    yL.ticks(4).forEach(t => {
      gGrid.append('line').attr('x1', mL).attr('x2', W - mR).attr('y1', yL(t)).attr('y2', yL(t)).attr('stroke', th.hairline);
      gGrid.append('text').attr('x', mL - 6).attr('y', yL(t)).attr('text-anchor', 'end').attr('dominant-baseline', 'central').attr('font-size', 10).attr('fill', th.muted).text(t);
    });
    gGrid.append('text').attr('x', 2).attr('y', mT + 2).attr('font-size', 10).attr('fill', th.muted).attr('dominant-baseline', 'hanging').text('loss');
    if (yD) {
      const y0 = yD(0);
      gGrid.append('line').attr('x1', mL).attr('x2', W - mR).attr('y1', y0).attr('y2', y0).attr('stroke', th.hairlineStrong);
      gGrid.append('text').attr('x', 2).attr('y', yD.range()[1]).attr('font-size', 10).attr('fill', th.muted).attr('dominant-baseline', 'hanging').text('Δloss');
      gGrid.append('text').attr('x', mL - 6).attr('y', yD.range()[1] + 2).attr('text-anchor', 'end').attr('dominant-baseline', 'hanging').attr('font-size', 10).attr('fill', th.pos).text(`+${fmt(yD.domain()[1], 2)}`);
      gGrid.append('text').attr('x', mL - 6).attr('y', yD.range()[0]).attr('text-anchor', 'end').attr('font-size', 10).attr('fill', th.neg).text(`−${fmt(yD.domain()[1], 2)}`);
    }
    const bw = x.bandwidth(), rot = bw < 26, every = Math.max(1, Math.ceil(11 / x.step()));
    gAxis.selectAll('text').data(a.tokens.map((t, i) => ({ t, i })), d => d.i).join('text')
      .attr('font-size', 11).attr('fill', th.muted)
      .attr('transform', d => `translate(${x(d.i) + bw / 2},${H - labelH + 6}) rotate(${rot ? -50 : 0})`)
      .attr('text-anchor', rot ? 'end' : 'middle').attr('dominant-baseline', rot ? 'central' : 'hanging')
      .text(d => (d.i % every === 0 ? clip(vis(d.t.piece), 9) : ''));
    overlay.attr('x', mL).attr('y', mT).attr('width', W - mL - mR).attr('height', H - mT - labelH + 30);
    hl.attr('y', mT).attr('height', H - mT - labelH).attr('width', x.step()).attr('fill', alpha(th.accent, 0.08)).attr('stroke', alpha(th.accent, 0.5));
    void T;
  }

  function lossAt(i, s) { return s === 'before' || !a.ablation ? a.loss[i] : a.ablation.loss[i]; }

  function show(s, ms = 650, stagger = true) {
    state = a.ablation ? s : 'before';
    const T = a.tokens.length, bw = x.bandwidth();
    const lmax = yL.domain()[1];
    const items = d3.range(T).map(i => ({ i, v: lossAt(i, state), base: a.loss[i], d: a.ablation ? a.ablation.delta[i] : null }));
    const t = d3.transition().duration(dur(ms)).ease(d3.easeBackOut.overshoot(1.3));
    const delay = d => (stagger ? dur(Math.min(14, 600 / T)) * d.i : 0);
    gBars.selectAll('rect').data(items, d => d.i).join(e => e.append('rect').attr('y', yL(0)).attr('height', 0).attr('rx', 1.5))
      .attr('x', d => x(d.i)).attr('width', bw)
      .transition(t).delay(delay)
      .attr('y', d => (d.v == null ? yL(0) : yL(d.v))).attr('height', d => (d.v == null ? 0 : yL(0) - yL(d.v)))
      .attr('fill', d => (d.v == null ? th.zero : state === 'after' && d.d != null ? lossColor(d) : viridis(d.v / lmax)));
    gGhost.selectAll('line').data(a.ablation ? items.filter(d => d.base != null) : [], d => d.i).join('line')
      .attr('x1', d => x(d.i) - 1).attr('x2', d => x(d.i) + bw + 1).attr('y1', d => yL(d.base)).attr('y2', d => yL(d.base))
      .attr('stroke', th.text).attr('stroke-width', 1.5)
      .transition().duration(dur(300)).attr('stroke-opacity', state === 'after' ? 0.75 : 0);
    if (yD) {
      gDelta.selectAll('rect').data(items, d => d.i).join(e => e.append('rect').attr('y', yD(0)).attr('height', 0).attr('rx', 1))
        .attr('x', d => x(d.i)).attr('width', bw)
        .transition(t).delay(delay)
        .attr('y', d => (d.d == null || state === 'before' ? yD(0) : Math.min(yD(0), yD(d.d))))
        .attr('height', d => (d.d == null || state === 'before' ? 0 : Math.abs(yD(d.d) - yD(0))))
        .attr('fill', d => div((d.d ?? 0) / yD.domain()[1]));
    } else gDelta.selectAll('*').remove();
    // counter
    cancelCount();
    const from = parseFloat(counter.dataset.v ?? a.mean_loss), to = state === 'after' && a.ablation ? a.ablation.mean_loss : a.mean_loss;
    counter.innerHTML = `mean loss <b class="n">${fmt(from)}</b>${a.ablation ? ` <span>${state === 'after' ? `ablated L[${a.ablation.layers.join(',')}]` : 'normal'}</span>` : ''}`;
    const n = counter.querySelector('.n');
    cancelCount = countUp(n, from, to, ms + 200);
    counter.dataset.v = to;
    root.dataset.state = state;
  }

  function lossColor(d) {
    // after-state bars tinted by Δ (diverging); intensity follows |Δ|
    const dm = yD ? yD.domain()[1] : 1;
    return div(d.d / dm);
  }

  function highlightBar() {
    if (hot == null) { hl.attr('opacity', 0); readout.textContent = ''; return; }
    hl.attr('opacity', 1).attr('x', x(hot) - (x.step() - x.bandwidth()) / 2);
    const tk = a.tokens[hot];
    readout.innerHTML = `<b>${esc(vis(tk.piece))}</b> ${fmt(a.loss[hot])}${a.ablation && a.ablation.loss[hot] != null ? ` → ${fmt(a.ablation.loss[hot])} (${a.ablation.delta[hot] >= 0 ? '+' : ''}${fmt(a.ablation.delta[hot])})` : ''}`;
  }

  function idxAt(ev) {
    const [px] = d3.pointer(ev, svg.node());
    return Math.max(0, Math.min(a.tokens.length - 1, Math.floor((px - mL) / x.step())));
  }
  overlay.on('pointermove', ev => {
    const i = idxAt(ev);
    if (i !== hot) { hot = i; highlightBar(); opts.onHover?.(i); }
    const rows = [['loss', fmt(a.loss[i]), viridis((a.loss[i] ?? 0) / yL.domain()[1])]];
    if (a.ablation) {
      rows.push([`ablated L[${a.ablation.layers.join(',')}]`, fmt(a.ablation.loss[i])]);
      rows.push(['Δloss', a.ablation.delta[i] == null ? '–' : `${a.ablation.delta[i] >= 0 ? '▲ +' : '▼ '}${fmt(a.ablation.delta[i])}`, div((a.ablation.delta[i] ?? 0) / yD.domain()[1])]);
    }
    tip.show(tipHtml(`${esc(vis(a.tokens[i].piece))}<small>pos ${i}</small>`, rows, i === 0 ? 'first token has no loss' : 'Δ > 0: harder to predict without PLE'), ev);
  }).on('pointerleave', () => { hot = null; highlightBar(); tip.hide(); opts.onHover?.(null); })
    .on('click', ev => opts.onClick?.(idxAt(ev)));

  let over = false;
  root.addEventListener('pointerenter', () => { over = true; });
  root.addEventListener('pointerleave', () => { over = false; });
  const onKey = ev => {
    if (ev.code !== 'Space' || !a.ablation || (!over && document.activeElement !== root)) return;
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    ev.preventDefault();
    if (ev.type === 'keydown' && !ev.repeat && state !== 'before') show('before', 350, false);
    if (ev.type === 'keyup') show('after', 350, false);
  };
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKey);
  hold.addEventListener('pointerdown', () => a.ablation && show('before', 350, false));
  for (const e of ['pointerup', 'pointerleave']) hold.addEventListener(e, () => a.ablation && state === 'before' && show('after', 350, false));
  btn.addEventListener('click', play);

  function play() {
    show('before', 0, false);
    setTimeout(() => show('after', 900, true), dur(450));
  }

  function full(d) {
    a = d; counter.dataset.v = d.mean_loss;
    hold.style.display = btn.style.display = d.ablation ? '' : 'none';
    geometry(); staticParts();
    gBars.selectAll('*').remove(); gDelta.selectAll('*').remove(); gGhost.selectAll('*').remove();
    if (opts.autoplay !== false && d.ablation) play(); else show(d.ablation ? 'after' : 'before');
    root.setAttribute('aria-label', d.ablation
      ? `Ablating PLE layers ${d.ablation.layers.join(',')} changes the mean loss from ${fmt(d.mean_loss)} to ${fmt(d.ablation.mean_loss)}`
      : `Per-token loss, mean ${fmt(d.mean_loss)}`);
  }

  full(data);
  const off = onResize(el, () => { geometry(); staticParts(); show(state, 0, false); });
  return {
    update: full, play, show: s => show(s),
    highlight(i) { hot = i; highlightBar(); },
    destroy() { off(); cancelCount(); tip.hide(); document.removeEventListener('keydown', onKey); document.removeEventListener('keyup', onKey); el.innerHTML = ''; },
  };
}

const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
