// About tab: scroll-driven PLE data-flow diagram (IntersectionObserver), plus live numbers from the selected run.
import { countUp } from './ui.js';

const NS = 'http://www.w3.org/2000/svg';
const LAYERS = 6;
const layerY = l => 96 + l * 66;
const ROW_Y = 258;

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

/** Builds the repeated parts of the data-flow diagram (rows, slices, gates, blocks). */
function buildDiagram(svg) {
  const q = s => svg.querySelector(s);
  const mixColor = l => `color-mix(in oklab, var(--cyan) ${100 - l * 18}%, var(--violet))`;
  // table rows
  const rows = q('.fl-rows');
  for (let i = 0; i < 26; i++) {
    const y = 148 + i * 11;
    if (Math.abs(y - ROW_Y) < 6) continue;
    rows.append(el('line', { x1: 40, x2: 170, y1: y, y2: y, class: 'fl-tablerow' }));
  }
  // the highlighted row, split into 6 coloured segments
  const row = q('.fl-row');
  const segW = 130 / LAYERS;
  for (let l = 0; l < LAYERS; l++) {
    row.append(el('rect', { x: 40 + l * segW + 0.5, y: ROW_Y - 5, width: segW - 1, height: 10, rx: 2, fill: mixColor(l), class: 'fl-seg' }));
  }
  // blocks on the residual stream
  const blocks = q('.fl-blocks');
  for (let l = 0; l < LAYERS; l++) {
    const y = layerY(l);
    blocks.append(el('rect', { x: 426, y: y - 20, width: 88, height: 40, rx: 9, class: 'fl-block' }));
    blocks.append(el('text', { x: 470, y: y - 3, 'text-anchor': 'middle', class: 'fl-label fl-strong' }, `block ${l}`));
    blocks.append(el('text', { x: 470, y: y + 11, 'text-anchor': 'middle', class: 'fl-label fl-small' }, 'attn · mlp'));
  }
  // slices fan out from the row to each layer
  const slices = q('.fl-slices');
  for (let l = 0; l < LAYERS; l++) {
    const y = layerY(l), x0 = 40 + (l + 0.5) * segW;
    slices.append(el('path', {
      d: `M${x0} ${ROW_Y + (y < ROW_Y ? -5 : 5)} C ${x0 + 40} ${y}, 170 ${y}, 214 ${y}`,
      class: 'fl-draw fl-wire', pathLength: 1, stroke: mixColor(l), style: `--d:${l * 70}ms`,
    }));
    slices.append(el('rect', { x: 214, y: y - 8, width: 52, height: 16, rx: 4, fill: mixColor(l), class: 'fl-slice', style: `--d:${l * 70}ms` }));
    slices.append(el('text', { x: 240, y: y + 4, 'text-anchor': 'middle', class: 'fl-mono fl-small fl-on-slice' }, `E[t]${'₀₁₂₃₄₅'[l]}`));
  }
  // context taps (step 4)
  const taps = q('.fl-ctx-taps');
  for (let l = 0; l < LAYERS; l++) {
    const y = layerY(l);
    taps.append(el('line', { x1: 266, x2: 282, y1: y, y2: y, class: 'fl-link' }));
    taps.append(el('circle', { cx: 292, cy: y, r: 9, class: 'fl-node fl-node-ctx' }));
    taps.append(el('text', { x: 292, y: y + 4, 'text-anchor': 'middle', class: 'fl-op' }, '+'));
  }
  // gates (step 5)
  const gates = q('.fl-gates');
  for (let l = 0; l < LAYERS; l++) {
    const y = layerY(l);
    gates.append(el('line', { x1: 301, x2: 349, y1: y, y2: y, class: 'fl-link' }));
    gates.append(el('path', { d: `M426 ${y - 12} C 400 ${y - 30}, 372 ${y - 30}, 362 ${y - 11}`, class: 'fl-draw fl-gatewire', pathLength: 1, style: `--d:${l * 60}ms` }));
    gates.append(el('circle', { cx: 360, cy: y, r: 11, class: 'fl-node fl-node-gate', style: `--d:${l * 60}ms` }));
    gates.append(el('text', { x: 360, y: y + 4.5, 'text-anchor': 'middle', class: 'fl-op' }, '⊙'));
  }
  gates.append(el('text', { x: 392, y: layerY(0) - 30, 'text-anchor': 'middle', class: 'fl-label fl-amber' }, 'gelu(W_gate·h)'));
  // injection (step 6)
  const inj = q('.fl-inject');
  for (let l = 0; l < LAYERS; l++) {
    const y = layerY(l);
    inj.append(el('path', { d: `M371 ${y} H 424`, class: 'fl-draw fl-injwire', pathLength: 1, 'marker-end': 'url(#fl-arrow)', style: `--d:${l * 80}ms` }));
    inj.append(el('circle', { cx: 470, cy: y + 26, r: 3.2, class: 'fl-pulse', style: `--d:${l * 80}ms` }));
  }
  inj.append(el('text', { x: 396, y: layerY(LAYERS - 1) + 40, 'text-anchor': 'middle', class: 'fl-label fl-cyan' }, 'W_proj, RMSNorm'));
}

export function initAbout({ getRun, api }) {
  const fig = document.getElementById('flow');
  const steps = [...document.querySelectorAll('#about-steps .step')];
  if (!fig || !steps.length) return { refresh() {} };
  buildDiagram(fig);
  let current = 0;

  const setStep = n => {
    if (n === current) return;
    current = n;
    fig.dataset.step = n;
    fig.querySelectorAll('[data-s]').forEach(el => {
      const s = +el.dataset.s;
      el.classList.toggle('on', s <= n);
      el.classList.toggle('now', s === n);
    });
    steps.forEach(st => st.classList.toggle('active', +st.dataset.step === n));
    const cap = document.getElementById('flow-caption');
    const active = steps.find(st => +st.dataset.step === n);
    if (cap && active) cap.textContent = active.dataset.caption || '';
  };

  const io = new IntersectionObserver(entries => {
    // choose the step closest to the vertical middle of the viewport
    const visible = steps.map(st => ({ st, r: st.getBoundingClientRect() }))
      .filter(({ r }) => r.bottom > 0 && r.top < window.innerHeight);
    if (!visible.length || !entries.length) return;
    const mid = window.innerHeight * 0.5;
    visible.sort((a, b) => Math.abs((a.r.top + a.r.bottom) / 2 - mid) - Math.abs((b.r.top + b.r.bottom) / 2 - mid));
    setStep(+visible[0].st.dataset.step);
  }, { threshold: [0, 0.25, 0.5, 0.75, 1], rootMargin: '-20% 0px -20% 0px' });
  steps.forEach(st => io.observe(st));
  steps.forEach(st => st.addEventListener('click', () => setStep(+st.dataset.step)));
  setStep(1);

  // count up the result numbers the first time they scroll into view
  const counters = [...document.querySelectorAll('.result b[data-count]')];
  const cio = new IntersectionObserver(entries => entries.forEach(e => {
    if (!e.isIntersecting) return;
    cio.unobserve(e.target);
    const b = e.target;
    b.dataset.value = 0;
    countUp(b, +b.dataset.count, { decimals: +b.dataset.dec || 0, prefix: b.dataset.prefix || '', suffix: b.dataset.suffix || '', duration: 1400 });
  }), { threshold: 0.6 });
  counters.forEach(b => cio.observe(b));

  const set = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.textContent = v; };
  let lastRun = null;
  return {
    async refresh() {
      const run = getRun();
      if (!run || run.name === lastRun) return;
      lastRun = run.name;
      const c = run.config, M = n => `${(n / 1e6).toFixed(1)}M`;
      set('ab-V', c.vocab_size.toLocaleString('en-US'));
      set('ab-L', c.n_layer); set('ab-L2', c.n_layer); set('ab-L3', c.n_layer);
      set('ab-p', c.ple_dim); set('ab-p2', c.ple_dim);
      set('ab-Lp', (c.n_layer * c.ple_dim).toLocaleString('en-US'));
      set('ab-d', c.d_model);
      set('ab-table', run.params?.ple_table ? M(run.params.ple_table) : '0');
      set('ab-run', run.name);
      if (!run.has_ple) { set('ab-tid', 't'); return; }
      try {
        const t = await api(`/api/token?${new URLSearchParams({ run: run.name, q: ' dog' })}`);
        set('ab-tid', t.id); set('ab-tid2', t.id);
      } catch { /* the diagram works without the live id */ }
    },
    setStep,
  };
}
