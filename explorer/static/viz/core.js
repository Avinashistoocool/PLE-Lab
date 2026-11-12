// Shared helpers for the PLE Explorer viz modules: theme tokens, colour maps, motion, tooltip, sizing.
// No global state besides one lazily created tooltip element and one injected stylesheet link.
import d3 from '../vendor/d3/d3.js';

export { d3 };

// ---- theme (reads the "Instrument Panel" CSS custom properties, docs/DESIGN_IDEAS.md §6) -----------
const FALLBACK = {
  '--bg': '#0a0d14', '--bg-grid': '#111624', '--surface': '#10141e', '--surface-2': '#161b28', '--surface-3': '#1d2333',
  '--hairline': '#232a3b', '--hairline-strong': '#2f3850',
  '--text': '#e6e9f2', '--text-muted': '#8a93a8', '--text-faint': '#5b6478',
  '--accent': '#22d3ee', '--accent-dim': '#0e7490', '--ple': '#f5a524', '--gate': '#a78bfa', '--ctx': '#4fc3f7',
  '--pos': '#f97316', '--neg': '#3b82f6', '--zero': '#2a2f3d', '--ok': '#34d399', '--warn': '#fbbf24', '--err': '#fb7185',
  '--l0': '#f5a524', '--l1': '#4fc3f7', '--l2': '#2dd4a7', '--l3': '#f472b6', '--l4': '#a78bfa', '--l5': '#e8e36b',
  '--font-ui': '"Inter Variable","Inter","Segoe UI",system-ui,sans-serif',
  '--font-mono': '"JetBrains Mono Variable","JetBrains Mono","Cascadia Mono",Consolas,monospace',
};

export function cssVar(name, fallback = FALLBACK[name]) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Snapshot of all tokens; call per render so a theme change is picked up on the next update(). */
export function theme() {
  const g = n => cssVar(n);
  return {
    bg: g('--bg'), surface: g('--surface'), surface2: g('--surface-2'), surface3: g('--surface-3'),
    hairline: g('--hairline'), hairlineStrong: g('--hairline-strong'),
    text: g('--text'), muted: g('--text-muted'), faint: g('--text-faint'),
    accent: g('--accent'), ple: g('--ple'), gate: g('--gate'), ctx: g('--ctx'),
    pos: g('--pos'), neg: g('--neg'), zero: g('--zero'), ok: g('--ok'), warn: g('--warn'), err: g('--err'),
    layers: [0, 1, 2, 3, 4, 5].map(l => g(`--l${l}`)),
    fontUi: g('--font-ui'), fontMono: g('--font-mono'),
  };
}

export const layerColor = (th, l) => th.layers[((l % th.layers.length) + th.layers.length) % th.layers.length];

// ---- colour maps -------------------------------------------------------------------------------------
/** Viridis for magnitudes, t in [0,1]; starts at 0.18 so low values stay visible on a near-black bg. */
export const viridis = t => d3.interpolateViridis(0.18 + 0.82 * clamp01(t));
/** Blue (neg) ↔ surface-neutral ↔ orange (pos); returns f(t) with t in [-1, 1]. */
export function diverging(th = theme()) {
  const lo = d3.interpolateRgb(th.zero, th.neg), hi = d3.interpolateRgb(th.zero, th.pos);
  return t => {
    t = Math.max(-1, Math.min(1, Number.isFinite(t) ? t : 0));
    const k = Math.pow(Math.abs(t), 0.75);  // lift small values a little
    return t < 0 ? lo(k) : hi(k);
  };
}
export const clamp01 = t => Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
/** colour with alpha (any CSS colour d3 understands). */
export function alpha(color, a) {
  const c = d3.color(color);
  if (!c) return color;
  c.opacity = a;
  return c.formatRgb();
}

// ---- motion ------------------------------------------------------------------------------------------
export function reducedMotion() {
  const m = document.documentElement.dataset.motion;
  if (m === 'off' || m === 'reduced') return true;
  return !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
}
export const dur = ms => (reducedMotion() ? 0 : ms);
export const easeOut = d3.easeExpOut;
export const easeInOut = d3.easeCubicInOut;

/** rAF tween: fn(k) with k eased 0→1. Returns cancel(). Reduced motion: one call with k=1. */
export function tween(ms, fn, ease = d3.easeCubicOut) {
  ms = dur(ms);
  if (!ms) { fn(1); return () => {}; }
  let raf = 0, done = false;
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / ms);
    fn(ease(k));
    if (k < 1 && !done) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  return () => { done = true; cancelAnimationFrame(raf); };
}

// ---- text helpers ----------------------------------------------------------------------------------
export function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
/** Visible token text: leading spaces as '␣', newlines '⏎', tabs '⇥'. */
export const vis = p => String(p).replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
export function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  return (+v.toFixed(d)).toString();
}
export const signed = (v, d = 3) => (v == null ? '–' : `${v >= 0 ? '+' : '−'}${fmt(Math.abs(v), d)}`);

// ---- stylesheet + tooltip ----------------------------------------------------------------------------
export function ensureStyles() {
  if (document.getElementById('viz-css')) return;
  const link = document.createElement('link');
  link.id = 'viz-css'; link.rel = 'stylesheet';
  link.href = new URL('./viz.css', import.meta.url).href;
  document.head.appendChild(link);
}

let tipEl = null;
export const tip = {
  /** html: string (use tipHtml for the standard layout); ev: a pointer/mouse event. */
  show(html, ev) {
    ensureStyles();
    if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'viz-tip'; tipEl.setAttribute('role', 'tooltip'); document.body.appendChild(tipEl); }
    tipEl.innerHTML = html;
    tipEl.classList.add('on');
    const pad = 14, r = tipEl.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 4) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 4) y = ev.clientY - r.height - pad;
    tipEl.style.transform = `translate(${Math.max(4, x)}px, ${Math.max(4, y)}px)`;
  },
  hide() { if (tipEl) tipEl.classList.remove('on'); },
};
/** Standard tooltip body: mono header + key/value rows. rows: [[key, value, colour?], ...] */
export function tipHtml(head, rows = [], foot = '') {
  return `<div class="viz-tip-h">${head}</div>` +
    rows.map(([k, v, c]) => `<div class="viz-tip-r"><span>${c ? `<i style="background:${c}"></i>` : ''}${k}</span><b>${v}</b></div>`).join('') +
    (foot ? `<div class="viz-tip-f">${foot}</div>` : '');
}

// ---- sizing ------------------------------------------------------------------------------------------
export function widthOf(el, fallback = 600) {
  return Math.max(120, Math.floor(el.clientWidth || el.getBoundingClientRect().width || fallback));
}

/** Calls cb(width) when el's width changes by > 2px (debounced). Returns disconnect(). */
export function onResize(el, cb) {
  if (!window.ResizeObserver) return () => {};
  let last = widthOf(el), t = 0;
  const ro = new ResizeObserver(() => {
    clearTimeout(t);
    t = setTimeout(() => { const w = widthOf(el); if (Math.abs(w - last) > 2) { last = w; cb(w); } }, 120);
  });
  ro.observe(el);
  return () => { clearTimeout(t); ro.disconnect(); };
}

export function hidpiCanvas(canvas, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Root wrapper every chart renders into (keeps the user's container otherwise untouched). */
export function mount(el, cls, label) {
  ensureStyles();
  const root = document.createElement('div');
  root.className = `viz-root ${cls}`;
  if (label) { root.setAttribute('role', 'img'); root.setAttribute('aria-label', label); }
  el.innerHTML = '';
  el.appendChild(root);
  return root;
}

/** Segmented control: items [{value,label,title?}] → element; onChange(value). */
export function segmented(items, value, onChange, cls = '') {
  const wrap = document.createElement('div');
  wrap.className = `viz-seg ${cls}`;
  wrap.setAttribute('role', 'radiogroup');
  const set = v => wrap.querySelectorAll('button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.v === String(v))));
  for (const it of items) {
    const b = document.createElement('button');
    b.type = 'button'; b.dataset.v = String(it.value); b.setAttribute('role', 'radio');
    b.innerHTML = it.label; if (it.title) b.title = it.title;
    b.addEventListener('click', () => { set(it.value); onChange(it.value); });
    wrap.appendChild(b);
  }
  set(value);
  wrap.set = set;
  return wrap;
}

/** Animated number: counts el's text from a to b. */
export function countUp(el, a, b, ms = 700, d = 3) {
  return tween(ms, k => { el.textContent = fmt(a + (b - a) * k, d); }, d3.easeCubicOut);
}

export const nonNull = xs => xs.filter(v => v != null && Number.isFinite(v));
