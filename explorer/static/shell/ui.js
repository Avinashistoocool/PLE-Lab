// Shell UI helpers: theme, toasts, count-up tickers, info popovers, segmented controls, token text, skeletons.

const html = document.documentElement;
const THEME_KEY = 'ple.theme';
const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
export const reducedMotion = () => motionQuery.matches;

function escHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---- token text with visible whitespace ------------------------------------------
/** Plain-text token with visible whitespace (for canvas labels): space → ·, newline → ⏎, tab → ⇥. */
export function visTok(piece) {
  return String(piece).replace(/ /g, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
}
/** Escaped HTML for a token with dimmed whitespace glyphs. */
export function tokHTML(piece) {
  return String(piece).replace(/[ \n\t]|[^ \n\t]+/g, m => {
    if (m === ' ') return '<span class="ws">·</span>';
    if (m === '\n') return '<span class="ws">⏎</span>';
    if (m === '\t') return '<span class="ws">⇥</span>';
    return escHTML(m);
  });
}

/** Readable text colour (dark or light) for a CSS rgb() background. */
export function inkFor(rgb) {
  const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(rgb || '');
  if (!m) return null;
  const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * lin(+m[1]) + 0.7152 * lin(+m[2]) + 0.0722 * lin(+m[3]);
  return L > 0.2 ? '#06101c' : '#ffffff';
}

// ---- theme -------------------------------------------------------------------------
const themeListeners = new Set();
export const currentTheme = () => (html.dataset.theme === 'light' ? 'light' : 'dark');
export function setTheme(t) {
  html.dataset.theme = t;
  try { localStorage.setItem(THEME_KEY, t); } catch { /* private mode */ }
  document.querySelector('meta[name="color-scheme"]')?.setAttribute('content', t);
  themeListeners.forEach(fn => fn(t));
}
export const toggleTheme = () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
export const onTheme = fn => themeListeners.add(fn);

// ---- toasts ------------------------------------------------------------------------
const ICONS = {
  error: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8"/><path d="M10 5.5v5.5M10 13.6v.4"/></svg>',
  warn: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2.5 18.5 17h-17z"/><path d="M10 8v4.5M10 14.6v.4"/></svg>',
  info: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8"/><path d="M10 9v5.5M10 6v.4"/></svg>',
  ok: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8"/><path d="m6.5 10.2 2.4 2.4 4.6-5"/></svg>',
};
const TITLES = { error: 'Error', warn: 'Heads up', info: 'Info', ok: 'Done' };
const live = new Map();
/** kind: 'error' | 'warn' | 'info' | 'ok'. Errors stay longer and are announced assertively. */
export function toast(msg, kind = 'info', { timeout } = {}) {
  const host = document.getElementById('toasts');
  if (!host || !msg) return;
  const key = `${kind}:${msg}`;
  if (live.has(key)) { live.get(key).bump(); return; }
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.innerHTML = `<span class="toast-icon">${ICONS[kind] || ICONS.info}</span>
    <div class="toast-body"><b>${TITLES[kind] || 'Info'}</b><span></span></div>
    <button class="toast-x" type="button" aria-label="Dismiss notification">×</button><i class="toast-timer"></i>`;
  el.querySelector('.toast-body span').textContent = msg;
  const ms = timeout ?? (kind === 'error' ? 9000 : kind === 'warn' ? 7000 : 3200);
  let timer;
  const close = () => {
    clearTimeout(timer); live.delete(key);
    el.classList.add('out');
    setTimeout(() => el.remove(), reducedMotion() ? 0 : 260);
  };
  const arm = () => {
    clearTimeout(timer); timer = setTimeout(close, ms);
    el.style.setProperty('--toast-ms', `${ms}ms`);
    const bar = el.querySelector('.toast-timer');
    bar.style.animation = 'none'; void bar.offsetWidth; bar.style.animation = '';
  };
  el.querySelector('.toast-x').addEventListener('click', close);
  el.addEventListener('mouseenter', () => { clearTimeout(timer); el.classList.add('paused'); });
  el.addEventListener('mouseleave', () => { el.classList.remove('paused'); arm(); });
  live.set(key, { bump: () => { el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump'); arm(); }, close });
  host.appendChild(el);
  const all = [...live.values()];
  if (all.length > 4) all[0].close();
  arm();
}
export const clearToasts = kind => [...live.entries()].forEach(([k, v]) => { if (!kind || k.startsWith(`${kind}:`)) v.close(); });

// ---- count-up ticker -----------------------------------------------------------------
export function countUp(el, value, { decimals = 0, prefix = '', suffix = '', duration = 1100 } = {}) {
  if (!el) return;
  const fmtN = v => `${prefix}${v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
  if (value == null || !Number.isFinite(value)) { el.textContent = '–'; delete el.dataset.value; return; }
  const from = Number.isFinite(+el.dataset.value) && el.dataset.value !== undefined ? +el.dataset.value : 0;
  el.dataset.value = value;
  if (reducedMotion() || from === value) { el.textContent = fmtN(value); return; }
  const t0 = performance.now();
  const step = now => {
    if (+el.dataset.value !== value) return;
    const k = Math.min(1, (now - t0) / duration), e = 1 - (1 - k) ** 4;
    el.textContent = fmtN(from + (value - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---- info popovers -------------------------------------------------------------------
export function initPopovers(texts) {
  const pop = document.createElement('div');
  pop.className = 'popover'; pop.id = 'info-popover'; pop.setAttribute('role', 'dialog'); pop.hidden = true;
  document.body.appendChild(pop);
  let owner = null;
  const close = () => {
    if (!owner) return;
    owner.setAttribute('aria-expanded', 'false'); owner = null; pop.hidden = true;
  };
  const open = btn => {
    const t = texts[btn.dataset.info];
    if (!t) return;
    owner = btn; btn.setAttribute('aria-expanded', 'true');
    pop.innerHTML = `<h4>${escHTML(t.title)}</h4>${t.body}`;
    pop.setAttribute('aria-label', t.title);
    pop.hidden = false;
    const r = btn.getBoundingClientRect(), pr = pop.getBoundingClientRect();
    const x = Math.min(window.innerWidth - pr.width - 12, Math.max(12, r.left - 16));
    let y = r.bottom + 10;
    if (y + pr.height > window.innerHeight - 8) y = Math.max(8, r.top - pr.height - 10);
    pop.style.left = `${x}px`; pop.style.top = `${y}px`;
  };
  const decorate = () => document.querySelectorAll('.info[data-info]:not([aria-controls])').forEach(b => {
    b.type = 'button';
    b.setAttribute('aria-expanded', 'false'); b.setAttribute('aria-controls', 'info-popover');
    if (!b.getAttribute('aria-label')) b.setAttribute('aria-label', `Explain: ${texts[b.dataset.info]?.title ?? 'this view'}`);
    if (!b.textContent.trim()) b.textContent = '?';
  });
  decorate();
  document.addEventListener('click', ev => {
    const b = ev.target.closest('.info[data-info]');
    if (b) { ev.preventDefault(); ev.stopPropagation(); const same = owner === b; close(); if (!same) open(b); return; }
    if (!ev.target.closest('#info-popover')) close();
  });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && owner) { const o = owner; close(); o.focus(); } });
  window.addEventListener('scroll', close, { passive: true });
  window.addEventListener('resize', close);
  return { close, decorate, isOpen: () => !!owner };
}

// ---- segmented control bound to a <select> ---------------------------------------------
const segs = new Set();
export function segmented(select, { label, format } = {}) {
  const seg = document.createElement('div');
  seg.className = 'seg';
  seg.setAttribute('role', 'radiogroup');
  seg.setAttribute('aria-label', label || select.getAttribute('aria-label') || 'options');
  select.classList.add('seg-src');
  select.tabIndex = -1;
  select.setAttribute('aria-hidden', 'true');
  select.insertAdjacentElement('afterend', seg);
  const sync = () => {
    seg.querySelectorAll('button').forEach(b => {
      const on = b.dataset.v === select.value;
      b.setAttribute('aria-checked', String(on)); b.tabIndex = on ? 0 : -1; b.classList.toggle('on', on);
    });
    const on = seg.querySelector('button.on');
    if (on) {
      seg.style.setProperty('--seg-x', `${on.offsetLeft}px`);
      seg.style.setProperty('--seg-w', `${on.offsetWidth}px`);
      seg.classList.toggle('has-pill', on.offsetWidth > 0);
    }
  };
  const render = () => {
    seg.innerHTML = [...select.options].map(o =>
      `<button type="button" role="radio" data-v="${escHTML(o.value)}" title="${escHTML(o.title || o.text)}">${format ? format(o) : escHTML(o.text)}</button>`).join('');
    sync();
  };
  const choose = v => {
    if (select.value === v) return;
    select.value = v; sync();
    select.dispatchEvent(new Event('change', { bubbles: true }));
  };
  seg.addEventListener('click', ev => { const b = ev.target.closest('button[data-v]'); if (b) choose(b.dataset.v); });
  seg.addEventListener('keydown', ev => {
    const bs = [...seg.querySelectorAll('button')], i = bs.findIndex(b => b.dataset.v === select.value);
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[ev.key];
    if (!d || i < 0) return;
    ev.preventDefault();
    const nb = bs[(i + d + bs.length) % bs.length];
    choose(nb.dataset.v); nb.focus();
  });
  new MutationObserver(render).observe(select, { childList: true });
  select.addEventListener('change', sync);
  render();
  const api = { sync, render, el: seg };
  segs.add(api);
  return api;
}
export const syncSegments = () => segs.forEach(s => s.sync());

// ---- skeletons ---------------------------------------------------------------------------
export function skeleton(el, height = 160, kind = 'chart') {
  if (!el) return;
  if (kind === 'table') {
    el.innerHTML = `<tbody>${Array.from({ length: 6 }, (_, i) =>
      `<tr><td colspan="5"><div class="skel skel-line" style="width:${92 - i * 9}%"></div></td></tr>`).join('')}</tbody>`;
    return;
  }
  if (kind === 'chips') {
    el.innerHTML = Array.from({ length: 22 }, (_, i) =>
      `<span class="skel skel-chip" style="width:${28 + ((i * 37) % 50)}px"></span>`).join('');
    return;
  }
  el.innerHTML = `<div class="skel" style="height:${height}px" aria-hidden="true"></div>`;
}
