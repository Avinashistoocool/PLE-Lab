// Command palette (Ctrl/Cmd+K) with fuzzy matching and keyboard navigation.

function escHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Subsequence fuzzy score; returns null when not all query chars occur in order. */
export function fuzzy(query, text) {
  const q = query.toLowerCase().trim(), t = text.toLowerCase();
  if (!q) return { score: 0, hits: [] };
  const direct = t.indexOf(q);
  if (direct >= 0) return { score: 1000 - direct * 2 - t.length * 0.1, hits: Array.from(q, (_, i) => direct + i) };
  let ti = 0, score = 0, prev = -2;
  const hits = [];
  for (const ch of q) {
    if (ch === ' ') continue;
    const found = t.indexOf(ch, ti);
    if (found < 0) return null;
    score += found === prev + 1 ? 8 : 1;
    if (found === 0 || /[\s\-_:'"(]/.test(t[found - 1])) score += 6;
    hits.push(found); prev = found; ti = found + 1;
  }
  return { score: score - t.length * 0.05, hits };
}

function highlight(text, hits) {
  const set = new Set(hits);
  return Array.from(text, (ch, i) => (set.has(i) ? `<mark>${escHTML(ch)}</mark>` : escHTML(ch))).join('');
}

/**
 * commands(): [{id, title, group, hint?, keys?, icon?, run()}]
 * dynamic(query): extra commands that depend on the typed text (e.g. "Inspect token …").
 */
export function createPalette({ commands, dynamic }) {
  const root = document.getElementById('palette');
  const input = root.querySelector('input');
  const list = root.querySelector('[role=listbox]');
  let items = [], active = 0, lastFocus = null;

  const render = () => {
    const q = input.value;
    const scored = [];
    for (const c of commands()) {
      const m = fuzzy(q, c.title) || (c.keywords && fuzzy(q, c.keywords) ? { score: 1, hits: [] } : null);
      if (m) scored.push({ c, m });
    }
    scored.sort((a, b) => b.m.score - a.m.score);
    const dyn = q.trim() ? dynamic(q).map(c => ({ c, m: { score: 0, hits: [] } })) : [];
    // direct substring matches of real commands win over the free-text "inspect/analyze" fallbacks
    const strong = scored.filter(s => s.m.score >= 500);
    items = q.trim() ? [...strong, ...dyn.slice(0, 3), ...scored.filter(s => s.m.score < 500)] : scored;
    items = items.slice(0, 14);
    active = Math.min(active, Math.max(0, items.length - 1));
    let group = null;
    list.innerHTML = items.length ? items.map(({ c, m }, i) => {
      const head = !q.trim() && c.group !== group ? `<li class="pal-group" role="presentation">${escHTML((group = c.group))}</li>` : '';
      return `${head}<li role="option" id="pal-opt-${i}" data-i="${i}" class="pal-item${i === active ? ' active' : ''}" aria-selected="${i === active}">
        <span class="pal-icon" aria-hidden="true">${c.icon || '›'}</span>
        <span class="pal-title">${highlight(c.title, m.hits)}</span>
        ${c.hint ? `<span class="pal-hint">${escHTML(c.hint)}</span>` : ''}
        ${c.keys ? `<span class="pal-keys">${c.keys.map(k => `<kbd>${escHTML(k)}</kbd>`).join('')}</span>` : ''}
      </li>`;
    }).join('') : '<li class="pal-empty" role="presentation">No matching command. Try "map", "theme" or a token like " dog".</li>';
    input.setAttribute('aria-activedescendant', items.length ? `pal-opt-${active}` : '');
    list.querySelector('.pal-item.active')?.scrollIntoView({ block: 'nearest' });
  };

  const open = (prefill = '') => {
    lastFocus = document.activeElement;
    root.hidden = false;
    requestAnimationFrame(() => root.classList.add('open'));
    input.value = prefill; active = 0; render();
    input.focus();
  };
  const close = () => {
    if (root.hidden) return;
    root.classList.remove('open');
    root.hidden = true;
    if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
  };
  const exec = i => {
    const it = items[i];
    if (!it) return;
    close();
    it.c.run();
  };

  input.addEventListener('input', () => { active = 0; render(); });
  input.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); active = (active + 1) % Math.max(1, items.length); render(); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); active = (active - 1 + items.length) % Math.max(1, items.length); render(); }
    else if (ev.key === 'Enter') { ev.preventDefault(); exec(active); }
    else if (ev.key === 'Escape') { ev.preventDefault(); close(); }
    else if (ev.key === 'Tab') ev.preventDefault();
  });
  list.addEventListener('mousemove', ev => {
    const li = ev.target.closest('.pal-item');
    if (li && +li.dataset.i !== active) { active = +li.dataset.i; list.querySelectorAll('.pal-item').forEach(e => { const on = +e.dataset.i === active; e.classList.toggle('active', on); e.setAttribute('aria-selected', on); }); }
  });
  list.addEventListener('click', ev => { const li = ev.target.closest('.pal-item'); if (li) exec(+li.dataset.i); });
  root.addEventListener('mousedown', ev => { if (ev.target === root) close(); });

  return { open, close, isOpen: () => !root.hidden, toggle: () => (root.hidden ? open() : close()) };
}
