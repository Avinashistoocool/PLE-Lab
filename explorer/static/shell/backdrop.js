// Animated backdrop: six horizontal "per-layer" bands of glowing dots that ripple, with an occasional
// vertical token beam lighting up one column across all layers. Canvas 2D, ~30 fps, pauses when hidden.

export function startBackdrop(canvas, { reducedMotion = () => false } = {}) {
  const ctx = canvas.getContext('2d', { alpha: true });
  const BANDS = 6;
  let w = 0, h = 0, dpr = 1, cols = 0, raf = 0, last = 0, running = false;
  let sprites = {}, palette = null;
  const beams = [];
  let nextBeam = 1500;

  const readPalette = () => {
    const cs = getComputedStyle(document.documentElement);
    const get = (n, d) => (cs.getPropertyValue(n).trim() || d);
    palette = {
      a: get('--bg-dot-a', '77,225,255'), b: get('--bg-dot-b', '166,124,255'),
      alpha: parseFloat(get('--bg-dot-alpha', '0.55')) || 0.55,
    };
    sprites = {};
  };

  // Pre-rendered glow sprites keep per-frame cost to a few hundred drawImage calls.
  const sprite = (rgb, size) => {
    const key = `${rgb}|${size}`;
    if (sprites[key]) return sprites[key];
    const c = document.createElement('canvas'), s = Math.ceil(size * 2 * dpr);
    c.width = c.height = s;
    const g = c.getContext('2d'), r = s / 2;
    const grad = g.createRadialGradient(r, r, 0, r, r, r);
    grad.addColorStop(0, `rgba(${rgb},1)`);
    grad.addColorStop(0.25, `rgba(${rgb},0.55)`);
    grad.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    return (sprites[key] = c);
  };

  const resize = () => {
    dpr = Math.min(1.5, window.devicePixelRatio || 1);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
    cols = Math.max(24, Math.min(72, Math.round(w / 26)));
    sprites = {};
  };

  const mix = (a, b, t) => {
    const pa = a.split(',').map(Number), pb = b.split(',').map(Number);
    return pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(',');
  };

  const draw = (t) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'lighter';
    const top = h * 0.18, span = h * 0.7, gap = span / (BANDS - 1);
    const dx = w / (cols - 1);
    const s = t / 1000;
    for (let b = 0; b < BANDS; b++) {
      const rgb = mix(palette.a, palette.b, b / (BANDS - 1));
      const spr = sprite(rgb, 9);
      const y0 = top + b * gap;
      const phase = b * 0.9;
      for (let c = 0; c < cols; c++) {
        const x = c * dx;
        const wave = Math.sin(c * 0.22 + s * 0.55 + phase) + 0.6 * Math.sin(c * 0.071 - s * 0.31 + b);
        const y = y0 + wave * 7;
        let k = 0.18 + 0.22 * (0.5 + 0.5 * Math.sin(c * 0.37 + s * 0.8 + b * 1.7));
        for (const bm of beams) {
          const d = Math.abs(c - bm.col);
          if (d < 3) {
            const local = bm.age - b * 140;
            if (local > 0 && local < 1400) k += (1 - d / 3) * Math.sin(Math.PI * local / 1400) * 1.1;
          }
        }
        const size = (6 + k * 7) * dpr;
        ctx.globalAlpha = Math.min(1, k * palette.alpha * 1.6);
        ctx.drawImage(spr, x * dpr - size / 2, y * dpr - size / 2, size, size);
      }
    }
    // the beam itself: a thin vertical streak travelling down through the layers
    for (const bm of beams) {
      const x = bm.col * dx * dpr, prog = Math.min(1, bm.age / (BANDS * 140 + 700));
      const y1 = (top - 40 + (span + 80) * prog) * dpr;
      const g = ctx.createLinearGradient(0, y1 - 160 * dpr, 0, y1);
      g.addColorStop(0, `rgba(${palette.a},0)`);
      g.addColorStop(1, `rgba(${palette.a},${0.35 * (1 - prog * 0.6)})`);
      ctx.globalAlpha = 1; ctx.fillStyle = g;
      ctx.fillRect(x - 1 * dpr, y1 - 160 * dpr, 2 * dpr, 160 * dpr);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  };

  const frame = (now) => {
    raf = requestAnimationFrame(frame);
    if (now - last < 33) return; // ~30 fps is plenty for an ambient backdrop
    const dt = last ? Math.min(100, now - last) : 33;
    last = now;
    nextBeam -= dt;
    if (nextBeam <= 0) {
      beams.push({ col: 2 + Math.floor(Math.random() * (cols - 4)), age: 0 });
      nextBeam = 2600 + Math.random() * 3000;
    }
    for (const bm of beams) bm.age += dt;
    while (beams.length && beams[0].age > BANDS * 140 + 1600) beams.shift();
    draw(now);
  };

  const start = () => {
    if (running || reducedMotion() || document.hidden) return;
    running = true; last = 0; raf = requestAnimationFrame(frame);
  };
  const stop = () => { running = false; cancelAnimationFrame(raf); };
  // Reduced motion: the backdrop is off entirely.
  const still = () => { stop(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); };

  readPalette(); resize();
  if (reducedMotion()) still(); else start();
  window.addEventListener('resize', () => { resize(); if (!running && !reducedMotion()) draw(4000); });
  document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
  window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener?.('change', () => (reducedMotion() ? still() : start()));
  return {
    refresh() { readPalette(); if (!running && !reducedMotion()) draw(4000); },
    stop, start,
  };
}
