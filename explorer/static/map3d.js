/* PLE Explorer: interactive 3D token map (three.js r170, vendored, no build step).
 *
 * API
 * ---
 *   import { mountMap3D } from '/static/map3d.js';
 *   const map = await mountMap3D(containerEl, {
 *     run: 'ple-small',                       // run name (required)
 *     api: (path, body) => Promise<json>,     // optional; default: fetch(path) and throw on HTTP errors
 *     onTokenClick: id => {...},              // optional; double-click / "Open" button in the info card
 *     theme: { bg, fg, accent, accent2, muted } // optional CSS colours; default: the container's CSS vars
 *   });
 *   await map.setRun(name);      // load another run
 *   map.setLayer(l);             // l = 0..n_layer-1 or 'full' (tweened morph)
 *   map.highlight(query);        // find a token (' dog', 'dog', '#3290'), fly to it and pulse it
 *   map.resize();                // force a resize (a ResizeObserver also does this)
 *   map.destroy();               // stop RAF, dispose GPU resources, remove listeners/observers and DOM
 *
 * Data: GET /api/projection3d?run&n&space=shared (one PCA over all layer slices, so layers share axes)
 * and GET /api/trajectory?run&id&n&space&k (a token's path through the layers + its k neighbours).
 * For display every layer's cloud (and the full-row cloud, which has its own PCA) is centred and scaled
 * to the same RMS radius, so the morph shows tokens re-arranging rather than the cloud drifting.
 */
import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/addons/controls/OrbitControls.js';
import { EffectComposer } from './vendor/three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from './vendor/three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from './vendor/three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from './vendor/three/addons/renderers/CSS2DRenderer.js';

const R = 4;               // display RMS radius of every layer cloud
const N_LABELS = 40;
const K_NEIGH = 8;
const MORPH_MS = 800;
const FLY_MS = 1300;
const IDLE_RESUME_MS = 6000;

const CLUSTER_PALETTE = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#f87171',
  '#22d3ee', '#a3e635', '#fb923c', '#e879f9', '#2dd4bf', '#e2e8f0'];

// ---------------------------------------------------------------- helpers
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const vis = p => String(p).replace(/ /g, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥') || '∅';
const fmtInt = v => (v == null ? '–' : Number(v).toLocaleString('en-US'));
const fmt = (v, d = 3) => (v == null ? '–' : Number(v).toFixed(d));

function turbo(t) {
  t = clamp(t);
  const r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [clamp(r), clamp(g), clamp(b)];
}
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function viridis(t) {
  t = clamp(t) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i;
  return VIRIDIS[i].map((c, k) => (c + (VIRIDIS[i + 1][k] - c) * f) / 255);
}
const rampCss = (fn, a = 0, b = 1, steps = 12) => `linear-gradient(90deg, ${Array.from({ length: steps }, (_, i) => {
  const c = fn(a + (b - a) * i / (steps - 1)); return `rgb(${c.map(x => Math.round(x * 255)).join(',')})`;
}).join(',')})`;
const srgbToLinear = rgb => { const c = new THREE.Color(); c.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace); return c; };

function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]; }

async function defaultApi(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}

function readTheme(container, theme = {}) {
  const cs = getComputedStyle(container);
  const v = (name, fb) => (cs.getPropertyValue(name) || '').trim() || fb;
  return {
    bg: theme.bg || v('--bg', '#0b0d14'),
    fg: theme.fg || v('--text', '#d7dbe6'),
    accent: theme.accent || v('--accent', '#60a5fa'),
    accent2: theme.accent2 || v('--accent2', '#a78bfa'),
    muted: theme.muted || v('--muted', '#8b93a7'),
  };
}

// ---------------------------------------------------------------- CSS (injected once, ref-counted)
const STYLE_ID = 'm3d-style';
let styleRefs = 0;
const CSS = `
.m3d-root{position:relative;width:100%;height:100%;min-height:320px;overflow:hidden;border-radius:inherit;
  background:radial-gradient(ellipse at 50% 45%, #161b2e 0%, #0b0d16 55%, #05060a 100%);color:var(--m3d-fg);
  font:13px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;outline:none;user-select:none;-webkit-user-select:none}
.m3d-root canvas.m3d-canvas{position:absolute;inset:0;display:block;width:100%;height:100%;cursor:grab}
.m3d-root canvas.m3d-canvas.m3d-hovering{cursor:pointer}
.m3d-root canvas.m3d-canvas:active{cursor:grabbing}
.m3d-labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.m3d-label{font:500 11px/1 ui-monospace,SFMono-Regular,Consolas,monospace;color:#e6ebff;white-space:pre;
  padding:2px 5px;border-radius:4px;background:rgba(10,12,20,.45);border:1px solid rgba(255,255,255,.06);
  text-shadow:0 0 6px rgba(0,0,0,.9);pointer-events:none;transition:opacity .25s;margin-top:-14px}
.m3d-label.m3d-nb{color:#fff;background:rgba(20,24,40,.7);border-color:var(--m3d-accent2);font-size:12px}
.m3d-label.m3d-sel{color:#fff;background:linear-gradient(90deg,var(--m3d-accent),var(--m3d-accent2));border:none;
  font-size:13px;font-weight:700;padding:3px 8px;box-shadow:0 0 18px var(--m3d-accent);margin-top:-22px}
.m3d-label.m3d-lm{font-size:10px;font-weight:700;background:rgba(5,6,10,.8);color:var(--m3d-accent);
  border:1px solid var(--m3d-accent);border-radius:9px;min-width:12px;text-align:center;padding:2px 4px;margin-top:-18px}
.m3d-label.m3d-cl{font:600 11px system-ui,sans-serif;color:#fff;background:rgba(8,10,18,.72);border-left:3px solid var(--c);padding:3px 7px;transition:opacity .4s}
.m3d-label.m3d-axis{background:none;border:none;color:var(--m3d-muted);opacity:.55;font-size:10px}
.m3d-panel{position:absolute;background:rgba(14,16,26,.62);backdrop-filter:blur(12px) saturate(140%);
  -webkit-backdrop-filter:blur(12px) saturate(140%);border:1px solid rgba(255,255,255,.08);border-radius:12px;
  box-shadow:0 8px 32px rgba(0,0,0,.35);pointer-events:auto}
.m3d-title{position:absolute;left:18px;top:14px;pointer-events:none;max-width:min(560px,60%)}
.m3d-h{font-size:17px;font-weight:650;letter-spacing:.2px;color:#fff;text-shadow:0 2px 12px rgba(0,0,0,.6)}
.m3d-h b{background:linear-gradient(90deg,var(--m3d-accent),var(--m3d-accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.m3d-sub{color:var(--m3d-muted);font-size:12px;margin-top:3px}
.m3d-controls{right:14px;top:14px;padding:10px 12px;display:flex;flex-direction:column;gap:8px;width:230px}
.m3d-row{display:flex;align-items:center;gap:8px;justify-content:space-between}
.m3d-row>span{color:var(--m3d-muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.m3d-root select,.m3d-root input[type=text]{background:rgba(255,255,255,.05);color:var(--m3d-fg);border:1px solid rgba(255,255,255,.1);
  border-radius:7px;padding:4px 7px;font:inherit;outline:none}
.m3d-root select{flex:1;max-width:140px}
.m3d-root select option{background:#141724}
.m3d-root input[type=text]{width:100%;box-sizing:border-box;padding:6px 9px;font-family:ui-monospace,Consolas,monospace}
.m3d-root input[type=text]:focus,.m3d-root select:focus{border-color:var(--m3d-accent);box-shadow:0 0 0 3px rgba(96,165,250,.18)}
.m3d-status{font-size:11px;color:var(--m3d-muted);min-height:14px}
.m3d-switch{display:flex;align-items:center;gap:8px;cursor:pointer}
.m3d-switch input{appearance:none;-webkit-appearance:none;width:30px;height:17px;border-radius:9px;background:rgba(255,255,255,.12);
  position:relative;cursor:pointer;transition:background .2s;margin:0}
.m3d-switch input::after{content:"";position:absolute;left:2px;top:2px;width:13px;height:13px;border-radius:50%;background:#cfd6e6;transition:transform .2s}
.m3d-switch input:checked{background:var(--m3d-accent)}
.m3d-switch input:checked::after{transform:translateX(13px);background:#fff}
.m3d-timeline{left:50%;bottom:16px;transform:translateX(-50%);display:flex;align-items:center;gap:12px;padding:9px 14px;width:min(540px,calc(100% - 40px));box-sizing:border-box}
.m3d-play{width:34px;height:34px;border-radius:50%;border:none;cursor:pointer;color:#fff;font-size:13px;flex:none;
  background:linear-gradient(135deg,var(--m3d-accent),var(--m3d-accent2));box-shadow:0 0 16px rgba(96,165,250,.45)}
.m3d-play:hover{filter:brightness(1.15)}
.m3d-slider{flex:1;display:flex;flex-direction:column;gap:2px}
.m3d-slider input{width:100%;accent-color:var(--m3d-accent);margin:0;cursor:pointer}
.m3d-ticks{display:flex;justify-content:space-between;font-size:10px;color:var(--m3d-muted);padding:0 2px}
.m3d-ticks span{cursor:pointer;width:24px;text-align:center}
.m3d-ticks span.on{color:#fff;font-weight:700}
.m3d-layer{min-width:64px;text-align:right;font-weight:650;color:#fff;font-variant-numeric:tabular-nums}
.m3d-legend{left:14px;bottom:16px;padding:10px 12px;max-width:270px;max-height:calc(100% - 140px);overflow:auto;font-size:11.5px}
.m3d-legend h4,.m3d-card h4{margin:0 0 6px;font-size:10.5px;text-transform:uppercase;letter-spacing:.7px;color:var(--m3d-muted);font-weight:600}
.m3d-li{display:flex;align-items:center;gap:7px;padding:2px 4px;border-radius:6px;cursor:pointer;white-space:nowrap;overflow:hidden}
.m3d-li:hover,.m3d-li.on{background:rgba(255,255,255,.07)}
.m3d-li .tok{overflow:hidden;text-overflow:ellipsis;color:var(--m3d-fg);font-family:ui-monospace,Consolas,monospace;font-size:11px}
.m3d-sw{width:10px;height:10px;border-radius:50%;flex:none;box-shadow:0 0 8px currentColor}
.m3d-n{color:var(--m3d-muted);font-variant-numeric:tabular-nums;min-width:30px;text-align:right}
.m3d-ramp{height:10px;border-radius:5px;margin:4px 0 3px;position:relative}
.m3d-ramp i{position:absolute;top:-3px;bottom:-3px;width:2px;background:#fff;border-radius:1px;box-shadow:0 0 4px #000}
.m3d-rampl{display:flex;justify-content:space-between;color:var(--m3d-muted);font-variant-numeric:tabular-nums}
.m3d-note{color:var(--m3d-muted);font-size:10.5px;margin-top:6px;white-space:normal;line-height:1.35}
.m3d-card{right:14px;top:228px;width:262px;padding:12px 14px;max-height:calc(100% - 320px);overflow:auto;transition:opacity .25s,transform .25s}
.m3d-card.m3d-hidden{opacity:0;transform:translateX(16px);pointer-events:none}
.m3d-ch{display:flex;align-items:baseline;gap:8px}
.m3d-ch .tok{font:700 18px ui-monospace,Consolas,monospace;color:#fff;background:linear-gradient(90deg,var(--m3d-accent),var(--m3d-accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent;overflow:hidden;text-overflow:ellipsis;white-space:pre}
.m3d-ch .id{color:var(--m3d-muted);font-size:11px}
.m3d-x{margin-left:auto;background:none;border:none;color:var(--m3d-muted);cursor:pointer;font-size:16px;line-height:1}
.m3d-x:hover{color:#fff}
.m3d-meta{color:var(--m3d-muted);font-size:11.5px;margin:2px 0 10px}
.m3d-meta b{color:var(--m3d-fg);font-weight:600}
.m3d-bars{display:flex;align-items:flex-end;gap:5px;height:46px;position:relative;margin:2px 0 14px;padding-bottom:12px}
.m3d-bar{flex:1;border-radius:3px 3px 0 0;background:linear-gradient(180deg,var(--m3d-accent2),var(--m3d-accent));opacity:.55;position:relative;min-height:2px;cursor:pointer}
.m3d-bar.on{opacity:1;box-shadow:0 0 10px var(--m3d-accent)}
.m3d-bar span{position:absolute;bottom:-13px;left:0;right:0;text-align:center;font-size:9.5px;color:var(--m3d-muted)}
.m3d-init{position:absolute;left:0;right:0;border-top:1px dashed rgba(255,255,255,.35)}
.m3d-nb-row{display:flex;align-items:center;gap:7px;padding:3px 5px;border-radius:6px;cursor:pointer;font-size:12px}
.m3d-nb-row:hover{background:rgba(255,255,255,.08)}
.m3d-nb-row .tok{font-family:ui-monospace,Consolas,monospace;white-space:pre;overflow:hidden;text-overflow:ellipsis;flex:1}
.m3d-sim{width:52px;height:4px;border-radius:2px;background:rgba(255,255,255,.08);overflow:hidden;flex:none}
.m3d-sim i{display:block;height:100%;background:linear-gradient(90deg,var(--m3d-accent),var(--m3d-accent2))}
.m3d-simv{color:var(--m3d-muted);font-size:10.5px;width:34px;text-align:right;font-variant-numeric:tabular-nums}
.m3d-ov{display:flex;gap:3px;margin:4px 0 10px}
.m3d-ov div{flex:1;text-align:center;font-size:9.5px;color:var(--m3d-muted)}
.m3d-ov div i{display:block;height:16px;border-radius:3px;margin-bottom:2px;background:var(--m3d-accent2)}
.m3d-btn{width:100%;margin-top:10px;padding:7px 10px;border-radius:8px;border:1px solid rgba(255,255,255,.12);cursor:pointer;
  color:#fff;font:inherit;font-weight:600;background:linear-gradient(90deg,rgba(96,165,250,.25),rgba(167,139,250,.25))}
.m3d-btn:hover{background:linear-gradient(90deg,rgba(96,165,250,.4),rgba(167,139,250,.4))}
.m3d-tip{position:absolute;pointer-events:none;padding:7px 10px;border-radius:9px;background:rgba(10,12,20,.88);
  border:1px solid rgba(255,255,255,.12);box-shadow:0 6px 24px rgba(0,0,0,.5);font-size:11.5px;color:var(--m3d-muted);
  white-space:nowrap;transform:translate(14px,14px);z-index:5}
.m3d-tip b{font:700 14px ui-monospace,Consolas,monospace;color:#fff;white-space:pre}
.m3d-tip .m3d-sw{display:inline-block;width:8px;height:8px;margin-right:4px;vertical-align:middle}
.m3d-hidden{display:none}
.m3d-card.m3d-hidden{display:block}
.m3d-loading{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;
  color:var(--m3d-muted);pointer-events:none;transition:opacity .4s}
.m3d-loading.m3d-off{opacity:0}
.m3d-spinner{width:38px;height:38px;border-radius:50%;border:3px solid rgba(255,255,255,.08);border-top-color:var(--m3d-accent);
  border-right-color:var(--m3d-accent2);animation:m3d-spin 1s linear infinite}
@keyframes m3d-spin{to{transform:rotate(360deg)}}
.m3d-err{color:#fca5a5;max-width:420px;text-align:center}
.m3d-hint{position:absolute;right:16px;bottom:20px;color:var(--m3d-muted);font-size:10.5px;opacity:.7;pointer-events:none;text-align:right;line-height:1.6}
.m3d-hint kbd{font:inherit;color:var(--m3d-fg);background:rgba(255,255,255,.07);border-radius:4px;padding:0 4px}
@media (max-width:900px){.m3d-hint{display:none}.m3d-legend{max-width:180px}.m3d-card{top:auto;bottom:80px;max-height:40%}}
@media (prefers-reduced-motion:reduce){.m3d-spinner{animation-duration:3s}.m3d-card,.m3d-label{transition:none}}
`;

// ---------------------------------------------------------------- shaders
const PT_VERT = /* glsl */`
attribute vec3 aFrom;
attribute vec3 aColor;
attribute float aSize;
attribute float aHi;
uniform float uMix, uScale, uTime, uDim, uMinPx, uFogNear, uFogFar;
varying vec3 vColor;
varying float vAlpha;
varying float vHi;
void main() {
  vec3 p = mix(aFrom, position, uMix);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float s = aSize;
  if (aHi > 2.5) s *= 2.6 + 0.6 * sin(uTime * 5.0);
  else if (aHi > 1.5) s *= 2.2;
  else if (aHi > 0.5) s *= 1.6;
  gl_PointSize = max(uMinPx, s * uScale / -mv.z);
  gl_Position = projectionMatrix * mv;
  float fog = smoothstep(uFogNear, uFogFar, -mv.z);
  float a = mix(1.0, 0.16, fog);
  if (uDim > 0.5 && aHi < 0.5) a *= 0.16;
  vAlpha = a;
  vColor = aColor;
  vHi = aHi;
}`;
const PT_FRAG = /* glsl */`
uniform float uOpacity;
varying vec3 vColor;
varying float vAlpha;
varying float vHi;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  float core = smoothstep(0.42, 0.0, d);
  float halo = pow(1.0 - d, 2.4);
  float hi = step(0.5, vHi);
  vec3 col = vColor * (halo * 0.85 + core * 0.9) + vec3(1.0) * core * (0.18 + 0.5 * hi);
  gl_FragColor = vec4(col, uOpacity * vAlpha);
  #include <colorspace_fragment>
}`;
const TUBE_VERT = /* glsl */`
varying float vU;
varying float vRim;
void main() {
  vU = uv.x;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  vRim = 1.0 - abs(dot(n, normalize(-mv.xyz)));
  gl_Position = projectionMatrix * mv;
}`;
const TUBE_FRAG = /* glsl */`
uniform vec3 uC1, uC2;
uniform float uTime, uOpacity;
varying float vU;
varying float vRim;
void main() {
  vec3 c = mix(uC1, uC2, vU);
  float flow = pow(0.5 + 0.5 * sin((vU * 6.0 - uTime * 1.6) * 3.14159), 6.0);
  float a = (0.45 + 0.55 * flow) * (1.0 - 0.6 * vRim);
  gl_FragColor = vec4(c * (1.0 + flow * 0.8), a * uOpacity);
  #include <colorspace_fragment>
}`;

function pointsMaterial(extra = {}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uMix: { value: 1 }, uScale: { value: 500 }, uTime: { value: 0 }, uDim: { value: 0 }, uMinPx: { value: 1.5 },
      uFogNear: { value: 10 }, uFogFar: { value: 30 }, uOpacity: { value: 0.7 }, ...extra,
    },
    vertexShader: PT_VERT, fragmentShader: PT_FRAG,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

function ringTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 40, 64, 64, 62);
  grd.addColorStop(0, 'rgba(255,255,255,0)'); grd.addColorStop(0.55, 'rgba(255,255,255,1)'); grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.beginPath(); g.arc(64, 64, 63, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function backgroundTexture(bg) {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(256, 230, 10, 256, 256, 380);
  grd.addColorStop(0, '#1b2140'); grd.addColorStop(0.35, '#10142a'); grd.addColorStop(0.75, bg || '#080a12'); grd.addColorStop(1, '#030409');
  g.fillStyle = grd; g.fillRect(0, 0, 512, 512);
  // faint nebula tint
  const n = g.createRadialGradient(360, 150, 0, 360, 150, 220);
  n.addColorStop(0, 'rgba(120,80,200,0.10)'); n.addColorStop(1, 'rgba(120,80,200,0)');
  g.fillStyle = n; g.fillRect(0, 0, 512, 512);
  const m = g.createRadialGradient(130, 380, 0, 130, 380, 200);
  m.addColorStop(0, 'rgba(40,120,200,0.08)'); m.addColorStop(1, 'rgba(40,120,200,0)');
  g.fillStyle = m; g.fillRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ---------------------------------------------------------------- mount
export async function mountMap3D(container, ctx = {}) {
  const api = ctx.api || defaultApi;
  const theme = readTheme(container, ctx.theme);
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  const ac = new AbortController();
  const on = (el, ev, fn, opts = {}) => el.addEventListener(ev, fn, { ...opts, signal: ac.signal });
  const timers = new Set();
  const later = (fn, ms) => { const t = setTimeout(() => { timers.delete(t); fn(); }, ms); timers.add(t); return t; };

  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style'); st.id = STYLE_ID; st.textContent = CSS; document.head.appendChild(st);
  }
  styleRefs++;

  // ---- DOM
  const root = document.createElement('div');
  root.className = 'm3d-root'; root.tabIndex = 0;
  for (const [k, v] of Object.entries({ fg: theme.fg, accent: theme.accent, accent2: theme.accent2, muted: theme.muted, bg: theme.bg })) {
    root.style.setProperty(`--m3d-${k}`, v);
  }
  root.innerHTML = `
    <div class="m3d-labels" data-r="labels"></div>
    <div class="m3d-title"><div class="m3d-h">PLE token space <b>in 3D</b></div><div class="m3d-sub" data-r="sub">loading…</div></div>
    <div class="m3d-panel m3d-controls">
      <div class="m3d-row"><span>Colour</span><select data-r="color">
        <option value="cluster">k-means cluster</option><option value="freq">log frequency</option>
        <option value="norm">row norm</option><option value="maxlayer">layer of max norm</option></select></div>
      <div class="m3d-row"><span>Points</span><select data-r="n">
        <option value="1000">1,000</option><option value="2000" selected>2,000</option><option value="5000">5,000</option></select></div>
      <div class="m3d-row"><span>Bloom</span><label class="m3d-switch"><input type="checkbox" data-r="bloom" checked></label></div>
      <input type="text" data-r="search" placeholder="find token… e.g. ' dog'" spellcheck="false" autocomplete="off">
      <div class="m3d-status" data-r="status"></div>
    </div>
    <div class="m3d-panel m3d-legend" data-r="legend"></div>
    <div class="m3d-panel m3d-card m3d-hidden" data-r="card"></div>
    <div class="m3d-panel m3d-timeline">
      <button class="m3d-play" data-r="play" title="Play through the layers">▶</button>
      <div class="m3d-slider"><input type="range" min="0" max="6" step="1" value="0" data-r="slider"><div class="m3d-ticks" data-r="ticks"></div></div>
      <div class="m3d-layer" data-r="layerLabel">Layer 0</div>
    </div>
    <div class="m3d-tip m3d-hidden" data-r="tip"></div>
    <div class="m3d-hint"><kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom · <kbd>click</kbd> trace<br><kbd>double-click</kbd> open · <kbd>esc</kbd> clear</div>
    <div class="m3d-loading" data-r="loading"><div class="m3d-spinner"></div><span data-r="loadingText">projecting the table…</span></div>`;
  container.appendChild(root);
  const $ = name => root.querySelector(`[data-r="${name}"]`);
  const ui = Object.fromEntries(['labels', 'sub', 'color', 'n', 'bloom', 'search', 'status', 'legend', 'card', 'play', 'slider',
    'ticks', 'layerLabel', 'tip', 'loading', 'loadingText'].map(k => [k, $(k)]));

  const showError = msg => {
    ui.loading.classList.remove('m3d-off');
    ui.loading.innerHTML = `<div class="m3d-err">${esc(msg)}</div>`;
  };

  // ---- renderer
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  } catch (e) {
    showError('WebGL is not available in this browser, so the 3D map cannot be shown.');
    const noop = () => {};
    return { setRun: async () => {}, setLayer: noop, highlight: noop, resize: noop,
      destroy() { ac.abort(); root.remove(); if (--styleRefs === 0) document.getElementById(STYLE_ID)?.remove(); } };
  }
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.className = 'm3d-canvas';
  root.insertBefore(renderer.domElement, root.firstChild);

  const labelRenderer = new CSS2DRenderer({ element: ui.labels });

  const scene = new THREE.Scene();
  const bgTex = backgroundTexture(theme.bg);
  scene.background = bgTex;
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 400);
  camera.position.set(9, 5.5, 12.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minDistance = 1.5; controls.maxDistance = 60;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.45;
  controls.rotateSpeed = 0.7;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.5, 0.2);
  composer.addPass(bloomPass);
  const outputPass = new OutputPass();
  composer.addPass(outputPass);
  let bloomOn = true;

  // ---- static decor: starfield, grid, axes
  const decor = new THREE.Group(); scene.add(decor);
  {
    const N = 2200, pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, r = 70 + Math.random() * 90, s = Math.sqrt(1 - u * u);
      pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
      const b = 0.25 + Math.random() * 0.75, tint = Math.random();
      col.set([b * (0.75 + 0.25 * tint), b * 0.82, b * (1.0 - 0.15 * tint)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const stars = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.25, sizeAttenuation: false, vertexColors: true,
      transparent: true, opacity: 0.55, depthWrite: false }));
    stars.name = 'stars';
    decor.add(stars);

    const grid = new THREE.GridHelper(40, 40, 0x3b4a7a, 0x1d2440);
    grid.material.transparent = true; grid.material.opacity = 0.16; grid.material.depthWrite = false;
    grid.position.y = -9;
    decor.add(grid);

    const ax = new Float32Array([0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3]);
    const axc = new Float32Array([1, .4, .4, 1, .4, .4, .4, 1, .6, .4, 1, .6, .4, .6, 1, .4, .6, 1]);
    const ag = new THREE.BufferGeometry();
    ag.setAttribute('position', new THREE.BufferAttribute(ax, 3));
    ag.setAttribute('color', new THREE.BufferAttribute(axc, 3));
    const axes = new THREE.LineSegments(ag, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35, depthWrite: false }));
    axes.position.set(-7.5, -8.9, -7.5);
    decor.add(axes);
    ['PC1', 'PC2', 'PC3'].forEach((t, i) => {
      const el = document.createElement('div'); el.className = 'm3d-label m3d-axis'; el.textContent = t;
      const o = new CSS2DObject(el); o.position.set(i === 0 ? 3.4 : 0, i === 1 ? 3.4 : 0, i === 2 ? 3.4 : 0); axes.add(o);
    });
  }

  // ---- selection decor
  const ringTex = ringTexture();
  const selRing = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, color: new THREE.Color(theme.accent),
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
  selRing.visible = false; selRing.renderOrder = 10; scene.add(selRing);
  const ripples = [0, 1, 2].map(() => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, color: new THREE.Color(theme.accent2),
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, opacity: 0 }));
    s.visible = false; s.renderOrder = 11; scene.add(s); return s;
  });
  const trailGroup = new THREE.Group(); scene.add(trailGroup);
  const nbGeom = new THREE.BufferGeometry();
  nbGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(K_NEIGH * 6), 3));
  nbGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(K_NEIGH * 6), 3));
  const nbLines = new THREE.LineSegments(nbGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
    opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  nbLines.visible = false; nbLines.frustumCulled = false; scene.add(nbLines);

  const mkLabel = (cls, parent = scene) => {
    const el = document.createElement('div'); el.className = `m3d-label ${cls}`;
    const o = new CSS2DObject(el); o.visible = false; parent.add(o); return o;
  };
  const selLabel = mkLabel('m3d-sel');
  const nbLabels = Array.from({ length: K_NEIGH }, () => mkLabel('m3d-nb'));

  // ---- state
  const S = {
    run: ctx.run, n: 2000, data: null, nLayer: 6, layerPos: [], tf: [], cur: Infinity, color: 'cluster',
    points: null, geom: null, mat: null, hi: null, topLabels: [],
    morph: null, fly: null, mixE: 1,
    sel: null, selPos: null, selFrom: new THREE.Vector3(), selTo: new THREE.Vector3(), trajCache: new Map(),
    hover: -1, mouse: new THREE.Vector2(), mouseDirty: false, mouseIn: false, downAt: null,
    play: false, playNext: 0, focusCluster: null, searchSet: null, rippleT0: -1,
    visible: true, raf: 0, last: 0, destroyed: false, loadSeq: 0, colors: null,
  };
  const raycaster = new THREE.Raycaster();
  const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3();

  // ---- layer helpers
  const layerName = l => (l >= S.nLayer ? 'Full row' : `Layer ${l}`);
  const toLayerIndex = l => (l === 'full' ? S.nLayer : clamp(Math.round(Number(l)), 0, S.nLayer));
  function curPos(i, out) {
    const a = S.geom.attributes.aFrom.array, b = S.geom.attributes.position.array, m = S.mixE;
    return out.set(a[3 * i] + (b[3 * i] - a[3 * i]) * m, a[3 * i + 1] + (b[3 * i + 1] - a[3 * i + 1]) * m,
      a[3 * i + 2] + (b[3 * i + 2] - a[3 * i + 2]) * m);
  }
  function curSelPos(out) { return out.copy(S.selFrom).lerp(S.selTo, S.mixE); }

  function normaliseLayers(data) {
    const n = data.points.length, L = data.n_layer;
    const raws = [];
    for (let l = 0; l <= L; l++) {
      const a = new Float32Array(n * 3);
      data.points.forEach((p, i) => a.set(l < L ? p.coords[l] : p.full, i * 3));
      raws.push(a);
    }
    S.tf = []; S.layerPos = [];
    for (const a of raws) {
      const m = [0, 0, 0];
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) m[k] += a[3 * i + k] / n;
      let ss = 0;
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) ss += (a[3 * i + k] - m[k]) ** 2;
      const s = R / Math.max(1e-9, Math.sqrt(ss / n));
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[3 * i + k] = (a[3 * i + k] - m[k]) * s;
      S.tf.push({ m, s }); S.layerPos.push(out);
    }
  }
  const tfCoord = (l, c) => new THREE.Vector3((c[0] - S.tf[l].m[0]) * S.tf[l].s, (c[1] - S.tf[l].m[1]) * S.tf[l].s, (c[2] - S.tf[l].m[2]) * S.tf[l].s);

  // ---- colours
  const clusterCol = S_ => CLUSTER_PALETTE.map(c => new THREE.Color(c));
  const clusterColors = clusterCol();
  function colorFor(mode) {
    const pts = S.data.points, n = pts.length, out = new Float32Array(n * 3);
    let info = {};
    if (mode === 'cluster') {
      pts.forEach((p, i) => { const c = clusterColors[p.cluster % clusterColors.length]; out.set([c.r, c.g, c.b], i * 3); });
    } else if (mode === 'freq') {
      const v = pts.map(p => p.log_freq ?? 0), lo = Math.min(...v), hi = Math.max(...v);
      pts.forEach((p, i) => { const c = srgbToLinear(turbo(0.12 + 0.88 * (v[i] - lo) / Math.max(1e-9, hi - lo))); out.set([c.r, c.g, c.b], i * 3); });
      info = { lo, hi };
    } else if (mode === 'norm') {
      const L = S.nLayer, l = S.cur;
      const v = pts.map(p => (l < L ? p.norms[l] : p.full_norm));
      const sorted = [...v].sort((a, b) => a - b), lo = quantile(sorted, 0.02), hi = quantile(sorted, 0.98);
      pts.forEach((p, i) => { const c = srgbToLinear(viridis(0.08 + 0.92 * clamp((v[i] - lo) / Math.max(1e-9, hi - lo)))); out.set([c.r, c.g, c.b], i * 3); });
      const init = l < L ? S.data.init_norm : S.data.init_norm * Math.sqrt(L);
      info = { lo, hi, init };
    } else {
      const L = S.nLayer;
      const cols = Array.from({ length: L }, (_, l) => srgbToLinear(turbo(0.08 + 0.84 * (l + 0.5) / L)));
      pts.forEach((p, i) => { const c = cols[p.max_layer]; out.set([c.r, c.g, c.b], i * 3); });
      info = { cols };
    }
    return { out, info };
  }
  function applyColors() {
    if (!S.data) return;
    const { out, info } = colorFor(S.color);
    S.colors = out;
    S.geom.attributes.aColor.array.set(out); S.geom.attributes.aColor.needsUpdate = true;
    renderLegend(info);
  }
  const cssOf = c => `#${c.getHexString(THREE.SRGBColorSpace)}`;

  function renderLegend(info) {
    const d = S.data, pts = d.points;
    let h = '';
    if (S.color === 'cluster') {
      h = `<h4>k-means clusters (full row, cosine)</h4>` + d.clusters.map(c => `
        <div class="m3d-li${S.focusCluster === c.id ? ' on' : ''}" data-cluster="${c.id}">
          <span class="m3d-sw" style="background:${cssOf(clusterColors[c.id % clusterColors.length])};color:${cssOf(clusterColors[c.id % clusterColors.length])}"></span>
          <span class="m3d-n">${c.size}</span><span class="tok">${esc(c.top.slice(0, 5).map(vis).join('  '))}</span></div>`).join('')
        + `<div class="m3d-note">Click a cluster to isolate it. Clusters come from the full row, so they look tightest in the <b>Full row</b> view.</div>`;
    } else if (S.color === 'freq') {
      h = `<h4>log₁₀ training frequency</h4><div class="m3d-ramp" style="background:${rampCss(turbo, 0.12, 1)}"></div>
        <div class="m3d-rampl"><span>${fmt(info.lo, 1)}</span><span>${fmt(info.hi, 1)}</span></div>
        <div class="m3d-note">Point size also grows with log frequency. Data: ${esc(d.source)}.</div>`;
    } else if (S.color === 'norm') {
      const pos = clamp((info.init - info.lo) / Math.max(1e-9, info.hi - info.lo)) * 100;
      h = `<h4>${S.cur < S.nLayer ? `slice norm ‖E[t]<sub>${S.cur}</sub>‖` : 'full-row norm'}</h4>
        <div class="m3d-ramp" style="background:${rampCss(viridis, 0.08, 1)}"><i style="left:${pos}%" title="init norm"></i></div>
        <div class="m3d-rampl"><span>${fmt(info.lo, 2)}</span><span>init ${fmt(info.init, 2)}</span><span>${fmt(info.hi, 2)}</span></div>
        <div class="m3d-note">2–98th percentile. Tokens far above the init norm were moved most by training.</div>`;
    } else {
      const counts = Array(S.nLayer).fill(0); pts.forEach(p => counts[p.max_layer]++);
      h = `<h4>layer with the largest slice norm</h4>` + info.cols.map((c, l) => `
        <div class="m3d-li"><span class="m3d-sw" style="background:${cssOf(c)};color:${cssOf(c)}"></span>
        <span>L${l}</span><span class="m3d-n">${counts[l]}</span>
        <span class="m3d-sim" style="width:90px"><i style="width:${(100 * counts[l] / pts.length * 2).toFixed(1)}%;background:${cssOf(c)}"></i></span></div>`).join('')
        + `<div class="m3d-note">Which layer's PLE slice each token invests most in.</div>`;
    }
    ui.legend.innerHTML = h;
  }

  // ---- highlight attribute
  function refreshHi() {
    if (!S.data) return;
    const a = S.geom.attributes.aHi.array; a.fill(0);
    let dim = false;
    if (S.focusCluster != null) { dim = true; S.data.points.forEach((p, i) => { if (p.cluster === S.focusCluster) a[i] = 1; }); }
    if (S.searchSet) { dim = true; for (const i of S.searchSet) a[i] = 1; }
    if (S.sel) {
      dim = true;
      for (const nb of selNeighbours()) if (nb.index != null && nb.index < a.length) a[nb.index] = 1;
      if (S.sel.index != null && S.sel.index < a.length) a[S.sel.index] = 3;
    }
    if (S.hover >= 0) a[S.hover] = Math.max(a[S.hover], 2);
    S.mat.uniforms.uDim.value = dim ? 1 : 0;
    S.geom.attributes.aHi.needsUpdate = true;
  }

  // ---- build points
  function buildClusterLabels() {
    for (const o of S.clLabels || []) { o.removeFromParent(); o.element.remove(); }
    S.clLabels = S.data.clusters.filter(c => c.size > 0).map(c => {
      const o = mkLabel('m3d-cl');
      o.element.style.setProperty('--c', cssOf(clusterColors[c.id % clusterColors.length]));
      o.element.textContent = c.top.slice(0, 3).map(vis).join(' ');
      o.position.copy(tfCoord(S.nLayer, c.centroid)).multiplyScalar(1.35);
      return o;
    });
  }
  function disposePoints() {
    for (const o of S.clLabels || []) { o.removeFromParent(); o.element.remove(); }
    S.clLabels = [];
    if (S.points) { scene.remove(S.points); S.geom.dispose(); S.mat.dispose(); S.points = null; }
    for (const o of S.topLabels) { o.removeFromParent(); o.element.remove(); }
    S.topLabels = [];
  }
  function buildPoints() {
    disposePoints();
    const d = S.data, n = d.points.length;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(S.layerPos[S.cur].slice(), 3));
    g.setAttribute('aFrom', new THREE.BufferAttribute(S.layerPos[S.cur].slice(), 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const lf = d.points.map(p => p.log_freq ?? 0), lo = Math.min(...lf), hi = Math.max(...lf);
    const size = new Float32Array(n);
    const base = 0.07 * Math.sqrt(2000 / Math.max(500, n));
    d.points.forEach((p, i) => { size[i] = base * (0.7 + 2.6 * Math.pow((lf[i] - lo) / Math.max(1e-9, hi - lo), 1.6)); });
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aHi', new THREE.BufferAttribute(new Float32Array(n), 1));
    g.computeBoundingSphere();
    S.geom = g; S.mat = pointsMaterial();
    S.points = new THREE.Points(g, S.mat); S.points.frustumCulled = false;
    scene.add(S.points);
    S.mixE = 1; S.morph = null;
    for (let i = 0; i < Math.min(N_LABELS, n); i++) {
      const o = mkLabel('');
      o.element.textContent = vis(d.points[i].piece);
      o.userData.i = i; o.visible = true;
      S.topLabels.push(o);
    }
    buildClusterLabels();
    applyColors(); onResize();
  }

  // ---- data loading
  async function load(run, keepSel = true) {
    const seq = ++S.loadSeq;
    S.run = run;
    ui.loading.classList.remove('m3d-off');
    ui.loading.innerHTML = `<div class="m3d-spinner"></div><span>projecting ${esc(run)} · ${S.n.toLocaleString()} tokens…</span>`;
    let data;
    try {
      data = await api(`/api/projection3d?${new URLSearchParams({ run, n: S.n, space: 'shared' })}`);
    } catch (e) {
      if (seq === S.loadSeq && !S.destroyed) showError(`Could not load the 3D map: ${e.message}`);
      return;
    }
    if (seq !== S.loadSeq || S.destroyed) return;
    const prevSel = keepSel && S.sel ? S.sel.id : null;
    S.data = data; S.nLayer = data.n_layer; S.trajCache.clear();
    S.cur = Math.min(S.cur, S.nLayer); S.focusCluster = null; S.searchSet = null; S.hover = -1;
    clearSelection(false);
    normaliseLayers(data);
    buildTicks();
    buildPoints();
    updateHeader();
    ui.loading.classList.add('m3d-off');
    if (prevSel != null) selectToken(prevSel, {});
  }

  function updateHeader() {
    const d = S.data; if (!d) return;
    const l = S.cur, ev = l < S.nLayer ? d.explained_variance[l] : d.explained_variance_full;
    const evs = ev.map(e => `${(e * 100).toFixed(1)}%`).join(' · ');
    ui.sub.innerHTML = `${esc(S.run)} · ${d.n.toLocaleString()} most frequent tokens · ${l < S.nLayer
      ? `<b style="color:#fff">layer ${l}</b> slice in a shared PCA (all layers, same axes)` : '<b style="color:#fff">full row</b> (all layers concatenated, own PCA)'}
      <br>PC variance ${evs} — the map shows the dominant directions only`;
    ui.layerLabel.textContent = l < S.nLayer ? `Layer ${l}` : 'Full';
    ui.slider.max = String(S.nLayer); ui.slider.value = String(l);
    [...ui.ticks.children].forEach((t, i) => t.classList.toggle('on', i === l));
  }
  function buildTicks() {
    ui.ticks.innerHTML = Array.from({ length: S.nLayer + 1 }, (_, l) => `<span data-l="${l}">${l < S.nLayer ? `L${l}` : 'Full'}</span>`).join('');
  }

  // ---- layer morph
  function setLayer(l, { user = false } = {}) {
    if (!S.data) { S.cur = toLayerIndex(l); return; }
    l = toLayerIndex(l);
    if (user) stopPlay();
    const g = S.geom, from = g.attributes.aFrom, to = g.attributes.position;
    const sp = S.sel ? curSelPos(tmpV2.clone()) : null;
    if (S.mixE < 1) {
      const a = from.array, b = to.array, m = S.mixE;
      for (let i = 0; i < a.length; i++) a[i] += (b[i] - a[i]) * m;
    } else from.array.set(to.array);
    to.array.set(S.layerPos[l]);
    from.needsUpdate = true; to.needsUpdate = true; g.computeBoundingSphere();
    const changed = l !== S.cur;
    S.cur = l;
    if (S.sel) { S.selFrom.copy(sp); S.selTo.copy(S.selPos[l]); }
    if (reduced || !changed) { S.mixE = 1; S.mat.uniforms.uMix.value = 1; S.morph = null; }
    else { S.mixE = 0; S.mat.uniforms.uMix.value = 0; S.morph = { t0: performance.now(), dur: MORPH_MS }; }
    if (S.color === 'norm') applyColors();
    updateHeader();
    if (S.sel) { refreshHi(); renderCard(); }
    hideTip();
  }

  // ---- play
  function startPlay() { S.play = true; ui.play.textContent = '❚❚'; S.playNext = performance.now() + 300; idleStop(); }
  function stopPlay() { if (!S.play) return; S.play = false; ui.play.textContent = '▶'; idleResume(); }

  // ---- idle auto-rotate
  let idleTimer = null;
  function idleStop() { controls.autoRotate = false; if (idleTimer) { clearTimeout(idleTimer); timers.delete(idleTimer); idleTimer = null; } }
  function idleResume() {
    if (reduced) return;
    if (idleTimer) { clearTimeout(idleTimer); timers.delete(idleTimer); }
    idleTimer = later(() => { idleTimer = null; if (!S.play && !S.fly) controls.autoRotate = true; }, IDLE_RESUME_MS);
  }
  on(controls, 'start', () => { idleStop(); S.fly = null; });
  on(controls, 'end', idleResume);

  // ---- camera fly
  function flyTo(p, dist = 6.5) {
    idleStop();
    const dir = tmpV.copy(camera.position).sub(controls.target).normalize();
    const d = Math.min(dist, camera.position.distanceTo(controls.target));
    const toPos = p.clone().addScaledVector(dir, Math.max(2.5, d));
    if (reduced) { camera.position.copy(toPos); controls.target.copy(p); controls.update(); idleResume(); return; }
    S.fly = { t0: performance.now(), dur: FLY_MS, p0: camera.position.clone(), t0v: controls.target.clone(), p1: toPos, t1v: p.clone() };
  }
  function pulse() { S.rippleT0 = performance.now(); }

  // ---- selection / trajectory
  function selNeighbours() {
    if (!S.sel) return [];
    return S.cur < S.nLayer ? S.sel.tr.layers[S.cur].neighbors : S.sel.tr.full.neighbors;
  }
  function clearTrail() {
    for (const o of [...trailGroup.children]) {
      o.traverse(c => { c.geometry?.dispose(); c.material?.dispose(); if (c.isCSS2DObject) c.element.remove(); });
      o.removeFromParent();
    }
  }
  function clearSelection(update = true) {
    S.sel = null; S.selPos = null;
    clearTrail();
    selRing.visible = false; nbLines.visible = false; selLabel.visible = false;
    nbLabels.forEach(o => { o.visible = false; });
    ui.card.classList.add('m3d-hidden');
    if (update) refreshHi();
  }
  async function getTrajectory(id) {
    const key = `${S.run}|${S.n}|${id}`;
    if (!S.trajCache.has(key)) {
      S.trajCache.set(key, api(`/api/trajectory?${new URLSearchParams({ run: S.run, id, n: S.n, space: 'shared', k: K_NEIGH })}`)
        .catch(e => { S.trajCache.delete(key); throw e; }));
    }
    return S.trajCache.get(key);
  }
  async function selectToken(id, { fly = false, doPulse = true } = {}) {
    if (!S.data) return;
    const seq = S.loadSeq;
    let tr;
    try { tr = await getTrajectory(id); } catch (e) { ui.status.textContent = e.message; return; }
    if (S.destroyed || seq !== S.loadSeq) return;
    clearSelection(false);
    const L = S.nLayer;
    S.selPos = [...tr.layers.map(x => tfCoord(x.layer, x.coords)), tfCoord(L, tr.full.coords)];
    const idx = tr.index;
    S.sel = { id: tr.id, tr, index: idx, point: idx != null ? S.data.points[idx] : null };
    S.selFrom.copy(S.selPos[S.cur]); S.selTo.copy(S.selPos[S.cur]);
    buildTrail();
    selLabel.element.textContent = vis(tr.piece); selLabel.visible = true;
    selRing.visible = true; nbLines.visible = true;
    refreshHi(); renderCard();
    ui.card.classList.remove('m3d-hidden');
    if (fly) flyTo(S.selPos[S.cur]);
    if (doPulse) pulse();
  }
  function buildTrail() {
    const pts = S.selPos.slice(0, S.nLayer);
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const c1 = new THREE.Color(theme.accent), c2 = new THREE.Color(theme.accent2);
    const mk = (radius, opacity) => new THREE.Mesh(new THREE.TubeGeometry(curve, 240, radius, 10, false),
      new THREE.ShaderMaterial({ uniforms: { uC1: { value: c1 }, uC2: { value: c2 }, uTime: { value: 0 }, uOpacity: { value: opacity } },
        vertexShader: TUBE_VERT, fragmentShader: TUBE_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    const core = mk(0.028, 1.0), glow = mk(0.12, 0.16);
    core.name = 'trail-core'; glow.name = 'trail-glow';
    trailGroup.add(glow, core);
    // layer markers
    const g = new THREE.BufferGeometry();
    const n = pts.length;
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.flatMap(p => [p.x, p.y, p.z])), 3));
    g.setAttribute('aFrom', g.attributes.position.clone());
    const col = new Float32Array(n * 3);
    pts.forEach((_, i) => { const c = c1.clone().lerp(c2, i / Math.max(1, n - 1)); col.set([c.r, c.g, c.b], i * 3); });
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(n).fill(0.16), 1));
    g.setAttribute('aHi', new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    const markers = new THREE.Points(g, pointsMaterial({ uOpacity: { value: 1 } }));
    markers.name = 'trail-markers'; markers.frustumCulled = false;
    trailGroup.add(markers);
    pts.forEach((p, l) => {
      const o = mkLabel('m3d-lm', trailGroup); o.element.textContent = String(l); o.position.copy(p); o.visible = true;
      o.userData.layer = l;
    });
  }
  function updateSelectionFrame(now) {
    if (!S.sel) return;
    const p = curSelPos(tmpV);
    selRing.position.copy(p);
    const dist = camera.position.distanceTo(p);
    const s = dist * 0.045 * (1 + 0.12 * Math.sin(now / 260));
    selRing.scale.set(s, s, s);
    selLabel.position.copy(p);
    const nbs = selNeighbours();
    const pa = nbGeom.attributes.position.array, ca = nbGeom.attributes.color.array;
    const c1 = new THREE.Color(theme.accent);
    nbs.slice(0, K_NEIGH).forEach((nb, k) => {
      const q = curPos(nb.index, tmpV2);
      pa.set([p.x, p.y, p.z, q.x, q.y, q.z], k * 6);
      const w = clamp(nb.sim * 1.6, 0.25, 1);
      ca.set([c1.r * w, c1.g * w, c1.b * w, 0.5 * w, 0.45 * w, 0.9 * w], k * 6);
      const o = nbLabels[k]; o.position.copy(q); o.visible = true;
      const txt = vis(nb.piece); if (o.element.textContent !== txt) o.element.textContent = txt;
    });
    for (let k = nbs.length; k < K_NEIGH; k++) { pa.fill(0, k * 6, k * 6 + 6); nbLabels[k].visible = false; }
    nbGeom.setDrawRange(0, Math.min(nbs.length, K_NEIGH) * 2);
    nbGeom.attributes.position.needsUpdate = true; nbGeom.attributes.color.needsUpdate = true;
    for (const o of trailGroup.children) if (o.isCSS2DObject) o.element.style.opacity = o.userData.layer === S.cur ? '1' : '0.55';
    const fullView = S.cur >= S.nLayer;
    for (const o of trailGroup.children) if (o.material?.uniforms?.uTime) {
      o.material.uniforms.uTime.value = now / 1000;
      o.material.uniforms.uOpacity.value = (o.name === 'trail-glow' ? 0.16 : 1.0) * (fullView ? 0.3 : 1);
    }
  }

  function renderCard() {
    const sel = S.sel; if (!sel) return;
    const tr = sel.tr, p = sel.point, L = S.nLayer;
    const norms = tr.layers.map(x => x.norm), maxN = Math.max(...norms, S.data.init_norm) * 1.08;
    const nbs = selNeighbours();
    const initPct = (S.data.init_norm / maxN) * 100;
    ui.card.innerHTML = `
      <div class="m3d-ch"><span class="tok">${esc(vis(tr.piece))}</span><span class="id">#${tr.id}</span>
        <button class="m3d-x" data-act="close" title="Clear selection (Esc)">✕</button></div>
      <div class="m3d-meta">${p ? `rank <b>${p.rank + 1}</b> · <b>${fmtInt(p.freq)}</b>× in training · cluster
        <span class="m3d-sw" style="display:inline-block;width:8px;height:8px;background:${cssOf(clusterColors[p.cluster % clusterColors.length])}"></span> <b>${p.cluster}</b>`
        : `outside the top ${S.data.n.toLocaleString()} — placed by projection`}</div>
      <h4>slice norm per layer <span style="text-transform:none;letter-spacing:0">(dashed: init)</span></h4>
      <div class="m3d-bars">${norms.map((v, l) => `<div class="m3d-bar${l === S.cur ? ' on' : ''}" data-l="${l}" style="height:${(v / maxN * 100).toFixed(1)}%" title="layer ${l}: ${fmt(v)}"><span>${l}</span></div>`).join('')}
        <div class="m3d-init" style="bottom:calc(12px + ${initPct.toFixed(1)}% * 34 / 46)"></div></div>
      <h4>nearest in ${S.cur < L ? `layer ${S.cur}` : 'the full row'} (cosine)</h4>
      ${nbs.map(nb => `<div class="m3d-nb-row" data-id="${nb.id}"><span class="tok">${esc(vis(nb.piece))}</span>
        <span class="m3d-sim"><i style="width:${(clamp(nb.sim) * 100).toFixed(0)}%"></i></span><span class="m3d-simv">${fmt(nb.sim, 2)}</span></div>`).join('')}
      <h4 style="margin-top:10px">neighbour overlap, layer l → l+1</h4>
      <div class="m3d-ov">${tr.neighbor_overlap.map((o, l) => `<div title="${(o * 100).toFixed(0)}% of the ${K_NEIGH} neighbours shared"><i style="opacity:${(0.12 + 0.88 * o).toFixed(2)}"></i>${l}→${l + 1}</div>`).join('')}</div>
      <div class="m3d-note">Low overlap = each layer's slice gives this token a different neighbourhood. The glowing path is the token's position in layers 0→${L - 1}.</div>
      ${ctx.onTokenClick ? `<button class="m3d-btn" data-act="open">Open in token view ↗</button>` : ''}`;
  }

  // ---- search
  async function highlight(query) {
    query = String(query ?? '');
    S.searchSet = null;
    if (!query.trim()) { ui.status.textContent = ''; refreshHi(); return; }
    if (!S.data) return;
    const pts = S.data.points;
    let idx = pts.findIndex(p => p.piece === query);
    if (idx < 0 && !query.startsWith(' ')) idx = pts.findIndex(p => p.piece === ' ' + query);
    if (idx < 0) { const t = query.trim().toLowerCase(); idx = pts.findIndex(p => p.piece.trim().toLowerCase() === t); }
    if (idx < 0 && /^#\d+$/.test(query.trim())) {
      const id = Number(query.trim().slice(1));
      idx = pts.findIndex(p => p.id === id);
      if (idx < 0) { ui.status.textContent = `#${id} is outside the map, projected`; await selectToken(id, { fly: true }); return; }
    }
    if (idx >= 0) {
      ui.status.textContent = `${vis(pts[idx].piece)} · rank ${idx + 1}`;
      await selectToken(pts[idx].id, { fly: true });
      return;
    }
    const t = query.trim().toLowerCase();
    const matches = [];
    pts.forEach((p, i) => { if (p.piece.toLowerCase().includes(t)) matches.push(i); });
    if (matches.length) {
      S.searchSet = new Set(matches.slice(0, 500));
      ui.status.textContent = `${matches.length} tokens contain “${query.trim()}”`;
      await selectToken(pts[matches[0]].id, { fly: true });
      S.searchSet = new Set(matches.slice(0, 500)); refreshHi();
      return;
    }
    try {
      const info = await api(`/api/token?${new URLSearchParams({ run: S.run, q: query })}`);
      ui.status.textContent = `${vis(info.piece)} is outside the top ${S.data.n.toLocaleString()}, projected`;
      await selectToken(info.id, { fly: true });
    } catch (e) { ui.status.textContent = `not found: ${e.message}`; }
  }

  // ---- hover / tooltip
  function hideTip() { if (S.hover >= 0) { S.hover = -1; refreshHi(); } ui.tip.classList.add('m3d-hidden'); renderer.domElement.classList.remove('m3d-hovering'); }
  function doHover() {
    S.mouseDirty = false;
    if (!S.points || !S.mouseIn || S.morph || S.fly) return;
    const dist = camera.position.distanceTo(controls.target);
    raycaster.params.Points.threshold = dist * 0.012;
    raycaster.setFromCamera(S.mouse, camera);
    const hits = raycaster.intersectObject(S.points, false);
    let best = -1, bd = Infinity;
    for (const h of hits) {
      const score = h.distanceToRay + h.distance * 0.002;
      if (score < bd) { bd = score; best = h.index; }
    }
    if (best === S.hover) return;
    S.hover = best; refreshHi();
    if (best < 0) { ui.tip.classList.add('m3d-hidden'); renderer.domElement.classList.remove('m3d-hovering'); return; }
    renderer.domElement.classList.add('m3d-hovering');
    const p = S.data.points[best], c = clusterColors[p.cluster % clusterColors.length];
    const norm = S.cur < S.nLayer ? p.norms[S.cur] : p.full_norm;
    ui.tip.innerHTML = `<b>${esc(vis(p.piece))}</b> &nbsp;#${p.id}<br>rank ${p.rank + 1} · ${fmtInt(p.freq)}×<br>
      <span class="m3d-sw" style="background:${cssOf(c)};color:${cssOf(c)}"></span>cluster ${p.cluster} · norm ${fmt(norm, 2)}`;
    ui.tip.classList.remove('m3d-hidden');
  }
  function placeTip(e) {
    const r = root.getBoundingClientRect();
    let x = e.clientX - r.left, y = e.clientY - r.top;
    if (x > r.width - 200) x -= 200;
    if (y > r.height - 90) y -= 90;
    ui.tip.style.left = `${x}px`; ui.tip.style.top = `${y}px`;
  }

  // ---- events
  const cv = renderer.domElement;
  on(cv, 'pointermove', e => {
    const r = cv.getBoundingClientRect();
    S.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    S.mouseIn = true;
    if (e.buttons === 0) S.mouseDirty = true; else hideTip();
    placeTip(e);
  });
  on(cv, 'pointerleave', () => { S.mouseIn = false; hideTip(); });
  on(cv, 'pointerdown', e => { S.downAt = [e.clientX, e.clientY]; });
  on(cv, 'pointerup', e => {
    if (!S.downAt) return;
    const moved = Math.hypot(e.clientX - S.downAt[0], e.clientY - S.downAt[1]);
    S.downAt = null;
    if (moved > 5 || e.button !== 0) return;
    if (S.hover >= 0) { idleStop(); idleResume(); selectToken(S.data.points[S.hover].id, {}); }
    else if (S.sel) clearSelection();
  });
  on(cv, 'dblclick', () => {
    const id = S.hover >= 0 ? S.data.points[S.hover].id : S.sel?.id;
    if (id != null && ctx.onTokenClick) ctx.onTokenClick(id);
  });
  on(cv, 'wheel', () => { idleStop(); idleResume(); }, { passive: true });
  on(root, 'keydown', e => {
    if (e.key === 'Escape') { clearSelection(); S.searchSet = null; S.focusCluster = null; refreshHi(); if (S.data) renderLegend(colorFor(S.color).info); }
  });
  on(ui.slider, 'input', () => setLayer(Number(ui.slider.value), { user: true }));
  on(ui.ticks, 'click', e => { const l = e.target.closest('[data-l]')?.dataset.l; if (l != null) setLayer(Number(l), { user: true }); });
  on(ui.play, 'click', () => (S.play ? stopPlay() : startPlay()));
  on(ui.color, 'change', () => { S.color = ui.color.value; S.focusCluster = null; applyColors(); refreshHi(); });
  on(ui.n, 'change', () => { S.n = Number(ui.n.value); load(S.run); });
  on(ui.bloom, 'change', () => { bloomOn = ui.bloom.checked; });
  on(ui.search, 'keydown', e => { e.stopPropagation(); if (e.key === 'Enter') highlight(ui.search.value); if (e.key === 'Escape') { ui.search.value = ''; highlight(''); } });
  on(ui.legend, 'click', e => {
    const c = e.target.closest('[data-cluster]'); if (!c) return;
    const id = Number(c.dataset.cluster);
    S.focusCluster = S.focusCluster === id ? null : id;
    renderLegend(colorFor(S.color).info); refreshHi();
    if (S.focusCluster != null) {
      const c = new THREE.Vector3(); let n = 0;
      S.data.points.forEach((p, i) => { if (p.cluster === id) { c.add(curPos(i, tmpV2)); n++; } });
      if (n) flyTo(c.divideScalar(n), 9);
    }
  });
  on(ui.card, 'click', e => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'close') return clearSelection();
    if (act === 'open') return ctx.onTokenClick?.(S.sel.id);
    const bar = e.target.closest('.m3d-bar'); if (bar) return setLayer(Number(bar.dataset.l), { user: true });
    const row = e.target.closest('[data-id]'); if (row) selectToken(Number(row.dataset.id), { fly: true });
  });
  on(ui.card, 'dblclick', e => { const row = e.target.closest('[data-id]'); if (row) ctx.onTokenClick?.(Number(row.dataset.id)); });

  // ---- resize / visibility
  function onResize() {
    const w = Math.max(1, root.clientWidth), h = Math.max(1, root.clientHeight);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    composer.setPixelRatio(dpr); composer.setSize(w, h);
    bloomPass.resolution.set(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const scale = (h * dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const mats = [S.mat, ...trailGroup.children.filter(o => o.isPoints).map(o => o.material)].filter(Boolean);
    for (const m of mats) { m.uniforms.uScale.value = scale; m.uniforms.uMinPx.value = 1.6 * dpr; }
    S.uScale = scale; S.dpr = dpr;
  }
  const ro = new ResizeObserver(() => onResize()); ro.observe(root);
  const io = new IntersectionObserver(es => { S.inView = es[es.length - 1].isIntersecting; updateRunning(); });
  S.inView = true; io.observe(root);
  on(document, 'visibilitychange', updateRunning);
  function updateRunning() {
    const run = S.inView && !document.hidden && !S.destroyed;
    if (run && !S.raf) { S.last = performance.now(); S.raf = requestAnimationFrame(frame); }
    if (!run && S.raf) { cancelAnimationFrame(S.raf); S.raf = 0; }
  }

  // ---- frame loop
  function frame(now) {
    S.raf = requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - S.last) / 1000); S.last = now;
    if (S.morph) {
      const t = clamp((now - S.morph.t0) / S.morph.dur);
      S.mixE = ease(t); S.mat.uniforms.uMix.value = S.mixE;
      if (t >= 1) { S.morph = null; S.mixE = 1; S.mouseDirty = true; }
    }
    if (S.fly) {
      const t = clamp((now - S.fly.t0) / S.fly.dur), e = ease(t);
      camera.position.lerpVectors(S.fly.p0, S.fly.p1, e);
      controls.target.lerpVectors(S.fly.t0v, S.fly.t1v, e);
      if (t >= 1) { S.fly = null; idleResume(); }
    }
    if (S.play && !S.morph && now >= S.playNext) {
      setLayer((S.cur + 1) % (S.nLayer + 1));
      S.playNext = now + (reduced ? 1600 : MORPH_MS + 900);
    }
    controls.update(dt);
    const camDist = camera.position.distanceTo(controls.target);
    if (S.mat) {
      S.mat.uniforms.uTime.value = now / 1000;
      S.mat.uniforms.uFogNear.value = Math.max(0.1, camDist - 2);
      S.mat.uniforms.uFogFar.value = camDist + 12;
    }
    for (const o of trailGroup.children) if (o.isPoints) {
      o.material.uniforms.uTime.value = now / 1000;
      o.material.uniforms.uFogNear.value = 1e3; o.material.uniforms.uFogFar.value = 2e3;
    }
    if (S.mouseDirty) doHover();
    // labels: follow the morph and fade with distance
    if (S.data) {
      const near = camDist * 0.75, far = camDist * 1.35;
      const sel = !!S.sel;
      const showCl = S.color === 'cluster' && S.cur >= S.nLayer && !S.morph && !sel && S.focusCluster == null;
      for (const o of S.clLabels) { o.visible = showCl; }
      for (const o of S.topLabels) {
        curPos(o.userData.i, o.position);
        const d = camera.position.distanceTo(o.position);
        const op = clamp((far - d) / (far - near)) * (sel || S.searchSet || S.focusCluster != null ? 0.18 : showCl ? 0.45 : 0.95);
        o.element.style.opacity = op.toFixed(2);
        o.visible = op > 0.03;
      }
    }
    updateSelectionFrame(now);
    // ripples
    if (S.rippleT0 >= 0 && S.sel) {
      const p = curSelPos(tmpV), dist = camera.position.distanceTo(p);
      let alive = false;
      ripples.forEach((r, k) => {
        const t = (now - S.rippleT0 - k * 320) / 1300;
        if (t < 0 || t > 1) { r.visible = false; if (t < 0) alive = true; return; }
        alive = true; r.visible = true; r.position.copy(p);
        const s = dist * (0.03 + 0.22 * ease(t)); r.scale.set(s, s, s);
        r.material.opacity = (1 - t) * 0.9;
      });
      if (!alive) S.rippleT0 = -1;
    } else ripples.forEach(r => { r.visible = false; });
    decor.children[0].rotation.y += dt * 0.004;
    if (bloomOn) composer.render(dt); else renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }

  // ---- go
  onResize();
  updateRunning();
  if (ctx.run) await load(ctx.run, false);
  if (!reduced) idleResume();
  if (!reduced) controls.autoRotate = true;

  return {
    async setRun(name) { await load(name, false); },
    setLayer(l) { setLayer(l, { user: true }); },
    highlight(q) { ui.search.value = q ?? ''; return highlight(q); },
    resize: onResize,
    destroy() {
      if (S.destroyed) return;
      S.destroyed = true; S.loadSeq++;
      if (S.raf) cancelAnimationFrame(S.raf); S.raf = 0;
      for (const t of timers) clearTimeout(t); timers.clear();
      ac.abort(); ro.disconnect(); io.disconnect();
      controls.dispose();
      clearTrail(); disposePoints();
      scene.traverse(o => {
        o.geometry?.dispose();
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { m.map?.dispose(); m.dispose(); });
      });
      bgTex.dispose(); ringTex.dispose();
      bloomPass.dispose(); outputPass.dispose(); composer.dispose();
      renderer.dispose(); renderer.forceContextLoss();
      scene.clear();
      root.remove();
      if (--styleRefs === 0) document.getElementById(STYLE_ID)?.remove();
    },
  };
}
