// Neighbour force graph (DESIGN_IDEAS #3, Token-tab variant). SVG + d3-force.
import { d3, theme, layerColor, tip, tipHtml, esc, vis, fmt, mount, widthOf, onResize, reducedMotion, alpha } from './core.js';

/**
 * data: /api/neighbor_graph response
 *   {id, piece, layer: int|'all', layers: [..], nodes: [{id, piece, root, layers[], sim, shared}],
 *    links: [{source, target, sim, layer (-1 = full row), kind: 'root'|'peer'}]}
 * opts: {height?: 420, onNodeClick?(node), onHover?(node|null)}
 * returns {update(data), focus(id|null), destroy()}
 * Root is pinned at the centre; other nodes are draggable (drag pins them until double-click).
 */
export function render(el, data, opts = {}) {
  const root = mount(el, 'viz-force', 'Neighbour graph');
  const H = opts.height || 420;
  const svg = d3.select(root).append('svg').attr('height', H);
  const defs = svg.append('defs');
  const glow = defs.append('filter').attr('id', `vz-glow-${Math.random().toString(36).slice(2)}`).attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
  glow.append('feGaussianBlur').attr('stdDeviation', 4).attr('result', 'b');
  const fm = glow.append('feMerge'); fm.append('feMergeNode').attr('in', 'b'); fm.append('feMergeNode').attr('in', 'SourceGraphic');
  const gRings = svg.append('g'), gLinks = svg.append('g'), gNodes = svg.append('g');
  const legend = document.createElement('div'); legend.className = 'viz-legend';
  root.append(legend);
  let th = theme(), W = widthOf(el), cur = null, nodes = [], links = [];
  const sim = d3.forceSimulation().alphaDecay(0.035).velocityDecay(0.32);
  sim.on('tick', ticked);

  function key(n) { return n.id; }

  function update(d) {
    th = theme(); W = widthOf(el);
    svg.attr('width', W).attr('viewBox', [-W / 2, -H / 2, W, H]);
    const old = new Map(nodes.map(n => [n.id, n]));
    cur = d;
    const all = d.layers.length > 1;
    nodes = d.nodes.map(n => {
      const o = old.get(n.id);
      const a = Math.random() * 2 * Math.PI;
      return { ...n, x: o ? o.x : Math.cos(a) * 30, y: o ? o.y : Math.sin(a) * 30, vx: 0, vy: 0, fx: n.root ? 0 : null, fy: n.root ? 0 : null };
    });
    const byId = new Map(nodes.map(n => [n.id, n]));
    links = d.links.map(l => ({ ...l, source: byId.get(l.source), target: byId.get(l.target) })).filter(l => l.source && l.target);
    const maxSim = d3.max(nodes.filter(n => !n.root), n => n.sim) || 1, minSim = d3.min(nodes.filter(n => !n.root), n => n.sim) || 0;
    const R = Math.min(W, H) / 2 - 40;
    const dist = s => 50 + (R - 50) * (1 - (s - minSim) / (maxSim - minSim || 1));
    sim.nodes(nodes)
      .force('link', d3.forceLink(links).distance(l => (l.kind === 'root' ? dist(l.sim) : 70)).strength(l => (l.kind === 'root' ? 0.5 / (all ? l.target.shared : 1) : 0.05 + 0.15 * Math.max(0, l.sim))))
      .force('charge', d3.forceManyBody().strength(-180))
      .force('collide', d3.forceCollide().radius(n => pillW(n) / 2 + 4).strength(0.9))
      .force('x', d3.forceX(0).strength(0.02)).force('y', d3.forceY(0).strength(0.03));

    // sim rings (radial guides)
    const ticks = d3.ticks(minSim, maxSim, 3).filter(t => t > minSim && t < maxSim);
    gRings.selectAll('g').data(ticks, t => t).join(e => { const g = e.append('g'); g.append('circle'); g.append('text'); return g; })
      .call(g => g.select('circle').attr('r', t => dist(t)).attr('fill', 'none').attr('stroke', th.hairline).attr('stroke-dasharray', '2 4'))
      .call(g => g.select('text').attr('y', t => -dist(t) - 3).attr('text-anchor', 'middle').attr('font-size', 9).attr('fill', th.faint).text(t => `cos ${t.toFixed(2)}`));

    const ls = gLinks.selectAll('line').data(links, l => `${l.source.id}-${l.target.id}-${l.layer}`).join('line')
      .attr('class', 'link')
      .attr('stroke', l => (l.kind === 'peer' ? th.hairlineStrong : all ? layerColor(th, l.layer) : th.accent))
      .attr('stroke-opacity', l => (l.kind === 'peer' ? 0.6 : 0.25 + 0.75 * clamp((l.sim - minSim) / (maxSim - minSim || 1))))
      .attr('stroke-width', l => (l.kind === 'peer' ? 1 : 1 + 2.5 * clamp((l.sim - minSim) / (maxSim - minSim || 1))))
      .attr('stroke-dasharray', l => (l.kind === 'peer' ? '3 3' : null));

    const ns = gNodes.selectAll('g.node').data(nodes, key).join(enter => {
      const g = enter.append('g').attr('class', 'node').attr('tabindex', 0).attr('role', 'button');
      g.append('rect').attr('class', 'pill');
      g.append('g').attr('class', 'dots');
      g.append('text').attr('class', 'lbl').attr('text-anchor', 'middle').attr('dominant-baseline', 'central');
      g.attr('opacity', 0).transition().duration(reducedMotion() ? 0 : 500).attr('opacity', 1);
      return g;
    });
    ns.attr('aria-label', n => `${n.piece}, cosine ${fmt(n.sim, 3)}`);
    ns.select('rect.pill')
      .attr('x', n => -pillW(n) / 2).attr('y', n => -pillH(n) / 2).attr('width', pillW).attr('height', pillH).attr('rx', n => pillH(n) / 2)
      .attr('fill', n => (n.root ? alpha(th.accent, 0.18) : th.surface2))
      .attr('stroke', n => (n.root ? th.accent : all ? layerColor(th, n.layers[0]) : alpha(th.accent, 0.25 + 0.6 * clamp((n.sim - minSim) / (maxSim - minSim || 1)))))
      .attr('stroke-width', n => (n.root ? 2 : 1.2))
      .attr('filter', n => (n.root ? `url(#${glow.attr('id')})` : null));
    ns.select('text.lbl').attr('font-size', n => (n.root ? 14 : 12)).attr('font-weight', n => (n.root ? 700 : 500))
      .attr('fill', n => (n.root ? th.text : th.text)).attr('y', n => (all && !n.root ? -2 : 0)).text(n => vis(n.piece));
    ns.select('g.dots').selectAll('circle').data(n => (all && !n.root ? n.layers.map(l => ({ l, n })) : []))
      .join('circle').attr('r', 2).attr('cy', d => pillH(d.n) / 2 - 4)
      .attr('cx', (d, i, arr) => (i - (arr.length - 1) / 2) * 6).attr('fill', d => layerColor(th, d.l));

    ns.on('pointerenter', (ev, n) => { focus(n.id); opts.onHover?.(n); })
      .on('pointermove', (ev, n) => showTip(ev, n))
      .on('pointerleave', () => { focus(null); tip.hide(); opts.onHover?.(null); })
      .on('click', (ev, n) => { if (ev.defaultPrevented) return; tip.hide(); opts.onNodeClick?.(n); })
      .on('keydown', (ev, n) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); opts.onNodeClick?.(n); } })
      .on('dblclick', (ev, n) => { if (!n.root) { n.fx = n.fy = null; sim.alpha(0.3).restart(); } })
      .call(d3.drag()
        .filter(ev => !ev.button)
        .on('start', (ev, n) => { if (n.root) return; if (!ev.active) sim.alphaTarget(0.25).restart(); n.fx = n.x; n.fy = n.y; tip.hide(); })
        .on('drag', (ev, n) => { if (n.root) return; n.fx = ev.x; n.fy = ev.y; })
        .on('end', (ev, n) => { if (!ev.active) sim.alphaTarget(0); }));

    legend.innerHTML = (all
      ? d.layers.map(l => `<span><i style="background:${layerColor(th, l)}"></i>L${l}</span>`).join('') + '<span>dots = layers that list the node</span>'
      : `<span><i style="background:${th.accent}"></i>${d.layer === -1 || d.layer === '-1' ? 'full row' : `layer ${d.layer}`}</span>`)
      + `<span><i style="background:none;border-top:1px dashed ${th.hairlineStrong};height:0;width:14px"></i>peer link (full-row cos)</span><span>drag · dbl-click unpins · click opens</span>`;

    if (reducedMotion()) { sim.alpha(1); for (let i = 0; i < 300; i++) sim.tick(); ticked(); sim.stop(); }
    else sim.alpha(1).restart();
  }

  function ticked() {
    const hw = W / 2 - 4, hh = H / 2 - 4;
    for (const n of nodes) {
      const pw = pillW(n) / 2, ph = pillH(n) / 2;
      n.x = Math.max(-hw + pw, Math.min(hw - pw, n.x));
      n.y = Math.max(-hh + ph, Math.min(hh - ph, n.y));
    }
    gLinks.selectAll('line').attr('x1', l => l.source.x).attr('y1', l => l.source.y).attr('x2', l => l.target.x).attr('y2', l => l.target.y);
    gNodes.selectAll('g.node').attr('transform', n => `translate(${n.x},${n.y})`);
  }

  function focus(id) {
    root.classList.toggle('focus', id != null);
    if (id == null) { gNodes.selectAll('g.node').classed('hot', false); gLinks.selectAll('line').classed('hot', false); return; }
    const nb = new Set([id]);
    links.forEach(l => { if (l.source.id === id) nb.add(l.target.id); if (l.target.id === id) nb.add(l.source.id); });
    gNodes.selectAll('g.node').classed('hot', n => nb.has(n.id));
    gLinks.selectAll('line').classed('hot', l => l.source.id === id || l.target.id === id);
  }

  function showTip(ev, n) {
    const rows = [];
    if (!n.root) {
      rows.push(['cosine (best)', fmt(n.sim, 4), th.accent]);
      if (cur.layers.length > 1) rows.push(['found in', n.layers.map(l => `L${l}`).join(' ')]);
      links.filter(l => l.kind === 'root' && l.target.id === n.id && cur.layers.length > 1)
        .forEach(l => rows.push([`cos L${l.layer}`, fmt(l.sim, 4), layerColor(th, l.layer)]));
      const peers = links.filter(l => l.kind === 'peer' && (l.source.id === n.id || l.target.id === n.id));
      if (peers.length) rows.push(['peers', peers.map(l => esc(vis((l.source.id === n.id ? l.target : l.source).piece))).join(', ')]);
    } else rows.push(['neighbours', String(nodes.length - 1)]);
    tip.show(tipHtml(`${esc(vis(n.piece))}<small>#${n.id}</small>`, rows, n.root ? 'selected token (pinned)' : 'click to open · drag to move'), ev);
  }

  update(data);
  const off = onResize(el, () => { W = widthOf(el); svg.attr('width', W).attr('viewBox', [-W / 2, -H / 2, W, H]); sim.alpha(0.3).restart(); });
  return {
    update,
    focus,
    destroy() { sim.stop(); off(); tip.hide(); el.innerHTML = ''; },
  };
}

const pillH = n => (n.root ? 30 : 24);
const pillW = n => Math.max(28, vis(n.piece).length * (n.root ? 9 : 7.6) + 18);
const clamp = t => Math.max(0, Math.min(1, t));
