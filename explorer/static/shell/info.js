// Plain-language explanations for the "?" popovers (condensed from docs/EXPLORER.md).

export const INFO = {
  norms: {
    title: 'Per-layer row norm',
    body: `<p>The length of this token's vector in each layer's slice, <code>‖E[t]<sub>l</sub>·√p‖</code>.</p>
      <p>The dashed line is the norm a slice has at initialisation (≈ 0.02·p). Bars well above it were moved by training; rare tokens stay near it.</p>
      <p class="pop-tip">Click a bar to list the token's neighbours in that layer.</p>`,
  },
  layersim: {
    title: 'Layer similarity',
    body: `<p>Cosine similarity between this token's slices in different layers, next to the average matrix over the most frequent tokens.</p>
      <p>High off-diagonal values mean layers store similar information; values near 0 mean every layer gets its own, unrelated vector. In <code>ple-small</code> the average is ≈ 0: the layers are independent.</p>`,
  },
  vectors: {
    title: 'Vectors',
    body: `<p>The raw <code>L × p</code> token part as a heatmap: one row per layer, one column per dimension (average-pooled if p &gt; 128).</p>
      <p>Blue is negative, warm (red/orange) positive; zero fades into the surface.</p>`,
  },
  neighbors: {
    title: 'Nearest tokens',
    body: `<p>Tokens whose slice points in the most similar direction (cosine) in the chosen layer, or across the full row.</p>
      <p>Similar slices mean that layer's PLE path receives the same signal for both tokens. Full rows are cleanly semantic (" dog" → puppy, cat, kitten); single slices are noisier.</p>
      <p class="pop-tip">Click a row to jump to that token.</p>`,
  },
  map: {
    title: 'Embedding map',
    body: `<p>A 2-D PCA projection of one layer's slice (or the full row) for the N most frequent tokens.</p>
      <p>Two components usually explain only a small share of the variance (shown in the toolbar), so treat it as a rough view of the dominant directions, often frequency or norm.</p>
      <p class="pop-tip">Hover for the token, click to open it.</p>`,
  },
  chips: {
    title: 'Token stream',
    body: `<p>Your text, split into GPT-2 tokens and coloured by:</p>
      <ul><li><b>gate</b>: mean |gelu(W<sub>gate</sub>h)| in the chosen layer, i.e. how much of the token's slice that layer reads here;</li>
      <li><b>loss</b>: −log p(token | previous tokens);</li>
      <li><b>Δloss</b>: ablated − normal loss. Positive (orange) = harder to predict without PLE.</li></ul>
      <p class="pop-tip">Hover for per-layer gates and the next-token top-5; click to pin.</p>`,
  },
  contrib: {
    title: 'PLE contribution',
    body: `<p><code>‖RMSNorm(W<sub>proj</sub>(gate ⊙ u<sub>l</sub>))‖ / ‖h‖</code>: the size of the PLE update relative to the residual stream it is added to.</p>
      <p>The numerator is roughly constant, so the ratio mostly falls as ‖h‖ grows with depth. The ablation is the better measure of how much a layer relies on PLE.</p>`,
  },
  gatemean: {
    title: 'Gate magnitude',
    body: `<p>The mean gate activation per layer over your text. High = the layer opens its PLE input wide.</p>
      <p class="pop-tip">Click a bar to colour the tokens by that layer.</p>`,
  },
  ablation: {
    title: 'Ablate PLE',
    body: `<p>Re-runs the model with the <em>token part</em> of the chosen layers set to zero (the context part stays), then compares losses.</p>
      <p>Over 25,500 validation predictions, ablating all layers costs <b>+0.621 nats</b>; layer 0 alone costs +0.090. One short text is noisy, so use it for intuition.</p>`,
  },
  selected: {
    title: 'Selected token',
    body: `<p>Gate magnitude per layer for the pinned token, and the model's top-5 guesses for the <em>next</em> token (green = the actual next token).</p>`,
  },
  hist: {
    title: 'Row-norm histogram',
    body: `<p>For every layer: how many of the 50k vocabulary rows have a given slice norm (log counts).</p>
      <p>The spike near the init norm is the tokens that never occur in the training data; trained tokens spread to the right.</p>`,
  },
  normfreq: {
    title: 'Row norm vs. frequency',
    body: `<p>Full-row norm against how often the token occurs in train.bin (log scale) for 3000 random tokens.</p>
      <p>Norms track log frequency (Spearman ρ = 0.87 over seen tokens). Rows never seen get no gradient from the loss and stay near init.</p>`,
  },
  quant: {
    title: 'int8 quantisation error',
    body: `<p>Relative L2 error <code>‖q·s − w‖/‖w‖</code> per layer of the per-row symmetric int8 table.</p>
      <p>About 0.72%; storing the table as int8 changes val loss from 2.01533 to 2.01535.</p>`,
  },
  compare: {
    title: 'Baseline vs. PLE',
    body: `<p>Per-token loss of a baseline run (no PLE) and a PLE run on the same text; <code>diff = baseline − PLE</code>.</p>
      <p>Orange = PLE predicted that token better. Only meaningful when both runs saw the same data and budget; one text is anecdotal.</p>`,
  },
  kpi: {
    title: 'Run at a glance',
    body: `<p>Size and quality of the selected run, read from its <code>config.json</code>. The PLE table is only ever <em>indexed</em> (never multiplied), so its parameters can live off the GPU.</p>`,
  },
  glyph: {
    title: 'Token glyph',
    body: `<p>The same <code>L × p</code> token part as the heatmap, wrapped into a ring: one ring per layer (inner = L0), one spoke per dimension.</p>
      <p>Each token gets a recognisable fingerprint. Hovering a cell highlights the same cell in the heatmap below.</p>`,
  },
  graph: {
    title: 'Neighbour graph',
    body: `<p>The token (centre) and its top-k nearest tokens by cosine in the layer chosen under “Nearest tokens” (or the full row). Thin links join neighbours that are also close to each other.</p>
      <p class="pop-tip">Drag nodes to untangle; click a node to open that token.</p>`,
  },
  river: {
    title: 'Gate river',
    body: `<p>A streamgraph of the PLE gate per layer along your text: the thicker a layer's band at a token, the more that layer's PLE path is open there.</p>
      <p class="pop-tip">Hover to highlight the matching chip; click to pin it.</p>`,
  },
  ablmorph: {
    title: 'Ablation morph',
    body: `<p>Per-token loss with the PLE token part (morphing to) zeroed in the ablated layers. Tokens that jump up rely on PLE.</p>
      <p class="pop-tip">Hold Space over the chart to see the un-ablated state.</p>`,
  },
  slope: {
    title: 'Slope chart',
    body: `<p>Each line joins a token's loss under the baseline (left) and the PLE run (right); the strip shows <code>Δ = baseline − PLE</code>.</p>
      <p>Orange (Δ &gt; 0) = PLE better. Sort by Δ to see the biggest wins and losses.</p>`,
  },
};
