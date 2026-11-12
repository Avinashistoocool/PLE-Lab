// Plain-language explanations for the "?" popovers (condensed from docs/EXPLORER.md).

export const INFO = {
  norms: {
    title: 'Per-layer row norm',
    body: `<p>The length of the vector of this token in the slice of each layer, <code>‖E[t]<sub>l</sub>·√p‖</code>.</p>
      <p>The dashed line is the norm of a slice at initialisation (≈ 0.02·p). Training moved the bars that are much higher than this line. Rare tokens stay near it.</p>
      <p class="pop-tip">Click a bar to show the neighbours of the token in that layer.</p>`,
  },
  layersim: {
    title: 'Layer similarity',
    body: `<p>The cosine similarity between the slices of this token in different layers, next to the average matrix over the most frequent tokens.</p>
      <p>High values off the diagonal mean that the layers keep similar information. Values near 0 mean that each layer gets its own, unrelated vector. In <code>ple-small</code>, the average is ≈ 0. Thus, the layers are independent.</p>`,
  },
  vectors: {
    title: 'Vectors',
    body: `<p>The raw <code>L × p</code> token part as a heatmap: one row for each layer, one column for each dimension (with average pooling if p &gt; 128).</p>
      <p>Blue is negative. Warm colours (red/orange) are positive. Zero has the colour of the surface.</p>`,
  },
  neighbors: {
    title: 'Nearest tokens',
    body: `<p>The tokens with a slice that has the most similar direction (cosine) in the selected layer, or in the full row.</p>
      <p>When two slices are similar, the PLE path of that layer receives the same signal for both tokens. Full rows are clearly semantic (" dog" → puppy, cat, kitten). Single slices have more noise.</p>
      <p class="pop-tip">Click a row to go to that token.</p>`,
  },
  map: {
    title: 'Embedding map',
    body: `<p>A 2-D PCA projection of the slice of one layer (or the full row) for the N most frequent tokens.</p>
      <p>Usually, two components explain only a small part of the variance (the toolbar shows this value). Thus, the map is only an approximate view of the dominant directions. Frequently, these directions are frequency or norm.</p>
      <p class="pop-tip">Move the pointer over a point to see the token. Click the point to open it.</p>`,
  },
  chips: {
    title: 'Token stream',
    body: `<p>Your text, divided into GPT-2 tokens. The colour shows one of these values:</p>
      <ul><li><b>gate</b>: mean |gelu(W<sub>gate</sub>h)| in the selected layer. This shows how much of the slice of the token that layer reads here;</li>
      <li><b>loss</b>: −log p(token | previous tokens);</li>
      <li><b>Δloss</b>: ablated − normal loss. A positive value (orange) means that the token is more difficult to predict without PLE.</li></ul>
      <p class="pop-tip">Move the pointer over a token to see the gates of each layer and the top-5 for the next token. Click a token to pin it.</p>`,
  },
  contrib: {
    title: 'PLE contribution',
    body: `<p><code>‖RMSNorm(W<sub>proj</sub>(gate ⊙ u<sub>l</sub>))‖ / ‖h‖</code>: the size of the PLE update, relative to the residual stream that receives it.</p>
      <p>The numerator is almost constant. Thus, the ratio mostly decreases when ‖h‖ increases with depth. The ablation is a better measure of how much a layer needs PLE.</p>`,
  },
  gatemean: {
    title: 'Gate magnitude',
    body: `<p>The mean gate activation of each layer over your text. A high value means that the layer opens its PLE input wide.</p>
      <p class="pop-tip">Click a bar to colour the tokens by that layer.</p>`,
  },
  ablation: {
    title: 'Ablate PLE',
    body: `<p>Runs the model again with the <em>token part</em> of the selected layers set to zero (the context part stays). Then it compares the losses.</p>
      <p>Over 25,500 validation predictions, the ablation of all layers costs <b>+0.621 nats</b>. Layer 0 only costs +0.090. One short text has much noise. Thus, use it only to get a general idea.</p>`,
  },
  selected: {
    title: 'Selected token',
    body: `<p>The gate magnitude of each layer for the pinned token, and the top-5 guesses of the model for the <em>next</em> token (green = the actual next token).</p>`,
  },
  hist: {
    title: 'Row-norm histogram',
    body: `<p>For each layer: the number of the 50k vocabulary rows that have a given slice norm (log counts).</p>
      <p>The peak near the initial norm shows the tokens that never occur in the training data. Trained tokens move to the right.</p>`,
  },
  normfreq: {
    title: 'Row norm vs. frequency',
    body: `<p>The full-row norm against the number of times that the token occurs in train.bin (log scale), for 3000 random tokens.</p>
      <p>The norms follow the log frequency (Spearman ρ = 0.87 over seen tokens). Rows that never occur get no gradient from the loss, and they stay near the initial value.</p>`,
  },
  quant: {
    title: 'int8 quantisation error',
    body: `<p>The relative L2 error <code>‖q·s − w‖/‖w‖</code> for each layer of the symmetric int8 table (one scale for each row).</p>
      <p>The error is about 0.72%. When you keep the table as int8, the val loss changes from 2.01533 to 2.01535.</p>`,
  },
  compare: {
    title: 'Baseline vs. PLE',
    body: `<p>The loss for each token of a baseline run (no PLE) and of a PLE run on the same text. <code>diff = baseline − PLE</code>.</p>
      <p>Orange = PLE predicted that token better. This comparison is meaningful only when the two runs used the same data and budget. One text is only an example.</p>`,
  },
  kpi: {
    title: 'Run at a glance',
    body: `<p>The size and quality of the selected run, from its <code>config.json</code>. The model only <em>indexes</em> the PLE table and never multiplies it. Thus, its parameters can stay off the GPU.</p>`,
  },
  glyph: {
    title: 'Token glyph',
    body: `<p>The same <code>L × p</code> token part as the heatmap, in the form of rings: one ring for each layer (inner = L0), one spoke for each dimension.</p>
      <p>Each token gets a fingerprint that you can recognise. When you move the pointer over a cell, the same cell in the heatmap below is highlighted.</p>`,
  },
  graph: {
    title: 'Neighbour graph',
    body: `<p>The token (centre) and its top-k nearest tokens by cosine in the layer that you select in “Nearest tokens” (or the full row). Thin links connect neighbours that are also near each other.</p>
      <p class="pop-tip">Drag the nodes to separate them. Click a node to open that token.</p>`,
  },
  river: {
    title: 'Gate river',
    body: `<p>A streamgraph of the PLE gate of each layer along your text. When the band of a layer is thicker at a token, the PLE path of that layer is more open there.</p>
      <p class="pop-tip">Move the pointer over the chart to highlight the related chip. Click to pin it.</p>`,
  },
  ablmorph: {
    title: 'Ablation morph',
    body: `<p>The loss for each token, with an animation to the state where the PLE token part is zero in the ablated layers. Tokens that move up need PLE.</p>
      <p class="pop-tip">Push and hold Space over the chart to see the state without ablation.</p>`,
  },
  slope: {
    title: 'Slope chart',
    body: `<p>Each line connects the loss of a token for the baseline (left) and for the PLE run (right). The strip shows <code>Δ = baseline − PLE</code>.</p>
      <p>Orange (Δ &gt; 0) = PLE is better. Sort by Δ to see the largest improvements and the largest losses.</p>`,
  },
};
