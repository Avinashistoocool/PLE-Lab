# PLE-Lab shared contract

All workstreams use `ple_lab/model.py` and `ple_lab/checkpoint.py`. Do not change their public behaviour unless you also update this file and `tests/test_core.py`.

## Model
- `GPTConfig(vocab_size, block_size, n_layer, n_head, d_model, dropout, ple_dim, ple_vocab_size)`. `ple_dim=0` gives the baseline (no PLE).
- PLE token part: `rows = model.ple_table(ids)` gives `[B,T,L*p]` raw (unscaled) rows. The model changes the shape to `[B,T,L,p]` and multiplies by `sqrt(p)`.
- Input for each layer: `u = (tok*mask + RMSNorm(W_ctx·x0·d^-0.5)) / sqrt(2)` → `[B,T,L,p]`.
- In block `l`, after attn+MLP: `h += RMSNorm(W_proj(gelu(W_gate h) ⊙ u_l))`. `block.ple.gate_activations(h)` gives the gate.
- `model(ids, targets=None, ple_mask=None, return_hidden=False)`. `ple_mask` must broadcast to `[B,T,L,1]`. It sets the token part of each layer to zero (ablation).

## Table source interface (the offload engine connects here)
You can assign any `nn.Module` to `model.ple_table` if it has `forward(ids: LongTensor[B,T]) -> Tensor[B,T,L*p]` and returns rows on `ids.device`. The default is `PLETableSource` (an `nn.Embedding` at `model.ple_table.table`).

## Run directory (`runs/<name>/`)
| file | content |
|---|---|
| `config.json` | `{"config": GPTConfig dict, ...extra (e.g. val_loss, tokenizer)}` |
| `dense.pt` | state_dict of all parameters **except** `ple_table.table.weight` |
| `ple_table.f16.bin` | raw float16, row-major `[V, L*p]` |
| `ple_table.i8.bin` + `.i8.scale.bin` | int8 rows + a float32 scale for each row (`row ≈ q*scale`) |
| `ple_table.json` | `{vocab, n_layer, ple_dim, row_width, files}` |

Helpers: `save_run(model, dir, extra)`, `load_run(dir, device, load_table=True)`, `load_table_memmap(dir, "f16"|"i8") -> (rows, scale|None, meta)`.

Tokenizer: tiktoken `gpt2` (vocab 50257; the model pads the vocab to 50304).
