# PLE-Lab shared contract

All workstreams build on `ple_lab/model.py` and `ple_lab/checkpoint.py`. Do not change their public behaviour without updating this file and `tests/test_core.py`.

## Model
- `GPTConfig(vocab_size, block_size, n_layer, n_head, d_model, dropout, ple_dim, ple_vocab_size)`. `ple_dim=0` = baseline (no PLE).
- PLE token part: `rows = model.ple_table(ids)` → `[B,T,L*p]` raw (unscaled) rows. Model reshapes to `[B,T,L,p]` and multiplies by `sqrt(p)`.
- Per-layer input: `u = (tok*mask + RMSNorm(W_ctx·x0·d^-0.5)) / sqrt(2)` → `[B,T,L,p]`.
- In block `l` after attn+MLP: `h += RMSNorm(W_proj(gelu(W_gate h) ⊙ u_l))`. `block.ple.gate_activations(h)` exposes the gate.
- `model(ids, targets=None, ple_mask=None, return_hidden=False)`; `ple_mask` broadcastable to `[B,T,L,1]` zeros token-part per layer (ablation).

## Table source interface (offload engine plugs in here)
Any `nn.Module` assigned to `model.ple_table` with `forward(ids: LongTensor[B,T]) -> Tensor[B,T,L*p]` returning rows on `ids.device`. Default = `PLETableSource` (an `nn.Embedding` at `model.ple_table.table`).

## Run directory (`runs/<name>/`)
| file | content |
|---|---|
| `config.json` | `{"config": GPTConfig dict, ...extra (e.g. val_loss, tokenizer)}` |
| `dense.pt` | state_dict of everything **except** `ple_table.table.weight` |
| `ple_table.f16.bin` | raw float16 row-major `[V, L*p]` |
| `ple_table.i8.bin` + `.i8.scale.bin` | int8 rows + float32 per-row scale (`row ≈ q*scale`) |
| `ple_table.json` | `{vocab, n_layer, ple_dim, row_width, files}` |

Helpers: `save_run(model, dir, extra)`, `load_run(dir, device, load_table=True)`, `load_table_memmap(dir, "f16"|"i8") -> (rows, scale|None, meta)`.

Tokenizer: tiktoken `gpt2` (vocab 50257, model vocab padded to 50304).
