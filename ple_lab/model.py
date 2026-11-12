"""Tiny GPT with optional Per-Layer Embeddings (PLE), following Gemma 3n's design.

PLE path (see docs/CONTRACT.md):
  u = (RMSNorm(W_ctx x0 * d^-0.5).view(L,p) + E_ple[ids].view(L,p) * sqrt(p)) / sqrt(2)
  in block l, after attn+mlp:  h = h + RMSNorm(W_proj(gelu(W_gate h) * u_l))
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    ple_dim: int = 0  # 0 disables PLE
    ple_vocab_size: int | None = None  # defaults to vocab_size

    @property
    def ple_vocab(self) -> int:
        return self.ple_vocab_size or self.vocab_size

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class PLEInject(nn.Module):
    """Per-layer gate: h += RMSNorm(W_proj(gelu(W_gate h) * u_l))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
        self.proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
        self.norm = nn.RMSNorm(cfg.d_model)

    def gate_activations(self, h):
        return F.gelu(self.gate(h), approximate="tanh")

    def forward(self, h, u_l):
        return h + self.norm(self.proj(self.gate_activations(h) * u_l))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.ple = PLEInject(cfg) if cfg.ple_dim > 0 else None

    def forward(self, x, u_l=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if self.ple is not None:
            x = self.ple(x, u_l)
        return x


class PLETableSource(nn.Module):
    """Default in-memory table. Offload engines replace this with the same interface:
    `forward(ids: LongTensor[B,T]) -> Tensor[B,T,L*p]` (raw rows, unscaled) on ids.device."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.table = nn.Embedding(cfg.ple_vocab, cfg.n_layer * cfg.ple_dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, ids):
        return self.table(ids)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        if cfg.ple_dim > 0:
            self.ple_table: nn.Module = PLETableSource(cfg)
            self.ple_ctx = nn.Linear(cfg.d_model, cfg.n_layer * cfg.ple_dim, bias=False)
            self.ple_ctx_norm = nn.RMSNorm(cfg.ple_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- PLE helpers (used by offload engine + explorer) ----
    def ple_token_part(self, ids):
        L, p = self.cfg.n_layer, self.cfg.ple_dim
        rows = self.ple_table(ids)
        return rows.view(*ids.shape, L, p).to(self.wte.weight.dtype) * math.sqrt(p)

    def ple_ctx_part(self, x0):
        L, p = self.cfg.n_layer, self.cfg.ple_dim
        c = self.ple_ctx(x0) * (self.cfg.d_model ** -0.5)
        return self.ple_ctx_norm(c.view(*x0.shape[:-1], L, p))

    def per_layer_inputs(self, ids, x0, ple_mask=None):
        """Returns u: [B,T,L,p]. ple_mask (broadcastable to [B,T,L,1]) zeroes the token part."""
        tok = self.ple_token_part(ids)
        if ple_mask is not None:
            tok = tok * ple_mask
        return (tok + self.ple_ctx_part(x0)) * (2 ** -0.5)

    def forward(self, ids, targets=None, ple_mask=None, return_hidden=False):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x0 = self.wte(ids) + self.wpe(pos)
        u = self.per_layer_inputs(ids, x0, ple_mask) if self.cfg.ple_dim > 0 else None
        x, hidden = x0, []
        for l, blk in enumerate(self.blocks):
            if return_hidden:
                hidden.append(x)
            x = blk(x, u[:, :, l] if u is not None else None)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def param_breakdown(self) -> dict:
        table = self.ple_table.table.weight.numel() if self.cfg.ple_dim and hasattr(self.ple_table, "table") else 0
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "ple_table": table, "dense": total - table}
