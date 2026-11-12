"""Generate text from a saved run.

Usage: python scripts/sample.py runs/ple-small --prompt "Once upon a time" --tokens 100 [--top-k 40 | --greedy]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ple_lab.checkpoint import load_run  # noqa: E402


@torch.no_grad()
def generate(model, ids, n, temperature=0.8, top_k=40, greedy=False, eot=50256):
    bs = model.cfg.block_size
    for _ in range(n):
        logits, _ = model(ids[:, -bs:])
        logits = logits[:, -1, :50257]
        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits.float(), -1), 1)
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == eot:
            break
    return ids


def main():
    import tiktoken

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    enc = tiktoken.get_encoding("gpt2")
    model = load_run(a.run, a.device)
    ids = torch.tensor([enc.encode_ordinary(a.prompt)], device=a.device)
    out = generate(model, ids, a.tokens, a.temperature, a.top_k, a.greedy)
    print(enc.decode([t for t in out[0].tolist() if t != 50256]))


if __name__ == "__main__":
    main()
