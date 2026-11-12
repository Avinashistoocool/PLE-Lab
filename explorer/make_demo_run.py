"""Create a tiny randomly initialised PLE run for developing the explorer.

    python -m explorer.make_demo_run [--out runs/demo-random]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ple_lab import GPT, GPTConfig, save_run


def make_demo_run(out: str | Path = "runs/demo-random", seed: int = 0, **overrides) -> Path:
    torch.manual_seed(seed)
    cfg = GPTConfig(**{"vocab_size": 50304, "block_size": 128, "n_layer": 4, "n_head": 4,
                       "d_model": 128, "ple_dim": 32, **overrides})
    model = GPT(cfg).eval()
    return save_run(model, out, {"val_loss": None, "tokenizer": "gpt2", "note": "random init (explorer demo)"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo-random")
    print("saved", make_demo_run(ap.parse_args().out))


if __name__ == "__main__":
    main()
