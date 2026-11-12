"""Train a (PLE-)GPT on tokenised .bin data.

Example:
  python -m ple_lab.train --name ple-small --ple-dim 64 --max-iters 3000
Logs to runs/<name>/log.jsonl and saves the run with ple_lab.checkpoint.save_run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoint import save_run
from .model import GPT, GPTConfig

# RMSNorm under bf16 autocast falls back to the unfused kernel; harmless but noisy.
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, d_model=64, block_size=64),
    "small": dict(n_layer=6, n_head=6, d_model=384, block_size=256),
}


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device, gen: np.random.Generator):
    ix = gen.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if torch.device(device).type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(it: int, max_iters: int, lr: float, warmup: int, min_lr_frac: float = 0.1) -> float:
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = min(1.0, (it - warmup) / max(1, max_iters - warmup))
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: GPT, lr: float, weight_decay: float, device_type: str):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # matrices in Linear layers decay; norms, embeddings (wte/wpe/PLE table) don't
        is_emb = name.startswith(("wte.", "wpe.", "ple_table."))
        (decay if p.dim() >= 2 and not is_emb else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": True} if device_type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), **kw)


@torch.no_grad()
def estimate_loss(model, data, cfg, batch_size, iters, device, ctx, seed=1234):
    model.eval()
    gen = np.random.default_rng(seed)  # fixed windows -> comparable val loss across evals/runs
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device, gen)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.device(device).type
    torch.manual_seed(args.seed)
    if dtype == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "cuda" else nullcontext()

    data_dir = Path(args.data)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    preset = PRESETS[args.preset]
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size or preset["block_size"],
        n_layer=args.n_layer or preset["n_layer"],
        n_head=args.n_head or preset["n_head"],
        d_model=args.d_model or preset["d_model"],
        dropout=args.dropout,
        ple_dim=args.ple_dim,
    )
    model = GPT(cfg).to(device)
    opt = make_optimizer(model, args.lr, args.weight_decay, dtype)
    fwd = model
    if args.compile:
        fwd = torch.compile(model)

    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    log_path.write_text("")
    breakdown = model.param_breakdown()
    print(f"[{args.name}] device={device} cfg={cfg.to_dict()} params={breakdown}")

    gen = np.random.default_rng(args.seed)
    tokens_per_iter = args.batch_size * args.grad_accum * cfg.block_size
    warmup = args.warmup if args.warmup is not None else max(1, args.max_iters // 20)
    t_start = time.time()
    t_win, tok_win = time.time(), 0
    val_loss, train_loss, history = None, None, []
    for it in range(args.max_iters + 1):
        last = it == args.max_iters
        if it % args.eval_interval == 0 or last:
            val_loss = estimate_loss(fwd, val_data, cfg, args.batch_size, args.eval_iters, device, ctx)
            dt = time.time() - t_win
            rec = {"iter": it, "train_loss": train_loss, "val_loss": val_loss,
                   "tokens_per_s": tok_win / dt if tok_win else None, "elapsed": time.time() - t_start,
                   "lr": lr_at(it, args.max_iters, args.lr, warmup), "tokens": it * tokens_per_iter}
            if dtype == "cuda":
                rec["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            history.append(rec)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tps = f"{rec['tokens_per_s']:.0f}" if rec["tokens_per_s"] else "-"
            tl = f"{train_loss:.4f}" if train_loss is not None else "-"
            print(f"[{args.name}] it {it}/{args.max_iters} train {tl} val {val_loss:.4f} tok/s {tps} "
                  f"elapsed {rec['elapsed']:.0f}s", flush=True)
            t_win, tok_win = time.time(), 0
        if last:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args.max_iters, args.lr, warmup)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device, gen)
            with ctx:
                _, loss = fwd(x, y)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        train_loss = loss_acc
        tok_win += tokens_per_iter

    elapsed = time.time() - t_start
    train_tps = [h["tokens_per_s"] for h in history if h["tokens_per_s"]]
    extra = {
        "val_loss": val_loss, "train_loss": train_loss, "tokenizer": "gpt2", "dataset": args.dataset,
        "iters": args.max_iters, "tokens_seen": args.max_iters * tokens_per_iter,
        "tokens_per_s": float(np.median(train_tps)) if train_tps else None, "elapsed_s": elapsed,
        "param_breakdown": breakdown,
        "train_args": {k: v for k, v in vars(args).items()},
    }
    save_run(model, run_dir, extra=extra)
    print(f"[{args.name}] done: val {val_loss:.4f} in {elapsed:.0f}s -> {run_dir}")
    return {"run_dir": str(run_dir), "history": history, **extra}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a GPT (optionally with PLE)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--ple-dim", type=int, default=0)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=None, help="default: 5%% of max-iters")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--data", default="data/tinystories")
    ap.add_argument("--dataset", default="tinystories")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", help="torch.compile (needs triton; usually unavailable on Windows)")
    return ap


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
