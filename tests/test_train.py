import json

import numpy as np
import torch

from ple_lab.checkpoint import load_run
from ple_lab.train import build_parser, lr_at, make_optimizer, train
from ple_lab.model import GPT, GPTConfig


def _synthetic_data(d, vocab=64, n=20000, seed=0):
    # highly predictable sequence: repeating pattern with a little noise
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, vocab, size=17)
    data = np.tile(pattern, n // 17 + 1)[:n].astype(np.uint16)
    data.tofile(d / "train.bin")
    data[: n // 4].tofile(d / "val.bin")


def _args(tmp_path, name, ple_dim):
    return build_parser().parse_args([
        "--name", name, "--preset", "tiny", "--ple-dim", str(ple_dim), "--vocab-size", "64",
        "--batch-size", "8", "--max-iters", "60", "--eval-interval", "20", "--eval-iters", "4",
        "--lr", "3e-3", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "runs"), "--device", "cpu",
    ])


def _run_and_check(tmp_path, name, ple_dim):
    (tmp_path / "data").mkdir(exist_ok=True)
    _synthetic_data(tmp_path / "data")
    res = train(_args(tmp_path, name, ple_dim))
    hist = res["history"]
    assert hist[-1]["val_loss"] < hist[0]["val_loss"] - 1.0
    run = tmp_path / "runs" / name
    lines = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert [l["iter"] for l in lines] == [0, 20, 40, 60]
    assert {"iter", "train_loss", "val_loss", "tokens_per_s", "elapsed"} <= set(lines[-1])
    meta = json.loads((run / "config.json").read_text())
    assert meta["tokenizer"] == "gpt2" and meta["iters"] == 60
    assert meta["param_breakdown"]["ple_table"] == (64 * 2 * ple_dim)
    model = load_run(run)
    x = torch.randint(0, 64, (2, 32))
    y = torch.roll(x, -1, 1)
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    return model


def test_train_baseline_reduces_loss_and_loads(tmp_path):
    m = _run_and_check(tmp_path, "base", 0)
    assert m.cfg.ple_dim == 0


def test_train_ple_reduces_loss_and_loads(tmp_path):
    m = _run_and_check(tmp_path, "ple", 8)
    assert m.cfg.ple_dim == 8 and m.ple_table.table.weight.shape == (64, 16)


def test_optimizer_groups_exclude_norms_and_embeddings():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, ple_dim=4))
    opt = make_optimizer(model, 1e-3, 0.1, "cpu")
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(model.wte.weight) not in decay_ids
    assert id(model.ple_table.table.weight) not in decay_ids
    assert id(model.ln_f.weight) not in decay_ids
    assert id(model.blocks[0].attn.qkv.weight) in decay_ids


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, 10) == 0.1
    assert abs(lr_at(9, 100, 1.0, 10) - 1.0) < 1e-9
    assert abs(lr_at(100, 100, 1.0, 10) - 0.1) < 1e-9
