"""Checkpoint format shared by trainer, offload engine and explorer. See docs/CONTRACT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig

TABLE_KEY = "ple_table.table.weight"


def save_run(model: GPT, run_dir: str | Path, extra: dict | None = None) -> Path:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != TABLE_KEY}
    torch.save(state, run / "dense.pt")
    meta = {"config": cfg.to_dict(), **(extra or {})}
    if cfg.ple_dim > 0:
        w = model.ple_table.table.weight.detach().float().cpu().numpy()
        save_table(w, run, cfg)
    (run / "config.json").write_text(json.dumps(meta, indent=2))
    return run


def save_table(w: np.ndarray, run: Path, cfg: GPTConfig) -> None:
    """fp16 raw row-major [V, L*p] + int8 per-row-symmetric variant."""
    w16 = w.astype(np.float16)
    w16.tofile(run / "ple_table.f16.bin")
    scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    q.tofile(run / "ple_table.i8.bin")
    scale.astype(np.float32).tofile(run / "ple_table.i8.scale.bin")
    meta = {
        "vocab": int(w.shape[0]), "n_layer": cfg.n_layer, "ple_dim": cfg.ple_dim,
        "row_width": int(w.shape[1]),
        "files": {"f16": "ple_table.f16.bin", "i8": "ple_table.i8.bin", "i8_scale": "ple_table.i8.scale.bin"},
    }
    (run / "ple_table.json").write_text(json.dumps(meta, indent=2))


def load_table_memmap(run_dir: str | Path, fmt: str = "f16"):
    """Returns (rows_memmap, scale_or_None, meta). Rows are unscaled (multiply by sqrt(p) happens in model)."""
    run = Path(run_dir)
    meta = json.loads((run / "ple_table.json").read_text())
    shape = (meta["vocab"], meta["row_width"])
    if fmt == "f16":
        return np.memmap(run / meta["files"]["f16"], dtype=np.float16, mode="r", shape=shape), None, meta
    if fmt == "i8":
        rows = np.memmap(run / meta["files"]["i8"], dtype=np.int8, mode="r", shape=shape)
        scale = np.fromfile(run / meta["files"]["i8_scale"], dtype=np.float32)
        return rows, scale, meta
    raise ValueError(fmt)


def load_run(run_dir: str | Path, device="cpu", load_table: bool = True) -> GPT:
    """Load a run. If load_table=False the PLE table stays as an (uninitialised) placeholder
    and the caller must attach an offload source via `model.ple_table = ...`."""
    run = Path(run_dir)
    meta = json.loads((run / "config.json").read_text())
    cfg = GPTConfig(**meta["config"])
    model = GPT(cfg)
    if not load_table and cfg.ple_dim:
        model.ple_table = torch.nn.Identity()  # placeholder; caller replaces
    state = torch.load(run / "dense.pt", map_location="cpu", weights_only=True)
    if cfg.ple_dim and load_table:
        rows, _, _ = load_table_memmap(run, "f16")
        state[TABLE_KEY] = torch.from_numpy(np.asarray(rows, dtype=np.float32))
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k != TABLE_KEY and not k.endswith("lm_head.weight")]
    if real_missing or unexpected:
        raise RuntimeError(f"bad checkpoint: missing={real_missing} unexpected={unexpected}")
    model.lm_head.weight = model.wte.weight
    return model.to(device).eval()
