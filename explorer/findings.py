"""Extract quantitative findings from a trained PLE run via the explorer's analysis functions.

    python -m explorer.findings --run runs/ple-small [--baseline runs/baseline-small]
        [--val data/tinystories/val.bin] [--train data/tinystories/train.bin]
        [--chunks 100] [--out results/explorer_findings.json]

Ablation numbers use `--chunks` windows of block_size tokens sampled (fixed seed) from val.bin;
each window gives block_size-1 next-token predictions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis as A

ROOT = Path(__file__).resolve().parent.parent
PROBE_TOKENS = [" the", " happy", " dog", " ran", " Lily", " said"]


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    xs = x[order]  # average ranks for ties
    bounds = np.flatnonzero(np.diff(xs)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(x)]):
        r[order[s:e]] = (s + e - 1) / 2
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def val_windows(val: Path, block: int, n: int, seed: int = 0) -> list[list[int]]:
    data = np.memmap(val, dtype=np.uint16, mode="r")
    starts = np.sort(np.random.default_rng(seed).choice(len(data) - block, size=n, replace=False))
    return [data[s:s + block].astype(np.int64).tolist() for s in starts]


def ablation_study(run: A.LoadedRun, windows: list[list[int]], base: A.LoadedRun | None) -> dict:
    L = run.cfg.n_layer
    masks: list = [None, "all"] + [[l] for l in range(L)]
    names = ["none", "all"] + [f"layer{l}" for l in range(L)]
    losses = {k: [] for k in names}
    contrib = []
    gate = []
    base_losses = []
    for w in windows:
        normal = None
        for name, m in zip(names, masks):
            r = A.analyze_ids(run, w, m)
            if normal is None:
                normal = r
                losses["none"].extend(r["loss"][1:])
                contrib.append(np.array(r["contrib_ratio"]))
                gate.append(np.array(r["gate"]))
            else:
                losses[name].extend(r["ablation"]["loss"][1:])
        if base is not None:
            ids = torch.tensor([w])
            with torch.no_grad():
                logits, _ = base.model(ids)
            base_losses.extend(A._per_token_loss(logits, ids)[1:])
    ref = np.array(losses["none"])
    out = {"n_windows": len(windows), "window": len(windows[0]), "n_predictions": int(len(ref)),
           "mean_loss": float(ref.mean())}
    for name in names[1:]:
        d = np.array(losses[name]) - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[f"ablate_{name}"] = {"mean_loss": float(np.mean(losses[name])), "mean_delta": float(d.mean()),
                                 "delta_stderr": float(se), "frac_tokens_worse": float((d > 0).mean())}
    per_layer = [out[f"ablate_layer{l}"]["mean_delta"] for l in range(L)]
    out["most_important_layer"] = int(np.argmax(per_layer))
    out["sum_of_single_layer_deltas"] = float(sum(per_layer))
    cr = np.concatenate(contrib)  # [N, L]
    out["contrib_ratio_mean_per_layer"] = cr.mean(0).tolist()
    out["contrib_ratio_mean_per_layer_excl_pos0"] = np.concatenate([c[1:] for c in contrib]).mean(0).tolist()
    out["gate_mean_per_layer"] = np.concatenate(gate).mean(0).tolist()
    if base is not None:
        b = np.array(base_losses)
        out["baseline_mean_loss_same_windows"] = float(b.mean())
        out["baseline_minus_ple"] = float(b.mean() - ref.mean())
    return out


def table_findings(run: A.LoadedRun, freq: A.Frequencies) -> dict:
    rows = run.token_rows()
    limit = min(rows.shape[0], A.GPT2_VOCAB)
    full = np.linalg.norm(rows[:limit].reshape(limit, -1), axis=-1)
    per_layer = np.linalg.norm(rows[:limit], axis=-1)  # [V, L]
    out: dict = {"init_full_norm_expected": 0.02 * run.cfg.ple_dim * np.sqrt(run.cfg.n_layer),
                 "full_norm": {"mean": float(full.mean()), "median": float(np.median(full)),
                               "max": float(full.max())},
                 "layer_mean_norm": per_layer.mean(0).tolist()}
    counts = freq.counts()
    if counts is not None:
        c = counts[:limit].astype(np.float64)
        lf = np.log10(c + 1)
        seen = c > 0
        out["freq_source"] = str(freq.bin_path)
        out["n_tokens_seen_in_train"] = int(seen.sum())
        out["spearman_norm_vs_log_freq_all"] = spearman(full, lf)
        out["spearman_norm_vs_log_freq_seen_only"] = spearman(full[seen], lf[seen])
        out["spearman_norm_vs_log_freq_per_layer_seen_only"] = [spearman(per_layer[seen, l], lf[seen])
                                                               for l in range(run.cfg.n_layer)]
        out["mean_full_norm_unseen"] = float(full[~seen].mean()) if (~seen).any() else None
        out["mean_full_norm_seen"] = float(full[seen].mean())
        top = np.argsort(-c)[:1000]
        out["mean_full_norm_top1000"] = float(full[top].mean())
        sim = A.layer_similarity(run, None, freq, 1000)["global"]
        out["layer_cos_top1000"] = sim
    L = run.cfg.n_layer
    nn = {}
    for t in PROBE_TOKENS:
        tid = A.encode(t)
        if len(tid) != 1:
            continue
        tid = tid[0]
        nn[t] = {"id": tid, "count": int(counts[tid]) if counts is not None else None}
        for layer in (0, L - 1, -1):
            r = A.neighbors(run, tid, layer, 8)
            nn[t][f"layer{layer}" if layer >= 0 else "full_row"] = [(n["piece"], n["sim"]) for n in r["neighbors"]]
    out["neighbors"] = nn
    out["quant_int8"] = A.quant_error(run)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "ple-small"))
    ap.add_argument("--baseline", default=str(ROOT / "runs" / "baseline-small"))
    ap.add_argument("--val", default=str(ROOT / "data" / "tinystories" / "val.bin"))
    ap.add_argument("--train", default=str(ROOT / "data" / "tinystories" / "train.bin"))
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "explorer_findings.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    run_path = Path(args.run)
    mgr = A.RunManager(run_path.parent, "cpu")
    run = mgr.get(run_path.name)
    base = None
    bp = Path(args.baseline)
    if bp.exists() and (bp / "config.json").exists():
        base = A.RunManager(bp.parent, "cpu").get(bp.name)
        if base.cfg.ple_dim != 0:
            base = None
    freq = A.Frequencies(Path(args.train), run_path.parent / ".explorer_cache")
    meta = json.loads((run_path / "config.json").read_text())
    windows = val_windows(Path(args.val), run.cfg.block_size, args.chunks)
    res = {
        "run": run_path.name, "config": run.cfg.to_dict(), "val_loss_reported": meta.get("val_loss"),
        "baseline": base.name if base else None,
        "baseline_val_loss_reported": json.loads((bp / "config.json").read_text()).get("val_loss") if base else None,
        "val_source": args.val,
        "ablation": ablation_study(run, windows, base),
        "table": table_findings(run, freq),
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["ablation"], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
