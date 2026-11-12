"""Compare every run under runs/: final val loss, params, throughput.

Usage: python scripts/compare_runs.py [--runs runs]
Prints a markdown table and writes runs/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for cfg_path in sorted(runs_dir.glob("*/config.json")):
        meta = json.loads(cfg_path.read_text())
        pb = meta.get("param_breakdown") or {}
        c = meta.get("config", {})
        rows.append({
            "name": cfg_path.parent.name,
            "ple_dim": c.get("ple_dim"),
            "n_layer": c.get("n_layer"), "d_model": c.get("d_model"),
            "val_loss": meta.get("val_loss"),
            "train_loss": meta.get("train_loss"),
            "params_total": pb.get("total"), "params_ple_table": pb.get("ple_table"), "params_dense": pb.get("dense"),
            "tokens_seen": meta.get("tokens_seen"),
            "tokens_per_s": meta.get("tokens_per_s"),
            "elapsed_s": meta.get("elapsed_s"),
        })
    return rows


def _m(n):
    return f"{n / 1e6:.2f}M" if isinstance(n, (int, float)) else "-"


def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) else "-"


def to_markdown(rows: list[dict]) -> str:
    hdr = "| run | ple_dim | val loss | params total | PLE table | dense | tokens seen | tokens/s | time |\n"
    hdr += "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {r['name']} | {r['ple_dim']} | {_f(r['val_loss'], '.4f')} | {_m(r['params_total'])} | "
        f"{_m(r['params_ple_table'])} | {_m(r['params_dense'])} | {_m(r['tokens_seen'])} | "
        f"{_f(r['tokens_per_s'], ',.0f')} | {_f((r['elapsed_s'] or 0) / 60, '.1f')} min |\n"
        for r in rows
    )
    return hdr + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    runs = Path(a.runs)
    rows = collect(runs)
    print(to_markdown(rows))
    (runs / "summary.json").write_text(json.dumps({"runs": rows}, indent=2))
    print(f"wrote {runs / 'summary.json'}")


if __name__ == "__main__":
    main()
