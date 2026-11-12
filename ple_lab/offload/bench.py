"""Offload-engine benchmark.

  python -m ple_lab.offload.bench --run runs/ple-small        # real trained model (+ val-loss fp16 vs int8)
  python -m ple_lab.offload.bench --synthetic                 # builds runs/synthetic-big (~310 MB fp16 table)
  python -m ple_lab.offload.bench --gemma3n-table             # table-only, 262144 x 8960 (~4.7 GB) - opt-in

Workloads (token ids from data/tinystories/val.bin):
  prefill   full-model forward, B=1,  T=256
  batched   full-model forward, B=16, T=256
  decode    256 consecutive steps, full-model forward on a sliding window of the last W=min(128, block_size)
            tokens of a contiguous val.bin stream (teacher-forced; this GPT has no KV cache). Each step re-looks-up
            W rows, W-1 of which were seen in the previous step, so caches see realistic reuse.
  decode-1  the same 256-step stream, table-only: one new token id per step (what a KV-cached decoder fetches).
T is clamped to the model's block_size. Tokens/s counts *new* tokens (decode: 1 per step).
Prefetch variants are skipped for decode: the next token is unknown until it is sampled.
Prints a markdown table and writes results/offload/<run>-<timestamp>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..checkpoint import load_run, save_run
from ..model import GPT, GPTConfig
from .sources import Prefetcher, attach, make_source, MmapTableSource

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tinystories"
MB = 1024 * 1024

WORKLOADS = {  # name: (B, T, warmup iters, measured iters)
    "prefill": (1, 256, 10, 60),
    "batched": (16, 256, 3, 20),
    "decode": (1, 128, 32, 256),
    "decode-1": (1, 1, 32, 256),
}
SLIDING = {"decode", "decode-1"}
TABLE_ONLY = {"decode-1"}

# (label, make_source kind, kwargs). Cache backing = mmap-f16 (the "flash" store).
SOURCES = [
    ("gpu", "gpu", {}),
    ("cpu", "cpu", {}),
    ("cpu-pinned", "cpu-pinned", {}),
    ("mmap-f16", "mmap-f16", {}),
    ("mmap-i8", "mmap-i8", {}),
    ("lru-1%", "cached-lru", {"capacity_frac": 0.01}),
    ("lru-5%", "cached-lru", {"capacity_frac": 0.05}),
    ("lru-20%", "cached-lru", {"capacity_frac": 0.20}),
    ("topk-5%", "static-topk", {"capacity_frac": 0.05}),
    ("mmap-f16+pf", "mmap-f16", {"prefetch": True}),
    ("lru-5%+pf", "cached-lru", {"capacity_frac": 0.05, "prefetch": True}),
]


def token_stream() -> np.ndarray:
    return np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")


def train_freq(vocab: int) -> np.ndarray:
    """Token counts from train.bin (never val, to avoid leaking the benchmark stream into static-topk)."""
    path = DATA / "train.bin" if (DATA / "train.bin").exists() else DATA / "val.bin"
    return np.bincount(np.memmap(path, dtype=np.uint16, mode="r"), minlength=vocab)[:vocab]


def make_batches(stream, B, T, n, offset, vocab, sliding=False, pin=False):
    """n [B,T] id tensors. Non-sliding: consecutive disjoint windows (rows of a batch 100k tokens apart).
    Sliding: window i = tokens [offset+i, offset+i+T) of one contiguous stream (one new token per step)."""
    out = []
    step = 1 if sliding else T
    for i in range(n):
        idx = (offset + np.arange(B)[:, None] * 100_003 + i * step + np.arange(T)[None, :]) % (len(stream) - 1)
        t = torch.from_numpy(stream[idx].astype(np.int64) % vocab)
        out.append(t.pin_memory() if pin else t)
    return out


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def bench_one(fn, src, batches, warm, dev, prefetch, new_tokens_per_iter=None):
    """fn(ids_cpu) runs one iteration. With prefetch, ids for i+1 are queued before running i."""
    def loop(bs, lat):
        if prefetch:
            src.prefetch(bs[0])
        for i, ids in enumerate(bs):
            t0 = time.perf_counter()
            if prefetch and i + 1 < len(bs):
                src.prefetch(bs[i + 1])
            fn(ids)
            sync(dev)
            if lat is not None:
                lat.append(time.perf_counter() - t0)

    loop(warm, None)
    if prefetch:
        src.drain()
    sync(dev)
    src.reset_stats()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    lat = []
    t0 = time.perf_counter()
    loop(batches, lat)
    total = time.perf_counter() - t0
    lat = np.array(lat) * 1e3
    toks = sum(b.numel() for b in batches) if new_tokens_per_iter is None else new_tokens_per_iter * len(batches)
    s = src.stats()
    return {
        "tokens_per_s": toks / total, "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "table_vram_bytes": int(s["resident_vram_bytes"]),
        "peak_cuda_alloc_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0,
        "hit_rate": s.get("hit_rate"), "h2d_bytes_per_token": s["h2d_bytes_per_token"],
        "h2d_index_bytes_per_token": s["h2d_index_bytes"] / max(s["tokens"], 1),
        "backing_bytes_per_token": s["backing_bytes_per_token"],
        "backing_page_bytes_per_token": s["backing_page_bytes_per_token"],
        "unique_rows_per_call": s["unique_rows"] / max(s["calls"], 1), "tokens": toks,
        "rows_looked_up": s["tokens"], "stats": s,
    }


def free(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def run_benchmarks(run_dir, dev, sources, workloads, table_only=False, model=None, amp=True):
    stream = token_stream()
    meta = json.loads((Path(run_dir) / "ple_table.json").read_text())
    V = meta["vocab"]
    freq = train_freq(V)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (dev.type == "cuda" and amp) else nullcontext()
    results = []
    block = model.cfg.block_size if model is not None else 10**9
    for wname in workloads:
        B, T, nw, nm = WORKLOADS[wname]
        T = min(T, block)
        sliding = wname in SLIDING
        tonly = table_only or wname in TABLE_ONLY
        warm = make_batches(stream, B, T, nw, 0, V, sliding, dev.type == "cuda")
        meas = make_batches(stream, B, T, nm, 1_000_000, V, sliding, dev.type == "cuda")
        for label, kind, kw in sources:
            if kw.get("prefetch") and sliding:
                continue  # next decode token is unknown before sampling; prefetch would be cheating
            kw = dict(kw)
            if kind == "static-topk":
                kw["freq"] = freq
            if kw.get("prefetch"):
                kw["verify"] = False  # the bench feeds batches in exactly the prefetched order
            free(dev)
            src = make_source(run_dir, kind, dev, **kw)
            if tonly:
                fn = lambda ids, src=src: src(ids.to(dev, non_blocking=True))
            else:
                attach(model, src)
                def fn(ids):
                    with ctx:
                        model(ids.to(dev, non_blocking=True))
            r = bench_one(fn, src, meas, warm, dev, isinstance(src, Prefetcher), B if sliding else None)
            if isinstance(src, Prefetcher):
                src.close()
            r.update(workload=wname, source=label, kind=kind, B=B, T=T, table_only=tonly)
            results.append(r)
            print(f"  {wname:8s} {label:12s} {r['tokens_per_s']:>11.0f} tok/s  p50 {r['p50_ms']:.3f} ms"
                  + (f"  hit {r['hit_rate']:.3f}" if r["hit_rate"] is not None else ""), flush=True)
            if model is not None:
                model.ple_table = torch.nn.Identity()
            del src
    free(dev)
    return results


def markdown(results):
    base = {r["workload"]: r["tokens_per_s"] for r in results if r["source"] == "gpu"}
    hdr = ("| workload | source | tok/s | vs gpu | p50 ms | p95 ms | table VRAM MB | peak alloc MB | hit rate "
           "| H2D B/tok | backing B/tok |")
    lines = [hdr, "|" + "---|" * 11]
    for r in results:
        hr = "-" if r["hit_rate"] is None else f"{100 * r['hit_rate']:.1f}%"
        rel = f"{100 * (r['tokens_per_s'] / base[r['workload']] - 1):+.0f}%" if r["workload"] in base else "-"
        lines.append(
            f"| {r['workload']} | {r['source']} | {r['tokens_per_s']:,.0f} | {rel} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} "
            f"| {r['table_vram_bytes'] / MB:.2f} | {r['peak_cuda_alloc_bytes'] / MB:.1f} | {hr} "
            f"| {r['h2d_bytes_per_token']:,.0f} | {r['backing_bytes_per_token']:,.0f} |")
    return "\n".join(lines)


@torch.no_grad()
def val_loss(model, dev, batches=50, bs=16, seed=1234):
    """Mean val loss over fixed random windows of val.bin, fp32 (no autocast) so only the table differs."""
    data = token_stream()
    T = model.cfg.block_size
    gen = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        ix = gen.integers(0, len(data) - T - 1, size=bs)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)).to(dev)
        losses.append(model(x, y)[1].item())
    return float(np.mean(losses))


def quality(run_dir, dev, batches):
    model = load_run(run_dir, dev, load_table=False)
    out = {}
    for kind in ("gpu", "mmap-i8"):
        attach(model, make_source(run_dir, kind, dev))
        out[f"val_loss_{kind}"] = val_loss(model, dev, batches)
    model.ple_table = torch.nn.Identity()
    out["delta_i8_minus_f16"] = out["val_loss_mmap-i8"] - out["val_loss_gpu"]
    out["batches"], out["batch_size"], out["block_size"] = batches, 16, model.cfg.block_size
    return out


def cost_model(cfg: GPTConfig, dense_params: int) -> dict:
    row = cfg.n_layer * cfg.ple_dim
    return {"ple_row_elems": row, "ple_bytes_per_token_f16": row * 2, "ple_bytes_per_token_i8": row + 4,
            "table_params": cfg.ple_vocab * row, "table_bytes_f16": cfg.ple_vocab * row * 2,
            "dense_params": dense_params, "dense_bytes_per_decode_token_f16": dense_params * 2,
            "ratio_dense_over_ple_f16": dense_params * 2 / max(row * 2, 1)}


def build_synthetic(out: Path) -> Path:
    if (out / "ple_table.json").exists():
        return out
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50304, block_size=256, n_layer=12, n_head=4, d_model=256, ple_dim=256)
    m = GPT(cfg).eval()
    print(f"building {out} params={m.param_breakdown()}", flush=True)
    save_run(m, out, {"note": "synthetic random-init model for offload benchmarks", "tokenizer": "gpt2"})
    return out


def gemma3n_table(out: Path, vocab: int, width: int) -> Path:
    """Random fp16 table of Gemma-3n-E2B-like shape, written in chunks; refuses if disk/RAM are tight."""
    nbytes = vocab * width * 2
    meta_path = out / "ple_table.json"
    if not (meta_path.exists() and json.loads(meta_path.read_text())["row_width"] == width
            and json.loads(meta_path.read_text())["vocab"] == vocab):
        out.mkdir(parents=True, exist_ok=True)
        free_disk = shutil.disk_usage(out).free
        if free_disk < nbytes * 1.5 + 2 * 1024**3:
            raise SystemExit(f"need {nbytes * 1.5 / 1e9:.1f} GB + 2 GB headroom free disk, have {free_disk / 1e9:.1f} GB")
        print(f"writing {nbytes / 1e9:.2f} GB table to {out} ...", flush=True)
        rng = np.random.default_rng(0)
        with open(out / "ple_table.f16.bin", "wb") as f:
            for s in range(0, vocab, 4096):
                n = min(4096, vocab - s)
                f.write((rng.standard_normal((n, width), dtype=np.float32) * 0.02).astype(np.float16).tobytes())
        meta_path.write_text(json.dumps({"vocab": vocab, "n_layer": 35, "ple_dim": width // 35, "row_width": width,
                                         "files": {"f16": "ple_table.f16.bin"}}))
    return out


def avail_ram() -> int | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("tot", "avail", "tpf", "apf", "tv", "av", "ave")]
        ms = MS(); ms.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return int(ms.avail)
    except Exception:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory with a PLE table")
    ap.add_argument("--synthetic", action="store_true", help="build/use runs/synthetic-big")
    ap.add_argument("--gemma3n-table", action="store_true", help="table-only gathers on a 262144x8960 fp16 table")
    ap.add_argument("--gemma3n-vocab", type=int, default=262144, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=os.environ.get("PLE_DEVICE", "cpu"),
                    help="cpu (default, safe while a GPU job runs) or cuda for the real offload benchmark")
    ap.add_argument("--sources", default=None, help="comma list of source labels (default: all)")
    ap.add_argument("--workloads", default="prefill,decode,decode-1,batched")
    ap.add_argument("--val-batches", type=int, default=50, help="0 disables the val-loss quality check")
    ap.add_argument("--out", default=str(ROOT / "results" / "offload"))
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    if dev.type != "cuda":
        # hide the GPU before anything can lazily initialise a CUDA context (pinned memory, streams, ...)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        assert not torch.cuda.is_initialized() and not torch.cuda.is_available(), "CUDA must stay unused with --device cpu"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    sources = SOURCES if not a.sources else [s for s in SOURCES if s[0] in a.sources.split(",")]
    workloads = a.workloads.split(",")
    t_start = time.time()
    report = {"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
              "torch": torch.__version__, "workloads": {k: dict(zip(("B", "T", "warmup", "iters"), v)) for k, v in WORKLOADS.items()},
              "decode_note": "decode = full-model forward on a sliding window of the last min(128, block) tokens, 1 new token/step "
                             "(no KV cache); decode-1 = table-only lookup of the 1 new token per step",
              "cache_backing": "mmap-f16", "static_topk_freq": "train.bin token counts"}

    if a.gemma3n_table:
        width = 8960
        need = a.gemma3n_vocab * width * 2
        ram = avail_ram()
        print(f"gemma3n table: {need / 1e9:.2f} GB fp16; available RAM {ram / 1e9 if ram else float('nan'):.1f} GB")
        run_dir = gemma3n_table(ROOT / "runs" / "synthetic-gemma3n-table", a.gemma3n_vocab, width)
        name = "gemma3n-table"
        # table-only: no model of this size exists here. cpu sources would need the full table in RAM.
        if ram is None or ram < need * 1.5:
            sources = [s for s in sources if s[1] not in ("cpu", "cpu-pinned")]
            print("  skipping cpu/cpu-pinned sources (not enough free RAM to hold the table)")
        sources = [s for s in sources if s[1] != "mmap-i8"]
        if dev.type == "cuda" and need > torch.cuda.mem_get_info(dev)[0] * 0.8:
            sources = [s for s in sources if s[1] != "gpu"]
            print("  skipping gpu source (table does not fit in free VRAM)")
        report.update(run=str(run_dir), mode="table-only", table_bytes_f16=need)
        report["results"] = run_benchmarks(run_dir, dev, sources, workloads, table_only=True)
    else:
        if a.synthetic:
            run_dir = build_synthetic(ROOT / "runs" / "synthetic-big")
        elif a.run:
            run_dir = Path(a.run)
        else:
            ap.error("give --run DIR, --synthetic or --gemma3n-table")
        name = run_dir.name
        model = load_run(run_dir, dev, load_table=False)
        dense = sum(p.numel() for p in model.parameters())
        report.update(run=str(run_dir), mode="full-model", config=model.cfg.to_dict(), cost_model=cost_model(model.cfg, dense),
                      run_meta={k: v for k, v in json.loads((run_dir / "config.json").read_text()).items() if k != "config"})
        print(f"[{name}] cost model: {json.dumps(report['cost_model'])}", flush=True)
        report["results"] = run_benchmarks(run_dir, dev, sources, workloads, model=model)
        del model
        free(dev)
        if a.val_batches and not a.synthetic:
            q = quality(run_dir, dev, a.val_batches)
            report["quality"] = q
            print(f"[{name}] val loss gpu-f16 {q['val_loss_gpu']:.4f}  mmap-i8 {q['val_loss_mmap-i8']:.4f}  "
                  f"delta {q['delta_i8_minus_f16']:+.5f}", flush=True)

    report["elapsed_s"] = time.time() - t_start
    for r in report["results"]:
        r.pop("stats", None) if False else None
    md = markdown(report["results"])
    report["markdown"] = md
    print()
    print(md)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {path}  ({report['elapsed_s']:.0f}s)")
    return report


if __name__ == "__main__":
    main()
