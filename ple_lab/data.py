"""TinyStories download + gpt2 tokenisation into uint16 memmap-able .bin files.

Usage: python -m ple_lab.data [--out data/tinystories] [--train-mb 300]
Idempotent: skips download/tokenisation steps whose outputs already exist.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStories-train.txt", "val": "TinyStories-valid.txt"}
EOT = 50256
SEP = "<|endoftext|>"


def download(url: str, dest: Path, max_bytes: int | None = None) -> Path:
    """Stream `url` into `dest`; if max_bytes is set, stop after that many bytes (cut at a story boundary later)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ple-lab"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    got, t0 = 0, time.time()
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            if max_bytes and got + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - got]
            f.write(chunk)
            got += len(chunk)
            if max_bytes and got >= max_bytes:
                break
    tmp.replace(dest)
    print(f"downloaded {dest.name}: {got / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    return dest


def split_stories(text: str, truncated: bool) -> list[str]:
    stories = [s.strip() for s in text.split(SEP)]
    if truncated and len(stories) > 1:
        stories = stories[:-1]  # last one is likely cut mid-story
    return [s for s in stories if s]


def tokenize(stories: list[str], batch: int = 20000) -> np.ndarray:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    parts = []
    for i in range(0, len(stories), batch):
        for ids in enc.encode_ordinary_batch(stories[i : i + batch], num_threads=8):
            ids.append(EOT)
            parts.append(np.asarray(ids, dtype=np.uint16))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def prepare(out: str | Path = "data/tinystories", train_mb: int = 300) -> dict:
    out = Path(out)
    raw = out / "raw"
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for split, fname in FILES.items():
        bin_path = out / f"{split}.bin"
        if bin_path.exists() and split in meta:
            print(f"{split}.bin exists ({meta[split]['tokens']:,} tokens), skipping")
            continue
        max_bytes = train_mb * 1_000_000 if split == "train" and train_mb > 0 else None
        txt = download(BASE + fname, raw / fname, max_bytes)
        t0 = time.time()
        text = txt.read_text(encoding="utf-8", errors="ignore")
        stories = split_stories(text, truncated=max_bytes is not None)
        ids = tokenize(stories)
        ids.tofile(bin_path)
        meta[split] = {"tokens": int(ids.size), "stories": len(stories), "source_bytes": txt.stat().st_size}
        print(f"{split}: {len(stories):,} stories -> {ids.size:,} tokens in {time.time() - t0:.0f}s")
        meta.update({"tokenizer": "gpt2", "eot": EOT, "dtype": "uint16", "dataset": "tinystories", "source": BASE})
        meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--train-mb", type=int, default=300, help="MB of train text to fetch (0 = full ~1.9 GB file)")
    a = ap.parse_args()
    print(json.dumps(prepare(a.out, a.train_mb), indent=2))


if __name__ == "__main__":
    main()
