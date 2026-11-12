"""Numerics for the PLE Explorer (kept free of any web framework code).

Conventions:
- "token part" = raw table row reshaped to [L, p] and multiplied by sqrt(p), exactly what the
  model feeds into u (before the /sqrt(2) mix with the context part).
- Per-token loss at position i is -log p(token_i | tokens_<i); position 0 has no loss (None).
- top-5 predictions at position i are the model's guesses for token i+1.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ple_lab import GPT, GPTConfig, load_run, load_table_memmap

GPT2_VOCAB = 50257
MAX_VEC_DIM = 128


class ExplorerError(ValueError):
    """User-facing error (mapped to HTTP 400 by the server)."""


class RunNotFound(ExplorerError):
    pass


# ---------------------------------------------------------------- tokenizer

_ENC: tiktoken.Encoding | None = None


def enc() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


def piece(tid: int) -> str:
    if 0 <= tid < GPT2_VOCAB:
        return enc().decode_single_token_bytes(tid).decode("utf-8", errors="replace")
    return f"<pad{tid}>"


def encode(text: str) -> list[int]:
    return enc().encode(text, allowed_special={"<|endoftext|>"})


# ---------------------------------------------------------------- token frequency

class Frequencies:
    """Token counts from a uint16 .bin file.

    Cached in memory and on disk as `<cache_dir>/freq_<pathtag>_<size>_<mtime_ns>.npy`, where pathtag
    identifies the source file. Any change of size or mtime (ns resolution) invalidates the cache;
    stale cache files for the same source are deleted. Writes are atomic (tmp file + rename)."""

    def __init__(self, bin_path: Path | None, cache_dir: Path):
        self.bin_path = Path(bin_path) if bin_path else None
        self.cache_dir = cache_dir
        self._counts: np.ndarray | None = None
        self._key: str | None = None
        self._lock = threading.Lock()

    def _tag(self) -> str:
        return hashlib.sha1(str(self.bin_path.resolve()).encode()).hexdigest()[:10]

    def key(self) -> str | None:
        if self.bin_path is None or not self.bin_path.is_file():
            return None
        st = self.bin_path.stat()
        if st.st_size < 2:
            return None
        return f"{self._tag()}_{st.st_size}_{st.st_mtime_ns}"

    def _compute(self) -> np.ndarray:
        n = self.bin_path.stat().st_size // 2  # ignore a trailing odd byte (file being written)
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r", shape=(n,))
        counts = np.zeros(65536, dtype=np.int64)
        step = 50_000_000
        for i in range(0, n, step):
            counts += np.bincount(data[i:i + step], minlength=65536)
        del data
        return counts

    def counts(self) -> np.ndarray | None:
        key = self.key()
        if key is None:
            return None
        with self._lock:
            if self._key == key and self._counts is not None:
                return self._counts
            cache = self.cache_dir / f"freq_{key}.npy"
            counts = None
            if cache.exists():
                try:
                    counts = np.load(cache)
                    if counts.shape != (65536,):
                        counts = None
                except (OSError, ValueError):
                    counts = None
            if counts is None:
                counts = self._compute()
                if self.key() != key:  # file changed while counting: serve, but don't persist
                    return counts
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                with open(tmp, "wb") as f:
                    np.save(f, counts)
                os.replace(tmp, cache)
                for old in self.cache_dir.glob(f"freq_{self._tag()}_*.npy"):
                    if old != cache:
                        old.unlink(missing_ok=True)
            self._counts, self._key = counts, key
            return counts

    def top_ids(self, n: int, vocab: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Most frequent ids (< vocab and < GPT2_VOCAB). Falls back to 0..n-1 without data."""
        limit = min(vocab, GPT2_VOCAB)
        c = self.counts()
        if c is None:
            return np.arange(min(n, limit)), None
        c = c[:limit]
        order = np.argsort(-c, kind="stable")[:n]
        return order, c[order]


# ---------------------------------------------------------------- runs

def discover_runs(runs_dir: Path) -> dict[str, Path]:
    if not runs_dir.exists():
        return {}
    return {p.name: p for p in sorted(runs_dir.iterdir()) if p.is_dir() and (p / "config.json").exists()}


def run_summary(name: str, path: Path) -> dict:
    meta = json.loads((path / "config.json").read_text())
    cfg = GPTConfig(**meta["config"])
    with torch.device("meta"):
        params = GPT(cfg).param_breakdown()
    extra = {k: v for k, v in meta.items() if k != "config"}
    return {
        "name": name, "config": cfg.to_dict(), "has_ple": cfg.ple_dim > 0,
        "val_loss": meta.get("val_loss"), "params": params,
        "has_i8": (path / "ple_table.i8.bin").exists(),
        "extra": {k: v for k, v in extra.items() if isinstance(v, (int, float, str, bool)) or v is None},
    }


@dataclass
class LoadedRun:
    name: str
    path: Path
    model: GPT
    device: str
    stamp: float
    rows: np.ndarray | None = None  # token part [V, L, p] float32
    caches: dict = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def cfg(self) -> GPTConfig:
        return self.model.cfg

    def require_ple(self) -> None:
        if self.cfg.ple_dim <= 0:
            raise ExplorerError(f"run '{self.name}' is a baseline (ple_dim=0): it has no PLE table")

    def token_rows(self) -> np.ndarray:
        self.require_ple()
        with self.lock:
            if self.rows is None:
                w = self.model.ple_table.table.weight.detach().float().cpu().numpy()
                L, p = self.cfg.n_layer, self.cfg.ple_dim
                self.rows = w.reshape(w.shape[0], L, p) * np.float32(math.sqrt(p))
            return self.rows

    @property
    def vocab(self) -> int:
        return self.token_rows().shape[0]

    def check_id(self, tid: int) -> None:
        if not 0 <= tid < self.vocab:
            raise ExplorerError(f"token id {tid} out of range [0, {self.vocab})")

    def check_layer(self, layer: int, allow_full: bool = True) -> None:
        lo = -1 if allow_full else 0
        if not lo <= layer < self.cfg.n_layer:
            raise ExplorerError(f"layer {layer} out of range [{lo}, {self.cfg.n_layer})")

    def normalized(self, layer: int) -> torch.Tensor:
        """Unit-normalised slice matrix [V, p] (layer=-1: full row) on the run's device, cached."""
        key = ("norm", layer)
        with self.lock:
            if key not in self.caches:
                rows = self.token_rows()
                x = rows.reshape(rows.shape[0], -1) if layer < 0 else rows[:, layer]
                t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
                self.caches[key] = F.normalize(t, dim=-1)
            return self.caches[key]


class RunManager:
    """Discovers runs and loads them lazily (reloading when the checkpoint changes)."""

    def __init__(self, runs_dir: Path, device: str = "cpu"):
        self.runs_dir = runs_dir
        self.device = device
        self._loaded: dict[str, LoadedRun] = {}
        self._lock = threading.Lock()

    def available(self) -> dict[str, Path]:
        return discover_runs(self.runs_dir)

    def get(self, name: str | None) -> LoadedRun:
        runs = self.available()
        if not name or name not in runs:
            raise RunNotFound(f"unknown run '{name}'. Available: {sorted(runs)}")
        path = runs[name]
        files = ("config.json", "dense.pt", "ple_table.f16.bin")
        stamp = max((path / f).stat().st_mtime_ns for f in files if (path / f).exists())
        with self._lock:
            cur = self._loaded.get(name)
            if cur is None or cur.stamp != stamp:
                try:
                    model = load_run(path, device=self.device)
                except Exception as e:  # partially written checkpoint, bad config, ...
                    raise ExplorerError(f"run '{name}' could not be loaded (incomplete or still being written?): {e}") from e
                cur = LoadedRun(name=name, path=path, model=model, device=self.device, stamp=stamp)
                self._loaded[name] = cur
            return cur


# ---------------------------------------------------------------- table views

def _r(a: np.ndarray, nd: int) -> list:
    """Round and convert to nested Python floats (via float64 so JSON stays short)."""
    return np.asarray(a, dtype=np.float64).round(nd).tolist()


def _downsample(v: np.ndarray, max_dim: int = MAX_VEC_DIM) -> np.ndarray:
    """Average-pool the last axis down to <= max_dim."""
    p = v.shape[-1]
    if p <= max_dim:
        return v
    f = math.ceil(p / max_dim)
    pad = f * math.ceil(p / f) - p
    v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, pad)])
    return v.reshape(*v.shape[:-1], -1, f).mean(-1)


def resolve_token(run: LoadedRun, q: str | None, tid: int | None) -> tuple[int, list[dict]]:
    """Returns (id, pieces of q). q may be text or '#123'; text resolves to its first token."""
    if tid is None:
        if not q:
            raise ExplorerError("provide a non-empty q (text or '#id') or id")
        if q.startswith("#") and q[1:].isdigit():
            tid = int(q[1:])
        else:
            ids = encode(q)
            if not ids:
                raise ExplorerError("query tokenized to nothing")
            run.check_id(ids[0])
            return ids[0], [{"id": i, "piece": piece(i)} for i in ids]
    run.check_id(tid)
    return tid, [{"id": tid, "piece": piece(tid)}]


def token_info(run: LoadedRun, q: str | None, tid: int | None, freq: Frequencies) -> dict:
    tid, pieces = resolve_token(run, q, tid)
    v = run.token_rows()[tid]  # [L, p]
    counts = freq.counts()
    return {
        "id": tid, "piece": piece(tid), "pieces": pieces,
        "norms": _r(np.linalg.norm(v, axis=-1), 5),
        "vectors": _r(_downsample(v), 4),
        "ple_dim": run.cfg.ple_dim, "n_layer": run.cfg.n_layer,
        "downsampled": v.shape[-1] > MAX_VEC_DIM,
        "init_norm": 0.02 * run.cfg.ple_dim,  # E||row_l * sqrt(p)|| at init: 0.02*sqrt(p)*sqrt(p)
        "frequency": int(counts[tid]) if counts is not None and tid < len(counts) else None,
    }


def neighbors(run: LoadedRun, tid: int, layer: int, k: int) -> dict:
    run.check_id(tid)
    run.check_layer(layer)
    k = max(1, min(int(k), 200))
    m = run.normalized(layer)
    limit = min(m.shape[0], GPT2_VOCAB)
    sims = m[:limit] @ m[tid]
    if tid < limit:
        sims[tid] = -2.0
    val, idx = torch.topk(sims, min(k, limit - 1))
    return {
        "id": tid, "piece": piece(tid), "layer": layer,
        "neighbors": [{"id": int(i), "piece": piece(int(i)), "sim": round(float(s), 4)}
                      for s, i in zip(val.cpu(), idx.cpu())],
    }


def _layer_cos(v: np.ndarray) -> np.ndarray:
    """v: [..., L, p] -> [..., L, L] cosine matrix."""
    n = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return n @ np.swapaxes(n, -1, -2)


def layer_similarity(run: LoadedRun, tid: int | None, freq: Frequencies, top_n: int = 1000) -> dict:
    rows = run.token_rows()
    out: dict = {"n_layer": run.cfg.n_layer}
    if tid is not None:
        run.check_id(tid)
        out.update(id=tid, piece=piece(tid), matrix=_r(_layer_cos(rows[tid]), 4))
    top_n = max(1, min(int(top_n), 20_000))
    key = ("global_sim", top_n, freq.key())
    with run.lock:
        if key not in run.caches:
            ids, _ = freq.top_ids(top_n, rows.shape[0])
            run.caches[key] = _r(_layer_cos(rows[ids]).mean(0), 4)
    out["global"] = run.caches[key]
    out["global_n"] = top_n
    out["global_source"] = "frequency" if freq.key() else "ids 0..N (no train.bin)"
    return out


def projection(run: LoadedRun, layer: int, n: int, freq: Frequencies) -> dict:
    run.check_layer(layer)
    n = max(10, min(int(n), 10_000))
    key = ("proj", layer, n, freq.key())
    with run.lock:
        if key in run.caches:
            return run.caches[key]
    rows = run.token_rows()
    ids, counts = freq.top_ids(n, rows.shape[0])
    x = rows[ids].reshape(len(ids), -1) if layer < 0 else rows[ids, layer]
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(x)
    res = {
        "layer": layer, "n": int(len(ids)),
        "source": "frequency" if counts is not None else "ids 0..n (no train.bin)",
        "explained_variance": [round(float(e), 4) for e in pca.explained_variance_ratio_],
        "points": [
            {"id": int(i), "piece": piece(int(i)), "x": round(float(a), 4), "y": round(float(b), 4),
             "norm": round(float(nm), 4), "freq": int(counts[j]) if counts is not None else None}
            for j, (i, (a, b), nm) in enumerate(zip(ids, xy, np.linalg.norm(x, axis=-1)))
        ],
    }
    with run.lock:
        run.caches[key] = res
    return res


PROJ3D_MAX_N = 5000
PROJ3D_K = 12


def _proj3d_model(run: LoadedRun, n: int, space: str, freq: Frequencies) -> dict:
    """Fitted 3-D PCA(s) + k-means for the top-n tokens, cached on the run.

    space='shared': one PCA fitted on all layers' slices stacked ([n*L, p]), so every layer lives in the
    same coordinates and positions can be morphed between layers. space='per_layer': one PCA per layer.
    The full concatenated row always gets its own PCA. Coordinates are divided by one scale per space
    (the max |coord| over the shared layers, or per layer), so they lie in [-1, 1]."""
    if space not in ("shared", "per_layer"):
        raise ExplorerError(f"space must be 'shared' or 'per_layer', got '{space}'")
    key = ("proj3d_model", n, space, freq.key())
    with run.lock:
        if key in run.caches:
            return run.caches[key]
    from sklearn.cluster import KMeans

    rows = run.token_rows()
    L, p = run.cfg.n_layer, run.cfg.ple_dim
    ids, counts = freq.top_ids(n, rows.shape[0])
    x = rows[ids].astype(np.float32)  # [n, L, p]
    full = x.reshape(len(ids), -1)
    ncomp = min(3, p, len(ids))

    def fit(a: np.ndarray) -> PCA:
        return PCA(n_components=ncomp, random_state=0).fit(a)

    def pad3(a: np.ndarray) -> np.ndarray:
        return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, 3 - a.shape[-1])])

    if space == "shared":
        pca = fit(x.reshape(-1, p))
        coords = pad3(pca.transform(x.reshape(-1, p)).reshape(len(ids), L, ncomp))
        scale = float(np.abs(coords).max()) or 1.0
        pcas, scales = [pca] * L, [scale] * L
        ev = [_r(pca.explained_variance_ratio_, 4)] * L
    else:
        pcas = [fit(x[:, l]) for l in range(L)]
        coords = np.stack([pad3(pcas[l].transform(x[:, l])) for l in range(L)], 1)
        scales = [float(np.abs(coords[:, l]).max()) or 1.0 for l in range(L)]
        ev = [_r(pc.explained_variance_ratio_, 4) for pc in pcas]
    coords = coords / np.array(scales, dtype=np.float32)[None, :, None]
    pca_full = fit(full)
    cf = pad3(pca_full.transform(full))
    scale_full = float(np.abs(cf).max()) or 1.0
    cf = cf / scale_full

    unit = full / np.maximum(np.linalg.norm(full, axis=-1, keepdims=True), 1e-12)
    k = max(1, min(PROJ3D_K, len(ids)))
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(unit)
    sub = torch.from_numpy(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))  # [n, L, p]
    sub_full = torch.from_numpy(unit)
    res = {
        "ids": ids, "counts": counts, "coords": coords.astype(np.float32), "full": cf.astype(np.float32),
        "norms": np.linalg.norm(x, axis=-1), "full_norm": np.linalg.norm(full, axis=-1),
        "labels": labels, "k": k, "pcas": pcas, "scales": scales, "pca_full": pca_full,
        "scale_full": scale_full, "ev": ev, "ev_full": _r(pca_full.explained_variance_ratio_, 4),
        "unit": sub, "unit_full": sub_full, "index": {int(t): j for j, t in enumerate(ids)},
    }
    with run.lock:
        run.caches[key] = res
    return res


def projection3d(run: LoadedRun, n: int, space: str, freq: Frequencies) -> dict:
    """3-D map of the top-n tokens in every layer (+ the full row). See docs/EXPLORER.md#3d-map."""
    run.require_ple()
    n = max(10, min(int(n), PROJ3D_MAX_N))
    key = ("proj3d", n, space, freq.key())
    with run.lock:
        if key in run.caches:
            return run.caches[key]
    m = _proj3d_model(run, n, space, freq)
    ids, counts, labels = m["ids"], m["counts"], m["labels"]
    logf = np.log10(counts.astype(np.float64) + 1) if counts is not None else np.zeros(len(ids))
    # float32 math, rounded in float64 so the JSON floats stay short (float32 -> float repr is long)
    coords = np.asarray(m["coords"], dtype=np.float64).round(4)
    full = np.asarray(m["full"], dtype=np.float64).round(4)
    norms = np.asarray(m["norms"], dtype=np.float32)
    clusters = []
    for c in range(m["k"]):
        members = np.nonzero(labels == c)[0]  # ids are sorted by frequency, so members[:8] are the top ones
        clusters.append({"id": c, "size": int(len(members)),
                         "top": [piece(int(ids[j])) for j in members[:8]],
                         "centroid": _r(full[members].mean(0), 4) if len(members) else [0, 0, 0]})
    points = [
        {"id": int(t), "piece": piece(int(t)), "rank": j,
         "freq": int(counts[j]) if counts is not None else None, "log_freq": round(float(logf[j]), 4),
         "coords": coords[j].tolist(), "full": full[j].tolist(),
         "norms": np.round(norms[j].astype(np.float64), 4).tolist(),
         "full_norm": round(float(m["full_norm"][j]), 4),
         "max_layer": int(norms[j].argmax()), "cluster": int(labels[j])}
        for j, t in enumerate(ids)
    ]
    res = {
        "n": int(len(ids)), "space": space, "n_layer": run.cfg.n_layer, "ple_dim": run.cfg.ple_dim,
        "source": "frequency" if counts is not None else "ids 0..n (no train.bin)",
        "explained_variance": m["ev"], "explained_variance_full": m["ev_full"],
        "k": m["k"], "clusters": clusters,
        "init_norm": 0.02 * run.cfg.ple_dim,
        "points": points,
    }
    with run.lock:
        run.caches[key] = res
    return res


def trajectory(run: LoadedRun, tid: int, n: int, space: str, freq: Frequencies, k: int = 8) -> dict:
    """Path of one token through the 3-D map across layers, plus its k nearest neighbours (cosine)
    among the map's top-n tokens in every layer and in the full row. Works for tokens outside the map."""
    run.require_ple()
    run.check_id(tid)
    n = max(10, min(int(n), PROJ3D_MAX_N))
    k = max(1, min(int(k), 32))
    m = _proj3d_model(run, n, space, freq)
    rows = run.token_rows()
    L = run.cfg.n_layer
    v = rows[tid].astype(np.float32)  # [L, p]
    coords = []
    for l in range(L):
        c = m["pcas"][l].transform(v[l:l + 1])[0]
        coords.append(np.pad(c, (0, 3 - len(c))) / m["scales"][l])
    cf = m["pca_full"].transform(v.reshape(1, -1))[0]
    cf = np.pad(cf, (0, 3 - len(cf))) / m["scale_full"]
    ids = m["ids"]
    self_idx = m["index"].get(tid)

    def nn(unit: torch.Tensor, q: np.ndarray) -> list[dict]:
        qn = torch.from_numpy(q / max(float(np.linalg.norm(q)), 1e-12))
        sims = unit @ qn
        if self_idx is not None:
            sims[self_idx] = -2.0
        val, idx = torch.topk(sims, min(k, len(ids) - (self_idx is not None)))
        return [{"id": int(ids[j]), "piece": piece(int(ids[j])), "index": int(j), "sim": round(float(s), 4)}
                for s, j in zip(val, idx)]

    layers = [{"layer": l, "coords": _r(coords[l], 4), "norm": round(float(np.linalg.norm(v[l])), 4),
               "neighbors": nn(m["unit"][:, l], v[l])} for l in range(L)]
    full = {"coords": _r(cf, 4), "norm": round(float(np.linalg.norm(v)), 4),
            "neighbors": nn(m["unit_full"], v.reshape(-1))}
    sets = [{x["id"] for x in lay["neighbors"]} for lay in layers]
    overlap = [round(len(sets[l] & sets[l + 1]) / k, 4) for l in range(L - 1)]
    return {"id": tid, "piece": piece(tid), "n": int(len(ids)), "space": space, "index": self_idx,
            "layers": layers, "full": full, "neighbor_overlap": overlap}


def quant_error(run: LoadedRun) -> list[dict] | None:
    """Per-layer relative L2 error and mean per-row max-abs error of the int8 table vs. the f16 table."""
    if not (run.path / "ple_table.i8.bin").exists():
        return None
    f16, _, meta = load_table_memmap(run.path, "f16")
    q, scale, _ = load_table_memmap(run.path, "i8")
    L, p, V = meta["n_layer"], meta["ple_dim"], meta["vocab"]
    err_sq, ref_sq, maxabs = np.zeros(L), np.zeros(L), np.zeros(L)
    for s in range(0, V, 8192):
        a = np.asarray(f16[s:s + 8192], dtype=np.float32).reshape(-1, L, p)
        b = (np.asarray(q[s:s + 8192], dtype=np.float32) * scale[s:s + 8192, None]).reshape(-1, L, p)
        d = b - a
        err_sq += (d.astype(np.float64) ** 2).sum((0, 2))
        ref_sq += (a.astype(np.float64) ** 2).sum((0, 2))
        maxabs += np.abs(d).max(-1).sum(0)
    return [{"layer": l, "rel_error": float(math.sqrt(err_sq[l] / max(ref_sq[l], 1e-30))),
             "mean_max_abs": float(maxabs[l] / V)} for l in range(L)]


def stats(run: LoadedRun, freq: Frequencies, bins: int = 40, sample: int = 3000) -> dict:
    key = ("stats", freq.key())
    with run.lock:
        if key in run.caches:
            return run.caches[key]
    rows = run.token_rows()
    limit = min(rows.shape[0], GPT2_VOCAB)
    norms = np.linalg.norm(rows[:limit], axis=-1)  # [V', L]
    full = np.linalg.norm(rows[:limit].reshape(limit, -1), axis=-1)
    edges = np.linspace(0.0, float(norms.max()) * 1.0001 + 1e-9, bins + 1)
    hists = [np.histogram(norms[:, l], bins=edges)[0].tolist() for l in range(norms.shape[1])]
    counts = freq.counts()
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(limit, size=min(sample, limit), replace=False))
    scatter: dict = {"ids": pick.tolist(), "norm": _r(full[pick], 4),
                     "pieces": [piece(int(i)) for i in pick], "freq": None}
    trend = None
    if counts is not None:
        c = counts[:limit]
        scatter["freq"] = c[pick].tolist()
        lb = np.floor(np.log10(c + 1) * 4) / 4  # quarter-decade buckets
        trend = [{"log10_freq": float(b), "mean_norm": float(full[lb == b].mean()), "count": int((lb == b).sum())}
                 for b in np.unique(lb)]
    res = {
        "n_layer": run.cfg.n_layer, "ple_dim": run.cfg.ple_dim, "vocab": limit,
        "init_norm": 0.02 * run.cfg.ple_dim,
        "init_full_norm": 0.02 * run.cfg.ple_dim * math.sqrt(run.cfg.n_layer),
        "hist": {"edges": _r(edges, 5), "counts": hists},
        "layer_mean_norm": _r(norms.mean(0), 5),
        "norm_vs_freq": scatter, "trend": trend,
        "quant": quant_error(run),
    }
    with run.lock:
        run.caches[key] = res
    return res


# ---------------------------------------------------------------- chart data (static/viz)

def norm_dist(run: LoadedRun, freq: Frequencies, bins: int = 48, gx: int = 36, aspect: float = 0.5) -> dict:
    """Data for the ridgeline + hexbin charts (explorer/static/viz).

    hist: per-layer slice-norm histograms split into tokens seen in train.bin (count > 0) and unseen.
    hexbin: all GPT-2 tokens binned on (x = log10(1+count), y = full-row norm). Coordinates are
    normalised so x spans [0, 1] and y spans [0, aspect]; hexagons (pointy-top, d3-hexbin layout) have
    radius 1 / (gx * sqrt(3)) in those units (gx hexagons across). Bin y grows upwards (data space)."""
    bins = max(8, min(int(bins), 200))
    gx = max(4, min(int(gx), 200))
    aspect = max(0.1, min(float(aspect), 4.0))
    key = ("norm_dist", bins, gx, round(aspect, 4), freq.key())
    with run.lock:
        if key in run.caches:
            return run.caches[key]
    rows = run.token_rows()
    limit = min(rows.shape[0], GPT2_VOCAB)
    L = run.cfg.n_layer
    norms = np.linalg.norm(rows[:limit], axis=-1)  # [V', L]
    full = np.linalg.norm(rows[:limit].reshape(limit, -1), axis=-1)
    counts = freq.counts()
    c = counts[:limit] if counts is not None else None
    seen = c > 0 if c is not None else np.ones(limit, dtype=bool)
    init = 0.02 * run.cfg.ple_dim
    edges = np.linspace(0.0, float(norms.max()) * 1.0001 + 1e-9, bins + 1)
    layers = []
    for l in range(L):
        x = norms[:, l]
        layers.append({
            "layer": l,
            "seen": np.histogram(x[seen], bins=edges)[0].tolist(),
            "unseen": np.histogram(x[~seen], bins=edges)[0].tolist(),
            "mean": round(float(x.mean()), 5), "mean_seen": round(float(x[seen].mean()), 5) if seen.any() else None,
            "median": round(float(np.median(x)), 5), "p99": round(float(np.percentile(x, 99)), 5),
            "frac_gt_2x_init": round(float((x > 2 * init).mean()), 5),
        })
    res: dict = {
        "n_layer": L, "ple_dim": run.cfg.ple_dim, "vocab": limit, "init_norm": init,
        "init_full_norm": init * math.sqrt(L), "has_freq": c is not None,
        "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()),
        "edges": _r(edges, 5), "layers": layers, "hexbin": None, "trend": None,
    }
    if c is not None:
        lx = np.log10(c.astype(np.float64) + 1)
        x0, x1 = 0.0, float(lx.max()) or 1.0
        y0, y1 = 0.0, float(full.max()) * 1.02 or 1.0
        nx = (lx - x0) / (x1 - x0)
        ny = (full - y0) / (y1 - y0) * aspect
        r = 1.0 / (gx * math.sqrt(3))
        dx, dy = r * 2 * math.sin(math.pi / 3), r * 1.5
        # d3-hexbin assignment: nearest of the two candidate row centres
        pj = ny / dy
        pi = nx / dx
        pj0 = np.round(pj)
        pi0 = np.round(pi - (pj0.astype(np.int64) & 1) / 2)
        pj1 = np.where(pj - pj0 < 0, pj0 - 1, pj0 + 1)
        pi1 = np.round(pi - (pj1.astype(np.int64) & 1) / 2)
        d0 = (pi - pi0 - (pj0.astype(np.int64) & 1) / 2) ** 2 * dx * dx + (pj - pj0) ** 2 * dy * dy
        d1 = (pi - pi1 - (pj1.astype(np.int64) & 1) / 2) ** 2 * dx * dx + (pj - pj1) ** 2 * dy * dy
        use1 = d1 < d0
        bj = np.where(use1, pj1, pj0).astype(np.int64)
        bi = np.where(use1, pi1, pi0).astype(np.int64)
        keys = (bj + 1) * 100_000 + (bi + 1)  # bi, bj >= -1 since nx, ny >= 0
        uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        order = np.argsort(-c, kind="stable")  # examples = most frequent members
        first: dict[int, list[int]] = {}
        for t in order:
            b = int(inv[t])
            lst = first.setdefault(b, [])
            if len(lst) < 3:
                lst.append(int(t))
        hb = []
        sum_n = np.bincount(inv, weights=full)
        sum_f = np.bincount(inv, weights=lx)
        for b, (k_, n_) in enumerate(zip(uniq, cnt)):
            j, i = divmod(int(k_), 100_000)
            j, i = j - 1, i - 1
            hb.append({"x": round((i + (j & 1) / 2) * dx, 5), "y": round(j * dy, 5), "n": int(n_),
                       "mean_norm": round(float(sum_n[b] / n_), 4), "mean_logf": round(float(sum_f[b] / n_), 4),
                       "examples": [{"id": t, "piece": piece(t), "freq": int(c[t]), "norm": round(float(full[t]), 4)}
                                    for t in first.get(b, [])]})
        res["hexbin"] = {"gx": gx, "aspect": aspect, "radius": r, "x_domain": [x0, x1], "y_domain": [y0, y1],
                         "x_label": "log10(1 + train count)", "y_label": "full-row norm", "bins": hb}
        lb = np.floor(lx * 4) / 4
        res["trend"] = [{"log10_freq": float(b), "mean_norm": round(float(full[lb == b].mean()), 4),
                         "count": int((lb == b).sum())} for b in np.unique(lb)]
    with run.lock:
        run.caches[key] = res
    return res


def neighbor_graph(run: LoadedRun, tid: int, layer: int | str = -1, k: int = 12, k2: int = 2) -> dict:
    """Ego graph for the force layout: root + top-k cosine neighbours in `layer` (-1 = full row) or in
    every layer (layer='all': nodes are the union, one root link per layer that found the node).
    Extra links connect each neighbour to its top-k2 most similar other nodes (full-row cosine)."""
    run.check_id(tid)
    L = run.cfg.n_layer
    k = max(1, min(int(k), 50))
    k2 = max(0, min(int(k2), 8))
    if isinstance(layer, str) and layer != "all":
        try:
            layer = int(layer)
        except ValueError:
            raise ExplorerError(f"layer must be an integer or 'all', got '{layer}'") from None
    layers = list(range(L)) if layer == "all" else [int(layer)]
    for l in layers:
        run.check_layer(l)
    limit = min(run.vocab, GPT2_VOCAB)
    nodes: dict[int, dict] = {tid: {"id": tid, "piece": piece(tid), "root": True, "layers": [], "sim": 1.0}}
    links = []
    for l in layers:
        m = run.normalized(l)
        sims = m[:limit] @ m[tid]
        if tid < limit:
            sims[tid] = -2.0
        val, idx = torch.topk(sims, min(k, limit - 1))
        for s, i in zip(val.cpu().tolist(), idx.cpu().tolist()):
            n = nodes.setdefault(i, {"id": i, "piece": piece(i), "root": False, "layers": [], "sim": -1.0})
            n["layers"].append(l)
            n["sim"] = max(n["sim"], round(s, 4))
            links.append({"source": tid, "target": i, "sim": round(s, 4), "layer": l, "kind": "root"})
    ids = list(nodes)
    if k2 and len(ids) > 2:
        m = run.normalized(-1)
        sub = m[torch.tensor(ids, device=m.device)]
        s = (sub @ sub.T).cpu()
        s.fill_diagonal_(-2.0)
        s[0, :] = -2.0
        s[:, 0] = -2.0
        seen_pairs = set()
        for a in range(1, len(ids)):
            val, idx = torch.topk(s[a], min(k2, len(ids) - 2))
            for v, b in zip(val.tolist(), idx.tolist()):
                pair = (min(a, b), max(a, b))
                if b == 0 or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                links.append({"source": ids[a], "target": ids[b], "sim": round(v, 4), "layer": -1, "kind": "peer"})
    for n in nodes.values():
        n["shared"] = len(n["layers"])
    return {"id": tid, "piece": piece(tid), "layer": layer, "layers": layers, "k": k, "k2": k2,
            "nodes": list(nodes.values()), "links": links}


# ---------------------------------------------------------------- model analysis

MAX_TEXT_CHARS = 100_000


def prepare_ids(run: LoadedRun, text: str | None) -> tuple[list[int], list[str]]:
    if not text or not text.strip():
        raise ExplorerError("text is empty")
    warnings: list[str] = []
    bs = run.cfg.block_size
    if len(text) > MAX_TEXT_CHARS:  # avoid tokenizing megabytes only to keep block_size tokens
        warnings.append(f"text was {len(text)} characters; only the first {MAX_TEXT_CHARS} were tokenized")
        text = text[:MAX_TEXT_CHARS]
    ids = encode(text)
    if not ids:
        raise ExplorerError("text tokenized to nothing")
    if len(ids) > bs:
        warnings.append(f"text was {len(ids)} tokens; truncated to block_size={bs}")
        ids = ids[:bs]
    if max(ids) >= run.cfg.vocab_size:
        raise ExplorerError(f"token id {max(ids)} exceeds model vocab {run.cfg.vocab_size}")
    if run.cfg.ple_dim > 0 and max(ids) >= run.cfg.ple_vocab:
        raise ExplorerError(f"token id {max(ids)} exceeds PLE table vocab {run.cfg.ple_vocab}")
    return ids, warnings


def parse_mask(mask: str | list[int] | None, n_layer: int) -> list[int] | None:
    if mask is None:
        return None
    if mask == "all":
        return list(range(n_layer))
    if isinstance(mask, str):
        raise ExplorerError("mask must be 'all' or a list of layer indices")
    layers = sorted({int(m) for m in mask})
    if not layers:
        return None
    bad = [l for l in layers if not 0 <= l < n_layer]
    if bad:
        raise ExplorerError(f"mask layers {bad} out of range [0, {n_layer})")
    return layers


def _per_token_loss(logits: torch.Tensor, ids: torch.Tensor) -> list[float | None]:
    """logits [1,T,V], ids [1,T] -> [None, loss_1, ..., loss_{T-1}]."""
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    nll = -lp.gather(-1, ids[0, 1:, None])[:, 0]
    return [None] + [round(float(x), 4) for x in nll.cpu()]


def _mean(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return round(sum(v) / len(v), 4) if v else None


@torch.no_grad()
def traced_forward(model: GPT, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mirrors GPT.forward block by block, recording gate magnitudes and PLE contribution ratios."""
    T = ids.shape[1]
    x0 = model.wte(ids) + model.wpe(torch.arange(T, device=ids.device))
    u = model.per_layer_inputs(ids, x0)
    x = x0
    gates, ratios = [], []
    for l, blk in enumerate(model.blocks):
        x = x + blk.attn(blk.ln1(x))
        x = x + blk.mlp(blk.ln2(x))
        g = blk.ple.gate_activations(x)
        c = blk.ple.norm(blk.ple.proj(g * u[:, :, l]))
        gates.append(g.abs().mean(-1)[0])
        ratios.append((c.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8))[0])
        x = x + c
    logits = model.lm_head(model.ln_f(x))
    return {"logits": logits, "gate": torch.stack(gates, -1), "ratio": torch.stack(ratios, -1)}


@torch.no_grad()
def analyze(run: LoadedRun, text: str | None, mask: str | list[int] | None = None) -> dict:
    run.require_ple()
    ids_list, warnings = prepare_ids(run, text)
    res = analyze_ids(run, ids_list, mask)
    res["warnings"] = warnings
    return res


@torch.no_grad()
def analyze_ids(run: LoadedRun, ids_list: list[int], mask: str | list[int] | None = None) -> dict:
    """analyze() on already tokenized ids (len <= block_size); used by the findings script."""
    run.require_ple()
    model, L = run.model, run.cfg.n_layer
    if not ids_list or len(ids_list) > run.cfg.block_size:
        raise ExplorerError(f"need 1..{run.cfg.block_size} token ids, got {len(ids_list)}")
    warnings: list[str] = []
    layers = parse_mask(mask, L)
    ids = torch.tensor([ids_list], device=run.device)
    tr = traced_forward(model, ids)
    probs = F.softmax(tr["logits"][0].float(), dim=-1)
    pv, pi = probs.topk(5, dim=-1)
    loss = _per_token_loss(tr["logits"], ids)
    res: dict = {
        "tokens": [{"id": i, "piece": piece(i)} for i in ids_list],
        "n_layer": L, "warnings": warnings,
        "gate": _r(tr["gate"].cpu().numpy(), 4),  # [T, L]
        "contrib_ratio": _r(tr["ratio"].cpu().numpy(), 4),  # [T, L]
        "contrib_ratio_mean": _r(tr["ratio"].mean(0).cpu().numpy(), 4),  # [L]
        "top5": [[{"id": int(i), "piece": piece(int(i)), "p": round(float(p), 4)} for p, i in zip(pr, ix)]
                 for pr, ix in zip(pv.cpu(), pi.cpu())],
        "loss": loss,
        "mean_loss": _mean(loss),
    }
    if layers:
        m = torch.ones(1, 1, L, 1, device=run.device)
        m[0, 0, layers] = 0
        logits_abl, _ = model(ids, ple_mask=m)
        abl = _per_token_loss(logits_abl, ids)
        delta = [None if a is None else round(a - b, 4) for a, b in zip(abl, loss)]
        res["ablation"] = {"layers": layers, "loss": abl, "mean_loss": _mean(abl),
                           "delta": delta, "mean_delta": _mean(delta)}
    return res


@torch.no_grad()
def per_token_loss(run: LoadedRun, text: str | None) -> dict:
    ids_list, warnings = prepare_ids(run, text)
    ids = torch.tensor([ids_list], device=run.device)
    logits, _ = run.model(ids)
    loss = _per_token_loss(logits, ids)
    return {"tokens": [{"id": i, "piece": piece(i)} for i in ids_list], "loss": loss,
            "mean_loss": _mean(loss), "warnings": warnings}


def compare(base: LoadedRun, ple: LoadedRun, text: str | None) -> dict:
    """diff = baseline_loss - ple_loss (positive = PLE predicts that token better)."""
    a, b = per_token_loss(base, text), per_token_loss(ple, text)
    n = min(len(a["loss"]), len(b["loss"]))
    diff = [None if x is None or y is None else round(x - y, 4) for x, y in zip(a["loss"][:n], b["loss"][:n])]
    return {
        "tokens": a["tokens"][:n],
        "baseline": {"run": base.name, "loss": a["loss"][:n], "mean_loss": _mean(a["loss"][:n])},
        "ple": {"run": ple.name, "loss": b["loss"][:n], "mean_loss": _mean(b["loss"][:n])},
        "diff": diff, "mean_diff": _mean(diff), "warnings": sorted(set(a["warnings"] + b["warnings"])),
    }
