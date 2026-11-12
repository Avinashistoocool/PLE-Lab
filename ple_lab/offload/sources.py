"""PLE table sources: where the [V, L*p] per-layer-embedding table lives and how rows reach the GPU.

Every source is an ``nn.Module`` implementing the contract in docs/CONTRACT.md:

    forward(ids: LongTensor[B, T]) -> Tensor[B, T, L*p]   (raw unscaled rows, on ids.device)

Off-GPU sources share one pipeline (``TableSource.forward``):

    ids -> CPU -> torch.unique (sorted, inverse, counts) -> fetch(unique ids) -> rows on device
        -> rows.index_select(inverse) -> [B, T, L*p]

so each distinct token row is read from the backing store and copied host->device at most once per call.
All tensors held by sources are plain attributes (not registered buffers), so ``model.to()`` /
``model.state_dict()`` never move or serialise them.

Byte accounting (``stats()``):
  backing_bytes       bytes read from the off-GPU store (RAM table or memmap file), row payload (+ int8 scales)
  backing_page_bytes  mmap only: 4 KiB pages touched (what the OS actually has to fault in when cold)
  h2d_bytes           row payload copied host->device
  h2d_index_bytes     index tensors copied host->device (inverse map, cache slots)
  resident_vram_bytes table / cache bytes permanently held on the GPU
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..checkpoint import load_table_memmap

PAGE_BYTES = 4096


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class TableSource(nn.Module):
    """Base class: unique -> fetch -> scatter, plus stats. Subclasses implement ``fetch``."""

    kind = "base"

    def __init__(self, vocab: int, width: int, device="cpu", out_dtype=torch.float16, pinned: bool = True):
        super().__init__()
        self.vocab, self.width = int(vocab), int(width)
        self.device = torch.device(device)
        self.out_dtype = out_dtype
        self.pinned = bool(pinned) and self.device.type == "cuda"
        self.reset_stats()

    # ---- stats ----
    def reset_stats(self) -> None:
        self._s = dict(calls=0, tokens=0, unique_rows=0, backing_bytes=0, backing_page_bytes=0,
                       h2d_bytes=0, h2d_index_bytes=0)

    def resident_bytes(self) -> int:
        return 0

    def stats(self) -> dict:
        s = dict(self._s)
        t = max(s["tokens"], 1)
        s.update(kind=self.kind, resident_vram_bytes=self.resident_bytes(),
                 h2d_bytes_per_token=s["h2d_bytes"] / t, backing_bytes_per_token=s["backing_bytes"] / t,
                 backing_page_bytes_per_token=s["backing_page_bytes"] / t)
        return s

    def _account(self, tokens: int, unique: int) -> None:
        self._s["calls"] += 1
        self._s["tokens"] += int(tokens)
        self._s["unique_rows"] += int(unique)

    # ---- transfer helpers ----
    def _stage(self, shape, dtype) -> torch.Tensor:
        """Host staging buffer. Pinned buffers come from PyTorch's caching host allocator, which keeps
        them alive until pending non_blocking copies complete, so a fresh buffer per call is safe."""
        return torch.empty(shape, dtype=dtype, pin_memory=self.pinned)

    def _h2d(self, t: torch.Tensor, index: bool = False) -> torch.Tensor:
        if self.device.type == "cpu":
            return t
        self._s["h2d_index_bytes" if index else "h2d_bytes"] += _nbytes(t)
        return t.to(self.device, non_blocking=t.is_pinned())

    # ---- interface ----
    def fetch(self, uids: torch.Tensor, counts: torch.Tensor | None = None) -> torch.Tensor:
        """uids: sorted unique int64 CPU ids [n]. Returns rows [n, width] (out_dtype) on self.device."""
        raise NotImplementedError

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        flat_cpu = flat if flat.device.type == "cpu" else flat.cpu()  # D2H sync; avoid via Prefetcher
        uids, inv, counts = torch.unique(flat_cpu, sorted=True, return_inverse=True, return_counts=True)
        self._account(flat.numel(), uids.numel())
        rows = self.fetch(uids, counts)
        inv_d = self._h2d(inv.to(torch.int32), index=True)
        return rows.index_select(0, inv_d).view(*ids.shape, self.width).to(ids.device)

    def extra_repr(self) -> str:
        return f"vocab={self.vocab}, width={self.width}, device={self.device}"


class GPUTableSource(TableSource):
    """Baseline: the whole fp16 table resident in VRAM; forward is a plain embedding gather."""

    kind = "gpu"

    def __init__(self, rows, device="cuda", out_dtype=torch.float16):
        rows = torch.as_tensor(np.asarray(rows))
        super().__init__(rows.shape[0], rows.shape[1], device, out_dtype)
        self.rows = rows.to(device=self.device, dtype=out_dtype).contiguous()

    @classmethod
    def from_run(cls, run_dir, device="cuda", **kw):
        rows, _, _ = load_table_memmap(run_dir, "f16")
        return cls(np.array(rows), device, **kw)

    def resident_bytes(self) -> int:
        return _nbytes(self.rows)

    def fetch(self, uids, counts=None):
        return self.rows.index_select(0, uids.to(self.device))

    def forward(self, ids):
        self._account(ids.numel(), 0)
        return F.embedding(ids.to(self.device), self.rows).to(ids.device)


class CPUTableSource(TableSource):
    """Table in (pageable) CPU RAM. Per call: CPU index_select of the unique rows into a staging buffer
    (pinned if ``pinned``) and a non_blocking H2D copy of just those rows."""

    def __init__(self, rows, device="cuda", pinned: bool = True, out_dtype=torch.float16):
        rows = torch.as_tensor(np.asarray(rows))
        super().__init__(rows.shape[0], rows.shape[1], device, out_dtype, pinned)
        self.rows = rows.to(torch.float16).contiguous()
        self.kind = "cpu-pinned" if self.pinned else "cpu"

    @classmethod
    def from_run(cls, run_dir, device="cuda", pinned=True, **kw):
        rows, _, _ = load_table_memmap(run_dir, "f16")
        return cls(np.array(rows), device, pinned, **kw)

    def fetch(self, uids, counts=None):
        buf = self._stage((uids.numel(), self.width), self.rows.dtype)
        torch.index_select(self.rows, 0, uids, out=buf)
        self._s["backing_bytes"] += _nbytes(buf)
        return self._h2d(buf).to(self.out_dtype)


class MmapTableSource(TableSource):
    """Table stays in a memory-mapped file (stand-in for phone flash). Only the requested rows are read
    (``np.take`` on the memmap, sorted ids -> mostly sequential access). ``fmt="i8"`` reads int8 rows plus
    float32 per-row scales, ships both to the device (half the bytes of fp16) and dequantises there."""

    def __init__(self, run_dir, fmt: str = "f16", device="cuda", pinned: bool = True, out_dtype=torch.float16):
        rows, scale, meta = load_table_memmap(run_dir, fmt)
        super().__init__(meta["vocab"], meta["row_width"], device, out_dtype, pinned)
        self.fmt, self.rows, self.scale = fmt, rows, scale
        self.kind = f"mmap-{fmt}"
        self.row_bytes = self.width * rows.dtype.itemsize

    def _pages(self, u: np.ndarray) -> int:
        if u.size == 0:
            return 0
        first = (u * self.row_bytes) // PAGE_BYTES
        last = (u * self.row_bytes + self.row_bytes - 1) // PAGE_BYTES
        return int((last - first + 1).sum() - (first[1:] == last[:-1]).sum())  # u is sorted & unique

    def fetch(self, uids, counts=None):
        u = uids.numpy()
        n = u.size
        buf = self._stage((n, self.width), torch.float16 if self.fmt == "f16" else torch.int8)
        np.take(self.rows, u, axis=0, out=buf.numpy())
        self._s["backing_bytes"] += _nbytes(buf)
        self._s["backing_page_bytes"] += self._pages(u) * PAGE_BYTES
        if self.fmt == "f16":
            return self._h2d(buf).to(self.out_dtype)
        sbuf = self._stage((n,), torch.float32)
        sbuf.numpy()[:] = self.scale[u]
        self._s["backing_bytes"] += _nbytes(sbuf)
        q, s = self._h2d(buf), self._h2d(sbuf)
        return (q.to(torch.float32) * s[:, None]).to(self.out_dtype)


class CachedTableSource(TableSource):
    """Fixed-size GPU row cache in front of any backing source.

    buffer [capacity, width] on the device + CPU-side slot map (id -> slot, slot -> id).
    policy:
      lru          evict the least-recently-used slot
      lfu          evict the least-frequently-used slot (frequency over the whole history); a miss only
                   replaces a victim if it has been seen more often (TinyLFU-style admission)
      static-topk  preload the ``capacity`` most frequent ids from ``freq`` (e.g. training-set counts);
                   never evicts, misses are served straight from the backing store
    Rows of the current call are never evicted by that same call, and misses are returned from the freshly
    fetched rows rather than from the buffer, so results are correct even if capacity < unique ids in a batch.
    """

    POLICIES = ("lru", "lfu", "static-topk")

    def __init__(self, backing: TableSource, capacity_rows: int, policy: str = "lru", freq=None):
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {self.POLICIES}")
        super().__init__(backing.vocab, backing.width, backing.device, backing.out_dtype)
        self.backing, self.policy = backing, policy
        self.capacity = int(max(1, min(int(capacity_rows), self.vocab)))
        self.kind = f"cached-{policy}" if policy != "static-topk" else "static-topk"
        self.buffer = torch.zeros(self.capacity, self.width, dtype=self.out_dtype, device=self.device)
        self.clear()
        if policy == "static-topk":
            if freq is None:
                raise ValueError("static-topk needs a frequency array")
            self._preload(np.asarray(freq))

    def clear(self) -> None:
        self.slot_of = np.full(self.vocab, -1, dtype=np.int64)
        self.id_of = np.full(self.capacity, -1, dtype=np.int64)
        self.last_used = np.zeros(self.capacity, dtype=np.int64)
        self.count = np.zeros(self.vocab, dtype=np.int64)
        self.tick = 0

    def _preload(self, freq: np.ndarray) -> None:
        freq = np.resize(freq.astype(np.int64), self.vocab) if freq.size != self.vocab else freq.astype(np.int64)
        top = np.sort(np.argsort(-freq, kind="stable")[: self.capacity])
        rows = self.backing.fetch(torch.from_numpy(top))
        self.buffer[: top.size] = rows
        self.slot_of[top] = np.arange(top.size)
        self.id_of[: top.size] = top
        self.preload_backing_bytes = self.backing._s["backing_bytes"]
        self.preload_h2d_bytes = self.backing._s["h2d_bytes"]
        self.backing.reset_stats()

    def reset_stats(self) -> None:
        super().reset_stats()
        self._s.update(hit_tokens=0, hit_unique=0, miss_unique=0, inserts=0, evictions=0)
        if hasattr(self, "backing"):
            self.backing.reset_stats()

    def resident_bytes(self) -> int:
        return _nbytes(self.buffer) + self.backing.resident_bytes()

    def stats(self) -> dict:
        b = self.backing.stats()
        for k in ("backing_bytes", "backing_page_bytes", "h2d_bytes"):
            self._s[k] = b[k]
        s = super().stats()
        s.update(capacity=self.capacity, policy=self.policy, backing_kind=b["kind"],
                 hit_rate=s["hit_tokens"] / max(s["tokens"], 1),
                 unique_hit_rate=s["hit_unique"] / max(s["hit_unique"] + s["miss_unique"], 1),
                 h2d_index_bytes=s["h2d_index_bytes"] + b["h2d_index_bytes"])
        return s

    def _dev_idx(self, a: np.ndarray) -> torch.Tensor:
        return self._h2d(torch.from_numpy(np.ascontiguousarray(a)), index=True)

    def fetch(self, uids, counts=None):
        u = uids.numpy()
        c = np.ones(u.size, dtype=np.int64) if counts is None else counts.numpy()
        self.tick += 1
        self.count[u] += c
        slots = self.slot_of[u]
        hit = slots >= 0
        hit_pos, miss_pos = np.nonzero(hit)[0], np.nonzero(~hit)[0]
        self._s["hit_tokens"] += int(c[hit_pos].sum())
        self._s["hit_unique"] += hit_pos.size
        self._s["miss_unique"] += miss_pos.size
        hit_slots = slots[hit_pos]
        self.last_used[hit_slots] = self.tick
        if miss_pos.size == 0:
            return self.buffer.index_select(0, self._dev_idx(hit_slots))
        miss_rows = self.backing.fetch(uids[torch.from_numpy(miss_pos)])
        if hit_pos.size == 0:
            out = miss_rows
        else:
            out = torch.empty(u.size, self.width, dtype=self.out_dtype, device=self.device)
            out[self._dev_idx(hit_pos)] = self.buffer.index_select(0, self._dev_idx(hit_slots))
            out[self._dev_idx(miss_pos)] = miss_rows
        if self.policy != "static-topk":
            self._admit(u[miss_pos], miss_rows)
        return out

    def _admit(self, miss_ids: np.ndarray, miss_rows: torch.Tensor) -> None:
        order = np.arange(miss_ids.size)
        if self.policy == "lfu":  # most frequent misses get the slots first
            order = np.argsort(-self.count[miss_ids], kind="stable")
        free = np.nonzero(self.id_of < 0)[0][: order.size]
        take, slots = order[: free.size], [free]
        rest = order[free.size:]
        if rest.size:
            cand = np.nonzero((self.id_of >= 0) & (self.last_used < self.tick))[0]

            k = min(rest.size, cand.size)
            if k:
                key = self.last_used[cand] if self.policy == "lru" else self.count[self.id_of[cand]]
                victims = cand[np.argsort(key, kind="stable")[:k]]
                if self.policy == "lfu":
                    ok = self.count[miss_ids[rest[:k]]] > self.count[self.id_of[victims]]
                    k = int(np.argmin(ok)) if not ok.all() else k
                    victims = victims[:k]
                self.slot_of[self.id_of[victims]] = -1
                self._s["evictions"] += k
                take = np.concatenate([take, rest[:k]])
                slots.append(victims)
        slots = np.concatenate(slots)
        if take.size == 0:
            return
        ids = miss_ids[take]
        self.buffer[self._dev_idx(slots)] = miss_rows.index_select(0, self._dev_idx(take))
        self.slot_of[ids] = slots
        self.id_of[slots] = ids
        self.last_used[slots] = self.tick
        self._s["inserts"] += take.size


class Prefetcher(TableSource):
    """Fetch rows for ids known ahead of time (the prompt / the next batch) on a background thread and,
    on CUDA, a dedicated copy stream, so backing reads + H2D overlap with GPU compute of the current batch.

    Usage:   pf.prefetch(next_ids_cpu)  ...  model(next_ids_gpu)   # forward consumes the oldest prefetch

    Limits (by design, keep it simple):
      * strictly FIFO: forward() consumes the oldest pending prefetch; if its ids do not match (checked when
        ``verify=True``, which costs a D2H copy of ids) all pending work is drained and the call falls back
        to a synchronous fetch, so results are always correct.
      * one worker thread; the GIL serialises Python-side work, only numpy/torch kernels and the async
        copy really overlap. Gains appear only when GPU compute per batch exceeds the fetch time.
      * useless for autoregressive decode: the next token is only known after sampling.
      * while prefetches are pending, don't call the wrapped source directly from another thread.
    """

    def __init__(self, source: TableSource, depth: int = 2, verify: bool = True, use_stream: bool = True):
        super().__init__(source.vocab, source.width, source.device, source.out_dtype)
        self.source, self.depth, self.verify = source, int(depth), verify
        self.kind = f"{source.kind}+prefetch"
        self.stream = torch.cuda.Stream(self.device) if (use_stream and self.device.type == "cuda") else None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-prefetch")
        self._queue: deque = deque()

    def reset_stats(self) -> None:
        super().reset_stats()
        self._s.update(prefetch_used=0, prefetch_fallback=0)
        if hasattr(self, "source"):
            self.source.reset_stats()

    def resident_bytes(self) -> int:
        return self.source.resident_bytes()

    def stats(self) -> dict:
        s = self.source.stats()
        s.update(kind=self.kind, prefetch_used=self._s["prefetch_used"], prefetch_fallback=self._s["prefetch_fallback"])
        return s

    def fetch(self, uids, counts=None):
        return self.source.fetch(uids, counts)

    def _work(self, ids_cpu: torch.Tensor, after: "torch.cuda.Event | None" = None):
        flat = ids_cpu.reshape(-1)
        uids, inv, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
        self.source._account(flat.numel(), uids.numel())
        if self.stream is None:
            rows = self.source.fetch(uids, counts)
            return rows, self.source._h2d(inv.to(torch.int32), index=True), None
        # order after work already queued on the consumer stream (e.g. reads of cache slots we may overwrite)
        self.stream.wait_event(after)
        with torch.cuda.stream(self.stream):
            rows = self.source.fetch(uids, counts)
            inv_d = self.source._h2d(inv.to(torch.int32).pin_memory(), index=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        return rows, inv_d, ev

    def prefetch(self, ids) -> None:
        """Queue a fetch for ids that a later forward() will request (ids may be on any device)."""
        if len(self._queue) >= self.depth:  # bound host/device memory: wait for the oldest one
            self._queue[0][1].result()
        ids_cpu = ids.detach().to("cpu").clone()
        after = None
        if self.stream is not None:
            after = torch.cuda.Event()
            after.record(torch.cuda.current_stream(self.device))
        self._queue.append((ids_cpu, self._pool.submit(self._work, ids_cpu, after)))

    def _sync(self, rows, inv_d, ev):
        if ev is not None:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(ev)
            rows.record_stream(cur)
            inv_d.record_stream(cur)

    def drain(self) -> None:
        while self._queue:
            _, fut = self._queue.popleft()
            self._sync(*fut.result())

    def forward(self, ids):
        if self._queue:
            ids_cpu, fut = self._queue[0]
            ok = tuple(ids_cpu.shape) == tuple(ids.shape) and (not self.verify or torch.equal(ids_cpu, ids.detach().cpu()))
            if ok:
                self._queue.popleft()
                rows, inv_d, ev = fut.result()
                self._sync(rows, inv_d, ev)
                self._s["prefetch_used"] += 1
                return rows.index_select(0, inv_d).view(*ids.shape, self.width).to(ids.device)
            self.drain()
        self._s["prefetch_fallback"] += 1
        return self.source(ids)

    def close(self) -> None:
        self.drain()
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------- factory
KINDS = ("gpu", "cpu", "cpu-pinned", "mmap-f16", "mmap-i8", "cached-lru", "cached-lfu", "static-topk")


def make_source(run_dir, kind: str, device="cuda", *, capacity: int | None = None, capacity_frac: float | None = None,
                backing: str = "mmap-f16", freq=None, prefetch: bool = False, prefetch_depth: int = 2,
                verify: bool = True, out_dtype=torch.float16) -> TableSource:
    """Build a table source for a saved run. Cache kinds take ``capacity`` (rows) or ``capacity_frac``
    (fraction of vocab) and a ``backing`` kind; ``static-topk`` needs ``freq`` ([V] token counts)."""
    run_dir = Path(run_dir)
    if kind == "gpu":
        src = GPUTableSource.from_run(run_dir, device, out_dtype=out_dtype)
    elif kind in ("cpu", "cpu-pinned"):
        src = CPUTableSource.from_run(run_dir, device, pinned=(kind == "cpu-pinned"), out_dtype=out_dtype)
    elif kind in ("mmap-f16", "mmap-i8"):
        src = MmapTableSource(run_dir, kind.split("-")[1], device, out_dtype=out_dtype)
    elif kind in ("cached-lru", "cached-lfu", "static-topk"):
        if backing not in ("cpu", "cpu-pinned", "mmap-f16", "mmap-i8"):
            raise ValueError(f"bad backing {backing!r}")
        inner = make_source(run_dir, backing, device, out_dtype=out_dtype)
        if capacity is None:
            if capacity_frac is None:
                raise ValueError("cache needs capacity or capacity_frac")
            capacity = max(1, int(round(capacity_frac * inner.vocab)))
        policy = "static-topk" if kind == "static-topk" else kind.split("-")[1]
        src = CachedTableSource(inner, capacity, policy, freq=freq)
    else:
        raise ValueError(f"unknown source kind {kind!r}; choose from {KINDS}")
    if prefetch:
        src = Prefetcher(src, depth=prefetch_depth, verify=verify)
    return src


def attach(model, source: nn.Module):
    """Install ``source`` as ``model.ple_table`` after checking it matches the model's PLE shape."""
    cfg = model.cfg
    if not cfg.ple_dim:
        raise ValueError("model has no PLE (ple_dim=0)")
    width, vocab = getattr(source, "width", None), getattr(source, "vocab", None)
    if width is not None and width != cfg.n_layer * cfg.ple_dim:
        raise ValueError(f"source width {width} != n_layer*ple_dim {cfg.n_layer * cfg.ple_dim}")
    if vocab is not None and vocab < cfg.ple_vocab:
        raise ValueError(f"source vocab {vocab} < model ple vocab {cfg.ple_vocab}")
    model.ple_table = source
    return model
