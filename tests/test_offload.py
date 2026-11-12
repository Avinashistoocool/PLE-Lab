import os

import numpy as np
import pytest
import torch

from ple_lab import GPT, GPTConfig, load_run, save_run
from ple_lab.offload import (CachedTableSource, CPUTableSource, GPUTableSource, MmapTableSource, Prefetcher, attach,
                             make_source)

# CUDA cases are opt-in (PLE_TEST_CUDA=1) so the suite never touches a GPU that a training job is using.
_CUDA = os.environ.get("PLE_TEST_CUDA") == "1" and torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _CUDA, reason="CUDA unavailable or PLE_TEST_CUDA!=1"))]
V, L, P = 97, 3, 16


def tiny():
    return GPTConfig(vocab_size=V, block_size=16, n_layer=L, n_head=2, d_model=32, ple_dim=P)


@pytest.fixture
def run(tmp_path):
    torch.manual_seed(0)
    m = GPT(tiny()).eval()
    with torch.no_grad():  # larger rows so the PLE path visibly matters
        m.ple_table.table.weight.normal_(0, 0.5)
    save_run(m, tmp_path)
    return tmp_path, m


def ref_rows(m):
    return m.ple_table.table.weight.detach().half().float()  # what the fp16 file stores


def offloaded(run_dir, kind, device, **kw):
    m = load_run(run_dir, device, load_table=False)
    src = make_source(run_dir, kind, device, **kw)
    attach(m, src)
    return m, src


ALL = [("gpu", {}), ("cpu", {}), ("cpu-pinned", {}), ("mmap-f16", {}), ("mmap-i8", {}),
       ("cached-lru", {"capacity": 8}), ("cached-lfu", {"capacity": 8}),
       ("static-topk", {"capacity": 8, "freq": np.arange(V)[::-1]}),
       ("cached-lru", {"capacity": 8, "backing": "mmap-i8"}),
       ("mmap-f16", {"prefetch": True}), ("cached-lru", {"capacity": 8, "prefetch": True})]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind,kw", ALL, ids=lambda x: x if isinstance(x, str) else ",".join(sorted(x)))
def test_logits_match_in_memory(run, device, kind, kw):
    run_dir, ref = run
    m, src = offloaded(run_dir, kind, device, **kw)
    ref = ref.to(device)
    i8 = kind == "mmap-i8" or kw.get("backing") == "mmap-i8"
    # int8 per-row symmetric: |err| <= scale/2 = max|row|/254 ~ 0.5*4/254 ~ 8e-3 per element, which is
    # amplified by sqrt(p)=4 and the network; 5e-2 on logits of magnitude ~1 is still a tight check.
    atol = 5e-2 if i8 else 1e-2
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for step in range(4):  # several calls so caches see hits, misses and evictions
            ids = torch.randint(0, V, (2, 12), generator=g)
            ids[0, :4] = 5  # duplicates within a batch
            ids = ids.to(device)
            if isinstance(src, Prefetcher):
                src.prefetch(ids)
            a, _ = ref(ids)
            b, _ = m(ids)
            assert b.device == a.device
            torch.testing.assert_close(b, a, atol=atol, rtol=0)
    s = src.stats()
    assert s["tokens"] == 4 * 24
    if isinstance(src, Prefetcher):
        assert s["prefetch_used"] == 4 and s["prefetch_fallback"] == 0
        src.close()


@pytest.mark.parametrize("device", DEVICES)
def test_rows_exact(run, device):
    run_dir, m = run
    want = ref_rows(m).to(device)
    ids = torch.tensor([[3, 3, 96, 0, 3, 50]], device=device)
    for src in (CPUTableSource.from_run(run_dir, device, pinned=True), MmapTableSource(run_dir, "f16", device)):
        out = src(ids)
        assert out.device.type == device and out.shape == (1, 6, L * P)
        assert torch.equal(out.float(), want[ids])
    deq = MmapTableSource(run_dir, "i8", device)(ids).float()
    scale = want.abs().amax(1) / 127
    assert ((deq - want[ids]).abs() <= scale[ids][..., None] * 0.5 + 2e-3).all()


@pytest.mark.parametrize("device", DEVICES)
def test_cache_accounting_lru(run, device):
    run_dir, m = run
    want = ref_rows(m).to(device)
    src = make_source(run_dir, "cached-lru", device, capacity=4, backing="cpu")
    W = L * P

    def call(lst):
        ids = torch.tensor([lst], device=device)
        out = src(ids)
        assert torch.equal(out.float(), want[ids])

    call([1, 2, 1, 1])  # 2 misses (ids 1,2), 0 hits
    s = src.stats()
    assert (s["hit_tokens"], s["miss_unique"], s["evictions"], s["inserts"]) == (0, 2, 0, 2)
    assert s["backing_bytes"] == 2 * W * 2  # each unique row read once, fp16
    if device == "cuda":
        assert s["h2d_bytes"] == 2 * W * 2
    call([1, 2])  # all hits, nothing read
    s = src.stats()
    assert s["hit_tokens"] == 2 and s["backing_bytes"] == 2 * W * 2 and s["hit_rate"] == 2 / 6
    call([3, 4])  # fills cache: [1,2,3,4]
    call([5, 1])  # 1 hit; evict LRU among unused-this-call = 2 (tick 2) vs 3,4 (tick 3) -> 2
    s = src.stats()
    assert s["evictions"] == 1 and src.slot_of[2] == -1 and src.slot_of[5] >= 0 and src.slot_of[1] >= 0
    call([2])  # miss again
    assert src.stats()["miss_unique"] == 6
    assert s["resident_vram_bytes"] == 4 * W * 2
    src.reset_stats()
    assert src.stats()["tokens"] == 0 and src.stats()["backing_bytes"] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", ["lru", "lfu", "static-topk"])
def test_cache_smaller_than_batch(run, device, policy):
    run_dir, m = run
    want = ref_rows(m).to(device)
    freq = np.bincount(np.arange(V) % 7, minlength=V)
    src = make_source(run_dir, "cached-" + policy if policy != "static-topk" else policy, device, capacity=3,
                      backing="mmap-f16", freq=freq)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        ids = torch.randint(0, 40, (3, 20), generator=g).to(device)  # ~35 unique ids >> capacity 3
        assert torch.equal(src(ids).float(), want[ids])
    s = src.stats()
    assert s["tokens"] == 300 and 0 <= s["hit_rate"] <= 1
    assert (src.id_of >= 0).sum() <= 3
    if policy == "static-topk":
        assert s["evictions"] == 0 and sorted(src.id_of.tolist()) == [0, 1, 2]
    else:
        assert s["evictions"] > 0
    # slot map consistent
    for slot, tid in enumerate(src.id_of):
        if tid >= 0:
            assert src.slot_of[tid] == slot
            assert torch.equal(src.buffer[slot].float(), want[tid])


def test_lfu_keeps_frequent(run):
    run_dir, _ = run
    src = make_source(run_dir, "cached-lfu", "cpu", capacity=2, backing="cpu")
    for _ in range(5):
        src(torch.tensor([[7, 8]]))
    for i in range(10, 20):
        src(torch.tensor([[i]]))  # one-off ids must not displace the hot ones
    assert src.slot_of[7] >= 0 and src.slot_of[8] >= 0
    src(torch.tensor([[7, 8]]))
    assert src.stats()["hit_tokens"] == 10


def test_zipf_hit_rate_high(run):
    run_dir, _ = run
    rng = np.random.default_rng(0)
    ids = torch.from_numpy((rng.zipf(1.3, size=(50, 32)) - 1) % V)
    src = make_source(run_dir, "cached-lru", "cpu", capacity=10)
    for row in ids:
        src(row[None])
    assert src.stats()["hit_rate"] > 0.5


def test_prefetch_mismatch_falls_back(run):
    run_dir, m = run
    want = ref_rows(m)
    pf = Prefetcher(MmapTableSource(run_dir, "f16", "cpu"))
    pf.prefetch(torch.tensor([[1, 2, 3]]))
    out = pf(torch.tensor([[4, 5, 6]]))
    assert torch.equal(out.float(), want[torch.tensor([[4, 5, 6]])]) and pf.stats()["prefetch_fallback"] == 1
    pf.close()


def test_attach_validates(run):
    run_dir, _ = run
    m = load_run(run_dir, load_table=False)
    with pytest.raises(ValueError):
        attach(m, GPUTableSource(np.zeros((V, 7), np.float16), "cpu"))
    with pytest.raises(ValueError):
        make_source(run_dir, "nope", "cpu")
