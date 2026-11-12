import numpy as np
import pytest
import torch

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from explorer.server import create_app
from ple_lab import GPT, GPTConfig, save_run

L, P = 3, 8


def _cfg(ple: int) -> GPTConfig:
    return GPTConfig(vocab_size=50304, block_size=16, n_layer=L, n_head=2, d_model=16, ple_dim=ple)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    torch.manual_seed(0)
    save_run(GPT(_cfg(P)).eval(), root / "ple-small", {"val_loss": 3.21})
    save_run(GPT(_cfg(0)).eval(), root / "baseline-small", {"val_loss": 3.5})
    (root / "not-a-run").mkdir()
    freq = root / "train.bin"
    np.array([262, 262, 262, 11, 11, 13, 5, 0, 1], dtype=np.uint16).tofile(freq)
    return TestClient(create_app(root, "cpu", freq))


def test_runs(client):
    runs = {r["name"]: r for r in client.get("/api/runs").json()["runs"]}
    assert set(runs) == {"ple-small", "baseline-small"}
    assert runs["ple-small"]["val_loss"] == 3.21 and runs["ple-small"]["has_ple"]
    assert runs["ple-small"]["params"]["ple_table"] == 50304 * L * P
    assert not runs["baseline-small"]["has_ple"]


def test_token_text_and_id(client):
    r = client.get("/api/token", params={"run": "ple-small", "q": " the"}).json()
    assert r["id"] == 262 and r["piece"] == " the" and r["frequency"] == 3
    assert len(r["norms"]) == L and len(r["vectors"]) == L and len(r["vectors"][0]) == P
    r2 = client.get("/api/token", params={"run": "ple-small", "id": 262}).json()
    assert r2["norms"] == r["norms"]
    assert client.get("/api/token", params={"run": "ple-small", "q": "#11"}).json()["piece"] == ","
    assert len(client.get("/api/token", params={"run": "ple-small", "q": " the cat"}).json()["pieces"]) == 2


def test_neighbors(client):
    for layer in (-1, 0, L - 1):
        r = client.get("/api/neighbors", params={"run": "ple-small", "id": 262, "layer": layer, "k": 5}).json()
        sims = [n["sim"] for n in r["neighbors"]]
        assert len(sims) == 5 and sims == sorted(sims, reverse=True)
        assert 262 not in [n["id"] for n in r["neighbors"]]
        assert all(n["id"] < 50257 for n in r["neighbors"])


def test_layer_similarity(client):
    r = client.get("/api/layer_similarity", params={"run": "ple-small", "id": 262}).json()
    m, g = np.array(r["matrix"]), np.array(r["global"])
    assert m.shape == g.shape == (L, L)
    assert np.allclose(np.diag(m), 1, atol=1e-3) and np.allclose(m, m.T, atol=1e-4)
    assert r["global_source"] == "frequency"
    assert "matrix" not in client.get("/api/layer_similarity", params={"run": "ple-small"}).json()


def test_projection(client):
    r = client.get("/api/projection", params={"run": "ple-small", "layer": 1, "n": 50}).json()
    assert r["n"] == 50 and len(r["points"]) == 50 and r["points"][0]["id"] == 262
    assert {"x", "y", "piece", "freq"} <= set(r["points"][0])
    assert client.get("/api/projection", params={"run": "ple-small", "layer": 1, "n": 50}).json() == r  # cached


def test_stats(client):
    r = client.get("/api/stats", params={"run": "ple-small"}).json()
    assert len(r["hist"]["counts"]) == L and sum(r["hist"]["counts"][0]) == 50257
    assert len(r["quant"]) == L and all(q["rel_error"] < 0.05 for q in r["quant"])
    assert r["norm_vs_freq"]["freq"] is not None and r["trend"]


def test_analyze_and_ablation(client):
    text = "Once upon a time there was a cat."
    r = client.post("/api/analyze", json={"run": "ple-small", "text": text}).json()
    T = len(r["tokens"])
    assert T > 3 and "ablation" not in r
    assert np.array(r["gate"]).shape == (T, L) and np.array(r["contrib_ratio"]).shape == (T, L)
    assert len(r["contrib_ratio_mean"]) == L and len(r["top5"]) == T and len(r["top5"][0]) == 5
    assert r["loss"][0] is None and all(x > 0 for x in r["loss"][1:])
    a = client.post("/api/analyze", json={"run": "ple-small", "text": text, "mask": "all"}).json()["ablation"]
    assert a["layers"] == list(range(L)) and len(a["loss"]) == len(a["delta"]) == T
    assert a["delta"][1] == pytest.approx(a["loss"][1] - r["loss"][1], abs=2e-4)
    b = client.post("/api/analyze", json={"run": "ple-small", "text": text, "mask": [1]}).json()["ablation"]
    assert b["layers"] == [1]


def test_analyze_matches_model_forward(client):
    """The traced forward must give the same loss as the model's own forward."""
    text = "The dog ran home."
    r = client.post("/api/analyze", json={"run": "ple-small", "text": text, "mask": []}).json()
    c = client.post("/api/compare", json={"text": text}).json()
    assert r["loss"] == pytest.approx(c["ple"]["loss"], abs=1e-3)


def test_compare(client):
    r = client.post("/api/compare", json={"text": "Lily had a red ball."}).json()
    n = len(r["tokens"])
    assert len(r["baseline"]["loss"]) == len(r["ple"]["loss"]) == len(r["diff"]) == n
    assert r["diff"][1] == pytest.approx(r["baseline"]["loss"][1] - r["ple"]["loss"][1], abs=2e-4)


def test_truncation_warning(client):
    r = client.post("/api/analyze", json={"run": "ple-small", "text": "word " * 40}).json()
    assert len(r["tokens"]) == 16 and "truncated" in r["warnings"][0]


@pytest.mark.parametrize("method,url,kw", [
    ("get", "/api/token", {"params": {"run": "nope", "q": "a"}}),
    ("get", "/api/token", {"params": {"run": "baseline-small", "q": "a"}}),
    ("get", "/api/token", {"params": {"run": "ple-small", "q": ""}}),
    ("get", "/api/token", {"params": {"run": "ple-small", "id": 999999}}),
    ("get", "/api/neighbors", {"params": {"run": "ple-small", "id": 1, "layer": 99}}),
    ("get", "/api/neighbors", {"params": {"run": "baseline-small", "id": 1}}),
    ("get", "/api/projection", {"params": {"run": "ple-small", "layer": -5}}),
    ("get", "/api/stats", {"params": {"run": "baseline-small"}}),
    ("post", "/api/analyze", {"json": {"run": "ple-small", "text": "   "}}),
    ("post", "/api/analyze", {"json": {"run": "baseline-small", "text": "hi"}}),
    ("post", "/api/analyze", {"json": {"run": "ple-small", "text": "hi", "mask": [7]}}),
    ("post", "/api/analyze", {"json": {"run": "ple-small", "text": "hi", "mask": "some"}}),
    ("post", "/api/analyze", {"json": {"run": "nope", "text": "hi"}}),
    ("post", "/api/compare", {"json": {"text": ""}}),
    ("post", "/api/compare", {"json": {"text": "hi", "baseline": "ple-small", "ple": "ple-small"}}),
    ("post", "/api/compare", {"json": {"text": "hi", "baseline": "missing"}}),
])
def test_errors(client, method, url, kw):
    r = getattr(client, method)(url, **kw)
    assert r.status_code == 400, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_baseline_error_message(client):
    r = client.get("/api/token", params={"run": "baseline-small", "q": "a"})
    assert "baseline" in r.json()["detail"]


def test_static_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "PLE Explorer" in r.text
    assert client.get("/static/app.js").status_code == 200

# ---------------------------------------------------------------- numerics vs. the model's own forward

def _hooked_reference(model, ids):
    """Capture gate / PLE update / residual via hooks on the real GPT.forward."""
    rec = {"gate_in": [], "upd": []}
    hs = []
    for blk in model.blocks:
        hs.append(blk.ple.gate.register_forward_hook(lambda m, i, o: rec["gate_in"].append(i[0])))
        hs.append(blk.ple.norm.register_forward_hook(lambda m, i, o: rec["upd"].append(o)))
    try:
        with torch.no_grad():
            logits, _ = model(ids)
    finally:
        for h in hs:
            h.remove()
    return logits, rec


def test_traced_forward_matches_hooks(client):
    from explorer import analysis as A
    run = client.app.state.manager.get("ple-small")
    model = run.model
    ids = torch.tensor([A.encode("The dog ran to the park and saw a bird.")])
    ref_logits, rec = _hooked_reference(model, ids)
    tr = A.traced_forward(model, ids)
    assert torch.allclose(tr["logits"], ref_logits, atol=1e-5)
    for l, blk in enumerate(model.blocks):
        h = rec["gate_in"][l]  # hidden state AFTER attn+mlp of block l (input to W_gate)
        g = blk.ple.gate_activations(h).abs().mean(-1)[0]
        ratio = (rec["upd"][l].norm(dim=-1) / h.norm(dim=-1))[0]
        assert torch.allclose(tr["gate"][:, l], g, atol=1e-5)
        assert torch.allclose(tr["ratio"][:, l], ratio, atol=1e-5)


def test_gate_uses_post_mlp_hidden_not_block_input(client):
    """Guard against computing the gate from the block input instead of h after attn+mlp."""
    from explorer import analysis as A
    model = client.app.state.manager.get("ple-small").model
    ids = torch.tensor([A.encode("Once upon a time there was a cat.")])
    _, _, hidden = model(ids, return_hidden=True)
    tr = A.traced_forward(model, ids)
    wrong = model.blocks[1].ple.gate_activations(hidden[1]).abs().mean(-1)[0]
    assert not torch.allclose(tr["gate"][:, 1], wrong, atol=1e-4)


def test_ablation_matches_ple_mask_and_sign(client):
    from explorer import analysis as A
    run = client.app.state.manager.get("ple-small")
    text = "Lily went to the park with her mom."
    r = A.analyze(run, text, [0, 2])
    ids = torch.tensor([A.encode(text)])
    m = torch.ones(1, 1, L, 1)
    m[0, 0, [0, 2]] = 0
    with torch.no_grad():
        logits, _ = run.model(ids, ple_mask=m)
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
    a = r["ablation"]
    assert a["loss"][0] is None and a["delta"][0] is None
    assert np.allclose(a["loss"][1:], nll.numpy(), atol=1e-3)
    # delta = ablated - normal (positive: the token needed PLE)
    assert np.allclose(np.array(a["delta"][1:]), np.array(a["loss"][1:]) - np.array(r["loss"][1:]), atol=2e-4)
    assert a["mean_delta"] == pytest.approx(np.mean(a["delta"][1:]), abs=2e-4)


def test_very_long_text_is_truncated_with_warning(client):
    r = client.post("/api/analyze", json={"run": "ple-small", "text": "a " * 200_000})
    assert r.status_code == 200
    j = r.json()
    assert len(j["tokens"]) == 16 and any("truncated" in w for w in j["warnings"])
    assert any("characters" in w for w in j["warnings"])


# ---------------------------------------------------------------- caching

def test_frequency_cache_invalidates_when_file_changes(tmp_path):
    import os
    from explorer.analysis import Frequencies
    f = tmp_path / "train.bin"
    np.array([5, 5, 7], dtype=np.uint16).tofile(f)
    fr = Frequencies(f, tmp_path / "cache")
    c1 = fr.counts()
    assert c1[5] == 2 and c1[7] == 1
    st = f.stat()
    np.array([7, 7, 7], dtype=np.uint16).tofile(f)  # same size, new content
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    c2 = fr.counts()
    assert c2[5] == 0 and c2[7] == 3
    assert len(list((tmp_path / "cache").glob("freq_*.npy"))) == 1  # stale cache removed
    assert Frequencies(f, tmp_path / "cache").counts()[7] == 3  # fresh instance reads the disk cache
    f.unlink()
    assert fr.counts() is None and fr.key() is None


def test_frequency_cache_survives_corrupt_cache_file(tmp_path):
    from explorer.analysis import Frequencies
    f = tmp_path / "train.bin"
    np.array([1, 2, 2], dtype=np.uint16).tofile(f)
    fr = Frequencies(f, tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"freq_{fr.key()}.npy").write_bytes(b"garbage")
    assert fr.counts()[2] == 2


def test_run_reloads_when_checkpoint_changes_and_bad_run_is_400(tmp_path):
    import os
    root = tmp_path / "runs"
    torch.manual_seed(1)
    save_run(GPT(_cfg(P)).eval(), root / "r")
    c = TestClient(create_app(root, "cpu", None))
    n1 = c.get("/api/token", params={"run": "r", "id": 262}).json()["norms"]
    torch.manual_seed(2)
    save_run(GPT(_cfg(P)).eval(), root / "r")
    t = (root / "r" / "dense.pt").stat().st_mtime_ns + 10_000_000
    for name in ("dense.pt", "ple_table.f16.bin", "config.json"):
        os.utime(root / "r" / name, ns=(t, t))
    n2 = c.get("/api/token", params={"run": "r", "id": 262}).json()["norms"]
    assert n1 != n2
    assert c.get("/api/token", params={"run": "r", "q": " the"}).json()["frequency"] is None
    (root / "r" / "dense.pt").write_bytes(b"truncated")
    os.utime(root / "r" / "dense.pt", ns=(t + 10_000_000, t + 10_000_000))
    r = c.get("/api/token", params={"run": "r", "id": 1})
    assert r.status_code == 400 and "could not be loaded" in r.json()["detail"]


# ---------------------------------------------------------------- findings script helpers

def test_findings_helpers(client, tmp_path):
    from scipy.stats import spearmanr
    from explorer import findings as Fd
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.integers(0, 5, size=200).astype(float)  # ties in b
    assert Fd.spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-9)
    val = tmp_path / "val.bin"
    np.array(A_TOKENS * 20, dtype=np.uint16).tofile(val)
    run = client.app.state.manager.get("ple-small")
    base = client.app.state.manager.get("baseline-small")
    w = Fd.val_windows(val, 16, 3)
    assert len(w) == 3 and all(len(x) == 16 for x in w)
    r = Fd.ablation_study(run, w, base)
    assert r["n_predictions"] == 3 * 15 and len(r["contrib_ratio_mean_per_layer"]) == L
    assert {f"ablate_layer{l}" for l in range(L)} | {"ablate_all"} <= set(r)


A_TOKENS = [464, 3290, 4966, 1363, 13, 262, 3797, 373, 3772, 13]
