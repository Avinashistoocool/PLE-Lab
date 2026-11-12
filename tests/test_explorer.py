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


# ---------------------------------------------------------------- 3-D map

def test_projection3d_shapes_and_normalization(client):
    r = client.get("/api/projection3d", params={"run": "ple-small", "n": 60}).json()
    assert r["n"] == 60 and len(r["points"]) == 60 and r["space"] == "shared" and r["n_layer"] == L
    p0 = r["points"][0]
    assert p0["id"] == 262 and p0["rank"] == 0 and p0["freq"] == 3 and p0["log_freq"] > 0
    c = np.array([p["coords"] for p in r["points"]])
    f = np.array([p["full"] for p in r["points"]])
    assert c.shape == (60, L, 3) and f.shape == (60, 3)
    assert np.abs(c).max() <= 1.0 + 1e-6 and np.abs(c).max() > 0.99  # one shared scale
    assert np.abs(f).max() <= 1.0 + 1e-6
    assert all(len(p["norms"]) == L and 0 <= p["max_layer"] < L for p in r["points"])
    assert {p["cluster"] for p in r["points"]} <= set(range(r["k"])) and r["k"] == 12
    assert sum(cl["size"] for cl in r["clusters"]) == 60
    assert len(r["explained_variance"]) == L and len(r["explained_variance"][0]) == 3
    assert all(len(str(x).split(".")[-1]) <= 4 for x in c.ravel()[:50].tolist())  # rounded


def test_projection3d_shared_space_matches_pca_of_stacked_slices(client):
    from sklearn.decomposition import PCA
    from explorer import analysis as A
    run = client.app.state.manager.get("ple-small")
    r = client.get("/api/projection3d", params={"run": "ple-small", "n": 40}).json()
    ids = [p["id"] for p in r["points"]]
    x = run.token_rows()[ids]  # [n, L, p]
    y = PCA(n_components=3, random_state=0).fit_transform(x.reshape(-1, P)).reshape(40, L, 3)
    got = np.array([p["coords"] for p in r["points"]])
    y = y / np.abs(y).max()
    for k in range(3):  # PCA axes are defined up to sign
        s = np.sign((got[..., k] * y[..., k]).sum())
        assert np.allclose(got[..., k], s * y[..., k], atol=2e-3)
    pl = client.get("/api/projection3d", params={"run": "ple-small", "n": 40, "space": "per_layer"}).json()
    cpl = np.array([p["coords"] for p in pl["points"]])
    assert all(np.abs(cpl[:, l]).max() > 0.99 for l in range(L))  # each layer scaled separately


def test_projection3d_cap_cache_and_errors(client):
    from explorer import analysis as A
    run = client.app.state.manager.get("ple-small")
    r = client.get("/api/projection3d", params={"run": "ple-small", "n": 999999})
    assert r.status_code == 200 and r.json()["n"] == A.PROJ3D_MAX_N == 5000
    a = A.projection3d(run, 30, "shared", client.app.state.freq)
    assert A.projection3d(run, 30, "shared", client.app.state.freq) is a  # cached object
    for params in ({"run": "baseline-small"}, {"run": "ple-small", "space": "bogus"}, {"run": "nope"}):
        e = client.get("/api/projection3d", params=params)
        assert e.status_code == 400 and e.json()["detail"]
    assert "baseline" in client.get("/api/projection3d", params={"run": "baseline-small"}).json()["detail"]


def test_trajectory(client):
    t = client.get("/api/trajectory", params={"run": "ple-small", "id": 262, "n": 50}).json()
    p = client.get("/api/projection3d", params={"run": "ple-small", "n": 50}).json()
    assert t["id"] == 262 and t["index"] == 0 and len(t["layers"]) == L
    for l, lay in enumerate(t["layers"]):
        assert np.allclose(lay["coords"], p["points"][0]["coords"][l], atol=2e-4)
        sims = [n["sim"] for n in lay["neighbors"]]
        assert len(sims) == 8 and sims == sorted(sims, reverse=True)
        assert 262 not in [n["id"] for n in lay["neighbors"]]
        assert all(p["points"][n["index"]]["id"] == n["id"] for n in lay["neighbors"])
    assert np.allclose(t["full"]["coords"], p["points"][0]["full"], atol=2e-4)
    assert len(t["neighbor_overlap"]) == L - 1
    far = client.get("/api/trajectory", params={"run": "ple-small", "id": 40000, "n": 50}).json()
    assert far["index"] is None and len(far["layers"][0]["neighbors"]) == 8  # token outside the map
    for params in ({"run": "baseline-small", "id": 1}, {"run": "ple-small", "id": 999999}):
        assert client.get("/api/trajectory", params=params).status_code == 400


# ---------------------------------------------------------------- chart data (static/viz)

def test_norm_dist_ridgeline_and_hexbin(client):
    r = client.get("/api/norm_dist", params={"run": "ple-small", "bins": 20, "gx": 10}).json()
    assert r["n_layer"] == L and len(r["layers"]) == L and len(r["edges"]) == 21
    assert r["has_freq"] and r["n_seen"] == 6 and r["n_seen"] + r["n_unseen"] == 50257  # ids 0,1,5,11,13,262
    for lay in r["layers"]:
        assert len(lay["seen"]) == len(lay["unseen"]) == 20
        assert sum(lay["seen"]) == 6 and sum(lay["unseen"]) == 50257 - 6
        assert 0 <= lay["frac_gt_2x_init"] <= 1
    hb = r["hexbin"]
    assert sum(b["n"] for b in hb["bins"]) == 50257 and hb["gx"] == 10
    assert all(0 < len(b["examples"]) <= 3 for b in hb["bins"])
    assert all(-hb["radius"] * 2 <= b["x"] <= 1 + hb["radius"] * 2 for b in hb["bins"])
    top = max(hb["bins"], key=lambda b: b["x"])  # most frequent token lands in the right-most hex
    assert top["examples"][0]["id"] == 262
    # every token is assigned to its nearest hex centre (d3-hexbin semantics)
    from explorer import analysis as A
    run = client.app.state.manager.get("ple-small")
    full = np.linalg.norm(run.token_rows()[:50257].reshape(50257, -1), axis=-1)
    centres = np.array([[b["x"], b["y"]] for b in hb["bins"]])
    y = full[262] / hb["y_domain"][1] * hb["aspect"]
    d = ((centres - [1.0, y]) ** 2).sum(1)
    assert hb["bins"][int(d.argmin())]["examples"][0]["id"] == 262
    assert client.get("/api/norm_dist", params={"run": "baseline-small"}).status_code == 400
    assert A.norm_dist(run, client.app.state.freq, 20, 10) is A.norm_dist(run, client.app.state.freq, 20, 10)


def test_neighbor_graph(client):
    g = client.get("/api/neighbor_graph", params={"run": "ple-small", "id": 262, "layer": 1, "k": 5, "k2": 2}).json()
    root = [n for n in g["nodes"] if n["root"]]
    assert len(root) == 1 and root[0]["id"] == 262 and len(g["nodes"]) == 6
    nb = client.get("/api/neighbors", params={"run": "ple-small", "id": 262, "layer": 1, "k": 5}).json()
    assert [l["target"] for l in g["links"] if l["kind"] == "root"] == [n["id"] for n in nb["neighbors"]]
    ids = {n["id"] for n in g["nodes"]}
    assert all(l["source"] in ids and l["target"] in ids for l in g["links"])
    assert any(l["kind"] == "peer" for l in g["links"])
    a = client.get("/api/neighbor_graph", params={"run": "ple-small", "id": 262, "layer": "all", "k": 4, "k2": 0}).json()
    assert a["layers"] == list(range(L)) and sum(l["kind"] == "root" for l in a["links"]) == 4 * L
    assert all(n["shared"] == len(n["layers"]) >= 1 for n in a["nodes"] if not n["root"])
    for params in ({"layer": "bogus"}, {"layer": 99}, {"id": 999999}):
        q = {"run": "ple-small", "id": 262, **params}
        assert client.get("/api/neighbor_graph", params=q).status_code == 400


def test_map3d_static_assets(client):
    """The 3D map module, its demo page and the vendored three.js addons are served, and every import in
    them is a relative path (no bare 'three' specifier, which would need an import map)."""
    import re
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "explorer" / "static"
    files = [static / "map3d.js", *sorted((static / "vendor" / "three" / "addons").rglob("*.js"))]
    for f in files:
        rel = f.relative_to(static).as_posix()
        assert client.get(f"/static/{rel}").status_code == 200, rel
        for spec in re.findall(r"""^(?:import|export)\b[^'";]*?\bfrom\s*['"]([^'"]+)['"]""",
                               f.read_text(encoding="utf-8"), re.M):
            assert spec.startswith("."), (rel, spec)
            assert (f.parent / spec).resolve().is_file(), (rel, spec)
    assert client.get("/static/map3d-demo.html").status_code == 200
    assert "export async function mountMap3D" in (static / "map3d.js").read_text(encoding="utf-8")
