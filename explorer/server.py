"""FastAPI backend for the PLE Explorer.

Run:  python -m explorer.server --runs runs --port 8765 [--device cuda] [--data data/tinystories/train.bin]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis as A

STATIC = Path(__file__).parent / "static"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "tinystories" / "train.bin"


class AnalyzeBody(BaseModel):
    run: str
    text: str = ""
    mask: str | list[int] | None = None


class CompareBody(BaseModel):
    text: str = ""
    baseline: str = "baseline-small"
    ple: str = "ple-small"


def create_app(runs_dir: str | Path = "runs", device: str = "cpu",
               data_bin: str | Path | None = DEFAULT_DATA) -> FastAPI:
    runs_dir = Path(runs_dir)
    manager = A.RunManager(runs_dir, device)
    freq = A.Frequencies(Path(data_bin) if data_bin else None, runs_dir / ".explorer_cache")
    app = FastAPI(title="PLE Explorer")
    app.state.manager, app.state.freq = manager, freq

    @app.exception_handler(A.ExplorerError)
    async def _explorer_error(_: Request, exc: A.ExplorerError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def ple_run(name: str | None) -> A.LoadedRun:
        run = manager.get(name)
        run.require_ple()
        return run

    @app.get("/api/runs")
    def runs() -> dict:
        out = []
        for name, path in manager.available().items():
            try:
                out.append(A.run_summary(name, path))
            except Exception as e:  # a run being written by the trainer must not break the list
                out.append({"name": name, "error": str(e)})
        return {"runs": out, "device": device, "freq_source": str(freq.bin_path) if freq.key() else None}

    @app.get("/api/token")
    def token(run: str, q: str | None = None, id: int | None = None) -> dict:
        return A.token_info(ple_run(run), q, id, freq)

    @app.get("/api/neighbors")
    def neighbors(run: str, id: int, layer: int = -1, k: int = 20) -> dict:
        return A.neighbors(ple_run(run), id, layer, k)

    @app.get("/api/layer_similarity")
    def layer_similarity(run: str, id: int | None = None, top_n: int = 1000) -> dict:
        return A.layer_similarity(ple_run(run), id, freq, top_n)

    @app.get("/api/projection")
    def projection(run: str, layer: int = 0, n: int = 2000) -> dict:
        return A.projection(ple_run(run), layer, n, freq)

    @app.get("/api/projection3d")
    def projection3d(run: str, n: int = 2000, space: str = "shared") -> dict:
        return A.projection3d(ple_run(run), n, space, freq)

    @app.get("/api/trajectory")
    def trajectory(run: str, id: int, n: int = 2000, space: str = "shared", k: int = 8) -> dict:
        return A.trajectory(ple_run(run), id, n, space, freq, k)

    @app.get("/api/stats")
    def stats(run: str) -> dict:
        return A.stats(ple_run(run), freq)

    @app.get("/api/norm_dist")
    def norm_dist(run: str, bins: int = 48, gx: int = 36, aspect: float = 0.5) -> dict:
        return A.norm_dist(ple_run(run), freq, bins, gx, aspect)

    @app.get("/api/neighbor_graph")
    def neighbor_graph(run: str, id: int, layer: str = "-1", k: int = 12, k2: int = 2) -> dict:
        return A.neighbor_graph(ple_run(run), id, layer, k, k2)

    @app.post("/api/analyze")
    def analyze(body: AnalyzeBody) -> dict:
        return A.analyze(ple_run(body.run), body.text, body.mask)

    @app.post("/api/compare")
    def compare(body: CompareBody) -> dict:
        avail = manager.available()
        missing = [n for n in (body.baseline, body.ple) if n not in avail]
        if missing:
            raise A.ExplorerError(f"compare needs runs {missing} which do not exist yet")
        base, ple = manager.get(body.baseline), manager.get(body.ple)
        if base.cfg.ple_dim != 0:
            raise A.ExplorerError(f"'{body.baseline}' is not a baseline run (ple_dim={base.cfg.ple_dim})")
        ple.require_ple()
        return A.compare(base, ple, body.text)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="PLE Explorer server")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="uint16 token file for frequencies")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    import uvicorn
    uvicorn.run(create_app(args.runs, args.device, args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
