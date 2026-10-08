"""Serve the dashboard page and the backtest run archives it reads, on localhost only."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGE = Path(__file__).with_name("index.html")
DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "experiments/compare_forecasters/outputs"
# The only files served from a run folder; nothing else in the repository is reachable.
RUN_FILES = {
    "predictions.csv": "text/csv; charset=utf-8",
    "comparison.csv": "text/csv; charset=utf-8",
    "manifest.json": "application/json",
    "config.yaml": "text/yaml; charset=utf-8",
}


def list_runs(root: Path) -> list[str]:
    """Names of the subfolders of `root` holding a run archive (one with a predictions.csv)."""
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "predictions.csv").is_file())


def resolve(root: Path, url: str) -> tuple[Path, str] | None:
    """The file and content type a request path maps to, or None when it is not servable."""
    parts = unquote(urlsplit(url).path).strip("/").split("/")
    if parts in ([""], ["index.html"]):
        return PAGE, "text/html; charset=utf-8"
    if len(parts) == 3 and parts[0] == "runs":
        return _run_file(root, parts[1], parts[2])
    return None


def _run_file(root: Path, run: str, name: str) -> tuple[Path, str] | None:
    """A whitelisted file of a listed run; run and file names are matched, never joined blindly."""
    if name not in RUN_FILES or run not in list_runs(root):
        return None
    return root / run / name, RUN_FILES[name]


class _Handler(BaseHTTPRequestHandler):
    root: Path = DEFAULT_RUNS

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/runs":
            self._send(json.dumps(list_runs(self.root)).encode(), "application/json")
            return
        target = resolve(self.root, self.path)
        if target is None or not target[0].is_file():
            self.send_error(404)
            return
        self._send(target[0].read_bytes(), target[1])

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a rerun must show up on reload
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="folder of run archives")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    handler = type("Handler", (_Handler,), {"root": args.runs.resolve()})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"dashboard em {url}  (runs de {args.runs})  — Ctrl+C para parar")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
