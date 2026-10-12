"""Serve a read-only shape dashboard and filtered archive APIs on localhost."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from shape_dashboard.data import archive_file, list_runs, options, run_path, selected_points, view

STATIC = Path(__file__).parent
DEFAULT_RUNS = STATIC.resolve().parents[0] / "experiments/compare_shapes/outputs"
FILES = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/styles.css": ("styles.css", "text/css"),
}


def api(path: str, query: dict, root: Path) -> tuple[bytes, str]:
    """Route only read-only shape endpoints and named archive downloads."""
    actions = {
        "/api/runs": lambda: list_runs(root),
        "/api/options": lambda: options(root, query["run"]),
        "/api/view": lambda: view(root, query),
    }
    if path == "/api/export":
        return selected_points(root, query).to_csv(index=False).encode("utf-8-sig"), "text/csv"
    if path == "/api/config":
        source = archive_file(run_path(root, query["run"]), "config_snapshot.yaml")
        return source.read_bytes(), "text/yaml"
    if path not in actions:
        raise ValueError("Rota desconhecida.")
    return json.dumps(
        actions[path](), ensure_ascii=False, allow_nan=False
    ).encode(), "application/json"


class Handler(BaseHTTPRequestHandler):
    """Serve fixed frontend files and shape archives without exposing the repository."""

    root = DEFAULT_RUNS

    def do_GET(self) -> None:
        """Handle requests with explicit validation errors."""
        url = urlsplit(self.path)
        if url.path in FILES:
            filename, mime = FILES[url.path]
            self._send((STATIC / filename).read_bytes(), mime)
            return
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            body, mime = api(url.path, query, self.root)
        except (ValueError, KeyError, OSError) as error:
            self._send(
                json.dumps({"error": str(error)}, ensure_ascii=False).encode(),
                "application/json",
                400,
            )
            return
        self._send(body, mime)

    def _send(self, body: bytes, mime: str, status: int = 200) -> None:
        """Return noncached responses and name downloadable CSV/YAML files."""
        self.send_response(status)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._download_header(mime)
        self.end_headers()
        self.wfile.write(body)

    def _download_header(self, mime: str) -> None:
        """Name downloads without accepting filenames from the browser."""
        names = {"text/csv": "shape_filtered.csv", "text/yaml": "config_snapshot.yaml"}
        if mime in names:
            self.send_header("Content-Disposition", f'attachment; filename="{names[mime]}"')

    def log_message(self, format: str, *args) -> None:
        """Keep routine request logs out of the terminal."""


def main() -> None:
    """Launch an independent local server for a configured shape results root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    handler = type("ConfiguredHandler", (Handler,), {"root": args.runs.resolve()})
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Shape dashboard: {url} — Ctrl+C para parar", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    main()
