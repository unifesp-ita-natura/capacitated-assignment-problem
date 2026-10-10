"""Validate shape-only dashboard filtering, metrics, downloads and file boundaries."""

import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd
import pytest

from shape_dashboard.data import list_runs, options, predictions, run_path, view
from shape_dashboard.serve import Handler, api


@pytest.fixture
def runs(tmp_path):
    directory = tmp_path / "shape_test"
    directory.mkdir()
    rows = []
    for model, shares in [("sector", [0.2, 0.8]), ("uniform", [0.5, 0.5])]:
        for origin in ["0", "1"]:
            for offset, actual in enumerate([20, 80]):
                rows.append(
                    dict(
                        cd_setor="001",
                        CICLOS="2",
                        origin_cycle=origin,
                        candidate=model,
                        date=f"2026-01-0{offset + 1}",
                        actual=actual,
                        total_actual=100,
                        actual_share=actual / 100,
                        share_pred=shares[offset],
                        items_pred=100 * shares[offset],
                    )
                )
    pd.DataFrame(rows).to_csv(directory / "predictions.csv", index=False)
    (directory / "run_manifest.json").write_text('{"status":"completed"}')
    (directory / "config_snapshot.yaml").write_text("split:\n  min_train_cycles: 20\n")
    return tmp_path


def query(**extra):
    return dict(run="shape_test", cycle="2", origin="0", **extra)


def test_list_runs_and_missing_root(runs):
    assert list_runs(runs) == ["shape_test"]
    assert list_runs(runs / "absent") == []


def test_options_preserve_identifiers_and_origins(runs):
    result = options(runs, "shape_test")
    assert result["sectors"] == ["001"]
    assert result["origins_by_cycle"] == {"2": ["0", "1"]}
    assert result["manifest"]["status"] == "completed"


@pytest.mark.parametrize("name", ["../shape_test", "..", "missing", "shape_test/../shape_test"])
def test_run_path_refuses_traversal(runs, name):
    with pytest.raises(ValueError):
        run_path(runs, name)


def test_metrics_and_origin_not_double_counted(runs):
    result = view(runs, query())
    assert result["n_points"] == 4
    assert result["metrics"][0]["share_mae_pp"] == pytest.approx(0)
    assert result["metrics"][1]["share_mae_pp"] == pytest.approx(30)


def test_series_and_cumulative(runs):
    result = view(runs, query(model="uniform"))
    assert result["series"]["uniform"][-1]["pred_cumulative_pct"] == pytest.approx(100)
    assert result["series"]["uniform"][0]["error"] == pytest.approx(30)
    assert result["n_points"] == 2


def test_sectors_with_different_openings_align_relative_days(runs):
    source = runs / "shape_test" / "predictions.csv"
    frame = pd.read_csv(source, dtype={"cd_setor": str})
    frame["window_start"] = "2026-01-01"
    later = frame.assign(cd_setor="002", window_start="2026-01-10")
    later["date"] = (pd.to_datetime(later["date"]) + pd.Timedelta(days=9)).dt.strftime("%Y-%m-%d")
    pd.concat([frame, later]).to_csv(source, index=False)
    series = view(runs, query(model="sector"))["series"]["sector"]
    assert [point["relative_day"] for point in series] == [1, 2]
    assert [point["actual"] for point in series] == [40, 160]


def test_day_count_starts_at_declared_opening(runs):
    source = runs / "shape_test" / "predictions.csv"
    frame = pd.read_csv(source, dtype={"cd_setor": str})
    frame["window_start"] = "2025-12-30"
    frame.to_csv(source, index=False)
    series = view(runs, query(model="sector"))["series"]["sector"]
    assert series[0]["relative_day"] == 3


def test_empty_selection(runs):
    with pytest.raises(ValueError, match="Nenhum ponto"):
        view(runs, query(sector="unknown"))


def test_zero_total_serializes_null(runs):
    source = runs / "shape_test" / "predictions.csv"
    frame = pd.read_csv(source, dtype={"cd_setor": str})
    frame[["total_actual", "actual", "items_pred"]] = 0
    frame["actual_share"] = float("nan")
    frame.to_csv(source, index=False)
    body, _ = api("/api/view", query(), runs)
    assert json.loads(body)["metrics"][0]["share_mae_pp"] is None


def test_downloads(runs):
    csv, mime = api("/api/export", query(model="sector"), runs)
    assert mime == "text/csv"
    assert "uniform" not in csv.decode("utf-8-sig")
    yaml, _ = api("/api/config", {"run": "shape_test"}, runs)
    assert b"min_train_cycles" in yaml


@pytest.mark.parametrize("route", ["/api/runs", "/api/options", "/api/view"])
def test_json_endpoints(runs, route):
    body, mime = api(route, query(), runs)
    assert mime == "application/json"
    assert json.loads(body)


def test_unknown_endpoint(runs):
    with pytest.raises(ValueError, match="Rota"):
        api("/arbitrary-file", {}, runs)


def test_duplicate_points_rejected(runs):
    directory = runs / "shape_test"
    frame = predictions(directory)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(directory / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="duplicadas"):
        options(runs, "shape_test")


def test_incompatible_csv_rejected(runs):
    (runs / "shape_test" / "predictions.csv").write_text("candidate\nu\n")
    with pytest.raises(ValueError, match="faltam"):
        options(runs, "shape_test")


@pytest.fixture
def server(runs):
    handler = type("TestHandler", (Handler,), {"root": runs})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as service:
        thread = Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{service.server_port}"
        finally:
            service.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize(
    "path,mime",
    [
        ("/", "text/html"),
        ("/app.js", "text/javascript"),
        ("/styles.css", "text/css"),
        ("/api/runs", "application/json"),
        ("/api/export?run=shape_test&cycle=2&origin=0", "text/csv"),
        ("/api/config?run=shape_test", "text/yaml"),
    ],
)
def test_http_responses(server, path, mime):
    with urlopen(server + path, timeout=5) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == mime
        assert response.read()


def test_http_invalid_run(server):
    with pytest.raises(HTTPError) as error:
        urlopen(server + "/api/options?run=../secret", timeout=5)
    assert error.value.code == 400
