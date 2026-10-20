"""Tests for the dashboard server's run listing and its whitelist of servable files."""

from __future__ import annotations

from dashboard.serve import PAGE, list_runs, resolve


def _runs(tmp_path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "predictions.csv").write_text("cd_setor\n")
    (tmp_path / "run_a" / "notes.txt").write_text("private")
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "loose.csv").write_text("x\n")
    return tmp_path


def test_lists_only_folders_holding_predictions(tmp_path):
    assert list_runs(_runs(tmp_path)) == ["run_a"]
    assert list_runs(tmp_path / "missing") == []


def test_serves_the_page_and_whitelisted_run_files(tmp_path):
    root = _runs(tmp_path)
    assert resolve(root, "/")[0] == PAGE
    assert resolve(root, "/runs/run_a/predictions.csv?x=1")[0] == root / "run_a" / "predictions.csv"


def test_refuses_anything_outside_the_whitelist(tmp_path):
    root = _runs(tmp_path)
    for url in (
        "/runs/run_a/notes.txt",
        "/runs/not_a_run/predictions.csv",
        "/runs/../run_a/predictions.csv",
        "/runs/%2e%2e/predictions.csv",
        "/loose.csv",
        "/runs/run_a/../../loose.csv",
    ):
        assert resolve(root, url) is None, url
