"""Verify backtest archives preserve inputs, effective settings and execution status."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import yaml

from experiments.compare_forecasters import records
from experiments.compare_forecasters import run as runner


def _config(tmp_path, candidates=None):
    base = tmp_path / "base.csv"
    base.write_bytes(b"items\n100\n")
    config = {
        "split": {"min_train_cycles": 20, "horizon": 1},
        "candidates": candidates or [{"model": "naive", "strategy": "mean"}],
        "paths": {"base_csv": str(base), "run_dir": str(tmp_path / "teste_região")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path, config


def _read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_archive_records_effective_defaults_and_exact_features(tmp_path):
    candidates = [{"model": "lightgbm", "categorical_features": ["CD_RE", "CD_GV"]}]
    path, config = _config(tmp_path, candidates)
    directory, manifest = records.prepare_record(path, config)
    candidate = manifest["candidates"][0]
    assert candidate["effective_parameters"]["lags"] == [1, 2, 3, 4]
    assert candidate["features"][1:3] == ["CD_RE", "CD_GV"]
    assert len(candidate["features"]) == 14
    assert (directory / "config.yaml").read_bytes() == path.read_bytes()


def test_archive_hash_identifies_the_data_and_includes_default_split(tmp_path):
    path, config = _config(tmp_path)
    directory, manifest = records.prepare_record(path, config)
    assert manifest["data"]["sha256"] == hashlib.sha256(b"items\n100\n").hexdigest()
    assert _read_manifest(directory)["split"]["window"] == "expanding"
    assert manifest["candidates"][0]["features"] == []


def test_existing_run_directory_is_not_overwritten(tmp_path):
    path, config = _config(tmp_path)
    directory, _ = records.prepare_record(path, config)
    before = (directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        records.prepare_record(path, config)
    assert (directory / "manifest.json").read_bytes() == before


def test_duplicate_model_names_are_rejected_before_directory_creation(tmp_path):
    path, config = _config(tmp_path, [{"model": "naive"}, {"model": "naive"}])
    with pytest.raises(ValueError, match="names must be unique"):
        records.prepare_record(path, config)
    assert not (tmp_path / "teste_região").exists()


def test_runner_saves_both_outputs_and_completed_manifest(tmp_path, monkeypatch):
    path, config = _config(tmp_path)
    table = pd.DataFrame({"candidate": ["naive:mean"], "mae_common": [10]})
    predictions = pd.DataFrame({"actual": [100], "items_pred": [90]})
    monkeypatch.setattr(runner, "_execute", lambda config: (table, predictions))
    result = runner.run(path)
    directory = tmp_path / "teste_região"
    pd.testing.assert_frame_equal(result, table)
    assert pd.read_csv(directory / "predictions.csv").actual.tolist() == [100]
    assert _read_manifest(directory)["status"] == "complete"
    assert (directory / "comparison.csv").exists()


def test_failed_backtest_preserves_config_and_failure_status(tmp_path, monkeypatch):
    path, _ = _config(tmp_path)

    def fail(config):
        raise RuntimeError("model failed")

    monkeypatch.setattr(runner, "_execute", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        runner.run(path)
    manifest = _read_manifest(tmp_path / "teste_região")
    assert manifest["status"] == "failed"
    assert manifest["error"] == "model failed"


def test_legacy_configs_keep_csv_paths_and_gain_separate_archives(tmp_path, monkeypatch):
    path, config = _config(tmp_path)
    del config["paths"]["run_dir"]
    config["paths"].update(
        output_csv=str(tmp_path / "old.csv"), errors_csv=str(tmp_path / "errors.csv")
    )
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    frame = pd.DataFrame({"actual": [100]})
    monkeypatch.setattr(runner, "_execute", lambda config: (frame, frame))
    runner.run(path)
    runner.run(path)
    assert (tmp_path / "old.csv").exists()
    assert len(list(tmp_path.glob("run_*"))) == 2


def test_metadata_collection_tolerates_missing_package_and_git(monkeypatch):
    def missing_package(package):
        raise records.PackageNotFoundError(package)

    def missing_git(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(records, "version", missing_package)
    monkeypatch.setattr(records.subprocess, "check_output", missing_git)
    assert records._package_version("none") is None
    assert records._git_output("status") is None


def test_real_executor_archives_a_small_backtest(tmp_path):
    path, config = _config(tmp_path)
    config["paths"]["base_csv"] = "tests/fixtures/demand_sample.csv"
    config["split"]["min_train_cycles"] = 2
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    comparison = runner.run(path)
    manifest = _read_manifest(tmp_path / "teste_região")
    assert comparison.n_scored_common.iloc[0] > 0
    assert manifest["n_predictions"] == comparison.n_scored_own.iloc[0]
