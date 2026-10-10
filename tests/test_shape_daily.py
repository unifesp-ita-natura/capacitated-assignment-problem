"""Check CD-free daily curves, normalization and historical-only training."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.compare_shapes.run import run
from src.forecasting.shape_daily import build_shape_days, predict_shape
from src.forecasting.shape_evaluation import (
    ShapeCandidate,
    ShapeSplit,
    compare_shapes,
    evaluate_shapes,
)


@pytest.fixture
def raw():
    rows = []
    for cycle, start in enumerate(pd.date_range("2026-01-01", periods=3, freq="7D")):
        for offset, items in [(0, 20), (3, 80)]:
            rows.append(
                dict(
                    cd_setor="s",
                    CICLOS=str(cycle),
                    cd_cd="a",
                    data_pedido=start + pd.Timedelta(days=offset),
                    **{
                        "Dt Abertura": start,
                        "Dt Fechamento": start + pd.Timedelta(days=3),
                        "total_itens_mascarado": items,
                    },
                )
            )
    return pd.DataFrame(rows)


def test_daily_aggregation_and_zero_days(raw):
    extra = raw.iloc[[0]].assign(cd_cd="b", total_itens_mascarado=10)
    days = build_shape_days(pd.concat([raw, extra], ignore_index=True))
    assert "cd_cd" not in days
    assert len(days) == 12
    assert days.loc[days["CICLOS"].eq("0"), "actual"].tolist() == [30, 0, 0, 80]
    np.testing.assert_allclose(days.groupby(["cd_setor", "CICLOS"])["actual_share"].sum(), 1)


def test_shape_does_not_require_cd(raw):
    pd.testing.assert_frame_equal(
        build_shape_days(raw), build_shape_days(raw.drop(columns="cd_cd"))
    )


def test_organization_columns_preserved(raw):
    days = build_shape_days(raw.assign(CD_RE="r", CD_GV="g"))
    assert days["CD_RE"].eq("r").all()
    assert days["CD_GV"].eq("g").all()


def test_conflicting_shape_organization_rejected(raw):
    raw["CD_RE"] = "r"
    raw.loc[0, "CD_RE"] = "conflict"
    with pytest.raises(ValueError, match="conflicting"):
        build_shape_days(raw)


@pytest.mark.parametrize("model,expected", [("uniform", [0.25] * 4), ("sector", [0.2, 0, 0, 0.8])])
def test_expected_predictions(raw, model, expected):
    days = build_shape_days(raw)
    history = days[days["CICLOS"].eq("0")]
    windows = days[days["CICLOS"].eq("1")][
        ["cd_setor", "CICLOS", "window_start", "n_days"]
    ].drop_duplicates()
    forecast = predict_shape(history, windows, model, n_bins=4)
    np.testing.assert_allclose(forecast["share_pred"], expected)


def test_unknown_sector_uses_uniform(raw):
    days = build_shape_days(raw)
    windows = (
        days[["cd_setor", "CICLOS", "window_start", "n_days"]]
        .drop_duplicates()
        .assign(cd_setor="new")
    )
    forecast = predict_shape(days, windows)
    assert forecast["uniform_fallback"].all()
    np.testing.assert_allclose(forecast["share_pred"], 0.25)


@pytest.mark.parametrize(
    "candidate,metric,expected",
    [
        ("sector", "share_mae_pp", 0),
        ("uniform", "share_mae_pp", 27.5),
        ("uniform", "mae", 27.5),
        ("uniform", "wmape", 1.1),
        ("uniform", "total_variation", 0.55),
        ("uniform", "bias", 0),
        ("uniform", "n_days", 8),
    ],
)
def test_evaluation_and_metrics(raw, candidate, metric, expected):
    days = build_shape_days(raw)
    candidates = [
        ShapeCandidate(name="sector", model="sector", n_bins=4),
        ShapeCandidate(name="uniform", model="uniform"),
    ]
    scored = evaluate_shapes(days, candidates, ShapeSplit(min_train_cycles=1))
    comparison = compare_shapes(scored).set_index("candidate")
    assert comparison.loc[candidate, metric] == pytest.approx(expected)
    np.testing.assert_allclose(
        scored.groupby(["candidate", "origin_cycle", "CICLOS"])["items_pred"].sum(), 100
    )


@pytest.mark.parametrize("model", ["sector", "lightgbm"])
def test_target_demand_does_not_leak_into_prediction(raw, model):
    candidates = [
        ShapeCandidate(
            name=model, model=model, n_bins=4, learning={"features": ["weekday"], "n_estimators": 5}
        )
    ]
    original = evaluate_shapes(build_shape_days(raw), candidates, ShapeSplit(min_train_cycles=2))
    changed = raw.copy()
    changed.loc[changed["CICLOS"].eq("2"), "total_itens_mascarado"] = [90, 10]
    alternative = evaluate_shapes(
        build_shape_days(changed), candidates, ShapeSplit(min_train_cycles=2)
    )
    np.testing.assert_allclose(original["share_pred"], alternative["share_pred"])
    assert not original["actual_share"].equals(alternative["actual_share"])


def test_zero_cycle_reported_separately(raw):
    raw.loc[raw["CICLOS"].eq("2"), "total_itens_mascarado"] = 0
    scored = evaluate_shapes(
        build_shape_days(raw),
        [ShapeCandidate(name="u", model="uniform")],
        ShapeSplit(min_train_cycles=2),
    )
    metrics = compare_shapes(scored).iloc[0]
    assert metrics["n_zero_total_cycles"] == 1
    assert pd.isna(metrics["share_mae_pp"])
    assert pd.isna(metrics["wmape"])


@pytest.mark.parametrize("window,expected", [("expanding", 0.15), ("sliding", 0.1)])
def test_training_window(raw, window, expected):
    raw.loc[raw["CICLOS"].eq("1"), "total_itens_mascarado"] = [10, 90]
    scored = evaluate_shapes(
        build_shape_days(raw),
        [ShapeCandidate(name="s", model="sector", n_bins=4)],
        ShapeSplit(min_train_cycles=1, window=window),
    )
    last = scored[scored["CICLOS"].eq("2")]
    assert last.iloc[0]["share_pred"] == pytest.approx(expected)


def test_insufficient_cycles(raw):
    with pytest.raises(ValueError, match="not enough"):
        evaluate_shapes(
            build_shape_days(raw), [ShapeCandidate(name="u", model="uniform")], ShapeSplit()
        )


def test_negative_items_rejected(raw):
    raw.loc[0, "total_itens_mascarado"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        build_shape_days(raw)


def test_empty_positive_history_uses_uniform(raw):
    days = build_shape_days(raw)
    history = days.assign(total_actual=0, actual_share=np.nan)
    windows = days[["cd_setor", "CICLOS", "window_start", "n_days"]].drop_duplicates()
    forecast = predict_shape(history, windows)
    np.testing.assert_allclose(forecast["share_pred"], 0.25)
    assert forecast["uniform_fallback"].all()


def test_shorter_window_with_no_bin_mass_uses_uniform(raw):
    raw["total_itens_mascarado"] = [0, 100] * 3
    days = build_shape_days(raw)
    windows = (
        days[["cd_setor", "CICLOS", "window_start", "n_days"]].drop_duplicates().assign(n_days=1)
    )
    forecast = predict_shape(days, windows, n_bins=4)
    np.testing.assert_allclose(forecast["share_pred"], 1)
    assert forecast["uniform_fallback"].all()


def _config_file(raw, tmp_path, min_train=1):
    source = tmp_path / "base.csv"
    raw.assign(
        nm_ciclo="test",
        aa_ciclo=2026,
        dia_ciclo=1,
        total_pedidos_mascarado=1,
        total_volumes_mascarado=1,
    ).to_csv(source, index=False)
    config = {
        "split": {"min_train_cycles": min_train},
        "candidates": [{"name": "uniform", "model": "uniform"}],
        "paths": {"base_csv": str(source), "run_dir": str(tmp_path / "output")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_archives_and_refuses_overwrite(raw, tmp_path):
    config = _config_file(raw, tmp_path)
    directory = run(config)
    assert {path.name for path in directory.iterdir()} == {
        "predictions.csv",
        "comparison.csv",
        "config_snapshot.yaml",
        "run_manifest.json",
    }
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    with pytest.raises(FileExistsError):
        run(config)


def test_failed_run_records_error(raw, tmp_path):
    config = _config_file(raw, tmp_path, min_train=20)
    with pytest.raises(ValueError, match="not enough"):
        run(config)
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "not enough" in manifest["error"]


def test_learned_run_records_features(raw, tmp_path):
    path = _config_file(raw, tmp_path)
    config = yaml.safe_load(path.read_text())
    config["candidates"] = [
        {
            "name": "weekday",
            "model": "lightgbm",
            "learning": {"features": ["weekday"], "n_estimators": 2},
        }
    ]
    path.write_text(yaml.safe_dump(config))
    directory = run(path)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["features_by_candidate"]["weekday"]["features"] == [
        "cd_setor",
        "relative_position",
        "n_days",
        "weekday",
    ]


def test_multiple_horizons_keep_origins(raw):
    scored = evaluate_shapes(
        build_shape_days(raw),
        [ShapeCandidate(name="u", model="uniform")],
        ShapeSplit(min_train_cycles=1, horizon=2),
    )
    assert scored["CICLOS"].nunique() == 2
    assert scored["origin_cycle"].unique().tolist() == ["0"]


@pytest.mark.parametrize("n_days", [0, -1, 1.5, np.nan])
def test_invalid_query_windows(raw, n_days):
    days = build_shape_days(raw)
    windows = (
        days[["cd_setor", "CICLOS", "window_start", "n_days"]]
        .drop_duplicates()
        .assign(n_days=n_days)
    )
    with pytest.raises(ValueError, match="day counts"):
        predict_shape(days, windows)
