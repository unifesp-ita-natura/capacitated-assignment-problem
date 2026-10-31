"""Verify chosen hierarchy presets, closed-cycle recursion and conservative daily integration."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.forecast_combined.run import (
    CombinedConfig,
    _hierarchy_settings,
    _level_features,
    run,
)
from src.config.schema import ForecastConfig, NaiveParams
from src.forecasting.combined import forecast_combined
from src.forecasting.combined_inputs import prepare_histories, prepare_windows
from src.forecasting.hierarchy import forecast_distribution, forecast_hierarchy
from src.forecasting.hierarchy_config import HierarchyForecastParams
from src.forecasting.hierarchy_data import aggregate_nodes
from src.forecasting.hierarchy_future import _sector_shares, profile_settings
from src.forecasting.shape_evaluation import ShapeCandidate


@pytest.fixture
def raw():
    rows = []
    for cycle, opening in enumerate(pd.date_range("2026-01-01", periods=9, freq="7D"), 1):
        for sector, multiplier in [("01", 30), ("02", 10)]:
            rows.extend(_rows(cycle, opening, sector, multiplier))
    return pd.DataFrame(rows)


def _rows(cycle, opening, sector, multiplier):
    return [
        dict(
            cd_setor=sector,
            estado="SP",
            cidade="CIDADE",
            CD_RE="r",
            CD_GV="g",
            CICLOS=f"2026{cycle:02}",
            data_pedido=opening + pd.Timedelta(days=day),
            total_itens_mascarado=multiplier * (cycle + 1),
            total_pedidos_mascarado=1,
            total_volumes_mascarado=1,
            nm_ciclo="test",
            aa_ciclo=2026,
            dia_ciclo=day,
            **{"Dt Abertura": opening, "Dt Fechamento": opening + pd.Timedelta(days=2)},
        )
        for day in [0, 2]
    ]


@pytest.fixture
def windows():
    return pd.DataFrame(
        [
            dict(
                scenario_id="base",
                cd_setor=sector,
                CICLOS="202607",
                opening_date="2026-02-12",
                window_start="2026-02-12",
                window_end="2026-02-14",
            )
            for sector in ["01", "02"]
        ]
    )


def params(grouping="sector"):
    return HierarchyForecastParams(
        city_grouping=grouping,
        learning={"n_estimators": 3, "min_child_samples": 1, "num_leaves": 3},
    )


def predict(raw, windows, grouping="sector", as_of="2026-02-11"):
    return forecast_combined(
        raw, windows, as_of, 5, params(grouping), ShapeCandidate(model="sector", name="sector")
    )


def test_effective_presets_and_explicit_override():
    default = ForecastConfig().params
    assert isinstance(default, HierarchyForecastParams)
    assert default.city_grouping == "sector"
    assert default.learning.features == [
        "history",
        "calendar",
        "structure",
        "hierarchy",
        "intermittency",
        "city_recent_shares",
    ]
    assert "history" not in params("management").learning.features


def test_explicit_features_and_legacy_model_are_preserved():
    explicit = HierarchyForecastParams(learning={"features": ["calendar"]})
    assert explicit.learning.features == ["calendar"]
    assert isinstance(ForecastConfig(params={"model": "naive"}).params, NaiveParams)


def test_runner_default_and_effective_manifest():
    config = CombinedConfig(
        mode="validate",
        validation_cycles=["202607"],
        paths={"base_csv": "unused.csv", "run_dir": "unused"},
    )
    assert config.level.city_grouping == "sector"
    assert _hierarchy_settings(config)["learning"]["features"] == config.level.learning.features
    assert "node" in _level_features(config.level)


@pytest.mark.parametrize("grouping", ["sector", "management"])
def test_future_distribution_matches_backtest_candidate(raw, windows, grouping):
    history = prepare_histories(raw, pd.Timestamp("2026-02-11"), 5)
    query = (
        prepare_windows(windows)
        .merge(history.cities, on="cd_setor")
        .assign(CD_RE="r", CD_GV="g", items=0, closing_date=pd.NaT)
    )
    config = profile_settings(params(grouping), 5)
    past = aggregate_nodes(history.level, config.levels, grouping)
    target = aggregate_nodes(query, config.levels, grouping)
    expected = forecast_hierarchy(past, target, config).predictions["distribution_ets_lgbm"]
    pd.testing.assert_series_equal(forecast_distribution(past, target, config), expected)
    result = predict(raw, windows, grouping)
    np.testing.assert_allclose(
        result.level.aggregate_level_pred, result.level.level_node.map(expected)
    )


@pytest.mark.parametrize("grouping", ["sector", "management"])
def test_daily_totals_and_management_allocation(raw, windows, grouping):
    result = predict(raw, windows, grouping)
    daily = result.daily.groupby("serie").items_pred.sum().sort_index()
    np.testing.assert_allclose(daily, result.level.set_index("serie").level_pred.sort_index())
    totals = result.level.groupby("level_node").agg(
        total=("level_pred", "sum"),
        expected=("aggregate_level_pred", "first"),
        share=("sector_share", "sum"),
    )
    np.testing.assert_allclose(totals.total, totals.expected)
    np.testing.assert_allclose(totals.share, 1)


def test_management_shares_numeric_fallbacks():
    history = pd.DataFrame(
        dict(
            serie=["a", "b", "a", "b"],
            CICLOS=["1", "1", "2", "2"],
            opening_date=pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-08"] * 2),
            items=[30, 10, 0, 0],
        )
    )
    query = pd.DataFrame(dict(serie=["a", "b"], level_node=["group", "group"]))
    config = profile_settings(params("management"), 5).model_copy(update={"share_window": 1})
    np.testing.assert_allclose(_sector_shares(history, query, config), [0.75, 0.25])
    np.testing.assert_allclose(_sector_shares(history.assign(items=0), query, config), [0.5, 0.5])


def test_future_labels_and_other_scenarios_cannot_change_forecast(raw, windows):
    baseline = predict(raw, windows)
    changed = raw.copy()
    changed.loc[changed.CICLOS.ge("202607"), "total_itens_mascarado"] *= 1000
    pd.testing.assert_frame_equal(baseline.daily, predict(changed, windows).daily)
    alternative = windows.assign(scenario_id="later", window_start="2026-02-13")
    together = predict(raw, pd.concat([windows, alternative]))
    pd.testing.assert_frame_equal(
        baseline.daily, together.daily[together.daily.scenario_id.eq("base")].reset_index(drop=True)
    )


def test_recursive_path_keeps_only_projected_future_volumes(raw, windows):
    later = windows.assign(
        CICLOS="202608",
        opening_date="2026-02-19",
        window_start="2026-02-19",
        window_end="2026-02-21",
    )
    path = pd.concat([windows, later])
    baseline = predict(raw, path)
    changed = raw.copy()
    changed.loc[changed.CICLOS.ge("202607"), "total_itens_mascarado"] = 999999
    pd.testing.assert_frame_equal(baseline.daily, predict(changed, path).daily)
    assert baseline.daily.CICLOS.nunique() == 2


def _overlapping_history(raw):
    raw = raw.copy()
    raw.loc[raw.CICLOS.eq("202606"), "Dt Fechamento"] = pd.Timestamp("2026-02-15")
    raw.loc[raw.CICLOS.eq("202607"), "Dt Fechamento"] = pd.Timestamp("2026-02-22")
    latest_observation = (
        raw[raw.CICLOS.eq("202607")].head(1).assign(data_pedido=pd.Timestamp("2026-02-17"))
    )
    return pd.concat([raw, latest_observation], ignore_index=True)


def test_intermediate_cycle_has_older_closed_training_context(raw, windows):
    raw = _overlapping_history(raw)
    later = windows.assign(
        CICLOS="202608",
        opening_date="2026-02-19",
        window_start="2026-02-19",
        window_end="2026-02-21",
        emit=True,
    )
    path = pd.concat([windows.assign(emit=False, window_end="2026-02-22"), later])
    result = predict(raw, path, as_of="2026-02-18")
    assert set(result.daily.CICLOS) == {"202608"}


@pytest.mark.parametrize("grouping", ["sector", "management"])
def test_runner_saves_daily_forecast_and_effective_preset(tmp_path, raw, windows, grouping):
    base, schedule = tmp_path / "base.csv", tmp_path / "windows.csv"
    raw.to_csv(base, index=False)
    windows.to_csv(schedule, index=False)
    configuration = dict(
        as_of="2026-02-11",
        history_cycles=5,
        level=params(grouping).model_dump(),
        paths=dict(base_csv=str(base), windows_csv=str(schedule), run_dir=str(tmp_path / "output")),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    directory = run(path)
    record = json.loads((directory / "run_manifest.json").read_text())
    assert record["status"] == "completed"
    assert record["hierarchy_settings"]["city_grouping"] == grouping
    assert (directory / "daily_predictions.csv").is_file()


def test_missing_organization_rejected(raw, windows):
    with pytest.raises(ValueError, match="organization"):
        predict(raw.assign(CD_RE=None), windows)


def test_short_closed_history_rejected(raw, windows):
    raw = _overlapping_history(raw)
    raw = raw[raw.CICLOS.ne("202601")]
    later = windows.assign(
        CICLOS="202608",
        opening_date="2026-02-19",
        window_start="2026-02-19",
        window_end="2026-02-21",
        emit=True,
    )
    path = pd.concat([windows.assign(emit=False, window_end="2026-02-22"), later])
    with pytest.raises(ValueError, match="5 closed training cycles"):
        predict(raw, path, as_of="2026-02-18")


def test_subset_roster_does_not_inherit_other_sector_volume(raw, windows):
    result = predict(raw, windows[windows.cd_setor.eq("01")])
    scaled = raw.copy()
    scaled.loc[scaled.cd_setor.eq("02"), "total_itens_mascarado"] *= 100
    pd.testing.assert_frame_equal(
        result.daily, predict(scaled, windows[windows.cd_setor.eq("01")]).daily
    )
