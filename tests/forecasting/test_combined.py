"""Verify point-in-time level-shape integration, scenario isolation and fixed daily replay."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.forecast_combined.run import run
from src.config.schema import ArimaParams, ETSParams, LightGBMParams, NaiveParams
from src.forecasting.combined import combine_predictions, forecast_combined
from src.forecasting.combined_inputs import prepare_histories, prepare_windows
from src.forecasting.combined_validation import summarize_cities, validate_combined
from src.forecasting.shape_evaluation import ShapeCandidate

LEVEL = NaiveParams(strategy="mean")
SHAPE = ShapeCandidate(name="sector", model="sector", n_bins=3)


@pytest.fixture
def raw():
    rows = []
    for cycle, start in enumerate(pd.date_range("2026-01-01", periods=4, freq="7D"), 1):
        rows.extend(_city_rows(cycle, start, "MG", 10))
        rows.extend(_city_rows(cycle, start, "GO", 20))
    return pd.DataFrame(rows)


def _city_rows(cycle, start, state, multiplier):
    return [
        dict(
            cd_setor="001",
            estado=state,
            cidade="HOMONIMA",
            CICLOS=f"2026{cycle:02}",
            data_pedido=start + pd.Timedelta(days=day),
            total_itens_mascarado=multiplier * (day + 1),
            total_pedidos_mascarado=1,
            total_volumes_mascarado=1,
            CD_RE="r",
            CD_GV="g",
            nm_ciclo="test",
            aa_ciclo=2026,
            dia_ciclo=day,
            **{"Dt Abertura": start, "Dt Fechamento": start + pd.Timedelta(days=2)},
        )
        for day in range(3)
    ]


@pytest.fixture
def windows():
    return pd.DataFrame(
        [
            dict(
                scenario_id="base",
                cd_setor="001",
                CICLOS="202604",
                opening_date="2026-01-22",
                window_start="2026-01-22",
                window_end="2026-01-24",
            )
        ]
    )


def predict(raw, windows, **options):
    return forecast_combined(
        raw,
        windows,
        options.get("as_of", "2026-01-21"),
        options.get("history_cycles", 2),
        options.get("level", LEVEL),
        options.get("shape", SHAPE),
    )


def test_daily_city_totals_and_values(raw, windows):
    result = predict(raw, windows)
    assert len(result.daily) == 6
    assert set(result.daily["estado"]) == {"MG", "GO"}
    np.testing.assert_allclose(
        result.daily.loc[result.daily.estado.eq("MG"), "items_pred"], [10, 20, 30]
    )
    totals = result.daily.groupby("serie").items_pred.sum().sort_index()
    np.testing.assert_allclose(totals, result.level.set_index("serie").level_pred.sort_index())


def test_future_demand_cannot_change_forecast(raw, windows):
    baseline = predict(raw, windows)
    changed = raw.copy()
    changed.loc[changed.CICLOS.eq("202604"), "total_itens_mascarado"] *= 1000
    pd.testing.assert_frame_equal(baseline.daily, predict(changed, windows).daily)


def test_scenarios_have_separate_day_counts(raw, windows):
    alternative = windows.assign(
        scenario_id="later", window_start="2026-01-23", window_end="2026-01-24"
    )
    result = predict(raw, pd.concat([windows, alternative]))
    assert result.daily.groupby("scenario_id").size().to_dict() == {"base": 6, "later": 4}
    np.testing.assert_allclose(
        result.shape.groupby(["scenario_id", "cd_setor", "CICLOS"]).share_pred.sum(), 1
    )


def test_multiple_cycles_keep_their_own_windows(raw, windows):
    next_cycle = windows.assign(
        CICLOS="202605",
        opening_date="2026-01-29",
        window_start="2026-01-29",
        window_end="2026-01-31",
    )
    result = predict(raw, pd.concat([next_cycle, windows]))
    assert result.daily.CICLOS.nunique() == 2
    assert result.daily.groupby("CICLOS").relative_day.max().tolist() == [3, 3]


def test_open_cycles_are_excluded_and_require_intermediate_windows(raw, windows):
    raw.loc[raw.CICLOS.eq("202603"), "Dt Fechamento"] = pd.Timestamp("2026-01-25")
    with pytest.raises(ValueError, match="intermediate"):
        predict(raw, windows)


def test_intermediate_cycles_are_forecast_but_not_emitted(raw, windows):
    raw.loc[raw.CICLOS.eq("202603"), "Dt Fechamento"] = pd.Timestamp("2026-01-25")
    bridge = windows.assign(
        CICLOS="202603",
        opening_date="2026-01-15",
        window_start="2026-01-15",
        window_end="2026-01-25",
        emit=False,
    )
    result = predict(raw, pd.concat([windows.assign(emit=True), bridge]))
    assert set(result.daily.CICLOS) == {"202604"}
    assert result.training_cycles == ["202601", "202602"]


@pytest.mark.parametrize("as_of", ["2026-01-22", "2026-01-23"])
def test_forecast_must_precede_first_cycle_opening(raw, windows, as_of):
    with pytest.raises(ValueError, match="before the first opening"):
        predict(raw, windows, as_of=as_of)


def test_insufficient_history_rejected(raw, windows):
    with pytest.raises(ValueError, match="complete closed cycles"):
        predict(raw, windows, history_cycles=20)


def test_unknown_sector_rejected(raw, windows):
    with pytest.raises(ValueError, match="no known city"):
        predict(raw, windows.assign(cd_setor="unknown"))


def test_missing_level_is_not_replaced_by_zero(raw, windows, monkeypatch):
    class Missing:
        name = "missing"

        def fit_predict(self, history, queries):
            return queries[["serie", "CICLOS", "window_start", "cycle_days"]].assign(
                items_pred=np.nan
            )

    monkeypatch.setattr("src.forecasting.combined.REGISTRY.build", lambda params: Missing())
    result = predict(raw, windows)
    assert result.daily["items_pred"].isna().all()
    assert result.daily["level_status"].eq("missing").all()


def test_invalid_shares_rejected(raw, windows):
    result = predict(raw, windows)
    broken = result.shape.assign(share_pred=0.1)
    with pytest.raises(ValueError, match="sum to one"):
        combine_predictions(result.level, broken)


def test_lightgbm_shape_is_combined_with_predicted_city_totals(raw, windows):
    params = ShapeCandidate(
        name="weekday",
        model="lightgbm",
        learning={"features": ["weekday", "CD_RE"], "n_estimators": 2},
    )
    result = predict(raw, windows, shape=params)
    np.testing.assert_allclose(result.shape.share_pred.sum(), 1)
    np.testing.assert_allclose(
        result.daily.groupby("serie").items_pred.sum().sort_index(),
        result.level.set_index("serie").level_pred.sort_index(),
    )


@pytest.mark.parametrize(
    "params",
    [
        ETSParams(alpha=0.3, cycle_days_exponent=0.0),
        ArimaParams(order=(0, 0, 0), trend="c"),
        LightGBMParams(
            n_estimators=2,
            lags=[1],
            rolling_windows=[1],
            min_child_samples=1,
            num_leaves=2,
            categorical_features=["estado"],
        ),
    ],
)
def test_supported_level_families_use_combined_contract(raw, windows, params):
    result = predict(raw, windows, level=params)
    assert result.daily.items_pred.notna().all()
    np.testing.assert_allclose(
        result.daily.groupby("serie").items_pred.sum().sort_index(),
        result.level.set_index("serie").level_pred.sort_index(),
    )


def test_requested_shape_organization_must_exist(raw, windows):
    params = ShapeCandidate(name="organization", model="lightgbm")
    with pytest.raises(ValueError, match="missing shape features"):
        predict(raw.drop(columns=["CD_RE", "CD_GV"]), windows, shape=params)


def test_shape_window_mismatch_rejected(raw, windows):
    result = predict(raw, windows)
    shifted = result.shape.assign(window_start=pd.Timestamp("2026-01-23"))
    with pytest.raises(ValueError, match="no matching shape"):
        combine_predictions(result.level, shifted)


def test_negative_level_rejected(raw, windows):
    result = predict(raw, windows)
    with pytest.raises(ValueError, match="nonnegative"):
        combine_predictions(result.level.assign(level_pred=-1), result.shape)


def test_reused_cycle_code_rejected(raw, windows):
    with pytest.raises(ValueError, match="cycle codes"):
        predict(raw, windows.assign(CICLOS="202603"))


@pytest.mark.parametrize("as_of", ["NaT", "2026-01-21T00:00:00Z"])
def test_invalid_forecast_timestamp_rejected(raw, windows, as_of):
    with pytest.raises(ValueError, match="local timestamp"):
        predict(raw, windows, as_of=as_of)


def test_duplicate_windows_rejected(windows):
    with pytest.raises(ValueError, match="one window"):
        prepare_windows(pd.concat([windows, windows]))


def test_invalid_window_dates_rejected(windows):
    with pytest.raises(ValueError, match="ordered|require opening_date"):
        prepare_windows(windows.assign(window_end="2026-01-01"))


def test_zero_city_cycles_remain_in_history(raw):
    keep = ~(raw.CICLOS.eq("202603") & raw.estado.eq("MG"))
    history = prepare_histories(raw[keep], pd.Timestamp("2026-01-21"), 2)
    quiet = history.level[history.level.CICLOS.eq("202603") & history.level.estado.eq("MG")]
    assert quiet["items"].tolist() == [0]


def test_fixed_validation_scores_predicted_totals(raw):
    result = validate_combined(raw, ["202604"], 2, LEVEL, SHAPE)
    assert len(result.predictions) == 6
    assert result.metrics.iloc[0]["mae"] == pytest.approx(0, abs=1e-10)
    assert result.metrics.iloc[0]["actual_volume_coverage"] == pytest.approx(1)


def test_validation_exposes_new_cities(raw):
    extra = raw[raw.CICLOS.eq("202604") & raw.estado.eq("MG")].assign(cidade="NEW")
    result = validate_combined(pd.concat([raw, extra]), ["202604"], 2, LEVEL, SHAPE)
    metrics = result.metrics.iloc[0]
    assert metrics["unpredicted_actual_items"] == 60
    assert metrics["actual_volume_coverage"] == pytest.approx(0.75)


def test_city_metrics_without_prediction_are_undefined(raw):
    extra = raw[raw.CICLOS.eq("202604") & raw.estado.eq("MG")].assign(cidade="NEW")
    result = validate_combined(pd.concat([raw, extra]), ["202604"], 2, LEVEL, SHAPE)
    city = summarize_cities(result.predictions).set_index("cidade").loc["NEW"]
    assert pd.isna(city["mae"])
    assert city["n_scored_days"] == 0


def test_validation_zero_days_are_scored(raw):
    keep = ~(raw.CICLOS.eq("202604") & raw.data_pedido.eq(pd.Timestamp("2026-01-23")))
    result = validate_combined(raw[keep], ["202604"], 2, LEVEL, SHAPE)
    zeros = result.predictions[result.predictions.date.eq(pd.Timestamp("2026-01-23"))]
    assert zeros.actual.eq(0).all()
    assert zeros.items_pred.gt(0).all()


def test_validation_rejects_incomplete_cycles(raw):
    with pytest.raises(ValueError, match="missing or incomplete"):
        validate_combined(raw, ["202605"], 2, LEVEL, SHAPE)


def _configuration(tmp_path, raw, windows, mode):
    base, schedule = tmp_path / "base.csv", tmp_path / "windows.csv"
    raw.to_csv(base, index=False)
    windows.to_csv(schedule, index=False)
    config = dict(
        mode=mode,
        history_cycles=2,
        level={"model": "naive", "strategy": "mean"},
        shape={"name": "sector", "model": "sector", "n_bins": 3},
        paths={"base_csv": str(base), "run_dir": str(tmp_path / "output")},
    )
    return _mode_configuration(config, schedule, mode)


def _mode_configuration(config, schedule, mode):
    if mode == "forecast":
        return dict(
            config, as_of="2026-01-21", paths=dict(config["paths"], windows_csv=str(schedule))
        )
    return dict(config, validation_cycles=["202604"])


@pytest.mark.parametrize(
    "mode,artifact", [("forecast", "daily_predictions.csv"), ("validate", "daily_validation.csv")]
)
def test_runner_archives_and_refuses_overwrite(tmp_path, raw, windows, mode, artifact):
    config = _configuration(tmp_path, raw, windows, mode)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    directory = run(path)
    assert (directory / artifact).is_file()
    record = json.loads((directory / "run_manifest.json").read_text())
    assert record["status"] == "completed"
    with pytest.raises(FileExistsError):
        run(path)


def test_runner_records_failure(tmp_path, raw, windows):
    config = _configuration(tmp_path, raw, windows, "forecast")
    config["history_cycles"] = 30
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="closed cycles"):
        run(path)
    record = json.loads((tmp_path / "output" / "run_manifest.json").read_text())
    assert record["status"] == "failed"
