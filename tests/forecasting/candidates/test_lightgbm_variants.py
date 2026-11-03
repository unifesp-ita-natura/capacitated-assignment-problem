"""Tests for the pooled LightGBM variants: objective, external UF features and dynamics."""

from __future__ import annotations

import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import (
    LightGBMDynamicsParams,
    LightGBMExternalParams,
    LightGBMObjectiveParams,
)
from src.forecasting.candidates.lightgbm_dynamics import _MOMENTUM_COLUMNS, _QUARTER_COLUMN
from src.forecasting.model import REGISTRY

CYCLES = ["202601", "202602", "202603", "202604", "202605", "202606"]
DATES = pd.to_datetime(
    ["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-04", "2026-06-01"]
)
SMALL = {"n_estimators": 20, "num_leaves": 3}
LAGS = {"lags": [1, 2], "rolling_windows": [3]}
VALUES = {"A": [100, 120, 110, 130, 125, 135], "B": [10, 12, 11, 13, 12, 14]}


def _history(extra: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    rows = []
    for sector, values in VALUES.items():
        for cycle, date, value in zip(CYCLES, DATES, values, strict=True):
            row = {"serie": sector, "CICLOS": cycle, "items": value, "opening_date": date}
            rows.append(row | (extra or {}).get(sector, {}))
    return pd.DataFrame(rows).assign(
        window_start=lambda frame: frame["opening_date"], cycle_days=21
    )


def _targets(cycles: list[str], dates: list[str]) -> pd.DataFrame:
    rows = [
        {"serie": sector, "CICLOS": cycle, "opening_date": pd.Timestamp(date)}
        for cycle, date in zip(cycles, dates, strict=True)
        for sector in VALUES
    ]
    return pd.DataFrame(rows).assign(
        window_start=lambda frame: frame["opening_date"], cycle_days=21
    )


TARGETS = _targets(["202607", "202608"], ["2026-07-06", "2026-08-03"])


@pytest.mark.parametrize("objective", ["tweedie", "quantile"])
def test_objective_candidate_trains_with_the_chosen_objective(objective):
    params = LightGBMObjectiveParams(objective=objective, **LAGS, **SMALL)
    candidate = REGISTRY.build(params)

    predictions = candidate.fit_predict(_history(), TARGETS)

    assert candidate._settings()["objective"] == objective
    assert candidate.name == f"lightgbm_{objective}"
    assert set(predictions["serie"]) == {"A", "B"}
    assert (predictions["items_pred"] >= 0).all()


def test_objective_settings_carry_only_the_parameters_of_the_chosen_objective():
    tweedie = REGISTRY.build(LightGBMObjectiveParams(objective="tweedie"))._settings()
    quantile = REGISTRY.build(
        LightGBMObjectiveParams(objective="quantile", quantile_alpha=0.8)
    )._settings()

    assert "tweedie_variance_power" in tweedie
    assert "alpha" not in tweedie
    assert quantile["alpha"] == 0.8
    assert "tweedie_variance_power" not in quantile


def test_external_candidate_uses_the_uf_columns_through_a_multi_step_forecast():
    extra = {
        "A": {"uf_populacao": 1.0e6, "uf_densidade": 50.0},
        "B": {"uf_populacao": 2.0e6, "uf_densidade": 80.0},
    }
    params = LightGBMExternalParams(
        external_features=["uf_populacao", "uf_densidade"], **LAGS, **SMALL
    )
    candidate = REGISTRY.build(params)

    predictions = candidate.fit_predict(_history(extra), TARGETS)

    assert candidate._columns[-2:] == ["uf_populacao", "uf_densidade"]
    assert candidate._carried_columns == ["uf_populacao", "uf_densidade"]
    assert len(predictions) == 4  # 2 sectors x 2 cycles
    assert predictions["items_pred"].notna().all()


def test_external_candidate_fails_clearly_when_the_panel_lacks_the_columns():
    candidate = REGISTRY.build(LightGBMExternalParams(**LAGS, **SMALL))

    with pytest.raises(KeyError):
        candidate.fit_predict(_history(), TARGETS)


def _derived_frame() -> pd.DataFrame:
    candidate = REGISTRY.build(LightGBMDynamicsParams(lags=[1, 2], rolling_windows=[3], **SMALL))
    featured = pd.DataFrame(
        {
            "lag_1": [10.0, 4.0],
            "lag_2": [6.0, 4.0],
            "rolling_mean_3": [0.0, 8.0],
            "opening_date": pd.to_datetime(["2026-02-02", "2026-08-03"]),
        }
    )
    return candidate._derive(featured)


def test_dynamics_derives_the_momentum_difference():
    assert list(_derived_frame()["mom_diff_1_2"]) == [4.0, 0.0]


def test_dynamics_momentum_ratio_is_nan_not_inf_on_a_zero_mean():
    ratio = _derived_frame()["mom_ratio_1_mean3"]

    assert pd.isna(ratio[0])
    assert ratio[1] == 0.5


def test_dynamics_derives_the_opening_quarter():
    assert list(_derived_frame()[_QUARTER_COLUMN]) == [1, 3]


def test_dynamics_feeds_momentum_columns_to_the_model():
    candidate = REGISTRY.build(LightGBMDynamicsParams(lags=[1, 2], rolling_windows=[3], **SMALL))

    assert set(_MOMENTUM_COLUMNS) <= set(candidate._columns)


def test_dynamics_candidate_predicts_every_sector():
    params = LightGBMDynamicsParams(**LAGS, **SMALL)

    predictions = REGISTRY.build(params).fit_predict(_history(), TARGETS)

    assert len(predictions) == 4
    assert (predictions["items_pred"] >= 0).all()


def test_dynamics_momentum_requires_the_lags_and_window_it_is_built_from():
    candidate = REGISTRY.build(LightGBMDynamicsParams(lags=[1], rolling_windows=[2], **SMALL))

    with pytest.raises(ValueError, match="momentum needs"):
        candidate.fit_predict(_history(), TARGETS)


def test_dynamics_name_lists_the_enabled_extras():
    params = LightGBMDynamicsParams(momentum=False, calendar_quarter=True)

    assert REGISTRY.build(params).name.endswith("+quarter")
