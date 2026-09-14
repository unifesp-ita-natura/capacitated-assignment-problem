"""Tests for the pooled LightGBM candidate, which implements the Strategy without the Adapter."""

from __future__ import annotations

import pandas as pd

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import LightGBMParams
from src.forecasting.model import REGISTRY

PARAMS = LightGBMParams(lags=[1, 2], rolling_windows=[2], n_estimators=20, num_leaves=3)

CYCLES = ["202601", "202602", "202603", "202604", "202605"]
DATES = pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-04"])


def _history(sectors: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for sector, values in sectors.items():
        for cycle, date, value in zip(CYCLES[: len(values)], DATES, values, strict=False):
            rows.append({"cd_setor": sector, "CICLOS": cycle, "items": value, "opening_date": date})
    return pd.DataFrame(rows)


def _targets(sectors: list[str], cycles: list[str], dates: list[str]) -> pd.DataFrame:
    rows = [
        {"cd_setor": sector, "CICLOS": cycle, "opening_date": pd.Timestamp(date)}
        for cycle, date in zip(cycles, dates, strict=True)
        for sector in sectors
    ]
    return pd.DataFrame(rows)


def test_pooled_candidate_predicts_every_sector_with_history():
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606"], ["2026-06-01"])

    predictions = REGISTRY.build(PARAMS).fit_predict(history, targets)

    assert set(predictions["cd_setor"]) == {"A", "B"}
    assert (predictions["items_pred"] >= 0).all()


def test_pooled_candidate_forecasts_each_step_of_a_longer_horizon():
    # Multi-step is recursive: the second cycle's lag_1 is the first cycle's
    # prediction, because its actual value is what the harness withholds.
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606", "202607"], ["2026-06-01", "2026-07-06"])

    predictions = REGISTRY.build(PARAMS).fit_predict(history, targets)

    assert len(predictions) == 4  # 2 sectors x 2 cycles
    assert set(predictions["CICLOS"]) == {"202606", "202607"}


def test_returns_nothing_when_history_is_too_short_to_build_one_complete_row():
    # 2 cycles can't fill lag_2 plus a 2-cycle rolling mean, so there's no
    # training row at all — the candidate declines instead of inventing.
    history = _history({"A": [100, 120], "B": [10, 12]})
    targets = _targets(["A", "B"], ["202603"], ["2026-03-02"])

    predictions = REGISTRY.build(PARAMS).fit_predict(history, targets)

    assert predictions.empty
    assert list(predictions.columns) == ["cd_setor", "CICLOS", "items_pred"]


def test_a_sector_with_no_prior_cycle_is_skipped_not_guessed():
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B", "NEW"], ["202606"], ["2026-06-01"])

    predictions = REGISTRY.build(PARAMS).fit_predict(history, targets)

    assert "NEW" not in set(predictions["cd_setor"])


def test_predictions_are_reproducible_for_a_fixed_random_state():
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606"], ["2026-06-01"])

    first = REGISTRY.build(PARAMS).fit_predict(history, targets)
    second = REGISTRY.build(PARAMS).fit_predict(history, targets)

    pd.testing.assert_frame_equal(first, second)


def test_candidate_name_records_the_shape_of_the_model():
    assert REGISTRY.build(PARAMS).name == "lightgbm:20x3"


def test_returns_nothing_when_no_target_sector_has_any_history():
    # The pooled model trains fine on A and B, but the only sector asked
    # about is brand new — there's no lag to predict it from, so nothing is
    # returned and the harness reports it as missing.
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["NEW"], ["202606"], ["2026-06-01"])

    predictions = REGISTRY.build(PARAMS).fit_predict(history, targets)

    assert predictions.empty
