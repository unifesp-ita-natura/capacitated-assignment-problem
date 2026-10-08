"""Tests for the pooled LightGBM candidate, which implements the Strategy without the Adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import LightGBMParams
from src.forecasting.candidates.lightgbm import (
    _as_model_frame,
    _carry_organization,
    _category_types,
)
from src.forecasting.model import QUERY_KEYS, REGISTRY

PARAMS = LightGBMParams(lags=[1, 2], rolling_windows=[2], n_estimators=20, num_leaves=3)

CYCLES = ["202601", "202602", "202603", "202604", "202605"]
DATES = pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-04"])


def _history(sectors: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for sector, values in sectors.items():
        for cycle, date, value in zip(CYCLES[: len(values)], DATES, values, strict=False):
            rows.append({"cd_setor": sector, "CICLOS": cycle, "items": value, "opening_date": date})
    return pd.DataFrame(rows).assign(
        window_start=lambda frame: frame["opening_date"], cycle_days=21
    )


def _targets(sectors: list[str], cycles: list[str], dates: list[str]) -> pd.DataFrame:
    rows = [
        {"cd_setor": sector, "CICLOS": cycle, "opening_date": pd.Timestamp(date)}
        for cycle, date in zip(cycles, dates, strict=True)
        for sector in sectors
    ]
    return pd.DataFrame(rows).assign(
        window_start=lambda frame: frame["opening_date"], cycle_days=21
    )


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
    assert list(predictions.columns) == [*QUERY_KEYS, "items_pred"]


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


PARAMS_WITH_COUNTS = LightGBMParams(
    lags=[1, 2], rolling_windows=[2], n_estimators=20, num_leaves=3, companion_lags=[1, 2]
)


def _history_with_counts(sectors: dict[str, list[float]]) -> pd.DataFrame:
    history = _history(sectors)
    return history.assign(
        orders=(history["items"] / 10).round(),
        volumes=(history["items"] / 5).round(),
    )


def test_pooled_candidate_uses_order_counts_without_needing_them_for_the_target_cycle():
    # The target cycle's own order count is as unknown as its item count, so
    # `targets` carries neither — only the lags, which reach into observed
    # cycles, may reach the model.
    history = _history_with_counts({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606"], ["2026-06-01"])

    predictions = REGISTRY.build(PARAMS_WITH_COUNTS).fit_predict(history, targets)

    assert set(predictions["cd_setor"]) == {"A", "B"}
    assert (predictions["items_pred"] >= 0).all()


def test_order_counts_candidate_forecasts_a_longer_horizon_recursively():
    history = _history_with_counts({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606", "202607"], ["2026-06-01", "2026-07-06"])

    predictions = REGISTRY.build(PARAMS_WITH_COUNTS).fit_predict(history, targets)

    assert len(predictions) == 4  # 2 sectors x 2 cycles


def test_candidate_name_distinguishes_the_order_count_variant():
    # The comparison table keys candidates by name, so the two rows of the
    # panel_order_counts experiment have to be tellable apart.
    assert REGISTRY.build(PARAMS_WITH_COUNTS).name == "lightgbm:20x3+orders"
    assert REGISTRY.build(PARAMS).name == "lightgbm:20x3"


def test_an_explicit_label_overrides_the_generated_name():
    # Two variants can differ only in a field the generated name doesn't
    # carry — rolling windows, say — and the comparison table keys rows by
    # name, so the config has to be able to name them.
    labelled = LightGBMParams(
        lags=[1, 2], rolling_windows=[2], n_estimators=20, num_leaves=3, label="short window"
    )

    assert REGISTRY.build(labelled).name == "short window"


RATIO_PARAMS = LightGBMParams(
    lags=[1, 2], rolling_windows=[2], n_estimators=20, num_leaves=3, target="ratio"
)


def test_ratio_target_still_returns_predictions_in_items():
    # The model fits items/level_ref, but what leaves the candidate must be
    # an item count, or the harness would score a ratio against a level.
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]})
    targets = _targets(["A", "B"], ["202606"], ["2026-06-01"])

    predictions = REGISTRY.build(RATIO_PARAMS).fit_predict(history, targets)

    assert set(predictions["cd_setor"]) == {"A", "B"}
    # A lives near 115 and B near 12; a ratio left untransformed would land
    # both near 1.
    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector["A"] > 50
    assert by_sector["B"] < 50


def test_ratio_target_is_recorded_in_the_candidate_name():
    assert REGISTRY.build(RATIO_PARAMS).name == "lightgbm:20x3:ratio"


def test_scenarios_of_one_cycle_differ_by_their_window():
    # 20 sectors whose items depend only on the window: 21-day cycles sell
    # 100, 14-day ones sell 50. Asked about both lengths for the next cycle,
    # the model must tell them apart.
    # The lengths are random, so no lag can anticipate them.
    lengths = np.random.default_rng(0).choice([14, 21], size=(20, 12))
    rows = []
    for s in range(20):
        for n in range(12):
            days = int(lengths[s, n])
            start = pd.Timestamp("2025-01-06") + pd.Timedelta(days=21 * n)
            rows.append(
                {
                    "cd_setor": f"S{s}",
                    "CICLOS": f"2025{n:02d}",
                    "items": 100.0 if days == 21 else 50.0,
                    "opening_date": start,
                    "window_start": start,
                    "cycle_days": days,
                }
            )
    params = LightGBMParams(
        lags=[1],
        rolling_windows=[2],
        n_estimators=50,
        num_leaves=3,
        learning_rate=0.3,
        min_child_samples=2,
    )
    next_start = pd.Timestamp("2025-01-06") + pd.Timedelta(days=21 * 12)
    targets = pd.DataFrame(
        {
            "cd_setor": ["S0", "S0"],
            "CICLOS": "202599",
            "opening_date": next_start,
            "window_start": next_start,
            "cycle_days": [21, 14],
        }
    )

    predictions = REGISTRY.build(params).fit_predict(pd.DataFrame(rows), targets)

    by_days = predictions.set_index("cycle_days")["items_pred"]
    assert by_days[21] > by_days[14] + 25


def test_organization_encoding_is_shared_and_unseen_categories_are_missing():
    history = pd.DataFrame({"cd_setor": ["A", "B"], "CD_RE": ["02", "01"]})
    types = _category_types(history, ["cd_setor", "CD_RE"])
    targets = pd.DataFrame({"cd_setor": ["B", "A"], "CD_RE": ["01", "99"]})
    encoded = _as_model_frame(targets, types)
    assert encoded.CD_RE.cat.categories.tolist() == ["01", "02"]
    assert encoded.CD_RE.cat.codes.tolist() == [0, -1]


def test_prediction_organization_comes_from_latest_history_by_sector():
    history = _history({"A": [100, 120, 110], "B": [10, 12, 11]}).assign(CD_RE="01")
    history.loc[(history.cd_setor == "A") & (history.CICLOS == "202603"), "CD_RE"] = "02"
    pending = _targets(["A", "B", "NEW"], ["202604"], ["2026-04-02"]).assign(CD_RE="99")
    carried = _carry_organization(pending, history.iloc[::-1], ["CD_RE"]).set_index("cd_setor")
    assert carried.CD_RE.loc["A"] == "02"
    assert carried.CD_RE.loc["B"] == "01"
    assert pd.isna(carried.CD_RE.loc["NEW"])


@pytest.mark.parametrize("target", ["level", "ratio"])
def test_organization_candidate_predicts_multiple_steps_without_target_metadata(target):
    params = PARAMS.model_copy(
        update={"categorical_features": ["CD_RE", "CD_GV"], "target": target}
    )
    history = _history({"A": [100, 120, 110, 130, 125], "B": [10, 12, 11, 13, 12]}).assign(
        CD_RE=lambda frame: frame.cd_setor.map({"A": "01", "B": "02"}),
        CD_GV=lambda frame: frame.cd_setor.map({"A": "001", "B": "002"}),
    )
    targets = _targets(["A", "B"], ["202606", "202607"], ["2026-06-01", "2026-07-06"])
    predictions = REGISTRY.build(params).fit_predict(history, targets)
    assert len(predictions) == 4
    assert predictions.items_pred.notna().all()


def test_organization_candidate_name_distinguishes_feature_selection():
    params = PARAMS.model_copy(update={"categorical_features": ["CD_RE", "CD_GV"]})
    assert REGISTRY.build(params).name == "lightgbm:20x3+CD_RE+CD_GV"
