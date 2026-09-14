"""Tests for the rolling-origin evaluation harness, the context driving every Strategy alike."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import ForecastCandidate

CYCLES = ["1", "2", "3", "4", "5", "6"]
OPENING_DATES = pd.to_datetime(
    ["2026-01-01", "2026-01-22", "2026-02-12", "2026-03-05", "2026-03-26", "2026-04-16"]
)
ITEMS_A = [100, 110, 120, 130, 140, 150]
ITEMS_B = [10, 10, 10, 10, 10, 10]


def _panel() -> pd.DataFrame:
    rows = []
    for cycle, date, a, b in zip(CYCLES, OPENING_DATES, ITEMS_A, ITEMS_B, strict=True):
        rows.append({"cd_setor": "A", "CICLOS": cycle, "items": a, "opening_date": date})
        rows.append({"cd_setor": "B", "CICLOS": cycle, "items": b, "opening_date": date})
    return pd.DataFrame(rows)


class _ConstantCandidate:
    """Predicts a fixed value for every (sector, cycle) asked about — error is hand-checkable."""

    name = "constant"

    def __init__(self, value: float, skip_sectors: frozenset[str] = frozenset()):
        self._value = value
        self._skip = skip_sectors

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        keep = targets[~targets["cd_setor"].isin(self._skip)]
        return pd.DataFrame(
            {"cd_setor": keep["cd_setor"], "CICLOS": keep["CICLOS"], "items_pred": self._value}
        )


class _SpyCandidate:
    """Records every (history, targets) call, so tests can inspect what the harness fed it."""

    name = "spy"

    def __init__(self):
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        self.calls.append((history.copy(), targets.copy()))
        return pd.DataFrame(
            {"cd_setor": targets["cd_setor"], "CICLOS": targets["CICLOS"], "items_pred": 0.0}
        )


def test_split_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="horizon"):
        RollingOriginSplit(horizon=0, min_train_cycles=1)


def test_split_rejects_non_positive_min_train_cycles():
    with pytest.raises(ValueError, match="min_train_cycles"):
        RollingOriginSplit(horizon=1, min_train_cycles=0)


def test_evaluate_produces_one_origin_per_valid_cutoff():
    split = RollingOriginSplit(horizon=1, min_train_cycles=3, window="expanding")

    result = evaluate(_ConstantCandidate(value=0), _panel(), split)

    # 6 cycles, min_train=3, horizon=1 -> origins at train sizes 3, 4, 5.
    assert len(result.origins) == 3
    assert [o.origin_cycle for o in result.origins] == ["3", "4", "5"]


def test_evaluate_never_lets_a_candidate_see_a_target_cycle():
    # Redundant with evaluation.py's own internal assertion by design: this
    # confirms it from the outside, against the actual data the harness
    # handed the candidate, not just that the assertion didn't fire.
    spy = _SpyCandidate()
    split = RollingOriginSplit(horizon=1, min_train_cycles=3, window="expanding")

    evaluate(spy, _panel(), split)

    assert len(spy.calls) == 3
    for history, targets in spy.calls:
        assert history["opening_date"].max() < targets["opening_date"].min()


def test_evaluate_computes_exact_mae_for_a_constant_candidate():
    split = RollingOriginSplit(horizon=1, min_train_cycles=3, window="expanding")

    result = evaluate(_ConstantCandidate(value=100), _panel(), split)
    summary = result.summary()

    # Targets are cycles 4,5,6: A = [130,140,150], B = [10,10,10] (constant
    # prediction is 100 throughout). Equal-weight-per-sector MAE:
    # sector A mean |actual-100| = mean(30,40,50) = 40
    # sector B mean |actual-100| = mean(90,90,90) = 90
    # overall = mean(40, 90) = 65, regardless of A and B having equal row counts.
    assert summary["mae"] == pytest.approx(65.0)
    assert summary["n_scored"] == 6
    assert summary["n_missing"] == 0

    scored = result.scored
    expected_rmse = np.sqrt(np.mean(scored["abs_error"] ** 2))
    expected_p90 = np.percentile(scored["abs_error"], 90)
    assert summary["rmse"] == pytest.approx(expected_rmse)
    assert summary["p90_abs_error"] == pytest.approx(expected_p90)


def test_evaluate_counts_a_skipped_sector_as_missing_not_silently_dropped():
    split = RollingOriginSplit(horizon=1, min_train_cycles=3, window="expanding")

    result = evaluate(_ConstantCandidate(value=0, skip_sectors=frozenset({"B"})), _panel(), split)
    summary = result.summary()

    assert summary["n_missing"] == 3  # sector B skipped at all 3 origins
    assert set(result.missing["cd_setor"]) == {"B"}
    assert set(result.scored["cd_setor"]) == {"A"}  # never silently mixed into A's error


def test_ranking_metric_matches_the_primary_metric_in_summary():
    result = evaluate(
        _ConstantCandidate(value=100), _panel(), RollingOriginSplit(horizon=1, min_train_cycles=3)
    )

    assert result.ranking_metric == result.summary()["mae"]


def test_summary_raises_a_clear_error_when_nothing_was_scored():
    # min_train_cycles leaves no room for any origin given 6 cycles and horizon=1.
    split = RollingOriginSplit(horizon=1, min_train_cycles=6)

    result = evaluate(_ConstantCandidate(value=0), _panel(), split)

    with pytest.raises(ValueError, match="nothing was scored"):
        result.summary()


def test_sliding_window_uses_only_the_most_recent_cycles():
    spy = _SpyCandidate()
    split = RollingOriginSplit(horizon=1, min_train_cycles=2, window="sliding")

    evaluate(spy, _panel(), split)

    for history, _ in spy.calls:
        assert history["CICLOS"].nunique() <= 2


def test_forecast_candidate_protocol_accepts_the_constant_candidate():
    # ForecastCandidate is a Protocol (structural typing) — this documents
    # that a plain class satisfying the shape is a valid Strategy without
    # inheriting from anything.
    candidate: ForecastCandidate = _ConstantCandidate(value=0)
    assert candidate.name == "constant"


def test_missing_is_an_empty_frame_with_the_right_columns_when_there_are_no_origins():
    # min_train_cycles=6 with 6 cycles and horizon=1 leaves no valid origin.
    split = RollingOriginSplit(horizon=1, min_train_cycles=6)

    result = evaluate(_ConstantCandidate(value=0), _panel(), split)

    assert result.origins == []
    assert result.missing.empty
    assert list(result.missing.columns) == ["cd_setor", "CICLOS", "origin_cycle"]


def test_mase_is_nan_for_a_fold_with_no_two_cycle_history_but_mae_stays_valid():
    # min_train_cycles=1: the first origin's training window has exactly one
    # cycle per sector, so there's no one-step diff to compute a naive
    # in-sample benchmark from (see _naive_in_sample_mae's docstring).
    split = RollingOriginSplit(horizon=1, min_train_cycles=1)

    result = evaluate(_ConstantCandidate(value=100), _panel(), split)
    first_fold = result.origins[0]

    assert np.isnan(first_fold.naive_in_sample_mae)
    assert not np.isnan(result.summary()["mae"])
