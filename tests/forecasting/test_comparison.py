"""Tests for the common-subset comparison that keeps candidate rankings honest."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.comparison import common_scored_keys, compare
from src.forecasting.evaluation import EvaluationResult, OriginResult


def _result(name: str, rows: list[tuple[str, str, float, float]]) -> EvaluationResult:
    """Build an EvaluationResult directly from (sector, cycle, actual, predicted) rows."""
    scored = pd.DataFrame(
        [
            {
                "cd_setor": sector,
                "CICLOS": cycle,
                "actual": actual,
                "items_pred": predicted,
                "abs_error": abs(actual - predicted),
                "origin_cycle": "o1",
            }
            for sector, cycle, actual, predicted in rows
        ]
    )
    origin = OriginResult(
        origin_cycle="o1",
        target_cycles=("c1",),
        scored=scored,
        missing=pd.DataFrame(columns=["cd_setor", "CICLOS", "origin_cycle"]),
        naive_in_sample_mae=1.0,
    )
    return EvaluationResult(candidate_name=name, origins=[origin])


def test_common_keys_are_the_intersection_across_candidates():
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 40)])
    narrow = _result("narrow", [("A", "c1", 100, 95)])

    keys = common_scored_keys([wide, narrow])

    assert keys == {("A", "c1", "o1")}


def test_common_keys_are_empty_without_results():
    assert common_scored_keys([]) == set()


def test_comparison_scores_every_candidate_on_the_same_points():
    # `narrow` skipped sector B, whose error `wide` had to carry. Ranking on
    # each candidate's own coverage would reward the skipping.
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 0)])
    narrow = _result("narrow", [("A", "c1", 100, 80)])

    table = compare([wide, narrow]).set_index("candidate")

    assert table.loc["wide", "n_scored_own"] == 2
    assert table.loc["wide", "n_scored_common"] == 1
    assert table.loc["wide", "mae_common"] == 10  # sector A only
    assert table.loc["narrow", "mae_common"] == 20


def test_comparison_ranks_by_the_common_subset_not_by_own_coverage():
    # `wide` looks worse on its own numbers (it carries sector B's big miss)
    # but is the better candidate on the points both actually predicted.
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 0)])
    narrow = _result("narrow", [("A", "c1", 100, 80)])

    table = compare([wide, narrow])

    assert table["mae_own"].iloc[0] > table["mae_own"].iloc[1]  # ranked first despite worse own MAE
    assert table["candidate"].iloc[0] == "wide"


def test_comparison_rejects_an_empty_candidate_list():
    with pytest.raises(ValueError, match="nothing to compare"):
        compare([])


def test_a_candidate_that_scored_nothing_gets_a_nan_common_error():
    scored_nothing = EvaluationResult(candidate_name="empty", origins=[])
    wide = _result("wide", [("A", "c1", 100, 90)])

    table = compare([wide, scored_nothing]).set_index("candidate")

    assert pd.isna(table.loc["empty", "mae_common"])


def _with_daily(result: EvaluationResult, cd_day_error: float) -> EvaluationResult:
    """Attach one CD-day row per scored point, off by `cd_day_error` items."""
    origin = result.origins[0]
    daily = origin.scored[["cd_setor", "CICLOS", "origin_cycle"]].assign(
        cd_cd="1", date=pd.Timestamp("2026-01-01"), actual=100.0, pred=100.0 - cd_day_error
    )
    replaced = OriginResult(**{**origin.__dict__, "daily": daily})
    return EvaluationResult(candidate_name=result.candidate_name, origins=[replaced])


def test_comparison_ranks_by_cd_day_error_when_results_carry_daily_forecasts():
    # `cycle_better` wins per cycle but loses per CD-day: the ranking follows the CD-day number.
    cycle_better = _with_daily(_result("cycle_better", [("A", "c1", 100, 99)]), 50.0)
    day_better = _with_daily(_result("day_better", [("A", "c1", 100, 80)]), 5.0)

    table = compare([cycle_better, day_better])

    assert table["candidate"].tolist() == ["day_better", "cycle_better"]
    assert table["mae_cd_day_common"].tolist() == [5.0, 50.0]


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("mae_common", 25),
        ("rmse_common", 650**0.5),
        ("p90_abs_error_common", 29),
        ("worst_origin_mae_common", 25),
        ("bias_common", -5),
        ("bias_pct_common", -100 / 30),
        ("wmape_common", 50 / 300),
        ("mase_common", 25),
    ],
)
def test_level_diagnostics_exclude_points_missing_from_other_candidates(column, expected):
    wide = _result("wide", [("A", "c1", 100, 120), ("B", "c1", 200, 170), ("C", "c1", 1000, 0)])
    narrow = _result("narrow", [("A", "c1", 100, 100), ("B", "c1", 200, 200)])
    row = compare([wide, narrow]).set_index("candidate").loc["wide"]
    assert row[column] == pytest.approx(expected)


def test_common_mase_uses_each_folds_training_scale():
    first = _result("model", [("A", "c1", 100, 110)]).origins[0]
    second = OriginResult(
        origin_cycle="o2",
        target_cycles=("c2",),
        scored=first.scored.assign(CICLOS="c2", origin_cycle="o2", items_pred=120, abs_error=20),
        missing=first.missing,
        naive_in_sample_mae=10,
    )
    result = EvaluationResult(candidate_name="model", origins=[first, second])
    row = compare([result]).iloc[0]
    assert row["mase_common"] == (10 / 1 + 20 / 10) / 2
    assert row["worst_origin_mae_common"] == 20


def test_all_level_diagnostics_are_nan_when_common_coverage_is_empty():
    result = _result("a", [("A", "c1", 100, 110)])
    empty = EvaluationResult(candidate_name="empty", origins=[])
    table = compare([result, empty])
    columns = [
        "rmse_common",
        "mase_common",
        "p90_abs_error_common",
        "worst_origin_mae_common",
        "bias_common",
        "bias_pct_common",
        "wmape_common",
    ]
    assert table[columns].isna().all().all()


def test_level_ranking_remains_mae_even_when_wmape_prefers_another_model():
    rows = [("A", "c1", 100, 150), ("A", "c2", 100, 150), ("B", "c1", 1000, 1000)]
    mae_winner = _result("mae_winner", rows)
    wmape_winner = _result(
        "wmape_winner", [("A", "c1", 100, 100), ("A", "c2", 100, 100), ("B", "c1", 1000, 940)]
    )
    table = compare([mae_winner, wmape_winner])
    assert table.candidate.tolist() == ["mae_winner", "wmape_winner"]
    assert table.wmape_common.iloc[0] > table.wmape_common.iloc[1]
