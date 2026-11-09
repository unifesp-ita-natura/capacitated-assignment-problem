"""Tests for the realized-demand scoring and spread helpers of the seed-stability experiment."""

import pandas as pd

from src.pipeline.stability import SeedOutcome, best_of_k, selection_cell, spread


def test_improvement_is_drop_relative_to_as_is():
    outcome = SeedOutcome(1, 100.0, 80.0, 50.0, 55.0, 3, 1.0)
    assert outcome.forecast_improvement_pct == 20.0
    assert outcome.realized_improvement_pct == -10.0


def test_spread_reports_cv():
    stats = spread(pd.Series([9.0, 11.0]))
    assert stats["mean"] == 10.0
    assert stats["min"] == 9.0 and stats["max"] == 11.0
    assert round(stats["cv_pct"], 2) == 14.14


def _cell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_to_be": [10.0, 20.0, 30.0],
            "realized_to_be": [5.0, 7.0, 9.0],
            "realized_improvement_pct": [30.0, 20.0, 10.0],
        }
    )


def test_selection_cell_picks_lowest_predicted_seed():
    cell = selection_cell(_cell())
    assert cell["spearman"] == 1.0
    assert cell["picked_improvement_pct"] == 30.0
    assert cell["mean_improvement_pct"] == 20.0
    assert cell["oracle_improvement_pct"] == 30.0


def test_best_of_k_grows_to_the_picked_value_with_k():
    curve = best_of_k(_cell(), [1, 3, 9], draws=200).set_index("k")["mean_improvement_pct"]
    assert 9 not in curve.index  # k above the number of seeds is skipped
    assert curve[3] == 30.0
    assert curve[1] < curve[3]
