"""Tests for scoring and ranking level and shape forecast strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.level import (
    LEVEL_STRATEGIES,
    LevelStrategy,
    level_linear_trend,
    level_naive_last,
)
from src.forecasting.scoring import (
    build_holdout_cycle_ids,
    score_and_select_strategies,
    score_level_strategies,
    score_level_strategy,
    score_shape_strategies,
    score_shape_strategy,
    select_best_level_strategy,
    select_best_shape_strategy,
)
from src.forecasting.shape import SHAPE_STRATEGIES
from src.generate.generator import generate_synthetic_demand


@pytest.fixture
def historical_cycle_totals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "sector": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
            "cycle_total": [10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 10.0, 0.0],
        }
    )


def test_score_level_strategy_computes_mae_and_wmape():
    forecast = pd.DataFrame({"sector": ["S1", "S2"], "forecast_cycle_total": [50.0, 0.0]})
    actual = pd.DataFrame({"sector": ["S1", "S2"], "actual_cycle_total": [40.0, 10.0]})

    mae, wmape = score_level_strategy(forecast, actual)

    assert mae == pytest.approx(10.0)
    assert wmape == pytest.approx(20.0 / 50.0)


def test_score_level_strategies_ranks_the_better_strategy_first(historical_cycle_totals):
    actual = pd.DataFrame({"sector": ["S1", "S2"], "actual_cycle_total": [50.0, 0.0]})
    strategies = {"naive_last": level_naive_last, "linear_trend": level_linear_trend}

    scoreboard = score_level_strategies(strategies, historical_cycle_totals, pd.DataFrame(), actual)

    assert list(scoreboard.index)[0] == "linear_trend"
    assert scoreboard["cycle_total_WMAPE"].is_monotonic_increasing


def test_score_level_strategies_skips_a_strategy_that_fails_to_fit(historical_cycle_totals):
    def broken_strategy(cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1) -> float:
        raise ValueError("not enough data to fit")

    strategies: dict[str, LevelStrategy] = {
        "naive_last": level_naive_last,
        "broken": broken_strategy,
    }
    actual = pd.DataFrame({"sector": ["S1", "S2"], "actual_cycle_total": [50.0, 0.0]})

    with pytest.warns(UserWarning, match="broken"):
        scoreboard = score_level_strategies(
            strategies, historical_cycle_totals, pd.DataFrame(), actual
        )

    assert "broken" not in scoreboard.index
    assert "naive_last" in scoreboard.index


def test_select_best_level_strategy_returns_the_lowest_wmape_strategy():
    scoreboard = pd.DataFrame(
        {"cycle_total_WMAPE": [0.4, 0.1, 0.2]},
        index=["naive_last", "linear_trend", "holt_ets"],
    )

    assert select_best_level_strategy(scoreboard) == "linear_trend"


def test_score_shape_strategy_computes_daily_mae_and_wmape():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.5, 0.5]})
    forecast_totals = pd.DataFrame(
        {"sector": ["S1"], "cycle_id": [1], "forecast_cycle_total": [20.0]}
    )
    start_day_by_sector = {"S1": 1}
    cycle_open_dates = {1: pd.Timestamp("2026-01-01")}
    actual_daily = pd.DataFrame(
        {
            "order_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            "actual_orders": [8, 12],
        }
    )

    strategy_daily, mae, wmape = score_shape_strategy(
        shape, forecast_totals, start_day_by_sector, cycle_open_dates, actual_daily
    )

    assert strategy_daily["forecast_orders"].tolist() == [10, 10]
    assert mae == pytest.approx(2.0)
    assert wmape == pytest.approx(4.0 / 20.0)


def test_score_shape_strategies_ranks_the_better_strategy_first():
    forecast_totals = pd.DataFrame(
        {"sector": ["S1"], "cycle_id": [1], "forecast_cycle_total": [20.0]}
    )
    start_day_by_sector = {"S1": 1}
    cycle_open_dates = {1: pd.Timestamp("2026-01-01")}
    actual_daily = pd.DataFrame(
        {
            "order_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            "actual_orders": [10, 10],
        }
    )
    strategies = {
        "perfect": pd.DataFrame(
            {"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.5, 0.5]}
        ),
        "off": pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.9, 0.1]}),
    }

    scoreboard = score_shape_strategies(
        strategies, forecast_totals, start_day_by_sector, cycle_open_dates, actual_daily
    )

    assert list(scoreboard.index)[0] == "perfect"
    assert scoreboard["daily_WMAPE"].is_monotonic_increasing


def test_select_best_shape_strategy_returns_the_lowest_wmape_strategy():
    scoreboard = pd.DataFrame(
        {"daily_WMAPE": [0.3, 0.05, 0.2]},
        index=["plain_average", "recency_weighted", "median"],
    )

    assert select_best_shape_strategy(scoreboard) == "recency_weighted"


def test_build_holdout_cycle_ids_takes_the_last_n_cycles_of_history():
    assert list(build_holdout_cycle_ids(n_cycles_history=6, n_cycles_horizon=2)) == [5, 6]


def test_build_holdout_cycle_ids_defaults_to_a_single_cycle():
    assert list(build_holdout_cycle_ids(n_cycles_history=6, n_cycles_horizon=1)) == [6]


def test_build_holdout_cycle_ids_rejects_a_horizon_with_no_training_history():
    with pytest.raises(ValueError):
        build_holdout_cycle_ids(n_cycles_history=4, n_cycles_horizon=4)


@pytest.fixture
def synthetic_demand():
    rng = np.random.default_rng(0)
    sectors = ["S01", "S02", "S03"]
    demand_level, demand_shape, cycle_starts, assignment = generate_synthetic_demand(
        rng, sectors, cycle_length=10, n_cycles=6
    )
    return demand_level, demand_shape, cycle_starts, assignment


def test_score_and_select_strategies_returns_valid_strategy_names(synthetic_demand):
    demand_level, demand_shape, cycle_starts, assignment = synthetic_demand

    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history=6
    )

    assert isinstance(best_level_name, str)
    assert isinstance(best_shape_name, str)
    assert best_level_name in LEVEL_STRATEGIES
    assert best_shape_name in SHAPE_STRATEGIES


def test_score_and_select_strategies_supports_a_multi_cycle_horizon(synthetic_demand):
    demand_level, demand_shape, cycle_starts, assignment = synthetic_demand

    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history=6, n_cycles_horizon=3
    )

    assert best_level_name in LEVEL_STRATEGIES
    assert best_shape_name in SHAPE_STRATEGIES


def test_score_and_select_strategies_survives_a_horizon_that_leaves_thin_training_history(
    synthetic_demand,
):
    # n_cycles_history=6, n_cycles_horizon=4 leaves only cycles 1-2 as training data,
    # too little for Holt/ARIMA/calendar-regression to fit — they should be skipped
    # (see score_level_strategies) rather than crashing the whole selection.
    demand_level, demand_shape, cycle_starts, assignment = synthetic_demand

    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history=6, n_cycles_horizon=4
    )

    assert best_level_name in LEVEL_STRATEGIES
    assert best_shape_name in SHAPE_STRATEGIES
