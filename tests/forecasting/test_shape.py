"""Tests for the within-window (shape) forecast strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.shape import (
    build_shape_observations,
    build_shape_strategies,
    forecast_with_shape,
    normalize_shape,
    score_shape_strategies,
    score_shape_strategy,
    select_best_shape_strategy,
    shape_last_campanha,
    shape_median,
    shape_plain_average,
    shape_recency_weighted,
    shape_shrinkage,
    to_calendar,
)


def test_build_shape_observations_computes_order_share():
    orders = pd.DataFrame(
        {
            "campanha_id": [1, 1],
            "sector": ["S1", "S1"],
            "offset": [0, 1],
            "orders": [3, 7],
        }
    )
    campanha_totals = pd.DataFrame({"campanha_id": [1], "sector": ["S1"], "campanha_total": [10]})

    observations = build_shape_observations(orders, campanha_totals)

    assert observations["order_share"].tolist() == pytest.approx([0.3, 0.7])


def test_normalize_shape_rescales_shares_to_sum_to_one():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.2, 0.2]})

    normalized = normalize_shape(shape)

    assert normalized["order_share"].tolist() == pytest.approx([0.5, 0.5])
    assert normalized.groupby("sector")["order_share"].sum().iloc[0] == pytest.approx(1.0)


@pytest.fixture
def shape_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["S1", "S1", "S1", "S1"],
            "campanha_id": [1, 1, 2, 2],
            "offset": [0, 1, 0, 1],
            "order_share": [0.6, 0.4, 0.2, 0.8],
        }
    )


def test_shape_plain_average_is_the_unweighted_mean_per_offset(shape_observations):
    shape = shape_plain_average(shape_observations)

    result = shape.set_index("offset")["order_share"]
    assert result[0] == pytest.approx(0.4)
    assert result[1] == pytest.approx(0.6)


def test_shape_median_is_robust_to_an_outlier_campanha():
    observations = pd.DataFrame(
        {
            "sector": ["S1"] * 6,
            "campanha_id": [1, 1, 2, 2, 3, 3],
            "offset": [0, 1, 0, 1, 0, 1],
            "order_share": [0.1, 0.9, 0.1, 0.9, 0.9, 0.1],
        }
    )

    median_shape = shape_median(observations)
    average_shape = shape_plain_average(observations)

    median_offset0 = median_shape.set_index("offset")["order_share"][0]
    average_offset0 = average_shape.set_index("offset")["order_share"][0]
    assert median_offset0 == pytest.approx(0.1)
    assert average_offset0 == pytest.approx((0.1 + 0.1 + 0.9) / 3)


def test_shape_recency_weighted_favors_the_most_recent_campanha():
    observations = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S1", "S1"],
            "campanha_id": [1, 1, 2, 2],
            "offset": [0, 1, 0, 1],
            "order_share": [0.0, 1.0, 1.0, 0.0],
        }
    )

    weighted = shape_recency_weighted(observations, half_life_campanhas=1.0)
    plain = shape_plain_average(observations)

    weighted_value = weighted.set_index("offset")["order_share"][0]
    plain_value = plain.set_index("offset")["order_share"][0]
    assert weighted_value > plain_value


def test_shape_last_campanha_uses_only_the_most_recent_campanha(shape_observations):
    shape = shape_last_campanha(shape_observations)

    result = shape.set_index("offset")["order_share"]
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(0.8)


def test_shape_shrinkage_with_zero_shrinkage_matches_plain_average():
    plain_average = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S2", "S2"],
            "offset": [0, 1, 0, 1],
            "order_share": [0.4, 0.6, 0.8, 0.2],
        }
    )

    shrunk = shape_shrinkage(plain_average, shrinkage=0.0)

    pd.testing.assert_series_equal(
        shrunk["order_share"].reset_index(drop=True),
        plain_average["order_share"].reset_index(drop=True),
    )


def test_shape_shrinkage_with_full_shrinkage_matches_the_global_curve():
    plain_average = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S2", "S2"],
            "offset": [0, 1, 0, 1],
            "order_share": [0.4, 0.6, 0.8, 0.2],
        }
    )

    shrunk = shape_shrinkage(plain_average, shrinkage=1.0)

    s1 = shrunk[shrunk["sector"].eq("S1")].set_index("offset")["order_share"]
    s2 = shrunk[shrunk["sector"].eq("S2")].set_index("offset")["order_share"]
    assert s1[0] == pytest.approx(s2[0])
    assert s1[1] == pytest.approx(s2[1])


def test_build_shape_strategies_returns_every_named_strategy(shape_observations):
    strategies = build_shape_strategies(shape_observations)

    assert set(strategies) == {
        "plain_average",
        "recency_weighted",
        "median",
        "last_campanha",
        "shrinkage",
    }


def test_to_calendar_maps_offsets_onto_slot_start_days():
    day_offsets = pd.DataFrame({"sector": ["S1", "S1", "S2"], "offset": [0, 1, 0]})
    start_day_by_sector = {"S1": 1, "S2": 6}
    campanha_open_date = pd.Timestamp("2026-01-01")

    calendar = to_calendar(day_offsets, campanha_open_date, start_day_by_sector)

    assert calendar["day_in_cycle"].tolist() == [1, 2, 6]
    assert calendar["order_date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-06"),
    ]


def test_forecast_with_shape_rounds_shares_times_totals_to_whole_orders():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.25, 0.75]})
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_campanha_total": [20.0]})

    forecast = forecast_with_shape(shape, forecast_totals)

    assert forecast["forecast_orders"].tolist() == [5, 15]


def test_score_shape_strategy_computes_daily_mae_and_wmape():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.5, 0.5]})
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_campanha_total": [20.0]})
    start_day_by_sector = {"S1": 1}
    campanha_open_date = pd.Timestamp("2026-01-01")
    actual_daily = pd.DataFrame(
        {
            "order_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            "actual_orders": [8, 12],
        }
    )

    strategy_daily, mae, wmape = score_shape_strategy(
        shape, forecast_totals, start_day_by_sector, campanha_open_date, actual_daily
    )

    assert strategy_daily["forecast_orders"].tolist() == [10, 10]
    assert mae == pytest.approx(2.0)
    assert wmape == pytest.approx(4.0 / 20.0)


def test_score_shape_strategies_ranks_the_better_strategy_first():
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_campanha_total": [20.0]})
    start_day_by_sector = {"S1": 1}
    campanha_open_date = pd.Timestamp("2026-01-01")
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
        strategies, forecast_totals, start_day_by_sector, campanha_open_date, actual_daily
    )

    assert list(scoreboard.index)[0] == "perfect"
    assert scoreboard["daily_WMAPE"].is_monotonic_increasing


def test_select_best_shape_strategy_returns_the_lowest_wmape_strategy():
    scoreboard = pd.DataFrame(
        {"daily_WMAPE": [0.3, 0.05, 0.2]},
        index=["plain_average", "recency_weighted", "median"],
    )

    assert select_best_shape_strategy(scoreboard) == "recency_weighted"
