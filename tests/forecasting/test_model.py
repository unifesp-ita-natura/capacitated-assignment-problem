"""Tests for forecasting models, strategy interface, adapter, and registry."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel

from src.forecasting.model import (
    CandidateRegistry,
    InsufficientHistoryError,
    forecast_future_cycles,
    forecast_with_shape,
    per_sector,
)


def test_forecast_with_shape_rounds_shares_times_totals_to_whole_orders():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.25, 0.75]})
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_cycle_total": [20.0]})

    forecast = forecast_with_shape(shape, forecast_totals)

    assert forecast["forecast_orders"].tolist() == [5, 15]


def _build_synthetic_demand() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cycles of demand_level/demand_shape rows for a single sector."""
    demand_level = pd.DataFrame(
        {
            "cd_setor": ["S1", "S1"],
            "ciclo": ["1", "2"],
            "cycle_duration": [3, 3],
            "total_pedidos": [30, 42],
            "total_volumes": [60, 84],
            "total_itens": [90, 126],
        }
    )
    demand_shape = pd.DataFrame(
        {
            "cd_setor": ["S1"] * 6,
            "ciclo": ["1", "1", "1", "2", "2", "2"],
            "relative_date": [0 / 3, 1 / 3, 2 / 3, 0 / 3, 1 / 3, 2 / 3],
            "share_pedidos": [0.2, 0.3, 0.5, 0.2, 0.3, 0.5],
        }
    )
    return demand_level, demand_shape


def test_forecast_future_cycles_returns_calendar_mapped_forecast_for_expected_sectors():
    demand_level, demand_shape = _build_synthetic_demand()
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    assignment = {"S1": (1, 1)}

    forecast = forecast_future_cycles(
        demand_level=demand_level,
        demand_shape=demand_shape,
        cycle_starts=cycle_starts,
        assignment=assignment,
        n_cycles_history=2,
        cycle_length=3,
        best_level_name="naive_last",
        best_shape_name="plain_average",
    )

    assert {"forecast_orders", "order_date"} <= set(forecast.columns)
    assert set(forecast["sector"].unique()) == {"S1"}
    assert forecast["cycle_id"].unique().tolist() == [3]


def test_forecast_future_cycles_covers_every_cycle_in_the_horizon():
    demand_level, demand_shape = _build_synthetic_demand()
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    assignment = {"S1": (1, 1)}

    forecast = forecast_future_cycles(
        demand_level=demand_level,
        demand_shape=demand_shape,
        cycle_starts=cycle_starts,
        assignment=assignment,
        n_cycles_history=2,
        cycle_length=3,
        best_level_name="naive_last",
        best_shape_name="plain_average",
        n_cycles_horizon=3,
    )

    assert sorted(forecast["cycle_id"].unique()) == [3, 4, 5]


HISTORY = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "B", "B"],
        "CICLOS": ["1", "2", "1", "2"],
        "items": [100, 200, 10, 20],
        "opening_date": pd.to_datetime(["2026-01-01", "2026-01-22", "2026-01-01", "2026-01-22"]),
    }
)
TARGETS = pd.DataFrame(
    {
        "cd_setor": ["A", "B"],
        "CICLOS": ["3", "3"],
        "opening_date": pd.to_datetime(["2026-02-12", "2026-02-12"]),
    }
)


def test_per_sector_applies_fn_independently_to_each_sector():
    candidate = per_sector(
        name="last", fn=lambda series, horizon: pd.Series([series.iloc[-1]] * horizon)
    )

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector["A"] == 200
    assert by_sector["B"] == 20


def test_per_sector_skips_a_sector_with_no_target_cycles_without_calling_fn():
    calls: list[str] = []

    def fn(series: pd.Series, horizon: int) -> pd.Series:
        calls.append("called")
        return pd.Series([series.iloc[-1]] * horizon)

    targets_missing_b = TARGETS[TARGETS["cd_setor"] == "A"]
    candidate = per_sector(name="last", fn=fn)

    predictions = candidate.fit_predict(HISTORY, targets_missing_b)

    assert set(predictions["cd_setor"]) == {"A"}
    assert calls == ["called"]


def test_per_sector_skips_a_sector_whose_fn_raises_insufficient_history():
    def fn(series: pd.Series, horizon: int) -> pd.Series:
        if series.iloc[-1] == 20:
            raise InsufficientHistoryError("not enough data")
        return pd.Series([series.iloc[-1]] * horizon)

    candidate = per_sector(name="picky", fn=fn)

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    assert set(predictions["cd_setor"]) == {"A"}


def test_per_sector_propagates_unexpected_errors():
    def fn(series: pd.Series, horizon: int) -> pd.Series:
        raise RuntimeError("real bug")

    candidate = per_sector(name="broken", fn=fn)

    with pytest.raises(RuntimeError, match="real bug"):
        candidate.fit_predict(HISTORY, TARGETS)


class _DummyParams(BaseModel):
    model: str = "dummy"


def test_registry_round_trips_a_registered_builder():
    registry = CandidateRegistry()

    @registry.register("dummy")
    def _build(params: _DummyParams):
        return per_sector(name="dummy", fn=lambda s, h: pd.Series([0] * h))

    candidate = registry.build(_DummyParams())

    assert candidate.name == "dummy"


def test_registry_raises_a_helpful_error_for_an_unknown_model():
    registry = CandidateRegistry()

    with pytest.raises(ValueError, match="no candidate registered"):
        registry.build(_DummyParams())


def test_registry_known_models_lists_registered_names():
    registry = CandidateRegistry()
    registry.register("dummy")(
        lambda params: per_sector(name="dummy", fn=lambda s, h: pd.Series([0] * h))
    )

    assert registry.known_models == ["dummy"]
