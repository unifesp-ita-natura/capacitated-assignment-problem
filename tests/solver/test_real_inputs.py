"""Tests building a solver instance from daily forecasts and the observed As-Is calendar."""

from __future__ import annotations

from itertools import chain

import pandas as pd

from src.solver.real_inputs import (
    build_real_instance,
    cycle_spacing,
    dominant_cd,
    infer_as_is,
    sector_curves,
)


def _daily() -> pd.DataFrame:
    rows = [("A", "202601", day, 10.0 * day) for day in range(1, 41)]
    rows += [("A", "202601", 1, 5.0), ("B", "202601", 1, 7.0), ("B", "202601", 2, None)]
    rows += [("B", "202601", day, 3.0) for day in range(3, 41)]
    return pd.DataFrame(rows, columns=["cd_setor", "CICLOS", "relative_day", "items_pred"])


def _raw() -> pd.DataFrame:
    openings = ["2026-01-05", "2026-01-05", "2026-01-12", "2026-01-05", "2026-01-26"]
    return pd.DataFrame(
        {
            "cd_setor": ["A", "A", "B", "C", "A"],
            "cd_cd": [2700, 5100, 5100, 2700, 5100],
            "total_itens_mascarado": [10, 90, 5, 1, 1],
            "CICLOS": ["202601"] * 4 + ["202602"],
            "Dt Abertura": pd.to_datetime(openings),
            "Dt Fechamento": pd.to_datetime(["2026-02-01"] * 5),
            "data_pedido": pd.to_datetime(["2026-01-06"] * 4 + ["2026-01-27"]),
        }
    )


def test_sector_curves_sum_cities_and_drop_missing() -> None:
    curves = sector_curves(_daily(), ["202601"])
    first_day = curves[(curves["sector"] == "A") & (curves["offset"] == 0)]
    assert first_day["forecast_items"].iloc[0] == 15.0
    assert not ((curves["sector"] == "B") & (curves["offset"] == 1)).any()


def test_dominant_cd_follows_item_volume() -> None:
    assert dominant_cd(_raw(), ["A", "B"]) == {"A": 5100, "B": 5100}


def test_infer_as_is_offsets_from_cycle_start_and_fills_missing() -> None:
    slots = infer_as_is(_raw(), "202601", ["A", "B", "Z"])
    assert slots["A"] == 1
    assert slots["B"] == 8
    assert slots["Z"] == 1  # median of observed slots


def test_cycle_spacing_is_median_days_between_openings() -> None:
    assert cycle_spacing(_raw(), ["202601", "202602"]) == 21


def _assert_steady_days_trim_the_ramps(instance) -> None:
    steady = list(chain.from_iterable(instance.days_by_cycle.values()))
    assert (min(steady), max(steady)) == (15, max(instance.days) - 14)


def _assert_demand_days_exist(instance) -> None:
    assert {day for _, day, _ in instance.projected_demand} <= set(instance.days)


def test_build_real_instance_is_consistent() -> None:
    real = build_real_instance(_daily(), _raw(), ["202601"], "202601", max_churn=0.5)
    instance = real.instance
    assert instance.sectors == [1, 2]
    assert instance.max_churn == 1.0
    assert set(instance.current_assignment.values()) == {1, 8}
    _assert_steady_days_trim_the_ramps(instance)
    _assert_demand_days_exist(instance)
    assert set(instance.cd_sectors) == {5100}


def test_max_sectors_keeps_the_largest() -> None:
    real = build_real_instance(_daily(), _raw(), ["202601"], "202601", max_churn=0.5, max_sectors=1)
    assert real.sector_ids == {"A": 1}
