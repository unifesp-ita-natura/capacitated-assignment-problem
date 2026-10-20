"""Tests for summing sector-city forecasts back into sector totals."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.sector_vs_city.run import city_sum
from src.forecasting.evaluation import EvaluationResult, OriginResult


def _origin(scored: pd.DataFrame, scale: float) -> OriginResult:
    return OriginResult(
        origin_cycle="202601",
        target_cycles=("202602",),
        scored=scored,
        missing=pd.DataFrame(columns=["serie", "CICLOS", "origin_cycle"]),
        naive_in_sample_mae=scale,
    )


def _summed() -> EvaluationResult:
    """Sector 1 has three cities but only two were predicted (the third is new)."""
    city_rows = pd.DataFrame(
        {
            "serie": ["1|SP|A", "1|SP|B", "2|RJ|C"],
            "cd_setor": ["1", "1", "2"],
            "CICLOS": ["202602"] * 3,
            "items_pred": [60.0, 30.0, 50.0],
        }
    )
    city = EvaluationResult("naive", [_origin(city_rows, scale=99.0)])
    sector = EvaluationResult("naive", [_origin(pd.DataFrame(), scale=7.0)])
    totals = pd.Series(
        [100.0, 40.0], index=pd.MultiIndex.from_tuples([("1", "202602"), ("2", "202602")])
    )
    return city_sum(city, sector, totals)


def test_city_forecasts_are_summed_and_scored_against_the_whole_sector_total():
    scored = _summed().scored.set_index("cd_setor")

    assert scored["items_pred"].to_dict() == {"1": 90.0, "2": 50.0}
    # The unpredicted city's items count as error instead of disappearing.
    assert scored["abs_error"].to_dict() == {"1": 10.0, "2": 10.0}


def test_city_sum_is_one_series_per_sector_on_the_sector_mase_scale():
    result = _summed()

    assert result.scored["serie"].tolist() == result.scored["cd_setor"].tolist()
    assert result.origins[0].naive_in_sample_mae == pytest.approx(7.0)
