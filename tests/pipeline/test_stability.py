"""Tests for the realized-demand scoring and spread helpers of the seed-stability experiment."""

import pandas as pd

from src.pipeline.stability import SeedOutcome, spread


def test_improvement_is_drop_relative_to_as_is():
    outcome = SeedOutcome(1, 100.0, 80.0, 50.0, 55.0, 3, 1.0)
    assert outcome.forecast_improvement_pct == 20.0
    assert outcome.realized_improvement_pct == -10.0


def test_spread_reports_cv():
    stats = spread(pd.Series([9.0, 11.0]))
    assert stats["mean"] == 10.0
    assert stats["min"] == 9.0 and stats["max"] == 11.0
    assert round(stats["cv_pct"], 2) == 14.14
