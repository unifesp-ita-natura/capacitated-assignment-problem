"""Tests for spreading a cycle forecast over days and scoring it per sector-day and per CD-day."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting import daily
from src.forecasting.dataset import (
    DailyBase,
    build_daily_base,
    build_item_panel,
    load_demand_base,
)
from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import QUERY_KEYS

FIXTURE_PATH = "tests/fixtures/demand_sample.csv"


def _base(rows: list[tuple[str, str, str, int, int]]) -> DailyBase:
    """(sector, cd, cycle, day offset, items) rows over 20-day windows opening 2026-01-01."""
    actuals = pd.DataFrame(rows, columns=["cd_setor", "cd_cd", "CICLOS", "offset", "items"])
    actuals["date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(actuals["offset"], unit="D")
    windows = actuals[["cd_setor", "CICLOS"]].drop_duplicates()
    windows = windows.assign(window_start=pd.Timestamp("2026-01-01"), n_days=20)
    return DailyBase(actuals.drop(columns="offset"), windows.reset_index(drop=True))


def _prediction(cycle: str, value: float, sector: str = "A") -> pd.DataFrame:
    return pd.DataFrame({"cd_setor": [sector], "CICLOS": [cycle], "items_pred": [value]})


def test_daily_items_add_back_up_to_the_cycle_panel():
    raw = load_demand_base(FIXTURE_PATH)

    base = build_daily_base(raw)
    panel = build_item_panel(raw).set_index(["cd_setor", "CICLOS"])["items"]

    by_cycle = base.actuals.groupby(["cd_setor", "CICLOS"])["items"].sum()
    assert by_cycle.to_dict() == panel.to_dict()


def test_every_order_falls_inside_its_window():
    base = build_daily_base(load_demand_base(FIXTURE_PATH))

    merged = base.actuals.merge(base.windows, on=["cd_setor", "CICLOS"])
    offset = (merged["date"] - merged["window_start"]).dt.days
    assert offset.between(0, merged["n_days"] - 1).all()


def test_uniform_curve_gives_every_day_the_same_share():
    base = _base([("A", "1", "c1", 0, 10), ("A", "1", "c2", 3, 10)])

    scored = daily.score_daily(_prediction("c2", 200.0), base, ["c1"], "uniform")

    assert scored["pred"].sum() == pytest.approx(200.0)
    assert scored["pred"].nunique() == 1  # 200 / 20 days on the single CD


def test_sector_curve_puts_the_forecast_where_the_history_sold():
    # Cycle c1 sold everything in the first tenth of the window (days 0-1).
    base = _base([("A", "1", "c1", 0, 60), ("A", "1", "c1", 1, 40), ("A", "1", "c2", 5, 10)])

    scored = daily.score_daily(_prediction("c2", 200.0), base, ["c1"], "sector")

    early = scored[scored["date"] < pd.Timestamp("2026-01-03")]
    assert early["pred"].sum() == pytest.approx(200.0)
    assert scored["pred"].sum() == pytest.approx(200.0)


def test_forecast_is_split_across_cds_by_the_historical_share():
    base = _base([("A", "1", "c1", 0, 75), ("A", "2", "c1", 0, 25), ("A", "1", "c2", 0, 1)])

    scored = daily.score_daily(_prediction("c2", 100.0), base, ["c1"], "uniform")

    by_cd = scored.groupby("cd_cd")["pred"].sum()
    assert by_cd["1"] == pytest.approx(75.0)
    assert by_cd["2"] == pytest.approx(25.0)


def test_a_day_with_no_order_scores_as_zero_actual_not_as_missing():
    base = _base([("A", "1", "c1", 0, 10), ("A", "1", "c2", 0, 40)])

    scored = daily.score_daily(_prediction("c2", 200.0), base, ["c1"], "uniform")

    assert len(scored) == 20  # every day of the window, not just the one with an order
    assert scored["actual"].sum() == 40


def test_cd_day_mae_matches_a_hand_calculation():
    frame = pd.DataFrame(
        {
            "cd_setor": ["A", "B", "A"],
            "cd_cd": ["1", "1", "2"],
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "actual": [10.0, 30.0, 5.0],
            "pred": [20.0, 10.0, 5.0],
        }
    )

    # CD 1 that day: actual 40 vs pred 30 -> 10. CD 2: 0. Each CD counts once.
    assert daily.cd_day_mae(frame) == pytest.approx(5.0)
    assert daily.cd_day_wape(frame) == pytest.approx(10 / 45)


def test_the_curve_never_reads_the_target_cycle():
    # Same history, wildly different target-cycle actuals: the forecast must not move.
    quiet = _base([("A", "1", "c1", 0, 10), ("A", "1", "c2", 19, 1)])
    loud = _base([("A", "1", "c1", 0, 10), ("A", "1", "c2", 0, 9999)])

    forecast = [
        daily.score_daily(_prediction("c2", 100.0), base, ["c1"], "sector")
        .groupby("date")["pred"]
        .sum()
        for base in (quiet, loud)
    ]

    pd.testing.assert_series_equal(*forecast)


class _Constant:
    name = "constant"

    def fit_predict(self, history, targets):
        return targets[QUERY_KEYS].assign(items_pred=100.0)


def test_evaluate_reports_daily_errors_only_when_given_a_daily_base():
    raw = load_demand_base(FIXTURE_PATH)
    panel, split = build_item_panel(raw), RollingOriginSplit(horizon=1, min_train_cycles=6)

    plain = evaluate(_Constant(), panel, split).summary()
    with_days = evaluate(_Constant(), panel, split, build_daily_base(raw)).summary()

    assert "mae_cd_day" not in plain
    assert with_days["mae_cd_day"] > 0
    assert with_days["mae"] == plain["mae"]  # the per-cycle number is untouched
