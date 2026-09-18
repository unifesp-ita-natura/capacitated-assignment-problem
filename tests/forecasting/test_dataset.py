"""Tests for the raw-base loader and the item/shape panel aggregations."""

from __future__ import annotations

import pytest

from src.forecasting.dataset import (
    build_item_panel,
    build_shape_panel,
    cycle_calendar,
    load_demand_base,
)

FIXTURE_PATH = "tests/fixtures/demand_sample.csv"


def test_load_demand_base_raises_on_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("cd_setor,total_itens_mascarado\n1,10\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load_demand_base(bad_csv)


def test_cycle_calendar_flags_the_trailing_partial_cycle():
    raw = load_demand_base(FIXTURE_PATH)
    calendar = cycle_calendar(raw)

    # 202614's closing date falls after the base's last observed order date,
    # so the base only partially covers it — see dataset.py's docstring.
    incomplete = set(calendar.loc[~calendar["is_complete"], "CICLOS"])
    assert "202614" in incomplete
    assert "202601" not in incomplete  # a genuinely low cycle, not a truncated one


def test_build_item_panel_drops_incomplete_cycles():
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)

    assert "202614" not in set(panel["CICLOS"])


def test_build_item_panel_sums_items_per_sector_and_cycle():
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)

    expected = (
        raw[raw["CICLOS"] != "202614"]
        .groupby(["cd_setor", "CICLOS"])["total_itens_mascarado"]
        .sum()
    )
    for _, row in panel.iterrows():
        assert row["items"] == expected[(row["cd_setor"], row["CICLOS"])]


def test_build_item_panel_opening_date_does_not_depend_on_block():
    # Regression guard for the leakage the team explicitly ruled out: dating
    # each sector by ITS OWN block's opening date would smuggle the
    # optimizer's decision variable into the forecast target.
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)

    # Every sector active in the same cycle must share the same opening_date.
    per_cycle_dates = panel.groupby("CICLOS")["opening_date"].nunique()
    assert (per_cycle_dates == 1).all()


def test_build_shape_panel_shares_sum_to_one_per_sector_cycle():
    raw = load_demand_base(FIXTURE_PATH)
    shape = build_shape_panel(raw)

    totals = shape.groupby(["cd_setor", "CICLOS"])["share"].sum()
    assert (totals.round(6) == 1.0).all()


def test_build_shape_panel_relative_position_is_in_unit_interval():
    raw = load_demand_base(FIXTURE_PATH)
    shape = build_shape_panel(raw)

    assert (shape["relative_position"] > 0).all()
    assert (shape["relative_position"] <= 1).all()
