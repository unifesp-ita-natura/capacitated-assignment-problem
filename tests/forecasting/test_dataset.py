"""Tests for the raw-base loader and the item/shape panel aggregations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.dataset import (
    build_item_panel,
    build_shape_panel,
    cycle_calendar,
    load_demand_base,
    sector_cycle_attributes,
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


def test_build_item_panel_sums_items_per_sector_city_and_cycle():
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)

    keys = ["cd_setor", "estado", "cidade", "CICLOS"]
    expected = raw[raw["CICLOS"] != "202614"].groupby(keys)["total_itens_mascarado"].sum()
    for _, row in panel.iterrows():
        assert row["items"] == expected[tuple(row[keys])]
    # The fixture splits its first sector over two cities: two series, not one.
    first = raw["cd_setor"].iloc[0]
    assert panel.loc[panel["cd_setor"] == first, "serie"].nunique() == 2
    assert panel["serie"].str.count("[|]").eq(2).all()


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


def test_build_item_panel_carries_order_and_volume_counts():
    # Stage 1 of the forecasting roadmap: these two series used to be
    # dropped at aggregation, and the previous cycle's order count predicts
    # the next cycle's items better than the item count itself does.
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)

    expected = (
        raw[raw["CICLOS"] != "202614"]
        .groupby(["cd_setor", "CICLOS"])["total_pedidos_mascarado"]
        .sum()
    )
    actual = panel.groupby(["cd_setor", "CICLOS"])["orders"].sum()

    assert actual.sort_index().equals(expected.sort_index().astype(actual.dtype))
    assert "volumes" in panel.columns


def test_loader_preserves_organization_codes_as_identifiers(tmp_path):
    raw = load_demand_base(FIXTURE_PATH).assign(CD_RE="01", CD_GV="007")
    path = tmp_path / "organization.csv"
    raw.to_csv(path, index=False)
    panel = build_item_panel(load_demand_base(path))
    assert panel.CD_RE.eq("01").all()
    assert panel.CD_GV.eq("007").all()


def test_organization_can_change_between_cycles():
    raw = load_demand_base(FIXTURE_PATH).assign(CD_RE="01", CD_GV="007")
    raw.loc[raw.CICLOS == "202601", "CD_GV"] = "008"
    panel = build_item_panel(raw)
    assert panel.loc[panel.CICLOS == "202601", "CD_GV"].eq("008").all()


def test_panel_rejects_conflicting_organization_within_sector_cycle():
    raw = load_demand_base(FIXTURE_PATH).assign(CD_RE="01", CD_GV="007")
    conflicting = pd.concat([raw, raw.iloc[[0]].assign(CD_RE="02")], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting"):
        build_item_panel(conflicting)


def test_sector_cycle_attributes_reports_the_cd_with_most_items():
    raw = load_demand_base(FIXTURE_PATH).assign(CD_RE="01")
    key = raw.iloc[0][["cd_setor", "CICLOS"]]
    is_key = (raw.cd_setor == key.cd_setor) & (raw.CICLOS == key.CICLOS)
    minor = raw[is_key].iloc[[0]].assign(cd_cd=9999, total_itens_mascarado=1)
    attributes = sector_cycle_attributes(pd.concat([raw, minor], ignore_index=True))
    row = attributes[(attributes.cd_setor == key.cd_setor) & (attributes.CICLOS == key.CICLOS)]
    assert len(row) == 1
    assert row.cd_cd.item() != 9999
    assert row.CD_RE.item() == "01"
    assert len(attributes) == len(raw[["cd_setor", "CICLOS"]].drop_duplicates())


def test_a_city_that_stops_buying_gets_zero_rows_only_after_its_first_purchase():
    raw = load_demand_base(FIXTURE_PATH)
    first = raw["cd_setor"].iloc[0]
    is_other = raw["cidade"] == "OUTRA CIDADE"
    cycles = sorted(raw.loc[raw["cd_setor"] == first, "CICLOS"].unique())
    # OUTRA CIDADE first buys in the 2nd cycle and skips the 4th.
    dropped = is_other & raw["CICLOS"].isin([cycles[0], cycles[3]])
    panel = build_item_panel(raw[~dropped])

    other = panel[panel["cidade"] == "OUTRA CIDADE"].set_index("CICLOS")
    assert cycles[0] not in other.index
    assert other.loc[cycles[3], ["items", "orders", "volumes"]].tolist() == [0, 0, 0]
    assert other.loc[cycles[3], "cycle_days"] > 0  # the sector's window still applies


def test_sector_panel_keeps_one_series_per_sector_with_the_same_totals():
    raw = load_demand_base(FIXTURE_PATH)
    by_sector = build_item_panel(raw, by_city=False)
    by_city = build_item_panel(raw)

    assert (by_sector["serie"] == by_sector["cd_setor"]).all()
    assert "cidade" not in by_sector
    totals = by_city.groupby(["cd_setor", "CICLOS"])["items"].sum()
    assert by_sector.set_index(["cd_setor", "CICLOS"])["items"].equals(totals)
