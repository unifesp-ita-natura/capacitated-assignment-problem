"""Tests for the raw-base loader and the item/shape panel aggregations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.dataset import (
    build_daily_base,
    build_item_panel,
    build_shape_panel,
    cycle_calendar,
    demand_unit_id,
    load_demand_base,
    resum_to_sector,
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
    actual = panel.set_index(["cd_setor", "CICLOS"])["orders"]

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
    first = raw.iloc[0].copy()
    first["CD_RE"] = "02"
    conflicting = pd.concat([raw, first.to_frame().T], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting"):
        build_item_panel(conflicting)


def _with_synthetic_states(raw: pd.DataFrame) -> pd.DataFrame:
    """Split every sector's rows roughly in half between two fake states."""
    estado = pd.Series("SP", index=raw.index)
    estado.loc[raw.index % 2 == 0] = "RJ"
    return raw.assign(estado=estado)


def test_demand_unit_id_returns_cd_setor_when_not_grouping_by_state():
    raw = load_demand_base(FIXTURE_PATH)
    assert demand_unit_id(raw, group_by_state=False).equals(raw["cd_setor"])


def test_demand_unit_id_raises_without_estado_column():
    raw = load_demand_base(FIXTURE_PATH)
    with pytest.raises(ValueError, match="estado"):
        demand_unit_id(raw, group_by_state=True)


def test_demand_unit_id_combines_sector_and_state():
    raw = _with_synthetic_states(load_demand_base(FIXTURE_PATH))
    unit = demand_unit_id(raw, group_by_state=True)
    assert set(unit) == {"3257|SP", "3257|RJ", "3260|SP", "3260|RJ", "3262|SP", "3262|RJ"}


def test_build_item_panel_group_by_state_splits_items_by_state():
    raw = _with_synthetic_states(load_demand_base(FIXTURE_PATH))
    panel = build_item_panel(raw, group_by_state=True)

    assert panel["cd_setor"].str.contains(r"\|").all()
    split = panel["cd_setor"].str.split("|", n=1, expand=True)
    assert set(split[1]) <= {"SP", "RJ"}
    assert set(split[0]) == {"3257", "3260", "3262"}


def test_resum_to_sector_undoes_a_group_by_state_split():
    raw = _with_synthetic_states(load_demand_base(FIXTURE_PATH))
    unsplit = build_item_panel(raw)
    split = build_item_panel(raw, group_by_state=True)

    resummed = resum_to_sector(split, ["items", "orders", "volumes"])
    left = resummed.set_index(["cd_setor", "CICLOS"])[["items", "orders", "volumes"]].sort_index()
    right = unsplit.set_index(["cd_setor", "CICLOS"])[["items", "orders", "volumes"]].sort_index()
    pd.testing.assert_frame_equal(left, right, check_like=True)


def test_resum_to_sector_is_a_no_op_on_an_unsplit_frame():
    raw = load_demand_base(FIXTURE_PATH)
    panel = build_item_panel(raw)
    assert resum_to_sector(panel, ["items"]) is panel


def test_build_shape_panel_group_by_state_resums_to_the_same_shares():
    raw = _with_synthetic_states(load_demand_base(FIXTURE_PATH))
    unsplit = build_shape_panel(raw)
    split = build_shape_panel(raw, group_by_state=True)

    resummed = resum_to_sector(split.drop(columns="share"), ["items"])
    totals = resummed.set_index(["cd_setor", "CICLOS", "relative_position"])["items"]
    expected = unsplit.set_index(["cd_setor", "CICLOS", "relative_position"])["items"]
    pd.testing.assert_series_equal(totals.sort_index(), expected.sort_index(), check_names=False)


def test_build_daily_base_group_by_state_resums_to_the_same_actuals():
    raw = _with_synthetic_states(load_demand_base(FIXTURE_PATH))
    unsplit = build_daily_base(raw)
    split = build_daily_base(raw, group_by_state=True)

    resummed = resum_to_sector(split.actuals, ["items"])
    left = resummed.set_index(["cd_setor", "cd_cd", "CICLOS", "date"])["items"].sort_index()
    right = unsplit.actuals.set_index(["cd_setor", "cd_cd", "CICLOS", "date"])["items"].sort_index()
    pd.testing.assert_series_equal(left, right, check_names=False)


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
