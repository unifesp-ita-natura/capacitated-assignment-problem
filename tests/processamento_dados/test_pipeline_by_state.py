"""Tests for the by-(sector, state) demand pipeline's pure transformation logic."""

from __future__ import annotations

import pandas as pd

from src.processamento_dados.pipeline_by_state import (
    _split_sector_state,
    build_raw_demand_with_state,
    merge_with_calendar,
)


def _raw_demand_csv(tmp_path):
    path = tmp_path / "raw_demand.csv"
    path.write_text(
        "data_pedido,cd_cd,cd_setor,rota,cidade,estado,nm_ciclo,aa_ciclo,"
        "total_pedidos_mascarado,total_volumes_mascarado,total_itens_mascarado\n"
        "17/09/2026,5700,S3336,700051,SANTA TEREZINHA,PR,13,2026,28,28,61\n"
        "18/09/2026,5100,S3336,5004,SANTA TEREZINHA,PR,13,2026,5,5,10\n"
    )
    return path


def test_build_raw_demand_with_state_strips_sector_prefix_and_builds_ciclo(tmp_path):
    raw = build_raw_demand_with_state(_raw_demand_csv(tmp_path))

    assert raw["cd_setor"].tolist() == [3336, 3336]
    assert raw["ciclo"].tolist() == ["202613", "202613"]
    assert raw["estado"].tolist() == ["PR", "PR"]
    assert raw["data_pedido"].iloc[0] == pd.Timestamp("2026-09-17")


def test_merge_with_calendar_derives_one_indexed_day_in_cycle(tmp_path):
    raw = build_raw_demand_with_state(_raw_demand_csv(tmp_path))
    calendar = pd.DataFrame(
        {
            "COD SETOR": [3336],
            "CICLOS": ["202613"],
            "BLOCO": ["BL-0"],
            "SUB BLOCO": [1],
            "Dt Abertura": ["2026-09-17"],
            "Dt Fechamento": ["2026-10-07"],
            "Qtde dias": [21],
        }
    )

    merged = merge_with_calendar(raw, calendar)

    assert merged["dia_ciclo"].tolist() == [1, 2]
    assert (merged["Dt Abertura"] == pd.Timestamp("2026-09-17")).all()


def test_merge_with_calendar_drops_orders_from_unknown_cycles(tmp_path):
    raw = build_raw_demand_with_state(_raw_demand_csv(tmp_path))
    empty_calendar = pd.DataFrame(
        columns=[
            "COD SETOR",
            "CICLOS",
            "BLOCO",
            "SUB BLOCO",
            "Dt Abertura",
            "Dt Fechamento",
            "Qtde dias",
        ]
    )

    merged = merge_with_calendar(raw, empty_calendar)

    assert merged.empty


def test_split_sector_state_recovers_the_real_sector_and_estado():
    panel = pd.DataFrame({"cd_setor": ["3336|PR", "3336|SC"], "items": [10, 20]})

    split = _split_sector_state(panel)

    assert split["cd_setor"].tolist() == [3336, 3336]
    assert split["estado"].tolist() == ["PR", "SC"]
