"""Tests for the shared IBGE readers, run against the repository's real per-UF files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.processamento_dados.ibge import (
    SIGLA_POR_COD,
    SIGLA_POR_NOME,
    UFS,
    indice_trimestre,
    ler_censo,
    ler_ibge_trimestral,
)

DATA_DIR = Path("src/processamento_dados")


def test_uf_tables_cover_the_27_federative_units_consistently():
    assert len(UFS) == 27
    assert SIGLA_POR_COD[35] == "SP"
    assert SIGLA_POR_NOME["Distrito Federal"] == "DF"


def test_indice_trimestre_is_monotonic_across_year_boundaries():
    assert indice_trimestre(2024, 4) + 1 == indice_trimestre(2025, 1)
    assert list(indice_trimestre(pd.Series([2024, 2024]), pd.Series([1, 2]))) == [8096, 8097]


def test_ler_censo_returns_one_row_per_uf_with_population_and_density():
    censo = ler_censo(DATA_DIR / "censo_2022.xlsx")

    assert set(censo.columns) == {"uf", "populacao", "densidade"}
    assert set(censo["uf"]) == set(UFS)
    assert (censo["populacao"] > 0).all()


def test_ler_ibge_trimestral_reads_both_sidra_layouts_and_drops_regions():
    for name, column in (("media_salarial.csv", "salario"), ("taxa_desemprego.csv", "desemprego")):
        tidy = ler_ibge_trimestral(DATA_DIR / name, column)

        assert list(tidy.columns) == ["uf", "q", column]
        assert set(tidy["uf"]) == set(UFS)  # Brasil and the grandes regiões are dropped
