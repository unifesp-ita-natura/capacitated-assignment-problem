"""Tests for the SIDRA macro-indicator fetcher's pure parsing and caching logic."""

from __future__ import annotations

import json

from src.processamento_dados.external_macro import (
    build_sidra_url,
    cache_response,
    load_cached_response,
    parse_sidra_response,
)

# Shaped like a real SIDRA `/values` response for table 6468 (taxa de desocupação),
# n3=35,41 (SP, PR), p=202601,202602: a header-label row followed by data rows, one
# of which carries SIDRA's "not available" sentinel.
SAMPLE_SIDRA_RESPONSE = [
    {
        "NC": "Nível Territorial (Código)",
        "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)",
        "MN": "Unidade de Medida",
        "V": "Valor",
        "D1C": "Unidade da Federação (Código)",
        "D1N": "Unidade da Federação",
        "D2C": "Variável (Código)",
        "D2N": "Variável",
        "D3C": "Trimestre (Código)",
        "D3N": "Trimestre",
    },
    {
        "NC": "3",
        "NN": "Unidade da Federação",
        "MC": "2",
        "MN": "%",
        "V": "6.0",
        "D1C": "35",
        "D1N": "São Paulo",
        "D2C": "4099",
        "D2N": "Taxa de desocupação",
        "D3C": "202601",
        "D3N": "1º trimestre 2026",
    },
    {
        "NC": "3",
        "NN": "Unidade da Federação",
        "MC": "2",
        "MN": "%",
        "V": "5.4",
        "D1C": "35",
        "D1N": "São Paulo",
        "D2C": "4099",
        "D2N": "Taxa de desocupação",
        "D3C": "202602",
        "D3N": "2º trimestre 2026",
    },
    {
        "NC": "3",
        "NN": "Unidade da Federação",
        "MC": "2",
        "MN": "%",
        "V": "...",
        "D1C": "41",
        "D1N": "Paraná",
        "D2C": "4099",
        "D2N": "Taxa de desocupação",
        "D3C": "202602",
        "D3N": "2º trimestre 2026",
    },
]


def test_build_sidra_url_includes_table_variable_ufs_and_periods():
    url = build_sidra_url("6468", "4099", ["SP", "PR"], "last 8")

    assert url == "https://apisidra.ibge.gov.br/values/t/6468/n3/35,41/v/4099/p/last 8"


def test_build_sidra_url_appends_classification_segment_when_given():
    url = build_sidra_url("5436", "5932", ["SP"], "202601", classification="2", category="6794")

    assert url.endswith("/c2/6794")


def test_parse_sidra_response_builds_tidy_frame():
    parsed = parse_sidra_response(SAMPLE_SIDRA_RESPONSE, "taxa_desocupacao")

    assert list(parsed.columns) == ["estado", "periodo", "indicador", "valor"]
    assert (parsed["indicador"] == "taxa_desocupacao").all()


def test_parse_sidra_response_drops_unavailable_values():
    parsed = parse_sidra_response(SAMPLE_SIDRA_RESPONSE, "taxa_desocupacao")

    # The Paraná/"..." row is dropped, leaving only the two valid São Paulo rows.
    assert parsed["estado"].tolist() == ["SP", "SP"]
    assert parsed["periodo"].tolist() == ["202601", "202602"]
    assert parsed["valor"].tolist() == [6.0, 5.4]


def test_cache_response_round_trips_through_load_cached_response(tmp_path):
    url = "https://apisidra.ibge.gov.br/values/t/6468/n3/35/v/4099/p/202601"
    data = SAMPLE_SIDRA_RESPONSE

    destination = cache_response(data, url, tmp_path)

    assert destination.exists()
    assert json.loads(destination.read_text())["url"] == url
    assert load_cached_response(url, tmp_path) == data


def test_load_cached_response_returns_none_when_not_cached(tmp_path):
    url = "https://apisidra.ibge.gov.br/values/t/6468/n3/35/v/4099/p/202601"

    assert load_cached_response(url, tmp_path) is None
