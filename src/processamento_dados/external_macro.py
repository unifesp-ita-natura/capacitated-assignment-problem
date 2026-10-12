"""Fetches UF-level unemployment rate and average income from IBGE's SIDRA API (PNAD Contínua)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SIDRA_BASE_URL = "https://apisidra.ibge.gov.br/values"
DEFAULT_RAW_CACHE_PATH = Path("data/external/raw")

# PNAD Contínua trimestral, table 6468: taxa de desocupação by UF (variable 4099).
# No classification needed — the table has no sex/age breakdown.
UNEMPLOYMENT_TABLE = "6468"
UNEMPLOYMENT_VARIABLE = "4099"
UNEMPLOYMENT_LABEL = "taxa_desocupacao"

# PNAD Contínua trimestral, table 5436: rendimento médio mensal real (habitual, trabalho
# principal) by UF (variable 5932), restricted to classification 2 ("Sexo"), category
# 6794 ("Total") to avoid the sex breakdown.
INCOME_TABLE = "5436"
INCOME_VARIABLE = "5932"
INCOME_CLASSIFICATION = "2"
INCOME_CATEGORY = "6794"
INCOME_LABEL = "rendimento_medio"

# IBGE territorial-level code for "Unidade da Federação" (UF), used as SIDRA's `n3`.
UF_TERRITORIAL_LEVEL = "n3"

# SIDRA's numeric UF codes, keyed by the two-letter UF abbreviation used elsewhere in
# this repo's demand data (`estado` column in demanda_level_by_state.csv`, etc.).
UF_CODES: dict[str, str] = {
    "RO": "11",
    "AC": "12",
    "AM": "13",
    "RR": "14",
    "PA": "15",
    "AP": "16",
    "TO": "17",
    "MA": "21",
    "PI": "22",
    "CE": "23",
    "RN": "24",
    "PB": "25",
    "PE": "26",
    "AL": "27",
    "SE": "28",
    "BA": "29",
    "MG": "31",
    "ES": "32",
    "RJ": "33",
    "SP": "35",
    "PR": "41",
    "SC": "42",
    "RS": "43",
    "MS": "50",
    "MT": "51",
    "GO": "52",
    "DF": "53",
}


def _build_period_segment(periods: str) -> str:
    """Build the `/p/<periods>` URL segment (e.g. `202501,202502` or `last 8`)."""
    return f"p/{periods}"


def _build_classification_segment(classification: str | None, category: str | None) -> str:
    """Build the optional `/c<classification>/<category>` URL segment, or `""` if unused."""
    if classification is None or category is None:
        return ""
    return f"c{classification}/{category}"


def build_sidra_url(
    table: str,
    variable: str,
    ufs: list[str],
    periods: str,
    classification: str | None = None,
    category: str | None = None,
) -> str:
    """Build a SIDRA `/values` API URL for a table/variable, a list of UFs, and a period range."""
    uf_codes = ",".join(UF_CODES[uf] for uf in ufs)
    segments = [
        SIDRA_BASE_URL,
        f"t/{table}",
        f"{UF_TERRITORIAL_LEVEL}/{uf_codes}",
        f"v/{variable}",
        _build_period_segment(periods),
        _build_classification_segment(classification, category),
    ]
    return "/".join(segment for segment in segments if segment)


def call_sidra_api(url: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Call the SIDRA API and return its decoded JSON body (a list of row dicts)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _cache_file_path(url: str, cache_path: Path) -> Path:
    """Derive a stable cache filename for a SIDRA request URL (hashed, since URLs are long)."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_path / f"sidra_{digest}.json"


def cache_response(data: list[dict[str, Any]], url: str, cache_path: Path) -> Path:
    """Write a SIDRA JSON response to disk, keyed by its request URL, and return the file path."""
    cache_path.mkdir(parents=True, exist_ok=True)
    destination = _cache_file_path(url, cache_path)
    destination.write_text(json.dumps({"url": url, "data": data}, ensure_ascii=False, indent=2))
    return destination


def load_cached_response(url: str, cache_path: Path) -> list[dict[str, Any]] | None:
    """Return a previously cached SIDRA response for this URL, or None if not cached."""
    destination = _cache_file_path(url, cache_path)
    if not destination.exists():
        return None
    return json.loads(destination.read_text())["data"]


def fetch_sidra_data(url: str, cache_path: Path = DEFAULT_RAW_CACHE_PATH) -> list[dict[str, Any]]:
    """Return a SIDRA response for `url`, using the on-disk cache when available."""
    cached = load_cached_response(url, cache_path)
    if cached is not None:
        logger.info("Using cached SIDRA response for %s", url)
        return cached

    logger.info("Calling SIDRA API: %s", url)
    data = call_sidra_api(url)
    cache_response(data, url, cache_path)
    return data


def parse_sidra_response(raw_rows: list[dict[str, Any]], indicador: str) -> pd.DataFrame:
    """Parse a SIDRA `/values` JSON body into a tidy `(estado, periodo, indicador, valor)` frame.

    SIDRA's first row is a header-label dict, not data, and is skipped. Rows whose value is
    the SIDRA "not available" sentinel (`"..."`, `"-"`) are dropped rather than coerced.
    """
    header, *rows = raw_rows
    uf_code_to_abbr = {code: abbr for abbr, code in UF_CODES.items()}
    frame = pd.DataFrame(rows)
    frame["estado"] = frame["D1C"].map(uf_code_to_abbr)
    frame["periodo"] = frame["D3C"]
    frame["valor"] = pd.to_numeric(frame["V"], errors="coerce")
    frame["indicador"] = indicador
    return frame.dropna(subset=["valor"])[["estado", "periodo", "indicador", "valor"]].reset_index(
        drop=True
    )


def fetch_unemployment_rate(
    ufs: list[str], periods: str, cache_path: Path = DEFAULT_RAW_CACHE_PATH
) -> pd.DataFrame:
    """Fetch SIDRA table 6468's UF-level taxa de desocupação for `ufs` over `periods`."""
    url = build_sidra_url(UNEMPLOYMENT_TABLE, UNEMPLOYMENT_VARIABLE, ufs, periods)
    return parse_sidra_response(fetch_sidra_data(url, cache_path), UNEMPLOYMENT_LABEL)


def fetch_average_income(
    ufs: list[str], periods: str, cache_path: Path = DEFAULT_RAW_CACHE_PATH
) -> pd.DataFrame:
    """Fetch SIDRA table 5436's UF-level rendimento médio mensal real for `ufs` over `periods`."""
    url = build_sidra_url(
        INCOME_TABLE,
        INCOME_VARIABLE,
        ufs,
        periods,
        classification=INCOME_CLASSIFICATION,
        category=INCOME_CATEGORY,
    )
    return parse_sidra_response(fetch_sidra_data(url, cache_path), INCOME_LABEL)


def fetch_macro_indicators(
    ufs: list[str], periods: str, cache_path: Path = DEFAULT_RAW_CACHE_PATH
) -> pd.DataFrame:
    """Fetch both indicators for `ufs` over `periods` and stack them into one tidy frame."""
    unemployment = fetch_unemployment_rate(ufs, periods, cache_path)
    income = fetch_average_income(ufs, periods, cache_path)
    return pd.concat([unemployment, income], ignore_index=True)
