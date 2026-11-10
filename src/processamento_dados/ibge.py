"""Shared readers for the IBGE per-UF files (SIDRA quarterly tables and the 2022 census)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

# Sigla -> (código IBGE, nome)
UFS = {
    "RO": (11, "Rondônia"),
    "AC": (12, "Acre"),
    "AM": (13, "Amazonas"),
    "RR": (14, "Roraima"),
    "PA": (15, "Pará"),
    "AP": (16, "Amapá"),
    "TO": (17, "Tocantins"),
    "MA": (21, "Maranhão"),
    "PI": (22, "Piauí"),
    "CE": (23, "Ceará"),
    "RN": (24, "Rio Grande do Norte"),
    "PB": (25, "Paraíba"),
    "PE": (26, "Pernambuco"),
    "AL": (27, "Alagoas"),
    "SE": (28, "Sergipe"),
    "BA": (29, "Bahia"),
    "MG": (31, "Minas Gerais"),
    "ES": (32, "Espírito Santo"),
    "RJ": (33, "Rio de Janeiro"),
    "SP": (35, "São Paulo"),
    "PR": (41, "Paraná"),
    "SC": (42, "Santa Catarina"),
    "RS": (43, "Rio Grande do Sul"),
    "MS": (50, "Mato Grosso do Sul"),
    "MT": (51, "Mato Grosso"),
    "GO": (52, "Goiás"),
    "DF": (53, "Distrito Federal"),
}
COD_POR_SIGLA = {s: c for s, (c, _) in UFS.items()}
SIGLA_POR_COD = {c: s for s, (c, _) in UFS.items()}
SIGLA_POR_NOME = {n: s for s, (_, n) in UFS.items()}


def indice_trimestre(ano, trimestre):
    """Inteiro monotônico: 1º tri 2024 -> 2024*4 + 0."""
    return np.asarray(ano) * 4 + (np.asarray(trimestre) - 1)


def _ler_csv_irregular(caminho: Path) -> pd.DataFrame:
    """Linhas com número de colunas diferente (títulos, notas): lê com csv e preenche."""
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas_csv = list(csv.reader(f, delimiter=";"))
    largura = max(len(linha) for linha in linhas_csv)
    return pd.DataFrame(
        [linha + [None] * (largura - len(linha)) for linha in linhas_csv],
        dtype=object,
    )


def _localizar_cabecalho_trimestres(bruto: pd.DataFrame) -> pd.Series:
    linha_cab = bruto.index[
        bruto.apply(lambda r: r.astype(str).str.contains("trimestre").any(), axis=1)
    ][0]
    return bruto.loc[linha_cab]


def _colunas_de_trimestre(cab: pd.Series) -> dict:
    cols_tri = {}
    for j, rot in cab.items():
        m = pd.Series([str(rot)]).str.extract(r"(\d)º trimestre (\d{4})").iloc[0]
        if m.notna().all():
            cols_tri[j] = int(indice_trimestre(int(m[1]), int(m[0])))
    return cols_tri


def _codigo_uf_da_linha(r: pd.Series, formato_nivel: bool) -> int | None:
    """Salário: (Nível, Cód., Nome, ...) com Nível == "UF"; desemprego: (Cód., Nome, ...)."""
    if formato_nivel:
        if str(r[0]).strip() != "UF":
            return None
        cod = r[1]
    else:
        cod = r[0]
    try:
        cod = int(str(cod).strip())
    except ValueError:
        return None
    return cod if cod in SIGLA_POR_COD else None  # ignora Brasil e Grandes Regiões (1 a 5)


def ler_ibge_trimestral(caminho: Path, nome_valor: str) -> pd.DataFrame:
    """Lê as tabelas SIDRA (salário e desemprego) e devolve uf, q, <nome_valor>."""
    bruto = _ler_csv_irregular(caminho)
    cab = _localizar_cabecalho_trimestres(bruto)
    cols_tri = _colunas_de_trimestre(cab)
    formato_nivel = str(cab[0]).strip() == "Nível"
    linha_cab = cab.name

    linhas = []
    for _, r in bruto.iloc[linha_cab + 1 :].iterrows():
        cod = _codigo_uf_da_linha(r, formato_nivel)
        if cod is None:
            continue
        for j, q in cols_tri.items():
            v = pd.to_numeric(str(r[j]).replace(",", "."), errors="coerce")  # X, ... => NaN
            linhas.append({"uf": SIGLA_POR_COD[cod], "q": q, nome_valor: v})
    return pd.DataFrame(linhas)


def ler_censo(caminho: Path) -> pd.DataFrame:
    """Lê população e densidade por UF do Censo 2022: colunas uf, populacao, densidade."""
    bruto = pd.read_excel(caminho, sheet_name=0, header=None, engine="calamine")
    linhas = []
    for _, r in bruto.iterrows():
        nome = str(r[0]).strip()
        if nome in SIGLA_POR_NOME:
            linhas.append(
                {
                    "uf": SIGLA_POR_NOME[nome],
                    "populacao": pd.to_numeric(r[1], errors="coerce"),
                    "densidade": pd.to_numeric(r[3], errors="coerce"),
                }
            )
    return pd.DataFrame(linhas)
