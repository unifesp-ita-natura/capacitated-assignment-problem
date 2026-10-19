"""Per-state (UF) socioeconomic features joined onto the level panel."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

_UFS = {
    "RO": (11, "Rondônia"), "AC": (12, "Acre"), "AM": (13, "Amazonas"),
    "RR": (14, "Roraima"), "PA": (15, "Pará"), "AP": (16, "Amapá"),
    "TO": (17, "Tocantins"), "MA": (21, "Maranhão"), "PI": (22, "Piauí"),
    "CE": (23, "Ceará"), "RN": (24, "Rio Grande do Norte"),
    "PB": (25, "Paraíba"), "PE": (26, "Pernambuco"), "AL": (27, "Alagoas"),
    "SE": (28, "Sergipe"), "BA": (29, "Bahia"), "MG": (31, "Minas Gerais"),
    "ES": (32, "Espírito Santo"), "RJ": (33, "Rio de Janeiro"),
    "SP": (35, "São Paulo"), "PR": (41, "Paraná"), "SC": (42, "Santa Catarina"),
    "RS": (43, "Rio Grande do Sul"), "MS": (50, "Mato Grosso do Sul"),
    "MT": (51, "Mato Grosso"), "GO": (52, "Goiás"), "DF": (53, "Distrito Federal"),
}
_SIGLA_POR_COD = {cod: sigla for sigla, (cod, _) in _UFS.items()}
_SIGLA_POR_NOME = {nome: sigla for sigla, (_, nome) in _UFS.items()}

EXTERNAL_COLUMNS = ["uf_populacao", "uf_densidade", "uf_salario_medio", "uf_desemprego"]

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "src" / "processamento_dados"


def _indice_trimestre(ano: int, trimestre: int) -> int:
    return int(ano) * 4 + (int(trimestre) - 1)


def _ler_sidra_trimestral(caminho: Path, nome_valor: str) -> pd.DataFrame:
    """Tidy a SIDRA quarterly table into columns uf, q, <nome_valor>."""
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas_csv = list(csv.reader(f, delimiter=";"))
    largura = max(len(linha) for linha in linhas_csv)
    bruto = pd.DataFrame(
        [linha + [None] * (largura - len(linha)) for linha in linhas_csv], dtype=object
    )
    linha_cab = bruto.index[
        bruto.apply(lambda r: r.astype(str).str.contains("trimestre").any(), axis=1)
    ][0]
    cab = bruto.loc[linha_cab]
    cols_tri = {}
    for j, rotulo in cab.items():
        m = pd.Series([str(rotulo)]).str.extract(r"(\d)º trimestre (\d{4})").iloc[0]
        if m.notna().all():
            cols_tri[j] = _indice_trimestre(int(m[1]), int(m[0]))
    formato_nivel = str(cab[0]).strip() == "Nível"
    linhas = []
    for _, r in bruto.iloc[linha_cab + 1 :].iterrows():
        cod = r[1] if formato_nivel else r[0]
        if formato_nivel and str(r[0]).strip() != "UF":
            continue
        try:
            cod = int(str(cod).strip())
        except ValueError:
            continue
        if cod not in _SIGLA_POR_COD:
            continue
        for j, q in cols_tri.items():
            valor = pd.to_numeric(str(r[j]).replace(",", "."), errors="coerce")
            linhas.append({"uf": _SIGLA_POR_COD[cod], "q": q, nome_valor: valor})
    return pd.DataFrame(linhas)


def _ler_censo(caminho: Path) -> pd.DataFrame:
    """Read the 2022 census sheet into columns uf, uf_populacao, uf_densidade."""
    bruto = pd.read_excel(caminho, sheet_name=0, header=None, engine="calamine")
    linhas = []
    for _, r in bruto.iterrows():
        nome = str(r[0]).strip()
        if nome in _SIGLA_POR_NOME:
            linhas.append(
                {
                    "uf": _SIGLA_POR_NOME[nome],
                    "uf_populacao": pd.to_numeric(r[1], errors="coerce"),
                    "uf_densidade": pd.to_numeric(r[3], errors="coerce"),
                }
            )
    return pd.DataFrame(linhas)


def load_uf_features(
    opening_dates: pd.Series,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """One static row per UF with the columns in EXTERNAL_COLUMNS, keyed by `estado`.

    Salary and unemployment are averaged over the quarters spanned by
    `opening_dates`; population and density come from the 2022 census.
    """
    directory = Path(data_dir) if data_dir is not None else _DEFAULT_DIR
    censo = _ler_censo(directory / "censo_2022.xlsx")
    salario = _ler_sidra_trimestral(directory / "media_salarial.csv", "uf_salario_medio")
    desemprego = _ler_sidra_trimestral(directory / "taxa_desemprego.csv", "uf_desemprego")

    dates = pd.to_datetime(opening_dates.dropna())
    q_min = _indice_trimestre(dates.min().year, dates.min().quarter)
    q_max = _indice_trimestre(dates.max().year, dates.max().quarter)

    tabela = censo.copy()
    for tidy, coluna in ((salario, "uf_salario_medio"), (desemprego, "uf_desemprego")):
        periodo = tidy[(tidy["q"] >= q_min) & (tidy["q"] <= q_max)]
        media = periodo.groupby("uf")[coluna].mean()
        tabela = tabela.merge(media.rename(coluna), on="uf", how="outer")

    tabela = tabela.rename(columns={"uf": "estado"})
    return tabela[["estado", *EXTERNAL_COLUMNS]]
