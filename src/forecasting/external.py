"""Per-state (UF) socioeconomic features joined onto the level panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.processamento_dados.ibge import indice_trimestre, ler_censo, ler_ibge_trimestral

EXTERNAL_COLUMNS = ["uf_populacao", "uf_densidade", "uf_salario_medio", "uf_desemprego"]

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "processamento_dados"


def load_uf_features(
    opening_dates: pd.Series,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """One static row per UF with the columns in EXTERNAL_COLUMNS, keyed by `estado`.

    Salary and unemployment are averaged over the quarters spanned by
    `opening_dates`; population and density come from the 2022 census.
    """
    directory = Path(data_dir) if data_dir is not None else _DEFAULT_DIR
    censo = ler_censo(directory / "censo_2022.xlsx").rename(
        columns={"populacao": "uf_populacao", "densidade": "uf_densidade"}
    )
    salario = ler_ibge_trimestral(directory / "media_salarial.csv", "uf_salario_medio")
    desemprego = ler_ibge_trimestral(directory / "taxa_desemprego.csv", "uf_desemprego")

    dates = pd.to_datetime(opening_dates.dropna())
    q_min = indice_trimestre(dates.min().year, dates.min().quarter)
    q_max = indice_trimestre(dates.max().year, dates.max().quarter)

    tabela = censo.copy()
    for tidy, coluna in ((salario, "uf_salario_medio"), (desemprego, "uf_desemprego")):
        periodo = tidy[(tidy["q"] >= q_min) & (tidy["q"] <= q_max)]
        media = periodo.groupby("uf")[coluna].mean()
        tabela = tabela.merge(media.rename(coluna), on="uf", how="outer")

    tabela = tabela.rename(columns={"uf": "estado"})
    return tabela[["estado", *EXTERNAL_COLUMNS]]
