from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet


def carregar_serie() -> pd.DataFrame:
    caminho_csv = Path(__file__).resolve().parent / "serie_itens_por_dia.csv"
    serie = pd.read_csv(caminho_csv, parse_dates=["data_pedido"])

    colunas_necessarias = {"data_pedido", "total_itens_mascarado"}
    colunas_faltantes = colunas_necessarias.difference(serie.columns)
    if colunas_faltantes:
        raise ValueError(f"Colunas ausentes em {caminho_csv.name}: {sorted(colunas_faltantes)}")

    serie["total_itens_mascarado"] = pd.to_numeric(serie["total_itens_mascarado"], errors="raise")
    return (
        serie.rename(columns={"data_pedido": "ds", "total_itens_mascarado": "y"})[["ds", "y"]]
        .set_index("ds")
        .asfreq("D", fill_value=0)
        .reset_index()
    )


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    erro = y_true - y_pred
    soma_y_true = np.abs(y_true).sum()

    return {
        "MAE": np.abs(erro).mean(),
        "WMAPE": (np.abs(erro).sum() / soma_y_true * 100 if soma_y_true != 0 else np.nan),
        "Bias": erro.mean(),
    }


def avaliar_janela_deslizante(
    serie: pd.DataFrame,
    dias_treino: int = 30,
    dias_teste: int = 30,
    passo: int = 30,
) -> pd.DataFrame:
    resultados, _ = avaliar_janela_deslizante_detalhada(
        serie, dias_treino=dias_treino, dias_teste=dias_teste, passo=passo
    )
    return resultados


def avaliar_janela_deslizante_detalhada(
    serie: pd.DataFrame,
    dias_treino: int = 30,
    dias_teste: int = 30,
    passo: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min(dias_treino, dias_teste, passo) <= 0:
        raise ValueError("dias_treino, dias_teste e passo devem ser positivos.")

    resultados = []
    previsoes_detalhadas = []
    inicio = 0

    while inicio + dias_treino + dias_teste <= len(serie):
        treino = serie.iloc[inicio : inicio + dias_treino]
        teste = serie.iloc[inicio + dias_treino : inicio + dias_treino + dias_teste]

        modelo = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=False,
        )
        modelo.fit(treino)
        previsao = modelo.predict(teste[["ds"]])[["ds", "yhat"]]
        comparacao = teste.merge(previsao, on="ds", validate="one_to_one")
        comparacao["janela"] = len(resultados) + 1
        comparacao["horizonte"] = range(1, len(comparacao) + 1)
        comparacao["residuo"] = comparacao["y"] - comparacao["yhat"]
        previsoes_detalhadas.append(
            comparacao[["ds", "y", "yhat", "residuo", "janela", "horizonte"]]
        )
        metricas = calculate_metrics(comparacao["y"], comparacao["yhat"])

        resultados.append(
            {
                "janela": len(resultados) + 1,
                "treino_inicio": treino["ds"].min(),
                "treino_fim": treino["ds"].max(),
                "teste_inicio": teste["ds"].min(),
                "teste_fim": teste["ds"].max(),
                **{chave.lower(): valor for chave, valor in metricas.items()},
            }
        )
        inicio += passo

    if not resultados:
        raise ValueError(
            "A série não possui dados suficientes para uma janela de "
            "30 dias de treino e 30 dias de teste."
        )

    return pd.DataFrame(resultados), pd.concat(previsoes_detalhadas, ignore_index=True)


def plotar_janelas_validacao(resultados: pd.DataFrame) -> Path:
    caminho_grafico = Path(__file__).resolve().parents[1] / "grafico_janelas_validacao.png"
    grafico = resultados.copy()
    grafico["janela"] = range(1, len(grafico) + 1)

    figura, eixo = plt.subplots(figsize=(14, 8))
    for _, janela in grafico.iterrows():
        treino_inicio = mdates.date2num(janela["treino_inicio"])
        treino_fim = mdates.date2num(janela["treino_fim"])
        teste_inicio = mdates.date2num(janela["teste_inicio"])
        teste_fim = mdates.date2num(janela["teste_fim"])
        altura = 0.35

        eixo.barh(
            janela["janela"],
            treino_fim - treino_inicio + 1,
            left=treino_inicio,
            height=altura,
            color="#4C78A8",
            label="Treino" if janela["janela"] == 1 else None,
        )
        eixo.barh(
            janela["janela"],
            teste_fim - teste_inicio + 1,
            left=teste_inicio,
            height=altura,
            color="#F58518",
            label="Teste" if janela["janela"] == 1 else None,
        )

    eixo.xaxis_date()
    eixo.xaxis.set_major_locator(mdates.AutoDateLocator())
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    eixo.set(
        title="Janelas de Validação Cruzada (Cross-Validation Cutoffs)",
        xlabel="Data",
        ylabel="Janela",
        yticks=grafico["janela"],
    )
    eixo.legend()
    eixo.grid(axis="x", alpha=0.3)
    figura.autofmt_xdate()
    figura.tight_layout()
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    return caminho_grafico


def plotar_comparacao_desempenho(resultados: pd.DataFrame) -> Path:
    caminho_grafico = Path(__file__).resolve().parents[1] / "grafico_comparacao_desempenho.png"
    figura, eixos = plt.subplots(1, 3, figsize=(16, 5))
    metricas = [
        ("mae", "MAE", "#4C78A8"),
        ("wmape", "WMAPE (%)", "#F58518"),
        ("bias", "Bias", "#54A24B"),
    ]

    for eixo, (coluna, rotulo, cor) in zip(eixos, metricas):
        eixo.bar(resultados["janela"].astype(str), resultados[coluna], color=cor)
        eixo.set_title(rotulo)
        eixo.set_xlabel("Janela de validação")
        eixo.set_ylabel(rotulo)
        eixo.grid(axis="y", alpha=0.3)

    figura.suptitle("Comparação de Desempenho entre Janelas")
    figura.tight_layout()
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    return caminho_grafico


def plotar_bias_residuos(previsoes: pd.DataFrame) -> Path:
    caminho_grafico = Path(__file__).resolve().parents[1] / "grafico_bias_residuos.png"
    grafico = previsoes.sort_values("ds").copy()
    grafico["bias_movel"] = grafico["residuo"].rolling(7, min_periods=1).mean()

    figura, eixo = plt.subplots(figsize=(14, 6))
    eixo.scatter(
        grafico["ds"],
        grafico["residuo"],
        s=18,
        alpha=0.55,
        color="#4C78A8",
        label="Resíduo (real - previsto)",
    )
    eixo.plot(
        grafico["ds"],
        grafico["bias_movel"],
        color="#E45756",
        linewidth=2,
        label="Bias móvel (7 dias)",
    )
    eixo.axhline(0, color="black", linewidth=1, linestyle="--")
    eixo.set(
        title="Bias ao Longo do Tempo (Residuals / Bias Plot)",
        xlabel="Data",
        ylabel="Resíduo / Bias",
    )
    eixo.xaxis.set_major_locator(mdates.AutoDateLocator())
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    eixo.legend()
    eixo.grid(alpha=0.3)
    figura.autofmt_xdate()
    figura.tight_layout()
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    return caminho_grafico


def plotar_metricas_por_horizonte(previsoes: pd.DataFrame) -> Path:
    caminho_grafico = Path(__file__).resolve().parents[1] / "grafico_metricas_por_horizonte.png"
    metricas_horizonte = (
        previsoes.groupby("horizonte", sort=True)
        .apply(
            lambda grupo: pd.Series(calculate_metrics(grupo["y"], grupo["yhat"])),
            include_groups=False,
        )
        .reset_index()
    )

    figura, eixos = plt.subplots(1, 3, figsize=(16, 5))
    metricas = [
        ("MAE", "MAE", "#4C78A8"),
        ("WMAPE", "WMAPE (%)", "#F58518"),
        ("Bias", "Bias", "#54A24B"),
    ]
    for eixo, (coluna, rotulo, cor) in zip(eixos, metricas):
        eixo.plot(
            metricas_horizonte["horizonte"],
            metricas_horizonte[coluna],
            marker="o",
            color=cor,
        )
        eixo.set_title(rotulo)
        eixo.set_xlabel("Horizonte (dias à frente)")
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.3)

    figura.suptitle("Métricas por Horizonte de Tempo (Horizon Plot)")
    figura.tight_layout()
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    return caminho_grafico


def main() -> None:
    serie = carregar_serie()
    resultados, previsoes = avaliar_janela_deslizante_detalhada(serie)
    caminhos_graficos = [
        plotar_janelas_validacao(resultados),
        plotar_comparacao_desempenho(resultados),
        plotar_bias_residuos(previsoes),
        plotar_metricas_por_horizonte(previsoes),
    ]

    print(resultados.to_string(index=False))
    print("\nMédias das métricas:")
    print(resultados[["mae", "wmape", "bias"]].mean().to_string())
    print("\nGráficos salvos em:")
    for caminho_grafico in caminhos_graficos:
        print(f"- {caminho_grafico}")


if __name__ == "__main__":
    main()
