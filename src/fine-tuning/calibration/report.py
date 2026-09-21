"""Agrega os CSVs da calibração numa tabela final e desenha a curva de convergência."""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

PARAM_NAMES = [
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "sector_bias",
    "destination_bias",
    "penalty_coefficient",
]


REQUIRED_COLUMNS = {"instance", "seed", "objective", "feasible", "config_id"}


def load_runs(pattern: str) -> pd.DataFrame:
    """Todas as execuções gravadas pelos scripts de calibração, num único frame.

    CSVs que não tenham as colunas de execução (as próprias saídas deste script,
    por exemplo) são ignorados, então o padrão pode ser um `results/*.csv` solto.
    """
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        if REQUIRED_COLUMNS.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"nenhum CSV de execuções casou com {pattern!r}")
    return pd.concat(frames, ignore_index=True)


def best_known_values(runs: pd.DataFrame) -> pd.Series:
    """Melhor objetivo viável já visto por instância (proxy do ótimo do MIP)."""
    return runs.loc[runs["feasible"]].groupby("instance")["objective"].min()


def add_gap(runs: pd.DataFrame, bkv: pd.Series) -> pd.DataFrame:
    """Gap percentual contra o BKV da instância; execuções inviáveis ficam como NaN."""
    reference = runs["instance"].map(bkv)
    gap = 100.0 * (runs["objective"] - reference) / reference
    return runs.assign(gap=np.where(runs["feasible"], gap, np.nan))


def summarize(runs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Média e desvio do gap, taxa de viabilidade, iterações e tempo por agrupamento."""
    grouped = runs.groupby(by)
    summary = grouped.agg(
        n=("gap", "size"),
        gap_medio=("gap", "mean"),
        gap_desvio=("gap", "std"),
        viavel_pct=("feasible", lambda column: 100.0 * column.mean()),
        objetivo_medio=("objective", "mean"),
        iteracoes=("iterations", "mean"),
        segundos=("seconds", "mean"),
    )
    return summary.sort_values("gap_medio")


def main_effects(runs: pd.DataFrame) -> pd.DataFrame:
    """Efeito principal de cada parâmetro: gap médio por nível observado."""
    rows = []
    for name in PARAM_NAMES:
        for level, group in runs.groupby(name):
            rows.append(
                {
                    "parametro": name,
                    "nivel": level,
                    "gap_medio": group["gap"].mean(),
                    "iteracoes": group["iterations"].mean(),
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows)


def plot_convergence(trace_path: str, output_path: str) -> None:
    """Energia corrente/melhor e temperatura ao longo das iterações."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = np.load(trace_path)  # (seeds, iterações, 3): corrente, melhor, temperatura
    iterations = np.arange(traces.shape[1])

    figure, (energy_ax, temperature_ax) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1]
    )
    for index, trace in enumerate(traces):
        energy_ax.plot(
            iterations, trace[:, 0], lw=0.5, alpha=0.45, label=f"corrente (seed {index + 1})"
        )
        energy_ax.plot(iterations, trace[:, 1], lw=1.8, label=f"melhor (seed {index + 1})")
    energy_ax.set_ylabel("Energia E(X)")
    energy_ax.set_title("Convergência do SA — melhor configuração calibrada")
    energy_ax.legend(fontsize=7, ncol=2)
    energy_ax.grid(alpha=0.3)

    temperature_ax.plot(iterations, traces[0][:, 2], color="tab:red")
    temperature_ax.set_yscale("log")
    temperature_ax.set_ylabel("Temperatura")
    temperature_ax.set_xlabel("Iteração")
    temperature_ax.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"[report] escrito {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/*.csv")
    parser.add_argument("--trace", default="results/trace.npy")
    parser.add_argument("--out-prefix", default="results/report")
    arguments = parser.parse_args()

    runs = load_runs(arguments.runs)
    bkv = best_known_values(runs)
    runs = add_gap(runs, bkv)

    print("Melhor valor conhecido (BKV) por instância:")
    print(bkv.round(1).to_string(), "\n")

    per_config = summarize(runs, ["config_id"]).join(runs.groupby("config_id").first()[PARAM_NAMES])
    per_config.to_csv(f"{arguments.out_prefix}_por_config.csv")

    validation = runs[runs["config_id"].str.match(r"default_repo|cal_")]
    if not validation.empty:
        table = summarize(validation, ["config_id", "instance"])
        table.to_csv(f"{arguments.out_prefix}_validacao.csv")
        print("Revalidação (por configuração e instância):")
        print(table.round(2).to_string(), "\n")
        print("Revalidação (agregada por configuração):")
        print(summarize(validation, ["config_id"]).round(2).to_string(), "\n")

    effects = main_effects(runs[runs["config_id"].str.startswith("screen")])
    effects.to_csv(f"{arguments.out_prefix}_efeitos.csv", index=False)
    print("Efeitos principais (varredura em grade):")
    print(effects.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

    plot_convergence(arguments.trace, f"{arguments.out_prefix}_convergencia.png")


if __name__ == "__main__":
    main()
