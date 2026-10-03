"""Agrega execuções do SA: gap, tabela por configuração, efeitos principais e regra de decisão."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.calibration.gap import MipReference, compute_gap, mip_reference
from src.solver.heuristics.simulated_annealing import AnnealingParams

P90 = 0.9
DEFAULT_PARAMS = AnnealingParams()
SEARCH_PARAMS = (
    "initial_temperature",
    "min_temperature",
    "sector_bias",
    "destination_bias",
    "penalty_coefficient",
)
MIP_COLUMNS = ("instance", "termination", "objective", "lower_bound", "upper_bound", "gap", "time")
CALIBRATED_PARAMS = (
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "sector_bias",
    "destination_bias",
    "penalty_coefficient",
)


def add_gaps(runs: pd.DataFrame, references: Mapping[str, MipReference]) -> pd.DataFrame:
    """Acrescenta `gap` (contra a referência do MIP) e `gap_vs_incumbent` a cada execução."""
    ref_value = runs["instance"].map(lambda name: references[name].value)
    incumbent = runs["instance"].map(lambda name: references[name].incumbent)
    return runs.assign(
        gap=_gaps(runs, ref_value),
        gap_vs_incumbent=_gaps(runs, incumbent),
        mip_ref_kind=runs["instance"].map(lambda name: references[name].kind),
    )


def _gaps(runs: pd.DataFrame, reference: pd.Series) -> list[float]:
    rows = zip(runs["objective"], runs["feasible"], reference, strict=True)
    return [compute_gap(obj, bool(feasible), ref) for obj, feasible, ref in rows]


def add_group(runs: pd.DataFrame, tight_instances: Iterable[str]) -> pd.DataFrame:
    """Marca cada execução como `tight` (capacidade restringe) ou `loose`."""
    tight = set(tight_instances)
    return runs.assign(group=np.where(runs["instance"].isin(tight), "tight", "loose"))


def summarize(runs: pd.DataFrame, by: Sequence[str] = ("config_id",)) -> pd.DataFrame:
    """Gap médio, desvio, P90, taxa de viabilidade, iterações e tempo por agrupamento."""
    grouped = runs.groupby(list(by), sort=False)
    summary = grouped.agg(
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        gap_p90=("gap", lambda g: g.quantile(P90)),
        gap_vs_incumbent_mean=("gap_vs_incumbent", "mean"),
        feasible_rate=("feasible", "mean"),
        iterations_mean=("iterations", "mean"),
        time_mean=("time", "mean"),
        n_runs=("gap", "size"),
    )
    return summary.reset_index()


def config_scores(runs: pd.DataFrame) -> pd.Series:
    """Score de cada configuração para o TPE: gap médio sobre todas as suas execuções."""
    return runs.groupby("config_id")["gap"].mean()


def main_effects(runs: pd.DataFrame, params: Sequence[str], n_bins: int = 4) -> pd.DataFrame:
    """Gap médio por faixa (quantis) de cada parâmetro — o efeito principal de cada um."""
    frames = [_effect_of(runs, param, n_bins) for param in params]
    return pd.concat(frames, ignore_index=True)


def _effect_of(runs: pd.DataFrame, param: str, n_bins: int) -> pd.DataFrame:
    levels = pd.qcut(runs[param], q=n_bins, duplicates="drop")
    effect = runs.groupby(levels, observed=True)["gap"].agg(["mean", "std", "size"])
    effect = effect.reset_index().rename(columns={param: "level", "mean": "gap_mean"})
    effect["level"] = effect["level"].map(interval_label)
    return effect.rename(columns={"std": "gap_std", "size": "n_runs"}).assign(param=param)


def interval_label(interval: pd.Interval) -> str:
    """Faixa legível para tabelas e eixos, com 3 algarismos significativos: `0.1–1.5`."""
    return f"{interval.left:.3g}–{interval.right:.3g}"


def complexity(config: Mapping[str, float]) -> int:
    """Quantos parâmetros a configuração muda em relação ao default do repositório."""
    return sum(
        1
        for name in CALIBRATED_PARAMS
        if name in config and not np.isclose(config[name], getattr(DEFAULT_PARAMS, name))
    )


def rank_by_rule(summary: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Ordena pela regra: menor gap médio; empate -> menor P90; empate -> mais simples.

    Duas configurações empatam num critério quando diferem por no máximo
    `tolerance` pontos percentuais. A ordem sai de escolher repetidamente a
    melhor das que restam, então a primeira linha é a recomendação.
    """
    remaining, ordered = summary.dropna(subset=["gap_mean"]), []
    while not remaining.empty:
        best = _pick_best(remaining, tolerance)
        ordered.append(best)
        remaining = remaining.drop(index=best)
    return summary.loc[ordered].reset_index(drop=True)


def _pick_best(summary: pd.DataFrame, tolerance: float):
    """Índice da melhor linha segundo os três critérios em cascata."""
    tied = _within(summary, "gap_mean", tolerance)
    tied = _within(tied, "gap_p90", tolerance)
    return tied.sort_values(["complexity", "gap_mean"]).index[0]


def _within(frame: pd.DataFrame, column: str, tolerance: float) -> pd.DataFrame:
    """Linhas a no máximo `tolerance` do melhor valor da coluna."""
    return frame[frame[column] <= frame[column].min() + tolerance]


def prepare_runs(
    runs: pd.DataFrame, references: Mapping[str, MipReference], tight_instances: Iterable[str]
) -> pd.DataFrame:
    """Execuções prontas para análise: com gap, gap contra o incumbente e grupo tight/loose."""
    return add_group(add_gaps(runs, references), tight_instances)


def configs_from_runs(runs: pd.DataFrame, params: Sequence[str]) -> dict[str, dict[str, float]]:
    """Os valores de `params` de cada configuração, lidos da primeira execução dela."""
    first = runs.drop_duplicates("config_id").set_index("config_id")[list(params)]
    return {config_id: dict(row) for config_id, row in first.iterrows()}


def history_from_runs(
    runs: pd.DataFrame, params: Sequence[str]
) -> list[tuple[dict[str, float], float]]:
    """Pares (configuração, score) para aquecer o TPE; configurações sem score ficam de fora."""
    scores = config_scores(runs).dropna()
    configs = configs_from_runs(runs, params)
    return [(configs[config_id], float(score)) for config_id, score in scores.items()]


def ranked_summary(runs: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Tabela por configuração, com complexidade, ordenada pela regra de decisão."""
    summary = summarize(runs)
    configs = configs_from_runs(runs, CALIBRATED_PARAMS)
    summary["complexity"] = summary["config_id"].map(lambda c: complexity(configs[c]))
    return rank_by_rule(summary, tolerance)


def best_of(runs: pd.DataFrame, tolerance: float) -> tuple[str, dict[str, float]]:
    """Id e parâmetros da configuração que a regra de decisão põe em primeiro."""
    best_id = ranked_summary(runs, tolerance)["config_id"].iloc[0]
    return best_id, configs_from_runs(runs[runs["config_id"] == best_id], CALIBRATED_PARAMS)[
        best_id
    ]


def mip_table(records: Iterable[Mapping]) -> pd.DataFrame:
    """Uma linha por instância: status e limites do MIP e a referência de gap usada."""
    return pd.DataFrame([_mip_row(record) for record in records])


def _mip_row(record: Mapping) -> dict:
    reference = mip_reference(record)
    row = {key: record.get(key) for key in MIP_COLUMNS}
    return {**row, "reference_kind": reference.kind, "reference": reference.value}


def always_feasible(runs: pd.DataFrame) -> pd.DataFrame:
    """Só as execuções de configurações que terminaram viáveis em todas as execuções.

    Separa o efeito de um parâmetro sobre a qualidade do efeito dele sobre a
    viabilidade: sem esse filtro, as execuções inviáveis (gap 100) dominam as médias.
    """
    return runs[runs.groupby("config_id")["feasible"].transform("all")]


def feasibility_by(runs: pd.DataFrame, param: str, n_bins: int = 6) -> pd.DataFrame:
    """Taxa de execuções viáveis por faixa (quantis) de um parâmetro."""
    levels = pd.qcut(runs[param], q=n_bins, duplicates="drop")
    table = runs.groupby(levels, observed=True)["feasible"].agg(["mean", "size"]).reset_index()
    table[param] = table[param].map(interval_label)
    return table.rename(columns={param: "level", "mean": "feasible_rate", "size": "n_runs"})
