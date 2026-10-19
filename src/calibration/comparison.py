"""Comparação Gurobi × HiGHS × SA nas mesmas instâncias: config, referência do gap e tabelas."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.calibration.config import BudgetSettings, ExperimentDescription, InstanceSpec
from src.calibration.gap import compute_gap

REFERENCE_AGREEMENT = 1e-4  # dois MIPs fechados a 1e-4 não podem discordar mais que isso


class SolverComparisonConfig(BaseModel):
    """O YAML de `configs/experiments/sa_calibration/solver_comparison.yaml`."""

    experiment_name: str
    description: ExperimentDescription
    output_dir: str
    n_workers: int = Field(default=0, ge=0)
    level_strategy: str
    shape_strategy: str
    budget: BudgetSettings = BudgetSettings()
    instances: dict[str, InstanceSpec]
    mip_solvers: list[str]
    mip_rel_gaps: list[float]  # a menor é a do ótimo provado (z_ref)
    # solver -> opções nativas extras (ex.: Threads, NodefileStart do Gurobi); textos aceitam
    # {instance} e {gap}, e os de caminho (NodefileDir, SolFiles, LogFile) têm a pasta criada
    mip_extra_options: dict[str, dict[str, float | int | str]] = {}
    sa_configs: dict[str, dict[str, float | str]]
    sa_seeds: list[int]

    @property
    def reference_gap(self) -> float:
        """Tolerância cujo ótimo provado é o denominador do gap (z_ref)."""
        return min(self.mip_rel_gaps)

    @property
    def loose_gap(self) -> float:
        """Tolerância da calibração anterior, reportada ao lado para contraste."""
        return max(self.mip_rel_gaps)


PATH_OPTIONS = {"NodefileDir": "dir", "SolFiles": "parent", "LogFile": "parent"}


def solve_options(
    config: SolverComparisonConfig, solver: str, instance: str, gap: float
) -> dict[str, float | int | str]:
    """Opções extras de `solver` num solve: {instance}/{gap} preenchidos, pastas criadas."""
    options = {
        key: value.format(instance=instance, gap=f"{gap:g}") if isinstance(value, str) else value
        for key, value in config.mip_extra_options.get(solver, {}).items()
    }
    _create_option_dirs(options)
    return options


def _create_option_dirs(options: Mapping[str, float | int | str]) -> None:
    """Cria a pasta de cada opção de caminho (o Gurobi não cria sozinho)."""
    for key in PATH_OPTIONS.keys() & options.keys():
        path = Path(str(options[key]))
        (path if PATH_OPTIONS[key] == "dir" else path.parent).mkdir(parents=True, exist_ok=True)


def load_comparison_config(path: str | Path) -> SolverComparisonConfig:
    """Lê e valida o YAML da comparação."""
    with open(path) as f:
        return SolverComparisonConfig.model_validate(yaml.safe_load(f))


def mip_record_name(instance: str, solver: str, mip_rel_gap: float) -> str:
    """Nome do JSON de um solve: `{instância}_{solver}_gap{tolerância}`."""
    return f"{instance}_{solver}_gap{mip_rel_gap:g}"


def proven_optimum(records: Sequence[Mapping], tolerance: float) -> float | None:
    """Melhor UB entre os MIPs que fecharam com `tolerance`; None se nenhum fechou.

    Levanta `ValueError` se dois MIPs fechados discordarem mais que a própria
    tolerância somada à `REFERENCE_AGREEMENT`: aí não há ótimo provado único e
    o gap não teria referência confiável.
    """
    closed = _closed_upper_bounds(records, tolerance)
    if not closed:
        return None
    best, worst = min(closed), max(closed)
    if (worst - best) / abs(best) > tolerance + REFERENCE_AGREEMENT:
        raise ValueError(f"MIPs fechados discordam: {sorted(closed)} (tolerância {tolerance})")
    return float(best)


def _closed_upper_bounds(records: Sequence[Mapping], tolerance: float) -> list[float]:
    """UBs dos MIPs que pediram `tolerance` e provaram otimalidade dentro dela."""
    return [
        r["upper_bound"]
        for r in records
        if r["mip_rel_gap"] == tolerance and r["termination"] == "optimal"
    ]


def best_lower_bound(records: Sequence[Mapping]) -> float | None:
    """Maior limite inferior entre os MIPs (o denominador quando nenhum fechou)."""
    bounds = [r["lower_bound"] for r in records if r.get("lower_bound") is not None]
    return max(bounds) if bounds else None


def group_by_instance(
    records: Sequence[Mapping], instances: Iterable[str]
) -> dict[str, list[Mapping]]:
    """Registros do MIP agrupados por instância (lista vazia se a instância não tem nenhum)."""
    return {name: [r for r in records if r["instance"] == name] for name in instances}


def reference_upper_bound(
    records: Sequence[Mapping], solver: str, tolerance: float
) -> float | None:
    """UB de `solver` com `tolerance` (ex.: o HiGHS a 0,01 da calibração anterior)."""
    for r in records:
        if r["solver"] == solver and r["mip_rel_gap"] == tolerance:
            return r["upper_bound"]
    return None


def mip_rows(records: Sequence[Mapping], z_ref: Mapping[str, float | None]) -> pd.DataFrame:
    """Uma linha por solve do MIP, com o gap da solução dele contra o ótimo provado."""
    rows = [
        {
            "instance": r["instance"],
            "method": f"{r['solver']} (gap {r['mip_rel_gap']:g})",
            "status": r["termination"],
            "objective": r["upper_bound"],
            "lower_bound": r["lower_bound"],
            "mip_residual_gap_pct": _pct(r["gap"]),
            "gap_vs_zref_pct": _relative_pct(r["upper_bound"], z_ref.get(r["instance"])),
            "time_s": r["time"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def sa_rows(
    runs: pd.DataFrame,
    z_ref: Mapping[str, float | None],
    other_refs: Mapping[str, Mapping[str, float | None]],
) -> pd.DataFrame:
    """Uma linha por (instância, config do SA): gap (média, desvio, P90), viabilidade e carga.

    `other_refs` mapeia um rótulo (ex.: `highs0.01`, `lb`) a outra referência por
    instância; cada uma vira a coluna `gap_vs_{rótulo}_pct` (só a média).
    """
    rows = []
    for (instance, config_id), group in runs.groupby(["instance", "config_id"], sort=True):
        gaps = _gaps(group, z_ref.get(instance))
        others = {
            f"gap_vs_{label}_pct": _gaps(group, ref.get(instance)).mean()
            for label, ref in other_refs.items()
        }
        rows.append(
            {
                "instance": instance,
                "method": f"SA {config_id} ({len(group)} seeds)",
                "status": f"{int(group['feasible'].sum())}/{len(group)} viáveis",
                "objective": group["objective"].mean(),
                "gap_vs_zref_pct": gaps.mean(),
                "gap_vs_zref_std": gaps.std(ddof=1),
                "gap_vs_zref_p90": gaps.quantile(0.9),
                **others,
                "peak": group["zmax"].mean(),
                "std_load": group["std_load"].mean(),
                "time_s": group["time"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _gaps(group: pd.DataFrame, reference: float | None) -> pd.Series:
    return pd.Series(
        [
            compute_gap(obj, bool(feasible), reference)
            for obj, feasible in zip(group["objective"], group["feasible"], strict=True)
        ],
        dtype=float,
    )


def _relative_pct(value: float | None, reference: float | None) -> float:
    if value is None or reference is None or reference == 0:
        return math.nan
    return 100.0 * (value - reference) / reference


def _pct(value: float | None) -> float:
    return math.nan if value is None else 100.0 * value
