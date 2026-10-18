"""Driver do experimento relax_and_fix: compara o Relax-and-Fix + Fix-and-Optimize com o MIP."""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from experiments.relax_and_fix.config import (
    InstanceSettings,
    RelaxAndFixConfig,
    WindowSettings,
    load_config,
)
from src.calibration import io
from src.calibration.config import InstanceSpec, MipSettings
from src.calibration.instances import CalibrationInstance, build_instance
from src.calibration.mip import (
    MipOutcome,
    assignment_from_model,
    finite_or_none,
    relative_gap,
)
from src.solver.matheuristics import RelaxAndFixParams, RelaxAndFixResult, solve
from src.solver.matheuristics.evaluation import capacity_utilization, evaluate_assignment
from src.solver.matheuristics.problem import AssignmentProblem, build_problem
from src.solver.matheuristics.subproblem import mip_gap_option, pyomo_subproblem_solver
from src.solver.mip.block_assignment import build_block_assignment_model

DEFAULT_CONFIG = "configs/experiments/relax_and_fix/full.yaml"
COLUMNS = [
    "instance",
    "n_sectors",
    "capacity",
    "method",
    "feasible",
    "objective",
    "construction_objective",
    "gap_vs_mip_pct",
    "gap_vs_best_pct",
    "time_s",
    "subproblems",
    "detail",
]


def problem_of(instance: CalibrationInstance) -> AssignmentProblem:
    """Os dados da instância indexados pro avaliador da matheurística."""
    kwargs = instance.mip_kwargs()
    return build_problem(**{k: v for k, v in kwargs.items() if k not in ("combinations", "days")})


def _spec(settings: InstanceSettings, capacity_multiplier: float) -> InstanceSpec:
    return InstanceSpec(
        n_sectors=settings.n_sectors,
        n_cycles_horizon=settings.n_cycles_horizon,
        n_cycles_history=settings.n_cycles_history,
        seed=settings.seed,
        capacity_multiplier=capacity_multiplier,
    )


def build_instance_for(name: str, settings: InstanceSettings) -> CalibrationInstance:
    """Instância sintética; a apertada reduz a capacidade até o As-Is passar do limite."""
    instance = build_instance(name, _spec(settings, 1.0))
    if settings.capacity == "loose":
        return instance
    problem = problem_of(instance)
    multiplier = settings.tight_headroom * capacity_utilization(problem, problem.as_is)
    return build_instance(name, _spec(settings, multiplier))


def _base_row(name: str, settings: InstanceSettings, method: str) -> dict:
    return {
        "instance": name,
        "n_sectors": settings.n_sectors,
        "capacity": settings.capacity,
        "method": method,
    }


def mip_row(
    name: str, settings: InstanceSettings, problem: AssignmentProblem, outcome: MipOutcome
) -> dict:
    """Linha do MIP, com a solução reavaliada pelo mesmo avaliador da matheurística."""
    row = _base_row(name, settings, "mip")
    evaluation = evaluate_assignment(problem, outcome.x) if outcome.x else None
    return {
        **row,
        "feasible": bool(evaluation and evaluation.feasible),
        "objective": evaluation.objective if evaluation else None,
        "construction_objective": None,
        "time_s": outcome.time,
        "subproblems": None,
        "detail": f"{outcome.termination}, LB={outcome.lower_bound}",
    }


def relax_and_fix_row(
    name: str, settings: InstanceSettings, window: WindowSettings, result: RelaxAndFixResult
) -> dict:
    """Linha de uma configuração (W, P) do Relax-and-Fix + Fix-and-Optimize."""
    construction = result.construction
    return {
        **_base_row(name, settings, _method_label(window)),
        "feasible": result.feasible,
        "objective": result.objective,
        "construction_objective": construction.objective if construction else None,
        "time_s": result.time,
        "subproblems": result.subproblems,
        "detail": result.stop_reason,
    }


def run_instance(name: str, settings: InstanceSettings, config: RelaxAndFixConfig) -> list[dict]:
    """O MIP (com limite de tempo) e cada configuração do R&F numa mesma instância."""
    instance = build_instance_for(name, settings)
    problem = problem_of(instance)
    outcome = reference_mip(instance, config.mip, settings.mip_time_limit)
    rows = [mip_row(name, settings, problem, outcome)]
    for window in config.windows:
        result = solve(
            **instance.mip_kwargs(),
            params=_params(config, window),
            solve_subproblem=pyomo_subproblem_solver(
                config.subproblem_solver, config.mip.mip_rel_gap
            ),
        )
        rows.append(relax_and_fix_row(name, settings, window, result))
        print(f"[{name}] {rows[-1]['method']}: obj={result.objective:.1f} t={result.time:.0f}s")
    return rows


def reference_mip(
    instance: CalibrationInstance, settings: MipSettings, time_limit: float
) -> MipOutcome:
    """O MIP completo com o solver pedido; igual a `solve_mip`, mas com a opção de gap dele."""
    model = build_block_assignment_model(**instance.mip_kwargs())
    solver = pyo.SolverFactory(settings.solver)
    start = time.perf_counter()
    results = solver.solve(
        model,
        timelimit=time_limit,
        options={mip_gap_option(settings.solver): settings.mip_rel_gap},
        load_solutions=False,
    )
    elapsed = time.perf_counter() - start
    lower = finite_or_none(results.problem.lower_bound)
    upper = finite_or_none(results.problem.upper_bound)
    return MipOutcome(
        instance=instance.name,
        solver=settings.solver,
        termination=str(results.solver.termination_condition),
        objective=upper,
        lower_bound=lower,
        upper_bound=upper,
        gap=relative_gap(lower, upper),
        time=elapsed,
        time_limit=time_limit,
        x=_incumbent(solver, model, upper),
    )


def _incumbent(solver, model: pyo.ConcreteModel, upper: float | None) -> dict[int, int]:
    if upper is None:
        return {}
    solver.load_vars()
    return assignment_from_model(model)


def with_solver(config: RelaxAndFixConfig, solver: str | None) -> RelaxAndFixConfig:
    """Troca o solver do MIP e dos subproblemas, gravando em outra pasta de saída."""
    if solver is None:
        return config
    return config.model_copy(
        update={
            "mip": config.mip.model_copy(update={"solver": solver}),
            "subproblem_solver": solver,
            "output_dir": f"{config.output_dir}_{solver}",
        }
    )


def _method_label(window: WindowSettings) -> str:
    group = f"_G{window.max_group_sectors}" if window.max_group_sectors else ""
    backtracks = f"_B{window.max_backtracks}" if window.max_backtracks != 2 else ""
    return f"rf_W{window.window}_P{window.step}{group}{backtracks}"


def _params(config: RelaxAndFixConfig, window: WindowSettings) -> RelaxAndFixParams:
    return RelaxAndFixParams(
        window=window.window,
        step=window.step,
        max_group_sectors=window.max_group_sectors,
        max_backtracks=window.max_backtracks,
        optimize_passes=config.optimize_passes,
        subproblem_time_limit=config.subproblem_time_limit,
        mip_rel_gap=config.mip.mip_rel_gap,
    )


def add_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Gap (%) de cada linha contra o MIP da instância e contra a melhor solução viável dela."""
    usable = table["feasible"] & table["objective"].notna()
    objective = table["objective"].where(usable)
    best = objective.groupby(table["instance"]).transform("min")
    mip = objective.where(table["method"] == "mip").groupby(table["instance"]).transform("max")
    return table.assign(
        gap_vs_best_pct=100 * (objective - best) / best,
        gap_vs_mip_pct=100 * (objective - mip) / mip,
    )


def run(config: RelaxAndFixConfig) -> pd.DataFrame:
    """Roda todas as instâncias do YAML e devolve a tabela de resultados com os gaps."""
    rows = []
    for name, settings in config.instances.items():
        rows.extend(run_instance(name, settings, config))
    return add_gaps(pd.DataFrame(rows))[COLUMNS]


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m experiments.relax_and_fix.run [--config caminho.yaml]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--solver",
        default=None,
        help="troca o solver do MIP e dos subproblemas (ex.: appsi_gurobi)",
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning, module="src.forecasting")
    config = with_solver(load_config(args.config), args.solver)
    table = run(config)
    io.write_table(table, Path(config.output_dir) / "results.csv")
    print(io.to_markdown(table))


if __name__ == "__main__":
    main()
