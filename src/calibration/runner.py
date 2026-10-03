"""Executa o SA por (configuração, instância, seed) com estagnação desligada e grava o traço."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.calibration.config import BudgetSettings
from src.calibration.instances import CalibrationInstance
from src.solver.heuristics.simulated_annealing import (
    AnnealingParams,
    AnnealingResult,
    cooling_rate_for_budget,
    iterations_for_schedule,
    solve,
)

TraceRow = tuple[int, float, float, float]  # (iteração, energia corrente, melhor energia, T)
_WORKER_INSTANCES: dict[str, CalibrationInstance] = {}


@dataclass(frozen=True)
class RunSpec:
    """Uma execução a fazer: qual configuração, em qual instância, com qual seed."""

    phase: str
    config_id: str
    config: Mapping[str, float]
    instance: str
    seed: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Identificador único da execução (usado para retomar fases interrompidas)."""
        return (self.phase, self.config_id, self.instance, self.seed)


@dataclass
class TraceRecorder:
    """Callback do SA que guarda (iteração, energia, melhor energia, T) a cada `every` iterações."""

    every: int
    rows: list[TraceRow] = field(default_factory=list)
    best: float = float("inf")

    def __call__(self, *, iteration: int, energy: float, temperature: float, assignment) -> None:
        """Assinatura esperada por `simulated_annealing.solve(callback=...)`."""
        self.best = min(self.best, energy)
        if iteration % self.every == 0:
            self.rows.append((iteration, energy, self.best, temperature))


@dataclass(frozen=True)
class SARun:
    """Uma execução concluída: o que foi pedido, os parâmetros efetivos, o resultado e o traço."""

    spec: RunSpec
    params: AnnealingParams
    result: AnnealingResult
    trace: list[TraceRow]


def resolve_params(config: Mapping[str, float], budget: BudgetSettings) -> AnnealingParams:
    """`AnnealingParams` da configuração, com estagnação desligada e orçamento fixo.

    Sem `cooling_rate` explícito, o alfa é derivado de (T0, T_min) para gastar o
    orçamento inteiro (`cooling_rate_for_budget`): o resfriamento deixa de
    controlar quantas iterações a busca tem.
    """
    values = dict(config)
    values.setdefault(
        "cooling_rate",
        cooling_rate_for_budget(
            budget.max_iterations, values["initial_temperature"], values["min_temperature"]
        ),
    )
    return AnnealingParams(
        **values,
        max_iterations=budget.max_iterations,
        stagnation_window=budget.max_iterations,
        stagnation_tolerance=0.0,
    )


def run_sa(instance: CalibrationInstance, spec: RunSpec, budget: BudgetSettings) -> SARun:
    """Uma execução do SA com traço de convergência."""
    params = resolve_params(spec.config, budget)
    recorder = TraceRecorder(every=budget.trace_every)
    result = solve(
        **instance.sa_kwargs(), params=params, rng=random.Random(spec.seed), callback=recorder
    )
    return SARun(spec=spec, params=params, result=result, trace=recorder.rows)


def run_record(run: SARun) -> dict:
    """A linha plana de uma execução, com os nomes de métrica do relatório (P_cap, P_churn...)."""
    params, result = run.params, run.result
    return {
        "phase": run.spec.phase,
        "config_id": run.spec.config_id,
        "instance": run.spec.instance,
        "seed": run.spec.seed,
        "initial_temperature": params.initial_temperature,
        "cooling_rate": params.cooling_rate,
        "min_temperature": params.min_temperature,
        "sector_bias": params.sector_bias,
        "destination_bias": params.destination_bias,
        "penalty_coefficient": params.penalty_coefficient,
        "scheduled_iterations": _scheduled_iterations(params),
        "objective": result.objective,
        "energy": result.energy,
        "zmax": result.zmax,
        "zmin": result.zmin,
        "std_load": result.std_load,
        "p_cap": result.capacity_penalty,
        "p_churn": result.churn_penalty,
        "feasible": result.feasible,
        "iterations": result.iterations,
        "time": result.time,
        "stop_reason": result.stop_reason,
    }


def _scheduled_iterations(params: AnnealingParams) -> int:
    """Iterações previstas pelo resfriamento, limitadas pelo orçamento."""
    schedule = iterations_for_schedule(
        params.initial_temperature, params.min_temperature, params.cooling_rate
    )
    return min(schedule, params.max_iterations)


def expand_specs(
    phase: str,
    configs: Mapping[str, Mapping[str, float]],
    instances: Iterable[str],
    seeds: Iterable[int],
) -> list[RunSpec]:
    """O produto configurações x instâncias x seeds, em ordem estável."""
    instance_list, seed_list = list(instances), list(seeds)
    return [
        RunSpec(phase, config_id, config, instance, seed)
        for config_id, config in configs.items()
        for instance in instance_list
        for seed in seed_list
    ]


def _init_worker(instances: Mapping[str, CalibrationInstance]) -> None:
    """Deixa as instâncias no processo filho uma única vez, em vez de a cada tarefa."""
    _WORKER_INSTANCES.update(instances)


def _run_task(spec: RunSpec, budget: BudgetSettings) -> SARun:
    return run_sa(_WORKER_INSTANCES[spec.instance], spec, budget)


def _executor(n_workers: int, instances: Mapping[str, CalibrationInstance]) -> Executor:
    workers = n_workers or os.cpu_count() or 1
    return ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(dict(instances),))


def run_many(
    instances: Mapping[str, CalibrationInstance],
    specs: Sequence[RunSpec],
    budget: BudgetSettings,
    n_workers: int,
    on_result: Callable[[SARun], None],
) -> None:
    """Roda `specs` em paralelo e entrega cada execução a `on_result` assim que termina."""
    with _executor(n_workers, instances) as pool:
        futures = [pool.submit(_run_task, spec, budget) for spec in specs]
        for future in as_completed(futures):
            on_result(future.result())
