"""Formulação em Pyomo do problema de atribuição setor -> Bloco/Subloco (desafio Natura)."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager

import pyomo.environ as pyo

from src.persistence import SolveResult
from src.solver.inputs import ProblemInstance, ProjectedDemand

try:
    import resource  # POSIX-only; see `_memory_limit`
except ImportError:  # Windows has no `resource` module
    resource = None

FALLBACK_SOLVER = "appsi_highs"
DEFAULT_TIME_LIMIT_SECONDS = 10 * 60
DEFAULT_MEMORY_LIMIT_GB = 8.0

logger = logging.getLogger(__name__)


@contextmanager
def _memory_limit(limit_gb: float) -> Generator[None]:
    """Best-effort `RLIMIT_AS` cap on the whole process while a solve runs.

    This is the only lever available for `appsi_highs`, which has no native
    memory-limit option at all (checked against its full option dump).
    It's POSIX-only, a no-op on Windows (no `resource` module), and
    unreliable on macOS (the kernel doesn't enforce `RLIMIT_AS` the way
    Linux does) — a blunt last resort, not a clean per-solve abort like
    Gurobi's own `MemLimit` (set separately in `_solve_options`). If it does
    trigger, the whole process can die rather than just failing one solve.
    """
    if resource is None:
        yield
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(limit_gb * 1024**3), hard))
    except (ValueError, OSError):
        logger.warning("Could not set RLIMIT_AS memory guard; continuing unconstrained.")
        yield
        return
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _solve_options(
    resolved_solver: str, memory_limit_gb: float, time_limit_seconds: float
) -> dict[str, object]:
    """Solver-specific options.

    Pyomo's generic `timelimit=` kwarg to `.solve()` is only honored by a
    handful of shell plugins (CPLEX, GLPK, SCIPAMPL, CBC); Gurobi's shell
    plugin (`pyomo.solvers.plugins.solvers.GUROBI`) never reads
    `self._timelimit` and silently ignores it, so `TimeLimit` must be passed
    through `options` instead, same as `MemLimit`.
    """
    if resolved_solver == "gurobi":
        return {"MemLimit": memory_limit_gb, "TimeLimit": time_limit_seconds}
    return {}


def build_block_assignment_model(instance: ProblemInstance) -> pyo.ConcreteModel:
    """Atribui cada setor a um Bloco/Subloco pra nivelar a demanda diária no horizonte.

    `instance.daily_capacity` é indexado por `(cd, dia)`, assumindo 0 pra chaves
    ausentes. `instance.days_by_cycle` agrupa `instance.days` por ciclo futuro
    (0-indexado): a amplitude z_max - z_min é calculada por ciclo e depois
    tirada a média, pra não confundir tendência entre ciclos com
    desbalanceamento dentro de um ciclo.
    """
    cd_sectors = {cd: list(members) for cd, members in instance.cd_sectors.items()}
    days_by_cycle = {h: list(members) for h, members in instance.days_by_cycle.items()}

    model = pyo.ConcreteModel()
    _add_sets(
        model, instance.sectors, instance.combinations, instance.days, cd_sectors, days_by_cycle
    )
    _add_variables(model)
    _add_objective(model)
    _add_assignment_constraint(model)
    _add_demand_range_constraints(model, instance.projected_demand, days_by_cycle)
    _add_capacity_constraints(model, cd_sectors, instance.daily_capacity, instance.projected_demand)
    _add_churn_constraint(model, instance.current_assignment, instance.max_churn)
    return model


def _resolve_solver(solver: str) -> tuple[pyo.SolverFactory, str]:
    """Pyomo solver interface for `solver`, falling back to `FALLBACK_SOLVER`.

    The fallback only triggers for `solver == "gurobi"` with no working
    license/install (checked via `.available()`, never by calling `.solve()`),
    so a locally-missing Gurobi license doesn't block running this project.
    """
    solver_interface = pyo.SolverFactory(solver)
    if solver == "gurobi" and not solver_interface.available(exception_flag=False):
        logger.warning(
            "Gurobi is unavailable (no license or install found); falling back to %s.",
            FALLBACK_SOLVER,
        )
        return pyo.SolverFactory(FALLBACK_SOLVER), FALLBACK_SOLVER
    return solver_interface, solver


def solve_block_assignment(
    model: pyo.ConcreteModel,
    solver: str = "gurobi",
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
) -> pyo.SolverResults:
    """Resolve `model` com `solver` via a interface de solvers do Pyomo.

    `solver` é qualquer nome de solver registrado no Pyomo. Faz fallback pra
    `FALLBACK_SOLVER` se `solver == "gurobi"` e não houver licença/instalação
    disponível. `time_limit_seconds` vira o `timelimit` padrão do Pyomo
    (honrado por `appsi_highs`) e também é repassado como `TimeLimit` nativo
    via `_solve_options` pro Gurobi, cujo plugin shell ignora o `timelimit`
    genérico — sem isso, nenhuma instância teria limite de tempo nessa solver.
    `memory_limit_gb` vira `MemLimit` nativo pro Gurobi; pro resto (ex.
    appsi_highs, sem opção nativa de memória) é só o guarda-corpo
    melhor-esforço de `_memory_limit`.
    """
    solver_interface, resolved_solver = _resolve_solver(solver)
    with _memory_limit(memory_limit_gb):
        return solver_interface.solve(
            model,
            timelimit=time_limit_seconds,
            options=_solve_options(resolved_solver, memory_limit_gb, time_limit_seconds),
        )


def run_block_assignment_mip(
    instance: ProblemInstance,
    solver: str = "gurobi",
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
) -> SolveResult:
    """Build and solve the block-assignment MIP, returning a timed `SolveResult`."""
    return solve_with_assignment(instance, solver, time_limit_seconds, memory_limit_gb)[0]


def extract_assignment(model: pyo.ConcreteModel) -> dict[int, int]:
    """The sector -> combination map encoded by the model's binary `x` (after a solve)."""
    return {s: d for s in model.S for d in model.D if pyo.value(model.x[s, d]) > 0.5}


def apply_warm_start(
    model: pyo.ConcreteModel,
    instance: ProblemInstance,
    assignment: Mapping[int, int],
) -> None:
    """Carrega `assignment` (setor -> combinação) como ponto inicial de x, z_max e z_min.

    z_max[h]/z_min[h] recebem o máximo/mínimo diário real do ciclo h sob essa
    atribuição, pra o ponto inicial ser completo (e não só parcial em x).
    """
    _warm_start_assignment(model, assignment)
    _warm_start_cycle_bounds(model, instance, assignment)


def _warm_start_assignment(model: pyo.ConcreteModel, assignment: Mapping[int, int]) -> None:
    """Fixa o valor inicial de x[s,d] em 1 pra combinação escolhida de cada setor, 0 nas demais."""
    for s, d in itertools.product(model.S, model.D):
        model.x[s, d].set_value(1 if assignment[s] == d else 0)


def _warm_start_cycle_bounds(
    model: pyo.ConcreteModel, instance: ProblemInstance, assignment: Mapping[int, int]
) -> None:
    """Fixa z_max[h]/z_min[h] no máximo/mínimo diário real do ciclo h sob `assignment`."""
    for h, days in instance.days_by_cycle.items():
        totals = [_daily_total(instance.projected_demand, assignment, a) for a in days]
        model.z_max[h].set_value(max(totals))
        model.z_min[h].set_value(min(totals))


def _daily_total(
    projected_demand: ProjectedDemand, assignment: Mapping[int, int], day: int
) -> float:
    """Demanda total do dia `day` sob `assignment`."""
    return sum(projected_demand.get((s, day, d), 0.0) for s, d in assignment.items())


def _warmstart_kwargs(resolved_solver: str, warm_start: bool) -> dict[str, object]:
    """`warmstart=True` só onde o plugin do Pyomo o honra (shell do Gurobi)."""
    if not warm_start:
        return {}
    if resolved_solver == "gurobi":
        return {"warmstart": True}
    logger.warning("%s does not take a warm start through Pyomo; solving cold.", resolved_solver)
    return {}


def solve_with_assignment(
    instance: ProblemInstance,
    solver: str = "gurobi",
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    warm_start: Mapping[int, int] | None = None,
) -> tuple[SolveResult, dict[int, int] | None]:
    """Like `run_block_assignment_mip`, but also returns the assignment (None if no solution).

    `warm_start` (setor -> combinação), se dado, vira o ponto inicial do MIP.
    """
    model = build_block_assignment_model(instance)
    solver_interface, resolved_solver = _resolve_solver(solver)
    if warm_start is not None:
        apply_warm_start(model, instance, warm_start)
    start = time.perf_counter()
    try:
        with _memory_limit(memory_limit_gb):
            pyomo_results = solver_interface.solve(
                model,
                timelimit=time_limit_seconds,
                options=_solve_options(resolved_solver, memory_limit_gb, time_limit_seconds),
                **_warmstart_kwargs(resolved_solver, warm_start is not None),
            )
    except RuntimeError as exc:
        # Both the legacy (`environ.SolverFactory`) and `appsi_highs`
        # interfaces raise this when the solver stopped (usually hitting
        # `time_limit_seconds`) without ever finding a feasible solution, so
        # there's nothing to load into `model`'s variables/objective.
        if "feasible solution was not found" not in str(exc):
            raise
        wall_time_seconds = time.perf_counter() - start
        logger.warning(
            "%s found no feasible solution within %ss: %s",
            resolved_solver,
            time_limit_seconds,
            exc,
        )
        return SolveResult(
            model_name="mip_block_assignment",
            solver=resolved_solver,
            status=str(pyo.SolverStatus.aborted),
            termination_condition=str(pyo.TerminationCondition.noSolution),
            objective=None,
            wall_time_seconds=wall_time_seconds,
        ), None
    wall_time_seconds = time.perf_counter() - start
    result = SolveResult(
        model_name="mip_block_assignment",
        solver=resolved_solver,
        status=str(pyomo_results.solver.status),
        termination_condition=str(pyomo_results.solver.termination_condition),
        objective=pyo.value(model.objective, exception=False),
        wall_time_seconds=wall_time_seconds,
    )
    return result, (extract_assignment(model) if result.objective is not None else None)


def _add_sets(
    model: pyo.ConcreteModel,
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, list[int]],
    days_by_cycle: Mapping[int, list[int]],
) -> None:
    """Declara S, D, A, C, H (§3.1: setores, combinações, dias, CDs, ciclos do horizonte)."""
    model.S = pyo.Set(initialize=list(sectors))
    model.D = pyo.Set(initialize=list(combinations))
    model.A = pyo.Set(initialize=list(days))
    model.C = pyo.Set(initialize=list(cd_sectors))
    model.H = pyo.Set(initialize=list(days_by_cycle))


def _add_variables(model: pyo.ConcreteModel) -> None:
    """Declara x (§3.3: 1 se o setor s usa a combinação d) e os limites z_max/z_min por ciclo."""
    model.x = pyo.Var(model.S, model.D, within=pyo.Binary)
    model.z_max = pyo.Var(model.H, within=pyo.Reals)
    model.z_min = pyo.Var(model.H, within=pyo.Reals)


def _add_objective(model: pyo.ConcreteModel) -> None:
    """Minimiza a média, entre os ciclos do horizonte, da amplitude de demanda
    diária z_max[h] - z_min[h] (§3.4).

    z_max/z_min ainda não ficam presos ao máximo/mínimo diário real por
    nenhuma restrição aqui — veja `_add_demand_range_constraints` pra essa
    outra metade do truque. A amplitude é calculada por ciclo (não globalmente
    sobre todo o horizonte) pra não confundir tendência entre ciclos com
    desbalanceamento dentro de um ciclo.
    """
    model.objective = pyo.Objective(
        expr=sum(model.z_max[h] - model.z_min[h] for h in model.H) / len(model.H),
        sense=pyo.minimize,
    )


def _add_assignment_constraint(model: pyo.ConcreteModel) -> None:
    """Cada setor escolhe exatamente uma combinação Bloco/Subloco (§3.5.1)."""

    def rule(m: pyo.ConcreteModel, s: int) -> bool:
        return sum(m.x[s, d] for d in m.D) == 1

    model.single_assignment = pyo.Constraint(model.S, rule=rule)


def _add_demand_range_constraints(
    model: pyo.ConcreteModel,
    projected_demand: ProjectedDemand,
    days_by_cycle: Mapping[int, list[int]],
) -> None:
    """Prende z_max[h]/z_min[h] como teto/piso da demanda total de cada dia do
    ciclo h (§3.5.2, §3.5.3).

    Combinado com o objetivo de minimizar a média de z_max[h] - z_min[h], isso
    força cada z_max[h] a virar o máximo diário real do ciclo h e cada z_min[h]
    o mínimo diário real do mesmo ciclo no ótimo — veja a explicação de
    epígrafe/hipógrafe na descrição do PR.
    """

    def total_demand(m: pyo.ConcreteModel, a: int) -> pyo.Expression:
        return sum(projected_demand.get((s, a, d), 0.0) * m.x[s, d] for s in m.S for d in m.D)

    upper_index = [(h, a) for h, days in days_by_cycle.items() for a in days]
    model.demand_upper_bound = pyo.Constraint(
        upper_index, rule=lambda m, h, a: m.z_max[h] >= total_demand(m, a)
    )
    model.demand_lower_bound = pyo.Constraint(
        upper_index, rule=lambda m, h, a: m.z_min[h] <= total_demand(m, a)
    )


def _add_capacity_constraints(
    model: pyo.ConcreteModel,
    cd_sectors: Mapping[int, list[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
) -> None:
    """Limita a demanda diária total de cada CD à sua capacidade (§3.5.4)."""

    def rule(m: pyo.ConcreteModel, c: int, a: int) -> bool:
        cd_demand = sum(
            projected_demand.get((s, a, d), 0.0) * m.x[s, d] for s in cd_sectors[c] for d in m.D
        )
        return cd_demand <= daily_capacity.get((c, a), 0.0)

    model.cd_capacity = pyo.Constraint(model.C, model.A, rule=rule)


def _add_churn_constraint(
    model: pyo.ConcreteModel,
    current_assignment: Mapping[int, int],
    max_churn: float,
) -> None:
    """Limita a max_churn o número de setores que mudam em relação ao As-Is (§3.5.5).

    `current_assignment` é o As-Is denso (setor -> combinação). Cada par
    (s, d) contribui com 1 - x[s,d] se era a atribuição As-Is do setor (0 se
    o setor manteve, 1 se abandonou) ou com x[s,d] caso contrário (1 se o
    setor escolheu essa combinação agora). Um setor que mudou contribui com
    exatamente 2 somando todos os seus pares (s, d); um que não mudou
    contribui com 0 — por isso o orçamento é 2 * max_churn, não max_churn.
    """

    def rule(m: pyo.ConcreteModel) -> bool:
        changed = sum(
            (1 if current_assignment.get(s) == d else 0)
            + (1 - 2 * (1 if current_assignment.get(s) == d else 0)) * m.x[s, d]
            for s in m.S
            for d in m.D
        )
        return changed <= 2 * max_churn

    model.churn_limit = pyo.Constraint(rule=rule)
