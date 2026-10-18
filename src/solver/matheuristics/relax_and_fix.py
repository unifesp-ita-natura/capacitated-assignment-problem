"""Relax-and-Fix + Fix-and-Optimize pro MIP de atribuição setor -> Bloco/Subloco."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyomo.environ as pyo

from src.persistence import SolveResult
from src.solver.matheuristics.evaluation import TOLERANCE, Evaluation, evaluate_assignment
from src.solver.matheuristics.problem import AssignmentProblem, ProjectedDemand, build_problem
from src.solver.matheuristics.subproblem import (
    SubproblemSolver,
    apply_variable_modes,
    pyomo_subproblem_solver,
    read_assignment,
)
from src.solver.matheuristics.windows import (
    RelaxAndFixWindow,
    fix_and_optimize_windows,
    relax_and_fix_windows,
    widen_window,
)
from src.solver.mip.block_assignment import build_block_assignment_model

COMPLETED = "completed"
NO_SOLUTION_WINDOW = "no_solution_window"


@dataclass(frozen=True)
class RelaxAndFixParams:
    """Parâmetros da matheurística; os valores são ponto de partida a calibrar."""

    window: int = 3  # W: grupos (CDs ou fatias de CD) com x binário em cada subproblema
    step: int = 2  # P: grupos congelados por iteração do Relax-and-Fix (P <= W)
    optimize_passes: int = 2  # varreduras do Fix-and-Optimize (0 desliga a fase 2)
    optimize_step: int = 1  # passo da janela no Fix-and-Optimize
    subproblem_time_limit: float | None = 60.0  # segundos por subproblema
    mip_rel_gap: float = 1e-4  # gap relativo de cada subproblema
    # None: cada CD é um grupo. N: cada CD é dividido em fatias de no máximo N setores, e a
    # janela passa a ser medida em setores, não em CDs (os CDs são muito desiguais).
    max_group_sectors: int | None = None
    # Se uma janela do R&F fica sem solução, tenta de novo descongelando os `step` últimos
    # grupos (e depois o dobro, ...) até esse número de vezes. 0 desliga o backtracking.
    max_backtracks: int = 2

    def __post_init__(self) -> None:
        if self.max_group_sectors is not None and self.max_group_sectors < 1:
            raise ValueError(f"max_group_sectors deve ser >= 1, recebi {self.max_group_sectors}")
        if self.max_backtracks < 0:
            raise ValueError(f"max_backtracks deve ser >= 0, recebi {self.max_backtracks}")


@dataclass(frozen=True)
class RelaxAndFixResult:
    """Melhor atribuição encontrada e como cada fase se saiu."""

    assignment: dict[int, int]  # setor -> combinação
    evaluation: Evaluation  # da solução final (depois do Fix-and-Optimize)
    construction: Evaluation | None  # da solução do Relax-and-Fix; None se ele falhou
    subproblems: int  # subproblemas MIP resolvidos nas duas fases
    stop_reason: str  # COMPLETED | NO_SOLUTION_WINDOW
    time: float  # segundos totais

    @property
    def objective(self) -> float:
        return self.evaluation.objective

    @property
    def feasible(self) -> bool:
        return self.evaluation.feasible


def solve(
    sectors: Sequence[int],
    combinations: Sequence[int],
    days: Sequence[int],
    cd_sectors: Mapping[int, Sequence[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[tuple[int, int], int],
    max_churn: float,
    days_by_cycle: Mapping[int, Sequence[int]],
    params: RelaxAndFixParams | None = None,
    solve_subproblem: SubproblemSolver | None = None,
    cd_order: Sequence[int] | None = None,
) -> RelaxAndFixResult:
    """Roda o Relax-and-Fix e depois o Fix-and-Optimize sobre grupos de setores (CDs ou fatias).

    Os argumentos de instância são os de `build_block_assignment_model` (por isso
    `CalibrationInstance.mip_kwargs()` serve direto). `cd_order` define a ordem em
    que os CDs entram nas janelas (padrão: a de `cd_sectors`); `solve_subproblem`
    troca o solver dos subproblemas (padrão: HiGHS via Pyomo).
    """
    params = params or RelaxAndFixParams()
    start = time.perf_counter()
    problem = build_problem(
        sectors, cd_sectors, daily_capacity, projected_demand, current_assignment,
        max_churn, days_by_cycle,
    )  # fmt: skip
    model = build_block_assignment_model(
        sectors=sectors,
        combinations=combinations,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=daily_capacity,
        projected_demand=projected_demand,
        current_assignment=current_assignment,
        max_churn=max_churn,
        days_by_cycle=days_by_cycle,
    )
    solver = solve_subproblem or pyomo_subproblem_solver(mip_rel_gap=params.mip_rel_gap)
    order = _cd_order(problem, cd_order)
    ctx = _Context(
        problem, model, _groups(problem, order, params.max_group_sectors), params,
        _Runner(solver, params.subproblem_time_limit),
    )  # fmt: skip
    constructed = _construct(ctx)
    if constructed is None:
        return _no_solution_window(ctx, start)
    return _finish(ctx, constructed, start)


def run_relax_and_fix(
    sectors: list[int],
    combinations: list[int],
    days: list[int],
    cd_sectors: dict[int, list[int]],
    daily_capacity: dict[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: dict[tuple[int, int], int],
    max_churn: float,
    days_by_cycle: dict[int, list[int]],
    solver: str = "appsi_highs",
    params: RelaxAndFixParams | None = None,
) -> SolveResult:
    """Roda a matheurística e devolve um `SolveResult` comparável ao do MIP e ao do SA."""
    params = params or RelaxAndFixParams()
    result = solve(
        sectors=sectors,
        combinations=combinations,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=daily_capacity,
        projected_demand=projected_demand,
        current_assignment=current_assignment,
        max_churn=max_churn,
        days_by_cycle=days_by_cycle,
        params=params,
        solve_subproblem=pyomo_subproblem_solver(solver, params.mip_rel_gap),
    )
    return SolveResult(
        model_name="relax_and_fix",
        solver=solver,
        status=result.stop_reason,
        termination_condition="feasible" if result.feasible else "infeasible",
        objective=result.objective if result.feasible else None,
        wall_time_seconds=result.time,
    )


class _Runner:
    """Chama o solver de subproblemas e conta quantas vezes foi chamado."""

    def __init__(self, solve_subproblem: SubproblemSolver, time_limit: float | None) -> None:
        self._solve = solve_subproblem
        self._time_limit = time_limit
        self.calls = 0

    def __call__(self, model: pyo.ConcreteModel) -> bool:
        self.calls += 1
        return self._solve(model, self._time_limit)


@dataclass(frozen=True)
class _Context:
    problem: AssignmentProblem
    model: pyo.ConcreteModel
    groups: dict[int, tuple[int, ...]]  # id do grupo -> setores; a ordem é a das janelas
    params: RelaxAndFixParams
    runner: _Runner


@dataclass(frozen=True)
class _Incumbent:
    assignment: dict[int, int]
    evaluation: Evaluation


def _cd_order(problem: AssignmentProblem, cd_order: Sequence[int] | None) -> tuple[int, ...]:
    order = tuple(cd_order) if cd_order is not None else tuple(problem.cd_sectors)
    if set(order) != set(problem.cd_sectors) or len(order) != len(problem.cd_sectors):
        raise ValueError("cd_order deve listar cada CD de cd_sectors exatamente uma vez")
    return order


def _groups(
    problem: AssignmentProblem, order: Sequence[int], max_group_sectors: int | None
) -> dict[int, tuple[int, ...]]:
    """Unidades das janelas, na ordem dos CDs: um grupo por CD ou fatias de até N setores."""
    parts = [p for cd in order for p in _split(problem.cd_sectors[cd], max_group_sectors)]
    return dict(enumerate(parts))


def _split(members: tuple[int, ...], max_size: int | None) -> list[tuple[int, ...]]:
    """Divide `members` em fatias de tamanho parecido, cada uma com no máximo `max_size`."""
    if max_size is None or len(members) <= max_size:
        return [members]
    n_parts = -(-len(members) // max_size)
    bounds = [i * len(members) // n_parts for i in range(n_parts + 1)]
    return [members[lo:hi] for lo, hi in zip(bounds, bounds[1:])]


def _sectors_of(groups: Mapping[int, tuple[int, ...]], ids: Sequence[int]) -> frozenset[int]:
    return frozenset(s for group_id in ids for s in groups[group_id])


# --- fase 1: Relax-and-Fix ---


def _construct(ctx: _Context) -> dict[int, int] | None:
    """Percorre as janelas congelando `C_step` a cada iteração; None se alguma não tem solução."""
    frozen: dict[int, int] = {}
    for window in relax_and_fix_windows(tuple(ctx.groups), ctx.params.window, ctx.params.step):
        solved = _solve_with_backtracking(ctx, window, frozen)
        if solved is None:
            return None
        frozen = solved
    return frozen


def _solve_with_backtracking(
    ctx: _Context, window: RelaxAndFixWindow, frozen: Mapping[int, int]
) -> dict[int, int] | None:
    """Resolve a janela; se ficar sem solução, descongela grupos já fixos e tenta de novo.

    Devolve o congelamento ampliado com `C_step` (que, nas tentativas alargadas, inclui os
    grupos descongelados), ou None se todas as tentativas falharem.
    """
    for attempt in _attempts(ctx, window):
        fixed = _restrict(ctx, frozen, attempt.fixed)
        if _solve_window(ctx, attempt, fixed):
            step_sectors = _sectors_of(ctx.groups, attempt.step)
            return {**fixed, **read_assignment(ctx.model, step_sectors)}
    return None


def _attempts(ctx: _Context, window: RelaxAndFixWindow) -> list[RelaxAndFixWindow]:
    """A janela e as versões com 1, 2, ... passos de grupos descongelados (sem repetição)."""
    n_fixed = len(window.fixed)
    undos = [min(ctx.params.step * i, n_fixed) for i in range(ctx.params.max_backtracks + 1)]
    return [widen_window(window, undo) for undo in dict.fromkeys(undos)]


def _restrict(
    ctx: _Context, frozen: Mapping[int, int], fixed_groups: Sequence[int]
) -> dict[int, int]:
    """Os valores congelados só dos grupos que seguem fixos."""
    return {s: frozen[s] for s in _sectors_of(ctx.groups, fixed_groups)}


def _solve_window(ctx: _Context, window: RelaxAndFixWindow, frozen: Mapping[int, int]) -> bool:
    """Congela C_fix, deixa C_int binário e relaxa C_relax; resolve o subproblema."""
    interior = _sectors_of(ctx.groups, window.interior)
    apply_variable_modes(ctx.model, fixed=frozen, binary=interior)
    return ctx.runner(ctx.model)


# --- fase 2: Fix-and-Optimize ---


def _optimize(ctx: _Context, start: _Incumbent) -> _Incumbent:
    """Varre as janelas até uma varredura não melhorar nada ou acabarem as varreduras."""
    incumbent = start
    for _ in range(ctx.params.optimize_passes):
        improved = _optimize_pass(ctx, incumbent)
        if improved is incumbent:
            return incumbent
        incumbent = improved
    return incumbent


def _optimize_pass(ctx: _Context, incumbent: _Incumbent) -> _Incumbent:
    """Uma varredura: reotimiza cada janela de CDs e só troca a incumbente se melhorar."""
    windows = fix_and_optimize_windows(
        tuple(ctx.groups), ctx.params.window, ctx.params.optimize_step
    )
    for cds in windows:
        incumbent = _better(incumbent, _reoptimize(ctx, incumbent, cds))
    return incumbent


def _reoptimize(ctx: _Context, incumbent: _Incumbent, cds: Sequence[int]) -> _Incumbent | None:
    """Libera os setores de `cds`, mantém os demais fixos na incumbente e resolve."""
    free = _sectors_of(ctx.groups, cds)
    fixed = {s: d for s, d in incumbent.assignment.items() if s not in free}
    apply_variable_modes(ctx.model, fixed=fixed, binary=free)
    if not ctx.runner(ctx.model):
        return None
    candidate = {**incumbent.assignment, **read_assignment(ctx.model, free)}
    return _Incumbent(candidate, evaluate_assignment(ctx.problem, candidate))


def _better(current: _Incumbent, candidate: _Incumbent | None) -> _Incumbent:
    if candidate is not None and _improves(candidate.evaluation, current.evaluation):
        return candidate
    return current


def _improves(new: Evaluation, old: Evaluation) -> bool:
    """Se `new` é viável e melhor que `old` (ou `old` é inviável)."""
    if not new.feasible:
        return False
    return not old.feasible or new.objective < old.objective - TOLERANCE


# --- resultado ---


def _finish(ctx: _Context, constructed: Mapping[int, int], start: float) -> RelaxAndFixResult:
    initial = _Incumbent(dict(constructed), evaluate_assignment(ctx.problem, constructed))
    final = _optimize(ctx, initial)
    return RelaxAndFixResult(
        assignment=final.assignment,
        evaluation=final.evaluation,
        construction=initial.evaluation,
        subproblems=ctx.runner.calls,
        stop_reason=COMPLETED,
        time=time.perf_counter() - start,
    )


def _no_solution_window(ctx: _Context, start: float) -> RelaxAndFixResult:
    """Um subproblema do Relax-and-Fix ficou sem solução (inviável ou sem achar uma no tempo).

    Devolve o As-Is. A causa exata (inviabilidade ou limite de tempo) fica no log do solver.
    """
    return RelaxAndFixResult(
        assignment=dict(ctx.problem.as_is),
        evaluation=evaluate_assignment(ctx.problem, ctx.problem.as_is),
        construction=None,
        subproblems=ctx.runner.calls,
        stop_reason=NO_SOLUTION_WINDOW,
        time=time.perf_counter() - start,
    )
