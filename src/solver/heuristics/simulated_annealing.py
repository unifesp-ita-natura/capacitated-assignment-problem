"""Simulated Annealing pra atribuição setor -> Bloco/Subloco (Seção 4 do desafio Natura)."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

import numpy as np

from src.persistence import SolveResult
from src.solver.inputs import ProblemInstance, ProjectedDemand

# chamado a cada iteração com iteration=, energy=, temperature=, assignment=
IterationCallback = Callable[..., None]
# a energia corrente é atualizada só com o setor movido; a cada tantas iterações é
# recalculada do zero, pra que o erro de ponto flutuante da soma incremental não acumule
RESYNC_EVERY = 10_000
# decisões com |ΔE| abaixo disso (relativo à energia) são empates em aritmética exata: o sinal
# vem do erro de arredondamento, então as duas energias são refeitas do zero antes de decidir,
# pra que a decisão (e o consumo do rng) seja a mesma da avaliação completa
TIE_TOLERANCE = 1e-9
# "range": amplitude z_max - z_min (o objetivo do MIP);
# "peak_std_breach": w_peak·z_max + w_std·desvio + w_range·amplitude
EnergyMode = Literal["range", "peak_std_breach"]


@dataclass(frozen=True)
class AnnealingParams:
    """Parâmetros do SA (Seção 4.1) — placeholders a calibrar na etapa de validação."""

    initial_temperature: float = 1000.0  # T0
    cooling_rate: float = 0.99  # alfa
    min_temperature: float = 0.01  # T_min
    max_iterations: int = 100_000  # MaxIter
    stagnation_window: int = 200  # L
    stagnation_tolerance: float = 1e-5  # epsilon
    sector_bias: float = 0.8  # beta — viés pros setores de CDs em violação
    destination_bias: float = 0.7  # gama — viés pra combinações com folga
    penalty_coefficient: float = 1e6  # rho — peso das penalidades de inviabilidade
    energy_mode: EnergyMode = "range"  # termo de balanço da energia (ver `EnergyMode`)
    w_peak: float = 1.0  # peso do pico z_max (só em "peak_std_breach")
    w_std: float = 1.0  # peso do desvio-padrão da demanda diária (só em "peak_std_breach")
    w_range: float = 0.0  # peso da amplitude z_max - z_min (só em "peak_std_breach")

    def __post_init__(self) -> None:
        if self.energy_mode not in get_args(EnergyMode):
            raise ValueError(f"energy_mode desconhecido: {self.energy_mode!r}")


@dataclass(frozen=True)
class AnnealingResult:
    """Melhor solução encontrada pelo SA e como a busca terminou."""

    assignment: dict[int, int]  # setor -> combinação
    energy: float  # E(X) da melhor solução (termo de balanço + penalidades)
    objective: float  # z_max - z_min da melhor solução (sem penalidades)
    capacity_penalty: float
    churn_penalty: float
    iterations: int
    stop_reason: str  # "min_temperature" | "stagnation" | "max_iterations"
    zmax: float = 0.0  # média, entre os ciclos, do pico de demanda diária da melhor solução
    zmin: float = 0.0  # média, entre os ciclos, do vale diário (zmax - zmin == objective)
    std_load: float = 0.0  # média, entre os ciclos, do desvio-padrão da demanda diária
    time: float = 0.0  # segundos de busca gastos por `solve`

    @property
    def feasible(self) -> bool:
        """Se a melhor solução respeita capacidade e churn (P_cap = P_churn = 0)."""
        return self.capacity_penalty == 0.0 and self.churn_penalty == 0.0


def solve(
    instance: ProblemInstance,
    valid_combinations: Mapping[int, Iterable[int]] | None = None,
    params: AnnealingParams | None = None,
    rng: random.Random | None = None,
    callback: IterationCallback | None = None,
) -> AnnealingResult:
    """Roda o Simulated Annealing (Seção 4) e devolve a melhor atribuição encontrada.

    `instance.current_assignment` mapeia setor -> combinação atual (o vetor
    As-Is), e também é o ponto de partida da busca. `valid_combinations` mapeia
    CD -> combinações permitidas (D_c); quando omitido, toda combinação vale
    pra todo CD. `instance.days_by_cycle` agrupa `instance.days` por ciclo
    futuro (0-indexado), pra que o objetivo (amplitude de demanda) seja
    calculado por ciclo e depois tirado a média, em vez de globalmente sobre
    todo o horizonte. `callback`, se dado, é chamado ao fim de cada iteração
    com `iteration`, `energy` (energia corrente), `temperature` e `assignment`
    (índices internos; não modifique o array).
    """
    params = params or AnnealingParams()
    rng = rng or random.Random()
    problem = _build_problem(
        instance.sectors,
        instance.combinations,
        instance.days,
        instance.cd_sectors,
        instance.daily_capacity,
        instance.projected_demand,
        instance.current_assignment,
        instance.max_churn,
        instance.days_by_cycle,
        valid_combinations,
    )
    start = time.perf_counter()
    result = _anneal(problem, params, rng, callback)
    return replace(result, time=time.perf_counter() - start)


def cooling_rate_for_budget(
    budget: int, initial_temperature: float, min_temperature: float
) -> float:
    """O alfa que leva T de `initial_temperature` a `min_temperature` em `budget` iterações.

    Separa o orçamento de iterações do esquema de resfriamento: com esse alfa o
    resfriamento geométrico gasta o orçamento inteiro (a menos de 1 iteração de
    arredondamento), em vez de parar em `min_temperature` muito antes dele.
    """
    return math.exp(math.log(min_temperature / initial_temperature) / budget)


def iterations_for_schedule(
    initial_temperature: float, min_temperature: float, cooling_rate: float
) -> int:
    """Iterações do resfriamento geométrico até T <= `min_temperature`.

    É `ceil(ln(T_min / T0) / ln(alfa))`, sem o teto de `max_iterations` — o
    número real de iterações de uma execução é o menor dos dois.
    """
    return math.ceil(math.log(min_temperature / initial_temperature) / math.log(cooling_rate))


def run_simulated_annealing(instance: ProblemInstance, seed: int) -> SolveResult:
    """Run the simulated-annealing heuristic, returning a timed `SolveResult`."""
    start = time.perf_counter()
    result = solve(instance, rng=random.Random(seed))
    wall_time_seconds = time.perf_counter() - start
    return SolveResult(
        model_name="simulated_annealing",
        solver="simulated_annealing",
        status=result.stop_reason,
        termination_condition=result.stop_reason,
        objective=result.objective,
        wall_time_seconds=wall_time_seconds,
    )


@dataclass(frozen=True)
class _Problem:
    demand: np.ndarray  # (n_setores, n_dias, n_combinações)  q[s,a,d]
    capacity: np.ndarray  # (n_cds, n_dias)                     Cap[c,a]
    sector_cd: np.ndarray  # (n_setores,)                        índice do CD de cada setor
    valid_combos: list[np.ndarray]  # por CD: índices de combinações válidas (D_c)
    as_is: np.ndarray  # (n_setores,)                        combinação atual de cada setor
    max_churn: float
    sector_labels: list[int]
    combo_labels: list[int]
    cycle_of_day: np.ndarray  # (n_dias,) índice do ciclo do horizonte (0-indexado) de cada dia
    cycle_days: list[np.ndarray]  # por ciclo: índices dos dias dele (mesma ordem de cycle_of_day)


@dataclass(frozen=True)
class _Evaluation:
    objective: float  # z_max - z_min
    capacity_penalty: float  # P_cap
    churn_penalty: float  # P_churn
    energy: float  # termo de balanço + rho * (P_cap + P_churn)
    load_by_cd: np.ndarray  # (n_cds, n_dias) — reaproveitado pela função de vizinhança
    daily_totals: np.ndarray  # (n_dias,) demanda total por dia — base da atualização incremental
    n_changed: int  # setores fora da combinação As-Is
    exact: bool = True  # False se veio da soma incremental (`_evaluate_move`)


@dataclass
class _SearchState:
    current: np.ndarray
    current_eval: _Evaluation
    best: np.ndarray
    best_eval: _Evaluation
    energy_window: deque[float] = field(default_factory=deque)


# --- avaliação da energia (Seção 4.1.3) ---


def _daily_load_by_sector(problem: _Problem, assignment: np.ndarray) -> np.ndarray:
    """Demanda de cada setor em cada dia, dada a combinação que ele usa. (n_setores, n_dias)."""
    sectors = np.arange(assignment.shape[0])
    return problem.demand[sectors, :, assignment]


def _load_by_cd(problem: _Problem, load_by_sector: np.ndarray) -> np.ndarray:
    """Agrega a demanda por CD por dia (Carga_{c,a}). (n_cds, n_dias)."""
    totals = np.zeros_like(problem.capacity)
    np.add.at(totals, problem.sector_cd, load_by_sector)
    return totals


def _capacity_penalty(problem: _Problem, load_by_cd: np.ndarray) -> float:
    """P_cap: soma dos excessos de capacidade sobre todos os CDs e dias."""
    return float(np.maximum(0.0, load_by_cd - problem.capacity).sum())


def _churn_penalty(problem: _Problem, changed: int) -> float:
    """P_churn: excesso de setores realocados (`changed`) sobre o teto 2M."""
    return max(0.0, 2.0 * changed - 2.0 * problem.max_churn)


def _balance_objective(problem: _Problem, daily_totals: np.ndarray) -> float:
    """Média, entre os ciclos do horizonte, da amplitude (máx - mín) da demanda
    diária de cada ciclo — evita confundir tendência entre ciclos com
    desbalanceamento dentro de um ciclo."""
    return _per_cycle_mean(problem, daily_totals, np.ptp)


def _per_cycle_mean(
    problem: _Problem, daily_totals: np.ndarray, reducer: Callable[[np.ndarray], float]
) -> float:
    """Aplica `reducer` à demanda diária de cada ciclo do horizonte e tira a média."""
    values = [reducer(daily_totals[days]) for days in problem.cycle_days]
    return float(np.mean(values))


def _balance_term(
    problem: _Problem, daily_totals: np.ndarray, objective: float, params: AnnealingParams
) -> float:
    """Parte da energia sem penalidades: a amplitude, ou pico + desvio + amplitude ponderados."""
    if params.energy_mode == "range":
        return objective
    peak = _per_cycle_mean(problem, daily_totals, np.max)
    std_load = _per_cycle_mean(problem, daily_totals, np.std)
    return params.w_peak * peak + params.w_std * std_load + params.w_range * objective


def _evaluate(problem: _Problem, assignment: np.ndarray, params: AnnealingParams) -> _Evaluation:
    """Objetivo, penalidades e energia total de uma atribuição (a "caixa preta" da Seção 4.1.5)."""
    load_by_sector = _daily_load_by_sector(problem, assignment)
    daily_totals = load_by_sector.sum(axis=0)
    load_by_cd = _load_by_cd(problem, load_by_sector)
    changed = int(np.count_nonzero(assignment != problem.as_is))
    return _evaluation_from(problem, daily_totals, load_by_cd, changed, params)


def _evaluate_move(
    problem: _Problem, base: _Evaluation, sector: int, old: int, new: int, params: AnnealingParams
) -> _Evaluation:
    """`_evaluate` da atribuição de `base` com `sector` movido de `old` pra `new`.

    Soma só a diferença de demanda do setor movido (|A| valores) nos totais
    diários e na linha do CD dele, em vez de reagregar os |S| setores: o custo
    por avaliação deixa de crescer com |S|.
    """
    delta = problem.demand[sector, :, new] - problem.demand[sector, :, old]
    load_by_cd = base.load_by_cd.copy()
    load_by_cd[problem.sector_cd[sector]] += delta
    as_is = problem.as_is[sector]
    changed = base.n_changed + int(new != as_is) - int(old != as_is)
    evaluation = _evaluation_from(problem, base.daily_totals + delta, load_by_cd, changed, params)
    return replace(evaluation, exact=False)


def _evaluation_from(
    problem: _Problem,
    daily_totals: np.ndarray,
    load_by_cd: np.ndarray,
    changed: int,
    params: AnnealingParams,
) -> _Evaluation:
    """Objetivo, penalidades e energia a partir das cargas já agregadas."""
    objective = _balance_objective(problem, daily_totals)
    capacity_penalty = _capacity_penalty(problem, load_by_cd)
    churn_penalty = _churn_penalty(problem, changed)
    balance = _balance_term(problem, daily_totals, objective, params)
    energy = balance + params.penalty_coefficient * (capacity_penalty + churn_penalty)
    return _Evaluation(
        objective, capacity_penalty, churn_penalty, energy, load_by_cd, daily_totals, changed
    )


# --- função de vizinhança (Seção 4.1.2) ---


def _violated_cds(problem: _Problem, load_by_cd: np.ndarray) -> np.ndarray:
    """B(X): CDs que estouram a capacidade em pelo menos um dia."""
    return np.where((load_by_cd > problem.capacity).any(axis=1))[0]


def _pick_sector(
    problem: _Problem, violated_cds: np.ndarray, params: AnnealingParams, rng: random.Random
) -> int:
    """Escolhe o setor a realocar: com prob. beta prioriza setores de CDs em violação (O(X))."""
    overloaded = np.where(np.isin(problem.sector_cd, violated_cds))[0]
    if overloaded.size and rng.random() < params.sector_bias:
        return int(rng.choice(overloaded))
    return rng.randrange(problem.as_is.shape[0])


def _slack_destinations(
    problem: _Problem, sector: int, assignment: np.ndarray, load_by_cd: np.ndarray
) -> np.ndarray:
    """F(X): combinações de D_c que mantêm o CD do setor sem violação em nenhum dia."""
    cd = int(problem.sector_cd[sector])
    candidates = problem.valid_combos[cd]
    without_sector = load_by_cd[cd] - problem.demand[sector, :, assignment[sector]]
    projected = without_sector[None, :] + problem.demand[sector][:, candidates].T
    feasible = (projected <= problem.capacity[cd]).all(axis=1)
    return candidates[feasible]


def _pick_destination(
    problem: _Problem,
    sector: int,
    assignment: np.ndarray,
    load_by_cd: np.ndarray,
    params: AnnealingParams,
    rng: random.Random,
) -> int:
    """Nova combinação: prob. gama sorteia entre as de folga (F(X)), senão qualquer de D_c."""
    cd = int(problem.sector_cd[sector])
    slack = _slack_destinations(problem, sector, assignment, load_by_cd)
    if slack.size and rng.random() < params.destination_bias:
        return int(rng.choice(slack))
    return int(rng.choice(problem.valid_combos[cd]))


def _neighbor(
    problem: _Problem,
    assignment: np.ndarray,
    evaluation: _Evaluation,
    params: AnnealingParams,
    rng: random.Random,
) -> tuple[int, int]:
    """Movimento que gera o vizinho: (setor, nova combinação); os |S| - 1 restantes ficam."""
    violated = _violated_cds(problem, evaluation.load_by_cd)
    sector = _pick_sector(problem, violated, params, rng)
    destination = _pick_destination(problem, sector, assignment, evaluation.load_by_cd, params, rng)
    return sector, destination


# --- laço principal (Seções 4.1.3 e 4.1.4) ---


def _accepts(delta_energy: float, temperature: float, rng: random.Random) -> bool:
    """Regra de Metropolis: aceita melhora sempre; aceita piora com prob. e^(-dE/T)."""
    if delta_energy < 0:
        return True
    return math.exp(-delta_energy / temperature) > rng.random()


def _stagnated(energy_window: Sequence[float], params: AnnealingParams) -> bool:
    """Parada por estagnação: desvio-padrão da energia na janela deslizante < epsilon."""
    if len(energy_window) < params.stagnation_window:
        return False
    return float(np.std(energy_window)) < params.stagnation_tolerance


def _near_tie(energy: float, other: float) -> bool:
    """Se a diferença entre duas energias está na escala do erro de arredondamento."""
    return abs(energy - other) <= TIE_TOLERANCE * max(1.0, abs(energy), abs(other))


def _exact(
    problem: _Problem, evaluation: _Evaluation, assignment: np.ndarray, params: AnnealingParams
) -> _Evaluation:
    """A avaliação completa de `assignment`, reaproveitando `evaluation` se ela já é completa."""
    return evaluation if evaluation.exact else _evaluate(problem, assignment, params)


def _accept(
    problem: _Problem,
    state: _SearchState,
    assignment: np.ndarray,
    evaluation: _Evaluation,
    params: AnnealingParams,
) -> _SearchState:
    """Move a busca pro vizinho aceito e atualiza a melhor solução global se for o caso."""
    best, best_eval = state.best, state.best_eval
    if _near_tie(evaluation.energy, best_eval.energy):
        evaluation = _exact(problem, evaluation, assignment, params)
        best_eval = _exact(problem, best_eval, best, params)
    state.energy_window.append(evaluation.energy)
    if evaluation.energy < best_eval.energy:
        best, best_eval = assignment.copy(), evaluation
    return _SearchState(assignment, evaluation, best, best_eval, state.energy_window)


def _candidate_evaluation(
    problem: _Problem,
    state: _SearchState,
    sector: int,
    candidate: np.ndarray,
    params: AnnealingParams,
) -> _Evaluation:
    """Avaliação do vizinho: incremental, ou completa (junto com a corrente) num quase empate.

    Mover o setor pra combinação em que ele já está devolve a própria avaliação
    corrente (ΔE = 0 exato, como na avaliação completa), sem refazer nada.
    """
    old, new = int(state.current[sector]), int(candidate[sector])
    if new == old:
        return state.current_eval
    evaluation = _evaluate_move(problem, state.current_eval, sector, old, new, params)
    if _near_tie(evaluation.energy, state.current_eval.energy):
        state.current_eval = _exact(problem, state.current_eval, state.current, params)
        evaluation = _exact(problem, evaluation, candidate, params)
    return evaluation


def _step(
    problem: _Problem,
    state: _SearchState,
    params: AnnealingParams,
    rng: random.Random,
    temperature: float,
) -> _SearchState:
    """Uma iteração do SA: gera vizinho, avalia e decide aceitação."""
    sector, destination = _neighbor(problem, state.current, state.current_eval, params, rng)
    candidate = state.current.copy()
    candidate[sector] = destination
    candidate_eval = _candidate_evaluation(problem, state, sector, candidate, params)
    delta = candidate_eval.energy - state.current_eval.energy
    if not _accepts(delta, temperature, rng):
        return state
    return _accept(problem, state, candidate, candidate_eval, params)


def _resync(problem: _Problem, state: _SearchState, params: AnnealingParams) -> _SearchState:
    """Recalcula do zero a avaliação corrente, descartando o erro acumulado da soma incremental."""
    state.current_eval = _evaluate(problem, state.current, params)
    return state


def _notify(
    callback: IterationCallback | None, iteration: int, state: _SearchState, temperature: float
) -> None:
    """Repassa a iteração corrente ao callback de instrumentação, se houver um."""
    if callback is not None:
        callback(
            iteration=iteration,
            energy=state.current_eval.energy,
            temperature=temperature,
            assignment=state.current,
        )


def _anneal(
    problem: _Problem,
    params: AnnealingParams,
    rng: random.Random,
    callback: IterationCallback | None = None,
) -> AnnealingResult:
    initial = _evaluate(problem, problem.as_is, params)
    state = _SearchState(
        current=problem.as_is.copy(),
        current_eval=initial,
        best=problem.as_is.copy(),
        best_eval=initial,
        energy_window=deque(maxlen=params.stagnation_window),
    )
    temperature = params.initial_temperature
    for iteration in range(params.max_iterations):
        if temperature <= params.min_temperature:
            return _build_result(problem, state, params, iteration, "min_temperature")
        state = _step(problem, state, params, rng, temperature)
        if (iteration + 1) % RESYNC_EVERY == 0:
            state = _resync(problem, state, params)
        _notify(callback, iteration, state, temperature)
        if _stagnated(state.energy_window, params):
            return _build_result(problem, state, params, iteration + 1, "stagnation")
        temperature *= params.cooling_rate
    return _build_result(problem, state, params, params.max_iterations, "max_iterations")


def _build_result(
    problem: _Problem,
    state: _SearchState,
    params: AnnealingParams,
    iterations: int,
    stop_reason: str,
) -> AnnealingResult:
    assignment = {
        problem.sector_labels[i]: problem.combo_labels[int(d)] for i, d in enumerate(state.best)
    }
    ev = _evaluate(problem, state.best, params)  # do zero: sem o erro da soma incremental
    zmax, zmin, std_load = _load_profile(problem, state.best)
    return AnnealingResult(
        assignment=assignment,
        energy=ev.energy,
        objective=ev.objective,
        capacity_penalty=ev.capacity_penalty,
        churn_penalty=ev.churn_penalty,
        iterations=iterations,
        stop_reason=stop_reason,
        zmax=zmax,
        zmin=zmin,
        std_load=std_load,
    )


def _load_profile(problem: _Problem, assignment: np.ndarray) -> tuple[float, float, float]:
    """(z_max, z_min, desvio-padrão) da demanda diária, cada um em média entre os ciclos."""
    daily_totals = _daily_load_by_sector(problem, assignment).sum(axis=0)
    return (
        _per_cycle_mean(problem, daily_totals, np.max),
        _per_cycle_mean(problem, daily_totals, np.min),
        _per_cycle_mean(problem, daily_totals, np.std),
    )


# --- construção do problema (mapeia rótulos externos -> índices de array) ---


def _cycle_of_day_array(
    days_by_cycle: Mapping[int, Iterable[int]], day_ix: Mapping[int, int]
) -> np.ndarray:
    """(n_dias,) índice do ciclo do horizonte (0-indexado) de cada dia, na ordem de `day_ix`."""
    cycle_of_day = np.zeros(len(day_ix), dtype=int)
    for h, day_labels in days_by_cycle.items():
        for a in day_labels:
            cycle_of_day[day_ix[a]] = h
    return cycle_of_day


def _build_problem(
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, Iterable[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[int, int],
    max_churn: float,
    days_by_cycle: Mapping[int, Iterable[int]],
    valid_combinations: Mapping[int, Iterable[int]] | None,
) -> _Problem:
    sector_labels = list(sectors)
    combo_labels = list(combinations)
    day_labels = list(days)
    cd_labels = list(cd_sectors)
    sector_ix = {s: i for i, s in enumerate(sector_labels)}
    combo_ix = {d: i for i, d in enumerate(combo_labels)}
    day_ix = {a: i for i, a in enumerate(day_labels)}
    cd_ix = {c: i for i, c in enumerate(cd_labels)}
    cycle_of_day = _cycle_of_day_array(days_by_cycle, day_ix)

    return _Problem(
        demand=_demand_array(projected_demand, sector_ix, day_ix, combo_ix),
        capacity=_capacity_array(daily_capacity, cd_ix, day_ix),
        sector_cd=_sector_cd_array(cd_sectors, sector_ix, cd_ix),
        valid_combos=_valid_combos_by_cd(
            valid_combinations, cd_labels, len(combo_labels), cd_ix, combo_ix
        ),
        as_is=_as_is_array(current_assignment, sector_ix, combo_ix),
        max_churn=float(max_churn),
        sector_labels=sector_labels,
        combo_labels=combo_labels,
        cycle_of_day=cycle_of_day,
        cycle_days=_cycle_days(cycle_of_day),
    )


def _cycle_days(cycle_of_day: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(cycle_of_day == h) for h in range(int(cycle_of_day.max()) + 1)]


def _demand_array(
    projected_demand: ProjectedDemand,
    sector_ix: Mapping[int, int],
    day_ix: Mapping[int, int],
    combo_ix: Mapping[int, int],
) -> np.ndarray:
    demand = np.zeros((len(sector_ix), len(day_ix), len(combo_ix)))
    for (s, a, d), qty in projected_demand.items():
        demand[sector_ix[s], day_ix[a], combo_ix[d]] = qty
    return demand


def _capacity_array(
    daily_capacity: Mapping[tuple[int, int], float],
    cd_ix: Mapping[int, int],
    day_ix: Mapping[int, int],
) -> np.ndarray:
    capacity = np.zeros((len(cd_ix), len(day_ix)))
    for (c, a), cap in daily_capacity.items():
        capacity[cd_ix[c], day_ix[a]] = cap
    return capacity


def _sector_cd_array(
    cd_sectors: Mapping[int, Iterable[int]],
    sector_ix: Mapping[int, int],
    cd_ix: Mapping[int, int],
) -> np.ndarray:
    sector_cd = np.zeros(len(sector_ix), dtype=int)
    for c, members in cd_sectors.items():
        for s in members:
            sector_cd[sector_ix[s]] = cd_ix[c]
    return sector_cd


def _as_is_array(
    current_assignment: Mapping[int, int],
    sector_ix: Mapping[int, int],
    combo_ix: Mapping[int, int],
) -> np.ndarray:
    as_is = np.zeros(len(sector_ix), dtype=int)
    for s, d in current_assignment.items():
        as_is[sector_ix[s]] = combo_ix[d]
    return as_is


def _valid_combos_by_cd(
    valid_combinations: Mapping[int, Iterable[int]] | None,
    cd_labels: list[int],
    n_combos: int,
    cd_ix: Mapping[int, int],
    combo_ix: Mapping[int, int],
) -> list[np.ndarray]:
    combos = [np.arange(n_combos) for _ in cd_labels]
    if valid_combinations is None:
        return combos
    for c, allowed in valid_combinations.items():
        combos[cd_ix[c]] = np.array([combo_ix[d] for d in allowed], dtype=int)
    return combos
