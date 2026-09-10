"""Simulated Annealing pra atribuição setor -> Bloco/Subloco (Seção 4 do desafio Natura)."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

ProjectedDemand = Mapping[tuple[int, int, int], float]  # (setor, dia, combinação) -> qtd


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


@dataclass(frozen=True)
class AnnealingResult:
    """Melhor solução encontrada pelo SA e como a busca terminou."""

    assignment: dict[int, int]  # setor -> combinação
    energy: float  # E(X) da melhor solução (objetivo + penalidades)
    objective: float  # z_max - z_min da melhor solução (sem penalidades)
    capacity_penalty: float
    churn_penalty: float
    iterations: int
    stop_reason: str  # "min_temperature" | "stagnation" | "max_iterations"


def solve(
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, Iterable[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[int, int],
    max_churn: float,
    valid_combinations: Mapping[int, Iterable[int]] | None = None,
    params: AnnealingParams | None = None,
    rng: random.Random | None = None,
) -> AnnealingResult:
    """Roda o Simulated Annealing (Seção 4) e devolve a melhor atribuição encontrada.

    `current_assignment` mapeia setor -> combinação atual (o vetor As-Is), e também
    é o ponto de partida da busca. `valid_combinations` mapeia CD -> combinações
    permitidas (D_c); quando omitido, toda combinação vale pra todo CD.
    """
    params = params or AnnealingParams()
    rng = rng or random.Random()
    problem = _build_problem(
        sectors,
        combinations,
        days,
        cd_sectors,
        daily_capacity,
        projected_demand,
        current_assignment,
        max_churn,
        valid_combinations,
    )
    return _anneal(problem, params, rng)


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


@dataclass(frozen=True)
class _Evaluation:
    objective: float  # z_max - z_min
    capacity_penalty: float  # P_cap
    churn_penalty: float  # P_churn
    energy: float  # objetivo + rho * (P_cap + P_churn)
    load_by_cd: np.ndarray  # (n_cds, n_dias) — reaproveitado pela função de vizinhança


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


def _churn_penalty(problem: _Problem, assignment: np.ndarray) -> float:
    """P_churn: excesso de setores realocados sobre o teto 2M."""
    changed = int(np.count_nonzero(assignment != problem.as_is))
    return max(0.0, 2.0 * changed - 2.0 * problem.max_churn)


def _evaluate(problem: _Problem, assignment: np.ndarray, penalty_coefficient: float) -> _Evaluation:
    """Objetivo, penalidades e energia total de uma atribuição (a "caixa preta" da Seção 4.1.5)."""
    load_by_sector = _daily_load_by_sector(problem, assignment)
    daily_totals = load_by_sector.sum(axis=0)
    load_by_cd = _load_by_cd(problem, load_by_sector)
    objective = float(daily_totals.max() - daily_totals.min())
    capacity_penalty = _capacity_penalty(problem, load_by_cd)
    churn_penalty = _churn_penalty(problem, assignment)
    energy = objective + penalty_coefficient * (capacity_penalty + churn_penalty)
    return _Evaluation(objective, capacity_penalty, churn_penalty, energy, load_by_cd)


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
) -> np.ndarray:
    """Gera um vizinho realocando um único setor (mantém os |S| - 1 restantes)."""
    violated = _violated_cds(problem, evaluation.load_by_cd)
    sector = _pick_sector(problem, violated, params, rng)
    destination = _pick_destination(problem, sector, assignment, evaluation.load_by_cd, params, rng)
    neighbor = assignment.copy()
    neighbor[sector] = destination
    return neighbor


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


def _accept(state: _SearchState, assignment: np.ndarray, evaluation: _Evaluation) -> _SearchState:
    """Move a busca pro vizinho aceito e atualiza a melhor solução global se for o caso."""
    state.energy_window.append(evaluation.energy)
    best, best_eval = state.best, state.best_eval
    if evaluation.energy < best_eval.energy:
        best, best_eval = assignment.copy(), evaluation
    return _SearchState(assignment, evaluation, best, best_eval, state.energy_window)


def _step(
    problem: _Problem,
    state: _SearchState,
    params: AnnealingParams,
    rng: random.Random,
    temperature: float,
) -> _SearchState:
    """Uma iteração do SA: gera vizinho, avalia e decide aceitação."""
    candidate = _neighbor(problem, state.current, state.current_eval, params, rng)
    candidate_eval = _evaluate(problem, candidate, params.penalty_coefficient)
    delta = candidate_eval.energy - state.current_eval.energy
    if not _accepts(delta, temperature, rng):
        return state
    return _accept(state, candidate, candidate_eval)


def _anneal(problem: _Problem, params: AnnealingParams, rng: random.Random) -> AnnealingResult:
    initial = _evaluate(problem, problem.as_is, params.penalty_coefficient)
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
            return _build_result(problem, state, iteration, "min_temperature")
        state = _step(problem, state, params, rng, temperature)
        if _stagnated(state.energy_window, params):
            return _build_result(problem, state, iteration + 1, "stagnation")
        temperature *= params.cooling_rate
    return _build_result(problem, state, params.max_iterations, "max_iterations")


def _build_result(
    problem: _Problem, state: _SearchState, iterations: int, stop_reason: str
) -> AnnealingResult:
    assignment = {
        problem.sector_labels[i]: problem.combo_labels[int(d)] for i, d in enumerate(state.best)
    }
    ev = state.best_eval
    return AnnealingResult(
        assignment=assignment,
        energy=ev.energy,
        objective=ev.objective,
        capacity_penalty=ev.capacity_penalty,
        churn_penalty=ev.churn_penalty,
        iterations=iterations,
        stop_reason=stop_reason,
    )


# --- construção do problema (mapeia rótulos externos -> índices de array) ---


def _build_problem(
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, Iterable[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[int, int],
    max_churn: float,
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
    )


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
