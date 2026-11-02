"""Testes da avaliação incremental do SA: mesma energia e mesmos movimentos da completa."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from src.solver.heuristics import simulated_annealing as sa
from src.solver.heuristics.simulated_annealing import AnnealingParams, solve
from tests.solver.heuristics.test_simulated_annealing_energy import (
    _LEGACY_PARAMS,
    _problem,
    _random_instance,
)

_MODES = {
    "range": {},
    "peak_std_breach": {"energy_mode": "peak_std_breach", "w_std": 1.5, "w_range": 0.3},
}

# Digest da sequência de atribuições correntes (uma por iteração) com `_LEGACY_PARAMS`,
# gravado com a avaliação completa: o SA original (commit ef55716) no modo "range" e o
# SA com `energy_mode` anterior à avaliação incremental no modo "peak_std_breach".
_FULL_EVALUATION_DIGESTS = {
    ("range", 1): "76372ab2bcb5dc8551c3c7cd6b4afe61",
    ("range", 2): "8623cf4302e319b4e7a29a5f81de0328",
    ("peak_std_breach", 1): "a4f3194078e92a59f58c4ca778724945",
    ("peak_std_breach", 2): "16f74830ff09739e03a005967f839ee6",
}


@pytest.mark.parametrize("mode", list(_MODES))
def test_incremental_evaluation_matches_full_evaluation_along_a_random_walk(mode):
    params = AnnealingParams(penalty_coefficient=5.0, **_MODES[mode])
    problem = _problem(_random_instance())
    rng = np.random.default_rng(0)
    assignment = problem.as_is.copy()
    current = sa._evaluate(problem, assignment, params)
    for _ in range(2000):
        sector = int(rng.integers(assignment.size))
        new = int(rng.integers(len(problem.combo_labels)))
        current = sa._evaluate_move(problem, current, sector, int(assignment[sector]), new, params)
        assignment[sector] = new
        _assert_same_evaluation(current, sa._evaluate(problem, assignment, params))


def _assert_same_evaluation(current, full) -> None:
    assert current.energy == pytest.approx(full.energy, rel=1e-12)
    assert current.objective == pytest.approx(full.objective, rel=1e-12, abs=1e-9)
    assert current.capacity_penalty == pytest.approx(full.capacity_penalty, abs=1e-9)
    assert current.n_changed == full.n_changed
    np.testing.assert_allclose(current.load_by_cd, full.load_by_cd, rtol=1e-12)


@pytest.mark.parametrize(("mode", "seed"), list(_FULL_EVALUATION_DIGESTS))
def test_moves_are_identical_to_the_full_evaluation(mode, seed):
    digest = hashlib.blake2b(digest_size=16)
    params = AnnealingParams(**{**_LEGACY_PARAMS.__dict__, **_MODES[mode]})

    solve(
        _random_instance(),
        params=params,
        rng=random.Random(seed),
        callback=lambda **kw: digest.update(kw["assignment"].astype("int64").tobytes()),
    )

    assert digest.hexdigest() == _FULL_EVALUATION_DIGESTS[mode, seed]


def test_result_energy_is_recomputed_from_scratch():
    result = solve(_random_instance(), params=_LEGACY_PARAMS, rng=random.Random(1))

    assert result.energy == 21.415885017813874  # bit a bit igual ao SA original
