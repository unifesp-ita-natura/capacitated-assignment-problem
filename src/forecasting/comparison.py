"""Rank candidates on the (sector, cycle, fold) points that all of them actually scored."""

from __future__ import annotations

import pandas as pd

from src.forecasting.evaluation import EvaluationResult, equal_weight_mae
from src.forecasting.metrics import PRIMARY_METRIC

SCORE_KEYS = ["cd_setor", "CICLOS", "origin_cycle"]


def _scored_keys(result: EvaluationResult) -> set[tuple[str, str, str]]:
    scored = result.scored
    return set(map(tuple, scored[SCORE_KEYS].to_numpy()))


def common_scored_keys(results: list[EvaluationResult]) -> set[tuple[str, str, str]]:
    """The (sector, cycle, fold) points every candidate produced a prediction for.

    Candidates legitimately skip different points: an ARIMA order may not fit
    a given sector, and a pooled model needs enough cycles to build its lag
    features before it can predict at all. Comparing each candidate's own
    headline error would then reward whichever one skipped the hardest
    points — so the ranking runs on the intersection.
    """
    if not results:
        return set()
    common = _scored_keys(results[0])
    for result in results[1:]:
        common &= _scored_keys(result)
    return common


def _restrict(scored: pd.DataFrame, keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    if scored.empty:
        return scored
    in_common = [tuple(row) in keys for row in scored[SCORE_KEYS].to_numpy()]
    return scored.loc[in_common]


def _candidate_row(
    result: EvaluationResult, keys: set[tuple[str, str, str]]
) -> dict[str, float | int | str]:
    """One comparison row. A candidate that scored nothing reports NaN rather than sinking the run.

    `EvaluationResult.summary()` refuses to aggregate an empty result, which
    is right for a single candidate's headline number but wrong here: one
    candidate that never managed a prediction shouldn't stop the others from
    being compared.
    """
    scored = result.scored
    restricted = _restrict(scored, keys)
    return {
        "candidate": result.candidate_name,
        "mae_own": equal_weight_mae(scored) if not scored.empty else float("nan"),
        "n_scored_own": len(scored),
        "n_missing": len(result.missing),
        "mae_common": equal_weight_mae(restricted) if not restricted.empty else float("nan"),
        "n_scored_common": len(restricted),
    }


def compare(results: list[EvaluationResult]) -> pd.DataFrame:
    """One row per candidate, sorted best-first by the primary metric on the common subset.

    `mae_own` is each candidate's error over everything it managed to
    predict; `mae_common` is its error over the points every candidate
    predicted, and is the only one of the two that's comparable across rows.
    `n_scored_own` vs `n_scored_common` shows how much of the difference is
    coverage rather than accuracy.
    """
    if not results:
        raise ValueError("nothing to compare: no evaluation results were passed")

    keys = common_scored_keys(results)
    table = pd.DataFrame([_candidate_row(result, keys) for result in results])
    return table.sort_values(f"{PRIMARY_METRIC}_common").reset_index(drop=True)
