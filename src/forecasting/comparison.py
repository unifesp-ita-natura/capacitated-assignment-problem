"""Rank candidates on the (sector, cycle, fold) points that all of them actually scored."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting import daily as daily_module
from src.forecasting.evaluation import EvaluationResult, equal_weight_mae
from src.forecasting.metrics import PRIMARY_METRIC, bias, bias_pct, mase, rmse, wmape

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


def _daily_columns(result: EvaluationResult, keys: set[tuple[str, str, str]]) -> dict[str, float]:
    """Per-day errors on the common subset; empty when the run had no daily base."""
    daily = _restrict(result.daily, keys)
    if daily.empty:
        return {}
    return {
        "mae_cd_day_common": daily_module.cd_day_mae(daily),
        "wape_cd_day_common": daily_module.cd_day_wape(daily),
        "mae_sector_day_common": daily_module.sector_day_mae(daily),
    }


def _common_mase(result: EvaluationResult, keys: set[tuple[str, str, str]]) -> float:
    """Average common-point fold MASE, retaining each fold's training-only scale."""
    values = []
    for origin in result.origins:
        scored = _restrict(origin.scored, keys)
        if not scored.empty:
            values.append(mase(scored["actual"], scored["items_pred"], origin.naive_in_sample_mae))
    return float(np.mean(values)) if values else float("nan")


def _common_level_columns(scored: pd.DataFrame) -> dict[str, float]:
    """Report level diagnostics on identical points; return NaN for empty coverage."""
    names = (
        "rmse_common",
        "p90_abs_error_common",
        "worst_origin_mae_common",
        "bias_common",
        "bias_pct_common",
        "wmape_common",
    )
    if scored.empty:
        return dict.fromkeys(names, float("nan"))
    actual, predicted = scored["actual"], scored["items_pred"]
    return {
        "rmse_common": rmse(actual, predicted),
        "p90_abs_error_common": float(np.percentile(scored["abs_error"], 90)),
        "worst_origin_mae_common": float(scored.groupby("origin_cycle")["abs_error"].mean().max()),
        "bias_common": bias(actual, predicted),
        "bias_pct_common": bias_pct(actual, predicted),
        "wmape_common": wmape(actual, predicted),
    }


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
        **_daily_columns(result, keys),
        **_common_level_columns(restricted),
        "mase_common": _common_mase(result, keys),
        "candidate": result.candidate_name,
        "mae_own": equal_weight_mae(scored) if not scored.empty else float("nan"),
        "n_scored_own": len(scored),
        "n_missing": len(result.missing),
        "mae_common": equal_weight_mae(restricted) if not restricted.empty else float("nan"),
        "n_scored_common": len(restricted),
    }


def compare(results: list[EvaluationResult]) -> pd.DataFrame:
    """One row per candidate, sorted best-first on the common subset.

    Ranks by the CD-day error when the results carry daily forecasts (the
    level the capacity constraint acts on), else by `PRIMARY_METRIC` per cycle.

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
    rank_by = "mae_cd_day_common" if "mae_cd_day_common" in table else f"{PRIMARY_METRIC}_common"
    return table.sort_values(rank_by).reset_index(drop=True)
