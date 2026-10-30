"""Compare configurable level hierarchies without changing existing level or shape pipelines."""

from dataclasses import dataclass, field

import pandas as pd

from src.forecasting.hierarchy_config import BASELINE_WINDOW, HierarchySettings
from src.forecasting.hierarchy_features import training_and_query
from src.forecasting.hierarchy_models import (
    ets_total,
    historical_shares,
    learned_values,
    moving_average,
    normalize_scores,
)
from src.forecasting.hierarchy_reconciliation import assert_coherent, distribute, reconcile


@dataclass
class HierarchyResult:
    """Predictions and fitted metadata for one complete scenario-cycle layout."""

    predictions: dict[str, pd.Series] = field(default_factory=dict)
    fitted: list[dict] = field(default_factory=list)


def _baselines(train: pd.DataFrame, target: pd.DataFrame, result: HierarchyResult) -> None:
    result.predictions[f"direct_ma_{BASELINE_WINDOW}"] = pd.Series(
        moving_average(train, target, BASELINE_WINDOW), index=target["node"]
    )


def _learned_predictions(
    train: pd.DataFrame, target: pd.DataFrame, config: HierarchySettings, result: HierarchyResult
) -> None:
    root = config.levels[0]
    root_values, fitted = ets_total(
        train[train.level.eq(root)], target[target.level.eq(root)], config
    )
    result.fitted.append({"level": root, "model": "ets", **fitted})
    base = pd.Series(root_values, index=target.loc[target.level.eq(root), "node"])
    shares = {}
    for level in config.levels[1:]:
        t, q = train[train.level.eq(level)], target[target.level.eq(level)]
        historical = historical_shares(t, q, config.share_window, config.learning.city_prior_cycles)
        scores, share_fit = learned_values(t, q, config.learning, "share", historical)
        shares[level] = normalize_scores(q, scores)
        volumes, volume_fit = learned_values(
            t, q, config.learning, "volume", moving_average(t, q, config.share_window)
        )
        base = pd.concat([base, pd.Series(volumes, index=q["node"])])
        result.fitted.extend([{**fit, "level": level} for fit in [share_fit, volume_fit]])
    result.predictions["distribution_ets_lgbm"] = distribute(
        target, root_values, shares, config.levels
    )
    reconciled, fitted = reconcile(
        train, target, base, config.levels[-1], config.reconciliation_max_iter
    )
    result.predictions["reconciliation_ets_lgbm"] = reconciled
    result.fitted.append({"candidate": "reconciliation_ets_lgbm", **fitted})


def forecast_hierarchy(
    history: pd.DataFrame, query: pd.DataFrame, config: HierarchySettings
) -> HierarchyResult:
    """Forecast one layout from past node panels; caller supplies explicit planned windows."""
    _validate_query(query, config.levels)
    _validate_history(history, query["opening_date"].min())
    train, target = training_and_query(history, query)
    result = HierarchyResult()
    _baselines(train, target, result)
    _learned_predictions(train, target, config, result)
    for name, predictions in result.predictions.items():
        if not name.startswith("direct_"):
            assert_coherent(target, predictions, config.levels[-1])
    return result


def _validate_query(query: pd.DataFrame, levels: list[str]) -> None:
    if query.empty:
        raise ValueError("the planned layout has no known cities")
    if query["CICLOS"].nunique() != 1 or query["node"].duplicated().any():
        raise ValueError("provide exactly one scenario-cycle and one row per node")
    if set(query["level"]) != set(levels):
        raise ValueError("query levels differ from the configured hierarchy")


def _validate_history(history: pd.DataFrame, opening: pd.Timestamp) -> None:
    if history.empty or history["closing_date"].isna().any():
        raise ValueError("historical nodes require known closing dates")
    if not history["closing_date"].lt(opening).all():
        raise ValueError("historical cycles must be closed before scenario first opening")
    if history["opening_date"].max() >= opening:
        raise ValueError("historical cycles must precede the scenario cycle")
