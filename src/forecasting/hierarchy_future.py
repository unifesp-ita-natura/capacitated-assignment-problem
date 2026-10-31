"""Forecast chosen hierarchy presets and preserve sector-city outputs for daily shapes."""

import pandas as pd

from src.forecasting.hierarchy import forecast_distribution
from src.forecasting.hierarchy_config import HierarchyForecastParams, HierarchySettings
from src.forecasting.hierarchy_data import aggregate_nodes, node_ids
from src.forecasting.hierarchy_models import normalize_scores

PANEL_COLUMNS = [
    "serie",
    "cd_setor",
    "estado",
    "cidade",
    "CD_RE",
    "CD_GV",
    "CICLOS",
    "opening_date",
    "window_start",
    "cycle_days",
    "closing_date",
    "items",
]


def profile_settings(params: HierarchyForecastParams, history_cycles: int) -> HierarchySettings:
    """Resolve all fitted parameters rather than choosing a model from future demand."""
    levels = ["region", "management", "city"]
    if params.city_grouping == "sector":
        levels = ["region", "management", "sector", "city"]
    return HierarchySettings(
        levels=levels,
        city_grouping=params.city_grouping,
        history_cycles=history_cycles,
        share_window=params.share_window,
        learning=params.learning,
    )


def _organization(history: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    latest = history.sort_values("opening_date").drop_duplicates("cd_setor", keep="last")
    frame = queries.merge(
        latest[["cd_setor", "CD_RE", "CD_GV"]], on="cd_setor", how="left", validate="many_to_one"
    )
    if frame[["CD_RE", "CD_GV"]].isna().any().any():
        raise ValueError("hierarchical forecast needs historical organization for every sector")
    return frame


def _closed_training(panel, opening, count):
    calendar = panel.groupby("CICLOS").agg(
        opening=("opening_date", "min"), closing=("closing_date", "max")
    )
    cycles = calendar[calendar.closing.lt(opening)].sort_values("opening").tail(count).index
    if len(cycles) < count:
        raise ValueError(f"hierarchical forecast needs {count} closed training cycles")
    return panel[panel.CICLOS.isin(cycles)].copy()


def _sector_shares(history, queries, config):
    """Split a management-city total using only known sector-city purchases."""
    if config.city_grouping == "sector":
        return pd.Series(1.0, index=queries.index).to_numpy()
    cycles = (
        history.groupby("CICLOS").opening_date.min().sort_values().tail(config.share_window).index
    )
    recent = history[history.CICLOS.isin(cycles)].groupby("serie").items.sum()
    full = history.groupby("serie").items.sum()
    scores = queries.serie.map(recent).fillna(0)
    zero_group = scores.groupby(queries.level_node).transform("sum").eq(0)
    scores = scores.where(~zero_group, queries.serie.map(full).fillna(0))
    siblings = pd.DataFrame({"node": queries.serie, "parent": queries.level_node})
    return normalize_scores(siblings, scores.to_numpy())


def _predict_cycle(history, query, config):
    history = history[history.cd_setor.isin(query.cd_setor)]
    nodes = aggregate_nodes(history, config.levels, config.city_grouping)
    pending = query.assign(items=0.0, closing_date=pd.NaT)
    targets = aggregate_nodes(pending, config.levels, config.city_grouping)
    values = forecast_distribution(nodes, targets, config)
    result = query.assign(level_node=node_ids(query, "city", config.city_grouping))
    result["aggregate_level_pred"] = result.level_node.map(values)
    result["sector_share"] = _sector_shares(history, result, config)
    result["level_pred"] = result.aggregate_level_pred * result.sector_share
    result["level_granularity"] = f"{config.city_grouping}_city"
    return result


def forecast_hierarchical_path(history, queries, params, history_cycles):
    """Advance scenarios with predicted volumes only; never append actual future demand."""
    config = profile_settings(params, history_cycles)
    panel = history[PANEL_COLUMNS].copy()
    queries = _organization(panel, queries)
    results = []
    for _, query in queries.sort_values("opening_date").groupby("CICLOS", sort=False):
        training = _closed_training(panel, query.opening_date.min(), history_cycles)
        result = _predict_cycle(training, query, config)
        results.append(result)
        projected = result.assign(items=result.level_pred, closing_date=result.window_end.max())
        panel = pd.concat([panel, projected[PANEL_COLUMNS]], ignore_index=True)
    return pd.concat(results, ignore_index=True)
