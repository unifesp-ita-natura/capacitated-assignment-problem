"""Replay closed-cycle hierarchy forecasts with active-city and full-volume diagnostics."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.dataset import build_item_panel, cycle_calendar
from src.forecasting.hierarchy import forecast_hierarchy
from src.forecasting.hierarchy_config import HierarchySettings, level_keys
from src.forecasting.hierarchy_data import (
    WINDOW_COLUMNS,
    aggregate_nodes,
    history_panel,
    known_roster,
    scenario_panel,
)
from src.forecasting.metrics import bias, bias_pct, wmape


@dataclass
class HierarchyEvaluation:
    """All artifacts needed to inspect one archived hierarchy experiment."""

    predictions: pd.DataFrame
    comparison: pd.DataFrame
    metrics_by_cycle: pd.DataFrame
    folds: pd.DataFrame
    fitted: list[dict]
    comparison_all_points: pd.DataFrame
    metrics_by_cycle_all_points: pd.DataFrame


def origin_plan(calendar: pd.DataFrame, config: HierarchySettings) -> list[dict]:
    """Select global closed cycles, not twenty positive purchases per city."""
    targets = _target_calendar(calendar, config)
    plans = [_origin(row, calendar, config.history_cycles) for row in targets.itertuples()]
    eligible = [plan for plan in plans if plan is not None]
    return eligible[-config.max_origins :] if config.max_origins else eligible


def _target_calendar(calendar: pd.DataFrame, config: HierarchySettings) -> pd.DataFrame:
    targets = calendar[calendar["is_complete"]].sort_values("opening_date")
    if config.first_target_cycle:
        targets = targets[targets["CICLOS"].ge(config.first_target_cycle)]
    if config.last_target_cycle:
        targets = targets[targets["CICLOS"].le(config.last_target_cycle)]
    return targets


def _origin(row, calendar: pd.DataFrame, count: int) -> dict | None:
    closed = calendar[calendar["is_complete"] & calendar["closing_date"].lt(row.opening_date)]
    cycles = closed.sort_values("opening_date")["CICLOS"].tail(count).tolist()
    if len(cycles) < count:
        return None
    return {"target_cycle": row.CICLOS, "as_of": row.opening_date, "training_cycles": cycles}


def _observed_windows(raw: pd.DataFrame, calendar: pd.DataFrame, cycle: str) -> pd.DataFrame:
    selected = raw[raw["CICLOS"].eq(cycle)]
    windows = selected.groupby("cd_setor", as_index=False).agg(
        window_start=("Dt Abertura", "min"), window_end=("Dt Fechamento", "max")
    )
    windows["cycle_days"] = (windows["window_end"] - windows["window_start"]).dt.days + 1
    opening = calendar.set_index("CICLOS").loc[cycle, "opening_date"]
    return windows.assign(CICLOS=cycle, opening_date=opening)[["cd_setor", *WINDOW_COLUMNS]]


def _scenario_layouts(
    observed: pd.DataFrame, scenarios: pd.DataFrame | None, cycle: str
) -> list[tuple[str, pd.DataFrame, bool]]:
    if scenarios is None:
        return [("historical", observed, True)]
    selected = scenarios[scenarios["CICLOS"].eq(cycle)]
    if selected.empty:
        raise ValueError(f"scenario CSV has no layout for evaluated cycle {cycle}")
    return [
        _checked_layout(observed, str(name), layout)
        for name, layout in selected.groupby("scenario_id")
    ]


def _checked_layout(observed: pd.DataFrame, name: str, layout: pd.DataFrame) -> tuple:
    if set(layout["cd_setor"]) != set(observed["cd_setor"]):
        raise ValueError("each scenario must include exactly the full planned sector roster")
    return name, layout, _same_layout(observed, layout)


def _same_layout(observed: pd.DataFrame, planned: pd.DataFrame) -> bool:
    columns = ["cd_setor", *WINDOW_COLUMNS]
    left = observed[columns].sort_values("cd_setor").reset_index(drop=True)
    right = planned[columns].sort_values("cd_setor").reset_index(drop=True)
    return left.shape == right.shape and left.eq(right).to_numpy().all()


def _actual_nodes(
    raw: pd.DataFrame,
    calendar: pd.DataFrame,
    cycle: str,
    roster: pd.DataFrame,
    levels: list[str],
    city_grouping: str = "sector",
) -> pd.DataFrame:
    actual = build_item_panel(raw[raw["CICLOS"].eq(cycle)], calendar)
    known = roster[["cd_setor", "CD_RE", "CD_GV"]].drop_duplicates("cd_setor")
    actual = actual.merge(known, on="cd_setor", how="left", suffixes=("", "_known"))
    for code in ["CD_RE", "CD_GV"]:
        actual[code] = actual[f"{code}_known"].fillna(actual[code])
    actual = actual.merge(calendar[["CICLOS", "closing_date"]], on="CICLOS")
    return aggregate_nodes(actual, levels, city_grouping)


def _identities(frame: pd.DataFrame, city_grouping: str = "sector") -> pd.DataFrame:
    rows = [
        dict(zip(level_keys(level, city_grouping), json.loads(node)[1:]))
        for level, node in zip(frame["level"], frame["node"])
    ]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _score(
    query: pd.DataFrame,
    actual: pd.DataFrame,
    predicted: pd.Series,
    observed: bool,
    city_grouping: str = "sector",
) -> pd.DataFrame:
    forecast = query[["node", "parent", "level"]].copy()
    forecast["items_pred"] = forecast["node"].map(predicted)
    labels = actual[["node", "level", "items"]].rename(columns={"items": "actual"})
    merged = forecast.merge(labels, on=["node", "level"], how="outer")
    merged["has_prediction"] = merged["items_pred"].notna()
    merged["items_pred"] = merged["items_pred"].fillna(0)
    merged["actual"] = merged["actual"].fillna(0).where(np.full(len(merged), observed))
    merged["abs_error"] = (merged["actual"] - merged["items_pred"]).abs()
    merged["included_in_primary_metrics"] = observed & _active_city_mask(merged)
    return _identities(merged.assign(is_observed=observed), city_grouping)


def _fold(
    raw: pd.DataFrame,
    calendar: pd.DataFrame,
    plan: dict,
    config: HierarchySettings,
    scenarios: pd.DataFrame | None,
) -> tuple[list[pd.DataFrame], list[dict]]:
    as_of, cycle = plan["as_of"], plan["target_cycle"]
    available = raw[raw["data_pedido"].lt(as_of)]
    cut = calendar.assign(is_complete=calendar["is_complete"] & calendar["closing_date"].lt(as_of))
    past = history_panel(available, cut, plan["training_cycles"])
    roster = known_roster(past)
    history = aggregate_nodes(past, config.levels, config.city_grouping)
    actual = _actual_nodes(raw, calendar, cycle, roster, config.levels, config.city_grouping)
    windows = _observed_windows(raw, calendar, cycle)
    frames, fitted = [], []
    for name, layout, observed in _scenario_layouts(windows, scenarios, cycle):
        query = aggregate_nodes(scenario_panel(roster, layout), config.levels, config.city_grouping)
        result = forecast_hierarchy(history, query, config)
        for candidate, values in result.predictions.items():
            scored = _score(query, actual, values, observed, config.city_grouping).assign(
                candidate=candidate,
                scenario_id=name,
                CICLOS=cycle,
                as_of=as_of,
                origin_cycle=plan["training_cycles"][-1],
            )
            frames.append(scored)
        fitted.extend([{**fit, "scenario_id": name, "CICLOS": cycle} for fit in result.fitted])
    return frames, fitted


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return _empty_metrics()
    actual, pred = frame["actual"], frame["items_pred"]
    positive = float(actual.sum())
    return {
        "mae": float(frame.groupby("node")["abs_error"].mean().mean()),
        "mae_pooled": float(frame["abs_error"].mean()),
        "wmape_pct": 100 * wmape(actual, pred),
        "bias": bias(actual, pred),
        "bias_pct": bias_pct(actual, pred),
        "n_points": len(frame),
        "n_nodes": frame["node"].nunique(),
        "n_missing": int((~frame["has_prediction"]).sum()),
        "coverage_points_pct": 100 * frame["has_prediction"].mean(),
        "coverage_volume_pct": 100 * actual[frame["has_prediction"]].sum() / positive
        if positive
        else np.nan,
    }


def _empty_metrics() -> dict:
    values = dict.fromkeys(
        [
            "mae",
            "mae_pooled",
            "wmape_pct",
            "bias",
            "bias_pct",
            "coverage_points_pct",
            "coverage_volume_pct",
        ],
        np.nan,
    )
    return {**values, "n_points": 0, "n_nodes": 0, "n_missing": 0}


def _active_city_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["level"].ne("city") | frame["actual"].gt(0)


def _group_metrics(frame: pd.DataFrame, city_active_only: bool) -> dict:
    mask = _active_city_mask(frame) if city_active_only else pd.Series(True, index=frame.index)
    selected = frame[mask]
    return {
        **_metrics(selected),
        "n_total_points": len(frame),
        "n_excluded_points": len(frame) - len(selected),
        "scoring_scope": "city_actual_positive"
        if city_active_only and frame["level"].eq("city").all()
        else "all_points",
    }


def summarize(
    predictions: pd.DataFrame, keys: list[str], *, city_active_only: bool = True
) -> pd.DataFrame:
    """Score positive-demand city periods by default; retain empty groups and unknown buyers."""
    observed = predictions[predictions["is_observed"]]
    rows = [
        {**dict(zip(keys, group)), **_group_metrics(frame, city_active_only)}
        for group, frame in observed.groupby(keys)
    ]
    return pd.DataFrame(rows)


def evaluate_hierarchy(
    raw: pd.DataFrame, config: HierarchySettings, scenarios: pd.DataFrame | None = None
) -> HierarchyEvaluation:
    """Evaluate every candidate on identical folds, layouts, identities and real demand."""
    calendar = cycle_calendar(raw)
    plans = origin_plan(calendar, config)
    if not plans:
        raise ValueError("not enough complete, closed cycles for this hierarchy backtest")
    frames, fitted = [], []
    for plan in plans:
        print(
            f"[hierarchy] target={plan['target_cycle']} last_closed={plan['training_cycles'][-1]}",
            flush=True,
        )
        predictions, parameters = _fold(raw, calendar, plan, config, scenarios)
        frames.extend(predictions)
        fitted.extend(parameters)
    predictions = pd.concat(frames, ignore_index=True)
    keys = ["level", "candidate"]
    cycle_keys = [*keys, "scenario_id", "CICLOS"]
    return HierarchyEvaluation(
        predictions=predictions,
        comparison=summarize(predictions, keys),
        metrics_by_cycle=summarize(predictions, cycle_keys),
        folds=pd.DataFrame(plans),
        fitted=fitted,
        comparison_all_points=summarize(predictions, keys, city_active_only=False),
        metrics_by_cycle_all_points=summarize(predictions, cycle_keys, city_active_only=False),
    )
