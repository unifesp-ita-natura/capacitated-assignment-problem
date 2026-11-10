"""Build past-only history, calendar, structure and intermittency features for every node."""

import numpy as np
import pandas as pd

BLOCKS = {
    "history": ["lag_1", "lag_2", "lag_3", "mean_3", "mean_5", "mean_10", "std_5", "trend_ratio"],
    "calendar": [
        "cycle_month",
        "cycle_number",
        "cycle_days",
        "days_min",
        "days_max",
        "opening_weekday_mean",
        "opening_day_mean",
        "days_since_previous_opening",
    ],
    "structure": ["known_children", "recent_active_children", "children_change"],
    "hierarchy": ["share_5", "share_10", "parent_mean_5", "previous_largest_share"],
    "city_recent_shares": ["share_1", "share_2", "share_3"],
    "intermittency": [
        "purchase_frequency_5",
        "cycles_since_purchase",
        "positive_mean_5",
        "n_history",
    ],
}


def feature_columns(blocks: list[str]) -> list[str]:
    """Always preserve node identity, selecting numeric feature blocks explicitly."""
    return ["node", "parent", *dict.fromkeys(c for block in blocks for c in BLOCKS[block])]


def rolling(values: pd.Series, nodes: pd.Series, window: int, operation: str) -> pd.Series:
    """Use vectorized group rolling rather than one Python callback per sparse city."""
    grouped = values.groupby(nodes).rolling(window, min_periods=1)
    return getattr(grouped, operation)().reset_index(level=0, drop=True).reindex(values.index)


def parent_totals(frame: pd.DataFrame) -> pd.Series:
    """Parent labels build historical share targets; only their shifted values become features."""
    lookup = frame.set_index(["node", "CICLOS"])["items"]
    keys = pd.MultiIndex.from_arrays([frame["parent"], frame["CICLOS"]])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=frame.index)


def _history_features(frame: pd.DataFrame) -> pd.DataFrame:
    nodes = frame["node"]
    previous = frame.groupby("node")["items"].shift(1)
    for lag in [1, 2, 3]:
        frame[f"lag_{lag}"] = frame.groupby("node")["items"].shift(lag)
    for window in [3, 5, 10]:
        frame[f"mean_{window}"] = rolling(previous, nodes, window, "mean")
    frame["std_5"] = rolling(previous, nodes, 5, "std")
    frame["trend_ratio"] = frame["mean_3"] / frame["mean_10"].replace(0, np.nan)
    frame["positive_mean_5"] = rolling(previous.where(previous > 0), nodes, 5, "mean")
    return frame


def _hierarchy_features(frame: pd.DataFrame) -> pd.DataFrame:
    nodes = frame["node"]
    previous = frame.groupby("node")["items"].shift(1)
    parent_previous = frame["parent_actual"].groupby(nodes).shift(1)
    for window in [1, 2, 3, 5, 10]:
        denominator = rolling(parent_previous, nodes, window, "sum")
        frame[f"share_{window}"] = rolling(previous, nodes, window, "sum") / denominator.replace(
            0, np.nan
        )
    columns = BLOCKS["city_recent_shares"]
    frame[columns] = frame[columns].where(frame["level"].eq("city"), axis=0)
    frame["parent_mean_5"] = rolling(parent_previous, nodes, 5, "mean")
    frame["previous_largest_share"] = frame.groupby("node")["largest_share"].shift(1)
    return frame


def _structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("node")
    frame["known_children"] = grouped["n_children"].shift(1)
    frame["children_change"] = frame["known_children"] - grouped["n_children"].shift(2)
    active = grouped["active_children"].shift(1)
    frame["recent_active_children"] = rolling(active, frame["node"], 5, "mean")
    return frame


def _intermittency_features(frame: pd.DataFrame) -> pd.DataFrame:
    nodes = frame["node"]
    previous = frame.groupby("node")["items"].shift(1)
    observed = frame["items"].notna().astype(int)
    frame["n_history"] = observed.groupby(nodes).cumsum() - observed
    positive = frame["items"].gt(0).astype(float).where(frame["items"].notna())
    frame["purchase_frequency_5"] = rolling(positive.groupby(nodes).shift(1), nodes, 5, "mean")
    position = frame.groupby("node").cumcount()
    last_purchase = position.where(previous > 0).groupby(nodes).ffill()
    frame["cycles_since_purchase"] = position - last_purchase + 1
    return frame


def build_features(nodes: pd.DataFrame) -> pd.DataFrame:
    """Compute features after sorting each node; the current label never enters its predictors."""
    frame = nodes.sort_values(["node", "opening_date"]).reset_index(drop=True).copy()
    frame["parent_actual"] = parent_totals(frame)
    frame["cycle_month"] = frame["opening_date"].dt.month
    frame["cycle_number"] = frame["CICLOS"].str[-2:].astype(int)
    previous = frame.groupby("node")["window_start"].shift(1)
    frame["days_since_previous_opening"] = (frame["window_start"] - previous).dt.days
    frame = _hierarchy_features(_history_features(frame))
    frame = _intermittency_features(_structure_features(frame))
    return _closed_history_features(frame)


def _closed_prefixes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    take = np.zeros(len(frame), dtype=int)
    previous = np.full(len(frame), -1, dtype=int)
    closes = frame["closing_date"].fillna(pd.Timestamp.max).to_numpy()
    openings = frame["opening_date"].to_numpy()
    for indices in frame.groupby("node", sort=False).indices.values():
        local = closes[indices]
        if (local[1:] < local[:-1]).any():
            raise ValueError("cycle closings must be chronological within each node")
        count = np.searchsorted(local, openings[indices], side="left")
        take[indices] = indices[count]
        previous[indices] = np.where(count > 0, indices[np.maximum(count - 1, 0)], -1)
    return take, previous


def _closed_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only already-closed labels even in each historical ML training example."""
    take, previous = _closed_prefixes(frame)
    columns = [
        column
        for block in ["history", "structure", "hierarchy", "intermittency", "city_recent_shares"]
        for column in BLOCKS[block]
    ]
    frame[columns] = frame[columns].iloc[take].to_numpy()
    dates = frame["window_start"].iloc[np.maximum(previous, 0)].to_numpy()
    gap = (frame["window_start"] - dates).dt.days.astype(float)
    frame["days_since_previous_opening"] = gap.where(previous >= 0)
    return frame


def training_and_query(
    history: pd.DataFrame, query: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append one query layout without manufacturing its demand or target shares."""
    pending = query.assign(items=np.nan, is_query=True)
    features = build_features(
        pd.concat([history.assign(is_query=False), pending], ignore_index=True)
    )
    training = features[~features["is_query"]].copy()
    target = features[features["is_query"]].copy()
    target["known_children"] = target["n_children"]
    return training, target
