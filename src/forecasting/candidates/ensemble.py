"""Combine several candidates: an average, or a combiner learned on the training window's end."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config.schema import EnsembleParams
from src.forecasting.model import QUERY_KEYS, REGISTRY, ForecastCandidate, align_predictions


def member_matrix(
    members: list[ForecastCandidate], history: pd.DataFrame, targets: pd.DataFrame
) -> np.ndarray:
    """One column per member: its forecast for each target row, NaN where it skipped."""
    columns = [align_predictions(targets, m.fit_predict(history, targets)) for m in members]
    return np.column_stack([c.to_numpy(dtype=float) for c in columns])


def validation_rows(
    members: list[ForecastCandidate], history: pd.DataFrame, n_cycles: int
) -> tuple[np.ndarray, np.ndarray]:
    """Member forecasts and actuals for the last `n_cycles` cycles of `history`.

    Each of those cycles is forecast one step ahead from the cycles before
    it only, the same way the harness forecasts the real target, so the
    combiner learns from forecasts as good as the ones it will combine.
    """
    dates = sorted(history["opening_date"].unique())
    features, actuals = [], []
    for date in dates[-n_cycles:]:
        past = history[history["opening_date"] < date]
        if past["opening_date"].nunique() < 2:
            continue
        rows = history[history["opening_date"] == date]
        features.append(member_matrix(members, past, rows[[*QUERY_KEYS, "opening_date"]]))
        actuals.append(rows["items"].to_numpy(dtype=float))
    if not features:
        return np.empty((0, len(members))), np.empty(0)
    return np.vstack(features), np.concatenate(actuals)


def nnls_weights(features: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Non-negative least-squares weights, one per member, no intercept."""
    from scipy.optimize import nnls

    return nnls(features, actuals)[0]


def fit_mlp(features: np.ndarray, actuals: np.ndarray, seed: int = 0):
    """A small MLP that corrects the members' mean: pred = mean * f(members / mean).

    Working in ratios to the members' own mean keeps the network scale-free,
    so one network serves sectors of any size. The loss is the absolute
    error in items (the ratio error weighted by the mean), the harness's MAE.
    """
    import torch

    torch.manual_seed(seed)
    scale = torch.tensor(features.mean(axis=1, keepdims=True), dtype=torch.float32)
    inputs = torch.tensor(features, dtype=torch.float32) / scale
    target = torch.tensor(actuals[:, None], dtype=torch.float32)
    width = features.shape[1]
    network = torch.nn.Sequential(
        torch.nn.Linear(width, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Softplus()
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(500):  # ponytail: fixed full-batch epochs; early stopping if it overfits
        optimizer.zero_grad()
        loss = (network(inputs) * scale - target).abs().mean()
        loss.backward()
        optimizer.step()

    def predict(rows: np.ndarray) -> np.ndarray:
        row_scale = torch.tensor(rows.mean(axis=1, keepdims=True), dtype=torch.float32)
        with torch.no_grad():
            out = network(torch.tensor(rows, dtype=torch.float32) / row_scale) * row_scale
        return out.numpy().ravel()

    return predict


class _EnsembleCandidate:
    def __init__(self, params: EnsembleParams) -> None:
        self._params = params
        self._members = [REGISTRY.build(member) for member in params.members]
        names = "+".join(member.name for member in self._members)
        self.name = params.label or f"ensemble:{params.combiner}({names})"

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        forecasts = member_matrix(self._members, history, targets)
        # Rows a member skipped fall back to the average of the others; a row
        # every member skipped stays NaN and is dropped as missing.
        average = np.nanmedian if self._params.combiner == "median" else np.nanmean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            combined = average(forecasts, axis=1)
        complete = ~np.isnan(forecasts).any(axis=1) & (np.nan_to_num(combined) > 0)
        if self._params.combiner in ("nnls", "mlp") and complete.any():
            combined[complete] = self._learned(history)(forecasts[complete])
        rows = targets[QUERY_KEYS].assign(items_pred=np.clip(combined, 0, None))
        return rows.dropna(subset=["items_pred"]).reset_index(drop=True)

    def _learned(self, history: pd.DataFrame):
        features, actuals = validation_rows(self._members, history, self._params.validation_cycles)
        usable = ~np.isnan(features).any(axis=1) & (features.mean(axis=1) > 0)
        features, actuals = features[usable], actuals[usable]
        if len(actuals) == 0:
            return lambda rows: rows.mean(axis=1)
        if self._params.combiner == "nnls":
            weights = nnls_weights(features, actuals)
            return lambda rows: rows @ weights
        return fit_mlp(features, actuals)


@REGISTRY.register("ensemble")
def build_ensemble(params: EnsembleParams) -> ForecastCandidate:
    """Build the combination; each member keeps its own window and opening-day handling."""
    return _EnsembleCandidate(params)
