"""Rolling-origin evaluation harness: the Strategy pattern context, driving candidates alike."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.forecasting.metrics import PRIMARY_METRIC, mase, rmse
from src.forecasting.model import ForecastCandidate


@dataclass(frozen=True)
class RollingOriginSplit:
    """Rolling-origin cross-validation parameters — see forecasting-volume-spec.tex section 4.6.

    `window="expanding"` trains on every cycle up to the origin (preferred
    with short history, per the spec); `"sliding"` trains on the most recent
    `min_train_cycles` cycles only.
    """

    horizon: int
    min_train_cycles: int
    window: Literal["expanding", "sliding"] = "expanding"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.min_train_cycles < 1:
            raise ValueError("min_train_cycles must be >= 1")


@dataclass(frozen=True)
class OriginResult:
    """One rolling-origin fold: which (sector, cycle) pairs were scored, which were skipped."""

    origin_cycle: str
    target_cycles: tuple[str, ...]
    scored: pd.DataFrame  # cd_setor, CICLOS, actual, items_pred, abs_error, origin_cycle
    missing: (
        pd.DataFrame
    )  # cd_setor, CICLOS, origin_cycle — target had an actual, candidate gave no prediction
    naive_in_sample_mae: (
        float  # one-cycle-ahead naive error on this fold's training window, for MASE
    )


@dataclass(frozen=True)
class EvaluationResult:
    """A candidate's full rolling-origin result, plus the aggregate summary used to compare it."""

    candidate_name: str
    origins: list[OriginResult]

    @property
    def scored(self) -> pd.DataFrame:
        if not self.origins:
            return pd.DataFrame(
                columns=["cd_setor", "CICLOS", "actual", "items_pred", "abs_error", "origin_cycle"]
            )
        return pd.concat([o.scored for o in self.origins], ignore_index=True)

    @property
    def missing(self) -> pd.DataFrame:
        if not self.origins:
            return pd.DataFrame(columns=["cd_setor", "CICLOS", "origin_cycle"])
        return pd.concat([o.missing for o in self.origins], ignore_index=True)

    def summary(self) -> dict[str, float | int]:
        """Aggregate error per the spec's rule: equal weight per sector, plus p90/worst fold."""
        scored = self.scored
        if scored.empty:
            raise ValueError(
                f"{self.candidate_name}: nothing was scored — check min_train_cycles/horizon "
                "against the panel's number of cycles"
            )
        per_sector_mae = scored.groupby("cd_setor")["abs_error"].mean()
        per_origin_mase = [
            mase(o.scored["actual"], o.scored["items_pred"], o.naive_in_sample_mae)
            for o in self.origins
            if not o.scored.empty
        ]
        return {
            "mae": float(per_sector_mae.mean()),  # equal weight per sector, not per row
            "rmse": rmse(scored["actual"], scored["items_pred"]),
            "mase": float(np.mean(per_origin_mase)) if per_origin_mase else float("nan"),
            "p90_abs_error": float(np.percentile(scored["abs_error"], 90)),
            "worst_origin_mae": float(scored.groupby("origin_cycle")["abs_error"].mean().max()),
            "n_scored": len(scored),
            "n_missing": len(self.missing),
            "n_sectors_scored": scored["cd_setor"].nunique(),
            "n_origins": len(self.origins),
        }

    @property
    def ranking_metric(self) -> float:
        """The number candidates are ranked by — `metrics.PRIMARY_METRIC`, decided in one place."""
        return self.summary()[PRIMARY_METRIC]


def _cycle_order(panel: pd.DataFrame) -> list[str]:
    """Cycles in chronological order, from `opening_date` rather than string-sorting `CICLOS`."""
    return (
        panel[["CICLOS", "opening_date"]]
        .drop_duplicates()
        .sort_values("opening_date")["CICLOS"]
        .tolist()
    )


def _training_cycles(cycles: list[str], origin_idx: int, split: RollingOriginSplit) -> list[str]:
    if split.window == "expanding":
        return cycles[:origin_idx]
    return cycles[max(0, origin_idx - split.min_train_cycles) : origin_idx]


def _naive_in_sample_mae(history: pd.DataFrame) -> float:
    """One-cycle-ahead naive absolute error, pooled across sectors, on the training window alone.

    Used only to scale MASE (metrics.mase) — never touches the holdout being
    scored, so the scale itself can't leak future data. Returns `nan` when
    no sector has 2+ cycles of training history yet (e.g. `min_train_cycles`
    of 1), which makes that fold's `mase` result `nan` too — `mae`/`rmse`
    stay valid regardless, since they don't depend on this denominator.
    """
    diffs = [
        np.abs(np.diff(series["items"].to_numpy()))
        for _, series in history.sort_values("opening_date").groupby("cd_setor")
        if len(series) >= 2
    ]
    if not diffs:
        return float("nan")
    return float(np.mean(np.concatenate(diffs)))


def _score_origin(
    candidate: ForecastCandidate,
    panel: pd.DataFrame,
    train_cycles: list[str],
    target_cycles: list[str],
) -> OriginResult:
    history = panel[panel["CICLOS"].isin(train_cycles)]
    assert (
        history["opening_date"].max()
        < panel[panel["CICLOS"].isin(target_cycles)]["opening_date"].min()
    ), "leakage: training history reaches into or past the target cycles"

    targets = panel[panel["CICLOS"].isin(target_cycles)][["cd_setor", "CICLOS", "opening_date"]]
    predictions = candidate.fit_predict(history, targets)

    actuals = panel[panel["CICLOS"].isin(target_cycles)][["cd_setor", "CICLOS", "items"]].rename(
        columns={"items": "actual"}
    )
    merged = actuals.merge(predictions, on=["cd_setor", "CICLOS"], how="left")
    is_missing = merged["items_pred"].isna()

    scored = merged.loc[~is_missing].copy()
    scored["abs_error"] = (scored["actual"] - scored["items_pred"]).abs()
    scored["origin_cycle"] = train_cycles[-1]

    missing = merged.loc[is_missing, ["cd_setor", "CICLOS"]].copy()
    missing["origin_cycle"] = train_cycles[-1]

    return OriginResult(
        origin_cycle=train_cycles[-1],
        target_cycles=tuple(target_cycles),
        scored=scored,
        missing=missing,
        naive_in_sample_mae=_naive_in_sample_mae(history),
    )


def evaluate(
    candidate: ForecastCandidate, panel: pd.DataFrame, split: RollingOriginSplit
) -> EvaluationResult:
    """Run rolling-origin evaluation of `candidate` against `panel` (`build_item_panel`'s shape).

    For every valid origin, trains only on cycles up to that origin, asks
    the candidate to predict the next `split.horizon` cycles, and scores
    against what actually happened. This is the only place in the codebase
    that performs the temporal split — a candidate never sees or chooses its
    own cutoff (see `model.ForecastCandidate`). `candidate` is a Strategy
    (any `ForecastCandidate`); this function is its context and behaves
    identically regardless of which technique is plugged in.
    """
    cycles = _cycle_order(panel)
    origins = [
        _score_origin(
            candidate,
            panel,
            train_cycles=_training_cycles(cycles, origin_idx, split),
            target_cycles=cycles[origin_idx : origin_idx + split.horizon],
        )
        for origin_idx in range(split.min_train_cycles, len(cycles) - split.horizon + 1)
    ]
    return EvaluationResult(candidate_name=candidate.name, origins=origins)
