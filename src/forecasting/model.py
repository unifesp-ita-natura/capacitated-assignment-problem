"""Strategy interface every forecasting technique implements, plus level+shape combiner.

Three design patterns do the work here, each solving a specific piece of
"add a technique without touching the harness":

- **Strategy** (`ForecastCandidate`) — the interface `evaluation.evaluate()`
  programs against. Naive, ARIMA, LightGBM, ... are interchangeable
  implementations selected at runtime; the harness (the Strategy pattern's
  "context") never branches on which one it got.
- **Adapter** (`per_sector`) — most candidates are naturally a function over
  one sector's series, not the panel-shaped Strategy interface. `per_sector`
  adapts that simpler shape to `ForecastCandidate` once, so per-sector
  techniques (naive, ARIMA, SARIMA) don't each reimplement the panel
  plumbing; a technique that pools across sectors (LightGBM — see
  docs/papers/forecasting-volume-spec.tex section 4.5) implements
  `ForecastCandidate` directly instead, since it needs the whole panel.
- **Registry/Factory** (`CandidateRegistry`) — maps a `ForecastParams`
  config's `model` discriminator to the builder that constructs the matching
  Strategy. Each technique registers itself from its own module
  (`candidates/naive.py` etc.) via `@REGISTRY.register("naive")`, so adding
  a technique means adding a module, never editing this file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from src.config.schema import ForecastParams
from src.forecasting.data import (
    DemandMetric,
    build_cycle_totals,
    build_shape_observations,
    to_calendar,
)
from src.forecasting.dataset import CYCLE_KEYS
from src.forecasting.level import LEVEL_STRATEGIES, forecast_cycle_total_with
from src.forecasting.shape import SHAPE_STRATEGIES
from src.generate.generator import future_cycle_starts, slot_start_day

# Columns build_item_panel() produces (src/forecasting/dataset.py) and every
# candidate's output must match, so the evaluation harness (evaluation.py)
# can join predictions back onto actuals without per-candidate glue.
PANEL_COLUMNS = ["cd_setor", "CICLOS", "items", "opening_date"]
PREDICTION_COLUMNS = ["cd_setor", "CICLOS", "items_pred"]
# A query is one scenario: "sector s, in cycle k, with its window opening on
# `window_start` and lasting `cycle_days` days". One (sector, cycle) can carry
# several queries, one per opening date the optimizer wants to compare.
SCENARIO_COLUMNS = ["window_start", "cycle_days"]
QUERY_KEYS = ["cd_setor", "CICLOS", *SCENARIO_COLUMNS]


def forecast_with_shape(shape: pd.DataFrame, forecast_cycle_totals: pd.DataFrame) -> pd.DataFrame:
    """Combine a within-window shape forecast with the cycle-total forecast into daily counts."""
    forecast = shape.merge(forecast_cycle_totals, on="sector")
    forecast["forecast_orders"] = np.rint(
        forecast["order_share"] * forecast["forecast_cycle_total"]
    ).astype(int)
    return forecast


def forecast_future_cycles(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles_history: int,
    cycle_length: int,
    best_level_name: str,
    best_shape_name: str,
    n_cycles_horizon: int = 1,
    metric: DemandMetric = "pedidos",
) -> pd.DataFrame:
    """Combined level+shape forecast for the next `n_cycles_horizon` cycles, mapped
    onto calendar dates. One level strategy and one shape strategy are applied
    throughout; only the strategies' own step-ahead behavior differs per cycle."""
    future_cycle_ids = [n_cycles_history + i for i in range(1, n_cycles_horizon + 1)]
    future_starts = future_cycle_starts(cycle_starts, cycle_length, n_cycles_horizon)
    all_cycles = pd.DataFrame(
        {
            "cycle_id": [*range(1, n_cycles_history + 1), *future_cycle_ids],
            "open_date": [*cycle_starts, *future_starts],
        }
    )

    cycle_totals = build_cycle_totals(demand_level, metric=metric)
    shape_observations = build_shape_observations(demand_shape, demand_level, metric=metric)

    forecast_cycle_totals = forecast_cycle_total_with(
        LEVEL_STRATEGIES[best_level_name],
        cycle_totals,
        all_cycles,
        steps=range(1, n_cycles_horizon + 1),
    )
    shape_forecast = SHAPE_STRATEGIES[best_shape_name](shape_observations)
    combined = forecast_with_shape(shape_forecast, forecast_cycle_totals)

    start_day_by_sector = {
        sector: slot_start_day(block, sublock) for sector, (block, sublock) in assignment.items()
    }
    cycle_open_dates = dict(zip(future_cycle_ids, future_starts))
    return to_calendar(combined, cycle_open_dates, start_day_by_sector)


class InsufficientHistoryError(Exception):
    """Raised by a per-sector forecast function to say it cannot predict for this sector's history.

    The Adapter (`per_sector`) catches only this exception and skips the
    sector, letting the harness count it as a missing prediction rather than
    fabricate a value. Any other exception is a real bug and propagates.
    """


class ForecastCandidate(Protocol):
    """Strategy interface: a technique that predicts item volume for a set of future cycles.

    `fit_predict` receives only the past (`history` never contains a cycle at
    or after the earliest target) and the target cycles to predict — never a
    training cutoff to compute itself. That split is the evaluation harness's
    job, in one place, so no candidate can leak future data by miscomputing
    its own cutoff.
    """

    name: str

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        """Predict items for each query in `targets` using only `history`.

        `history` and `targets` both have the shape of `build_item_panel`'s
        output (`targets` carries `CICLOS`, `opening_date`, `window_start` and
        `cycle_days` but no `items` column, since that's what's being
        predicted). A (sector, cycle) may appear in several target rows, one
        per scenario window. Returns `cd_setor`, `CICLOS`, `items_pred`, plus
        `window_start` and `cycle_days` if the candidate tells scenarios
        apart; without them, its prediction for a (sector, cycle) applies to
        every scenario of it (see `align_predictions`). A candidate is
        allowed to skip a sector it can't handle (e.g. too little history)
        rather than fabricate a value, and the harness accounts for the gap.
        """
        ...


class _PerSectorCandidate:
    """Adapter: wraps a per-sector function to satisfy `ForecastCandidate`."""

    def __init__(self, name: str, fn: Callable[[pd.Series, int], pd.Series]) -> None:
        self.name = name
        self._fn = fn

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        horizon = targets["CICLOS"].nunique()
        rows = []
        for sector, sector_history in history.sort_values("opening_date").groupby("cd_setor"):
            # A sector present in `history` isn't necessarily active in every
            # target cycle (not every sector orders every cycle) — skip it
            # here rather than call `self._fn` for nothing to score against.
            sector_targets = (
                targets[targets["cd_setor"] == sector]
                .sort_values("opening_date")
                .drop_duplicates("CICLOS")  # scenarios of one cycle share its forecast
            )
            if sector_targets.empty:
                continue

            series = sector_history.set_index("opening_date")["items"]
            try:
                predicted = self._fn(series, horizon)
            except InsufficientHistoryError:
                continue
            for cycle, value in zip(sector_targets["CICLOS"], predicted, strict=False):
                rows.append({"cd_setor": sector, "CICLOS": cycle, "items_pred": value})
        return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def per_sector(name: str, fn: Callable[[pd.Series, int], pd.Series]) -> ForecastCandidate:
    """Adapter: lift a per-sector forecast function to the panel-shaped Strategy interface.

    `fn` receives one sector's `items` series (indexed by `opening_date`,
    sorted ascending) and the number of steps to forecast, and returns a
    series of that length.
    """
    return _PerSectorCandidate(name=name, fn=fn)


def align_predictions(targets: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    """`items_pred` for each row of `targets`, in its order; NaN where the candidate skipped.

    Predictions keyed by the full query match their own scenario. Predictions
    keyed by (sector, cycle) alone come from a candidate that ignores the
    window, so each one is copied onto every scenario of that cycle.
    """
    keys = QUERY_KEYS if set(SCENARIO_COLUMNS) <= set(predictions.columns) else CYCLE_KEYS
    unique = predictions.drop_duplicates(keys)[[*keys, "items_pred"]]
    return targets[keys].merge(unique, on=keys, how="left")["items_pred"].set_axis(targets.index)


def forecast(
    candidate: ForecastCandidate, history: pd.DataFrame, queries: pd.DataFrame
) -> pd.DataFrame:
    """Answer scenario queries `(cd_setor, window_start, cycle_days)` from all of `history`.

    This is f(sector, opening day, cycle length) = expected items for the
    cycle, the call the optimizer makes for every opening date it weighs.
    Without `CICLOS`/`opening_date`, every query is taken as a scenario of
    the cycle right after `history`, dated by its earliest window start;
    pass them to ask about cycles further ahead. Returns `queries` with an
    `items_pred` column, NaN where the candidate could not answer.
    """
    queries = queries.copy()
    if "CICLOS" not in queries.columns:
        queries["CICLOS"] = "next"
        queries["opening_date"] = queries["window_start"].min()
    if history["opening_date"].max() >= queries["opening_date"].min():
        raise ValueError("every query must be for a cycle after the end of `history`")
    queries["items_pred"] = align_predictions(queries, candidate.fit_predict(history, queries))
    return queries


class CandidateRegistry:
    """Registry/Factory: maps a `ForecastParams.model` discriminator to the Strategy builder for it.

    Each forecasting technique registers its own builder — a
    `ForecastParams -> ForecastCandidate` function — under its `model`
    literal (matching the discriminator on the corresponding `*Params` class
    in `src/config/schema.py`), typically via `@REGISTRY.register("naive")`
    as a decorator in the technique's own module. Encapsulated in a class
    (rather than module-level globals) so a fresh registry can be built in
    tests without cross-test pollution from whichever candidate modules
    happened to be imported first.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[ForecastParams], ForecastCandidate]] = {}

    def register(
        self, model_name: str
    ) -> Callable[
        [Callable[[ForecastParams], ForecastCandidate]],
        Callable[[ForecastParams], ForecastCandidate],
    ]:
        def decorator(
            builder: Callable[[ForecastParams], ForecastCandidate],
        ) -> Callable[[ForecastParams], ForecastCandidate]:
            self._builders[model_name] = builder
            return builder

        return decorator

    def build(self, params: ForecastParams) -> ForecastCandidate:
        """Build the Strategy a `ForecastParams` config selects, via its `model` discriminator."""
        try:
            builder = self._builders[params.model]
        except KeyError as exc:
            raise ValueError(
                f"no candidate registered for model={params.model!r}; known: {self.known_models}"
            ) from exc
        return builder(params)

    @property
    def known_models(self) -> list[str]:
        return sorted(self._builders)


# The registry every shipped candidate registers into. Importing
# `src.forecasting.candidates` runs each technique module's
# `@REGISTRY.register(...)` decorator, populating this before `REGISTRY.build`
# is called.
REGISTRY = CandidateRegistry()
