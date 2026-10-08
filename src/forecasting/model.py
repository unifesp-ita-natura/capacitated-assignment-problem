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
  `window_scaled` adapts a candidate that only knows (sector, cycle) to
  answer each scenario window, and `opening_adjusted` makes its forecast
  depend on the day the window opens, so every candidate is keyed by
  (sector, cycle, opening date). `opening_scenarios` builds those queries.
- **Registry/Factory** (`CandidateRegistry`) — maps a `ForecastParams`
  config's `model` discriminator to the builder that constructs the matching
  Strategy. Each technique registers itself from its own module
  (`candidates/naive.py` etc.) via `@REGISTRY.register("naive")`, so adding
  a technique means adding a module, never editing this file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
# How many past cycles at one opening day it takes for that day's measured
# effect to count half: a day seen this often keeps half its raw effect, a
# day seen rarely stays near "no effect" (see `opening_factors`).
OPENING_DAY_PRIOR = 100


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

    `fn` receives one sector's series (indexed by `opening_date`, sorted
    ascending) and the number of steps to forecast, and returns a series of
    that length. The series is items per unit of window length (see
    `window_scaled`), so `fn` forecasts a level and each scenario scales it.
    """
    return opening_adjusted(window_scaled(_PerSectorCandidate(name=name, fn=fn)))


def cycle_days_exponent(history: pd.DataFrame) -> float:
    """How a sector's items scale with its window length: the β in items ~ cycle_days ** β.

    The slope of log items on log cycle_days, both taken relative to the
    sector's own mean, pooled over every sector. Clipped to [0, 1]: 0 is
    "the length doesn't matter", 1 is "items per day times days". 0 when the
    history never varied the length, since there is nothing to learn from.
    """
    # ponytail: sector effects only; a cycle-wide shock that coincides with
    # short windows leaks into β. Add cycle effects if β drifts between folds.
    usable = history[(history["items"] > 0) & (history["cycle_days"] > 0)]
    by_sector = usable["cd_setor"]
    days = np.log(usable["cycle_days"].astype(float))
    items = np.log(usable["items"].astype(float))
    days = days - days.groupby(by_sector).transform("mean")
    items = items - items.groupby(by_sector).transform("mean")
    spread = float((days * days).sum())
    if spread == 0:
        return 0.0
    return float(np.clip((days * items).sum() / spread, 0.0, 1.0))


class _WindowScaledCandidate:
    """Adapter: answers each scenario window with a candidate that forecasts per (sector, cycle).

    The wrapped candidate sees every past cycle's items divided by
    cycle_days ** β, so what it forecasts is a level per unit of window.
    Each scenario gets that level times its own cycle_days ** β, so two
    opening dates that give the sector windows of different length get
    different forecasts. β comes from the same history (`cycle_days_exponent`).
    """

    def __init__(self, inner: ForecastCandidate) -> None:
        self.name = inner.name
        self._inner = inner
        self.chosen_exponent: float | None = None

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        self.chosen_exponent = cycle_days_exponent(history)
        scale = history["cycle_days"].astype(float) ** self.chosen_exponent
        levels = self._inner.fit_predict(history.assign(items=history["items"] / scale), targets)
        levels = levels.drop_duplicates(CYCLE_KEYS)[[*CYCLE_KEYS, "items_pred"]]
        rows = targets[QUERY_KEYS].merge(levels, on=CYCLE_KEYS)
        rows["items_pred"] = rows["items_pred"] * rows["cycle_days"] ** self.chosen_exponent
        return rows


def window_scaled(candidate: ForecastCandidate) -> ForecastCandidate:
    """Adapter: make a candidate that forecasts per (sector, cycle) answer each scenario window."""
    return _WindowScaledCandidate(candidate)


def opening_scenarios(cycles: pd.DataFrame, days: Iterable[int]) -> pd.DataFrame:
    """Queries for the same cycle opening on different dates: one per (sector, cycle) and day.

    `cycles` has one row per (sector, cycle): `cd_setor`, `CICLOS`, the
    cycle's `opening_date` and the sector's `cycle_days`. Day `d` is the
    scenario where the sector's window opens `d` days after the cycle does.
    The window keeps its length and only moves.
    """
    offsets = pd.DataFrame({"opening_day": list(days)})
    scenarios = cycles[["cd_setor", "CICLOS", "opening_date", "cycle_days"]].merge(
        offsets, how="cross"
    )
    scenarios["window_start"] = scenarios["opening_date"] + pd.to_timedelta(
        scenarios.pop("opening_day"), unit="D"
    )
    return scenarios[[*QUERY_KEYS, "opening_date"]]


def opening_day(frame: pd.DataFrame) -> pd.Series:
    """How many days after its cycle opens each row's window opens."""
    return (frame["window_start"] - frame["opening_date"]).dt.days


def opening_factors(history: pd.DataFrame) -> pd.Series:
    """Multiplicative effect on items of opening `d` days after the cycle opens, indexed by d.

    Each past cycle's log items are taken relative to its sector (what the
    sector usually sells) and to its cycle (what everyone sold that cycle),
    so what is left compares a sector with itself when it opened on another
    day. The mean of that by opening day is shrunk toward 0 by
    `OPENING_DAY_PRIOR`, then exponentiated. A day never seen has no entry,
    which callers read as a factor of 1.
    """
    usable = history[history["items"] > 0]
    residual = np.log(usable["items"].astype(float))
    # ponytail: fixed 10 alternating passes for the two-way demeaning;
    # converge on a tolerance if the panel grows much more unbalanced.
    for _ in range(10):
        residual = residual - residual.groupby(usable["cd_setor"]).transform("mean")
        residual = residual - residual.groupby(usable["CICLOS"]).transform("mean")
    by_day = residual.groupby(opening_day(usable))
    return np.exp(by_day.sum() / (by_day.size() + OPENING_DAY_PRIOR))


class _OpeningAdjustedCandidate:
    """Adapter: makes a candidate's forecast depend on the day the scenario opens.

    The wrapped candidate sees every past cycle's items divided by the
    factor of the day it opened on (`opening_factors`), so it forecasts the
    sector's level as if the opening day didn't matter. Each scenario gets
    that forecast times its own day's factor. The wrapped candidate must
    already answer per scenario (see `window_scaled`).
    """

    def __init__(self, inner: ForecastCandidate) -> None:
        self._inner = inner
        self.name = inner.name
        self.opening_factors = pd.Series(dtype=float)

    def __getattr__(self, attribute: str):
        # What the wrapped candidate chose (alpha, exponent) stays readable.
        return getattr(self._inner, attribute)

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        self.opening_factors = opening_factors(history)
        level_history = history.assign(items=history["items"] / self._factor(history))
        predictions = self._inner.fit_predict(level_history, targets)
        predictions = predictions.drop_duplicates(QUERY_KEYS)[[*QUERY_KEYS, "items_pred"]]
        rows = targets[[*QUERY_KEYS, "opening_date"]].merge(predictions, on=QUERY_KEYS)
        rows["items_pred"] = rows["items_pred"] * self._factor(rows)
        return rows[[*QUERY_KEYS, "items_pred"]]

    def _factor(self, frame: pd.DataFrame) -> np.ndarray:
        return opening_day(frame).map(self.opening_factors).fillna(1.0).to_numpy()


def opening_adjusted(candidate: ForecastCandidate) -> ForecastCandidate:
    """Adapter: make a scenario-answering candidate's forecast depend on the opening day."""
    return _OpeningAdjustedCandidate(candidate)


def align_predictions(targets: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    """`items_pred` for each row of `targets`, in its order; NaN where the candidate skipped.

    Every prediction is keyed by its full query, so each scenario window gets
    its own value. A candidate that returns only (sector, cycle) is a bug:
    wrap it in `window_scaled`.
    """
    missing = set(SCENARIO_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing {sorted(missing)}; wrap it in window_scaled")
    unique = predictions.drop_duplicates(QUERY_KEYS)[[*QUERY_KEYS, "items_pred"]]
    merged = targets[QUERY_KEYS].merge(unique, on=QUERY_KEYS, how="left")
    return merged["items_pred"].set_axis(targets.index)


def forecast(
    candidate: ForecastCandidate, history: pd.DataFrame, queries: pd.DataFrame
) -> pd.DataFrame:
    """Answer scenario queries from all of `history`: one expected cycle total per query.

    This is f(sector, opening day, cycle length) = expected items for the
    cycle, the call the optimizer makes for every opening date it weighs.
    Each query names its sector (`cd_setor`), the cycle it schedules
    (`CICLOS`, with the cycle's `opening_date`) and the scenario window
    (`window_start`, `cycle_days`). The cycle is required rather than
    guessed: candidates read its code (LightGBM's cycle number) and order
    horizons by its date. Returns `queries` with an `items_pred` column, NaN
    where the candidate could not answer.
    """
    missing = set(QUERY_KEYS + ["opening_date"]) - set(queries.columns)
    if missing:
        raise ValueError(f"queries are missing {sorted(missing)}")
    if history["opening_date"].max() >= queries["opening_date"].min():
        raise ValueError("every query must be for a cycle after the end of `history`")
    answered = queries.copy()
    answered["items_pred"] = align_predictions(queries, candidate.fit_predict(history, queries))
    return answered


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
