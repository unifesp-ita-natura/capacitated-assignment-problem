"""Strategy interface every forecasting technique implements, plus its Adapter/Registry.

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

import pandas as pd

from src.config.schema import ForecastParams

# Columns build_item_panel() produces (src/forecasting/dataset.py) and every
# candidate's output must match, so the evaluation harness (evaluation.py)
# can join predictions back onto actuals without per-candidate glue.
PANEL_COLUMNS = ["cd_setor", "CICLOS", "items", "opening_date"]
PREDICTION_COLUMNS = ["cd_setor", "CICLOS", "items_pred"]


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
        """Predict items for each (sector, cycle) in `targets` using only `history`.

        `history` and `targets` both have the shape of `build_item_panel`'s
        output (`targets` carries `CICLOS` and `opening_date` but no `items`
        column, since that's what's being predicted). Returns a DataFrame
        with columns `cd_setor`, `CICLOS`, `items_pred` — one row per
        (sector, cycle) this candidate could predict for; a candidate is
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
            sector_targets = targets[targets["cd_setor"] == sector].sort_values("opening_date")
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
