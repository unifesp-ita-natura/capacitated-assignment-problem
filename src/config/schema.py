"""Pydantic schema for the top-level run configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Paths(BaseModel):
    instances: str = "data/instances"
    processed: str = "data/processed"
    results: str = "results"


class TrainWindow(BaseModel):
    lookback_periods: int = 52
    horizon_periods: int = 4


class NaiveParams(BaseModel):
    model: Literal["naive"] = "naive"
    strategy: Literal["last_value", "seasonal_naive", "mean"] = "last_value"
    season_length: int | None = None

    @model_validator(mode="after")
    def _season_length_valid(self):
        if self.strategy == "seasonal_naive":
            if not isinstance(self.season_length, int) or self.season_length <= 0:
                raise ValueError(
                    "season_length must be a positive integer when strategy is 'seasonal_naive'"
                )
        else:
            self.season_length = None
        return self


class ArimaParams(BaseModel):
    model: Literal["arima"] = "arima"
    order: tuple[int, int, int] = (1, 0, 0)  # (p, d, q)
    seasonal_order: tuple[int, int, int, int] | None = None  # (P, D, Q, s)
    trend: Literal["n", "c", "t", "ct"] | None = None


class LightGBMParams(BaseModel):
    model: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    lags: list[int] = [1, 2, 3, 4]


ForecastParams = Annotated[NaiveParams | ArimaParams | LightGBMParams, Field(discriminator="model")]


class ForecastConfig(BaseModel):
    params: ForecastParams = NaiveParams()
    train_window: TrainWindow = TrainWindow()


class RunConfig(BaseModel):
    project_name: str = "experiments-template"
    modeling: str = "pyomo"
    solver: str = "gurobi"
    paths: Paths = Paths()

    forecast: ForecastConfig = ForecastConfig()
