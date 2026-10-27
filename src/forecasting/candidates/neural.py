"""Neural candidates (N-HiTS, LSTM, MLP) trained here on every series at once via neuralforecast."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config.schema import NeuralParams
from src.forecasting.model import REGISTRY, ForecastCandidate, per_series_batch


def _network(params: NeuralParams, horizon: int):
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM, MLP, NHITS

    architecture = {"nhits": NHITS, "lstm": LSTM, "mlp": MLP}[params.architecture]
    return architecture(
        h=horizon,
        input_size=params.input_size,
        loss=MAE(),  # the harness ranks by MAE
        max_steps=params.max_steps,
        learning_rate=params.learning_rate,
        # Short series are padded at the start instead of being dropped.
        start_padding_enabled=True,
        scaler_type="robust",  # each window scaled by its own median and IQR
        random_seed=params.random_state,
        alias="network",
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )


def train_and_forecast(params: NeuralParams, series: list[pd.Series], horizon: int) -> np.ndarray:
    """Fit one network on every series and forecast `horizon` cycles ahead of each.

    Time is each series' own cycle position, so a cycle the series skipped
    is not read as a zero.
    """
    from neuralforecast import NeuralForecast

    for name in ("lightning.pytorch", "pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    frame = pd.concat(
        [
            pd.DataFrame({"unique_id": i, "ds": np.arange(len(s)), "y": s.to_numpy(dtype=float)})
            for i, s in enumerate(series)
        ],
        ignore_index=True,
    )
    forecaster = NeuralForecast(models=[_network(params, horizon)], freq=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lightning deprecation chatter, once per fit
        forecaster.fit(frame)
        predicted = forecaster.predict().sort_values(["unique_id", "ds"])
    return predicted["network"].to_numpy(dtype=float).reshape(len(series), horizon)


@REGISTRY.register("neural")
def build_neural(params: NeuralParams) -> ForecastCandidate:
    """Build the globally trained network the config selects, retrained at every origin."""

    def forecast(series: list[pd.Series], future: list[pd.Timestamp]) -> np.ndarray:
        return np.clip(train_and_forecast(params, series, len(future)), 0, None)

    return per_series_batch(f"{params.architecture}:in{params.input_size}", forecast)
