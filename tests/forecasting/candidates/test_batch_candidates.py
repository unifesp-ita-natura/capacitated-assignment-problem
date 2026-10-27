"""Tests for the batch adapter, the pretrained and neural candidates, and the ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401
from src.config.schema import EnsembleParams, NaiveParams, NeuralParams, PretrainedParams
from src.forecasting.candidates import pretrained
from src.forecasting.candidates.ensemble import fit_mlp, nnls_weights
from src.forecasting.model import REGISTRY, per_series_batch
from tests.forecasting.candidates._panels import flat, panel, split

HISTORY, TARGETS = split(panel(flat(100.0), flat(10.0)), 6)


def test_batch_adapter_sends_every_series_in_one_call_and_maps_steps_to_cycles():
    calls = []

    def last_value(series: list[pd.Series], future: list[pd.Timestamp]) -> np.ndarray:
        calls.append(len(series))
        return np.array([[s.iloc[-1]] * len(future) for s in series])

    predictions = per_series_batch("last", last_value).fit_predict(HISTORY, TARGETS)

    assert calls == [2]
    assert predictions.set_index("serie")["items_pred"].to_dict() == pytest.approx(
        {"A": 100.0, "B": 10.0}
    )
    assert set(predictions["CICLOS"]) == set(TARGETS["CICLOS"])


def test_pretrained_candidate_clips_negative_forecasts(monkeypatch):
    monkeypatch.setattr(
        pretrained, "chronos2", lambda series, horizon, point, context: np.full((2, 1), -5.0)
    )

    predictions = REGISTRY.build(PretrainedParams(name="chronos2")).fit_predict(HISTORY, TARGETS)

    assert (predictions["items_pred"] == 0).all()


def test_neural_candidate_trains_and_forecasts_every_series():
    params = NeuralParams(architecture="mlp", input_size=3, max_steps=2)

    predictions = REGISTRY.build(params).fit_predict(HISTORY, TARGETS)

    assert set(predictions["serie"]) == {"A", "B"}
    assert np.isfinite(predictions["items_pred"]).all()


def _ensemble(combiner: str) -> EnsembleParams:
    members = [NaiveParams(strategy="last_value"), NaiveParams(strategy="mean")]
    return EnsembleParams(members=members, combiner=combiner, validation_cycles=2)


@pytest.mark.parametrize("combiner", ["mean", "median", "nnls", "mlp"])
def test_ensemble_of_agreeing_members_reproduces_them(combiner):
    predictions = REGISTRY.build(_ensemble(combiner)).fit_predict(HISTORY, TARGETS)

    by_series = predictions.set_index("serie")["items_pred"]
    assert by_series["A"] == pytest.approx(100.0, rel=0.05)
    assert by_series["B"] == pytest.approx(10.0, rel=0.05)


def test_nnls_puts_the_weight_on_the_member_that_matches_the_actuals():
    features = np.array([[10.0, 50.0], [20.0, 10.0], [30.0, 90.0]])

    weights = nnls_weights(features, features[:, 0])

    assert weights == pytest.approx([1.0, 0.0], abs=1e-9)


def test_mlp_learns_a_constant_correction_on_top_of_the_members():
    features = np.array([[100.0, 100.0], [10.0, 10.0], [50.0, 50.0]] * 10)

    predict = fit_mlp(features, features[:, 0] * 1.2)

    assert predict(np.array([[200.0, 200.0]])) == pytest.approx([240.0], rel=0.05)
