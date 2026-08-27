"""Verify the mlflow adapter no-ops when unconfigured and logs when enabled."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.persistence import mlflow_tracking as tracking
from src.persistence.results import SolveResult


def _result(**overrides: object) -> SolveResult:
    defaults: dict[str, object] = {
        "model_name": "example",
        "solver": "gurobi",
        "status": "ok",
        "termination_condition": "optimal",
        "objective": 20.0,
        "wall_time_seconds": 0.05,
        "git_commit": "abc123",
    }
    defaults.update(overrides)
    return SolveResult(**defaults)


@contextmanager
def _fake_start_run(*, run_name: str, tags: dict[str, str]):
    yield MagicMock()


@pytest.fixture(autouse=True)
def _reset_enabled_flag() -> None:
    tracking._enabled = False
    yield
    tracking._enabled = False


def test_configure_tracking_without_uri_disables_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    active = tracking.configure_tracking()

    assert active is False
    assert tracking.tracking_enabled() is False


def test_configure_tracking_without_mlflow_installed_disables_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "mlflow", None)

    active = tracking.configure_tracking(tracking_uri="https://example.invalid")

    assert active is False
    assert tracking.tracking_enabled() is False


def test_log_solve_result_is_noop_when_tracking_disabled() -> None:
    tracking.log_solve_result(_result(), params={"n": 10})
    # No mlflow calls happen; absence of an exception is the assertion.


def test_configure_tracking_with_uri_enables_and_sets_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = MagicMock()
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)

    active = tracking.configure_tracking(
        tracking_uri="https://example.invalid",
        experiment_name="experiments-template",
    )

    assert active is True
    assert tracking.tracking_enabled() is True
    fake_mlflow.set_tracking_uri.assert_called_once_with("https://example.invalid")
    fake_mlflow.set_experiment.assert_called_once_with("experiments-template")


def test_log_solve_result_logs_run_name_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.side_effect = _fake_start_run
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)
    tracking._enabled = True

    tracking.log_solve_result(_result(), tags={"family": "mip"})

    fake_mlflow.start_run.assert_called_once()
    _, start_run_kwargs = fake_mlflow.start_run.call_args
    assert start_run_kwargs["run_name"] == "example"
    assert start_run_kwargs["tags"] == {
        "family": "mip",
        "status": "ok",
        "termination_condition": "optimal",
        "solver": "gurobi",
        "git_commit": "abc123",
    }


def test_log_solve_result_logs_params_metrics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.side_effect = _fake_start_run
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)
    tracking._enabled = True

    artifact_path = tmp_path / "results.csv"
    artifact_path.write_text("solver,objective\n", encoding="utf-8")

    tracking.log_solve_result(
        _result(),
        params={"n": 10, "density": 0.3},
        artifact_paths=[artifact_path],
    )

    fake_mlflow.log_params.assert_called_once_with({"n": "10", "density": "0.3"})
    fake_mlflow.log_metrics.assert_called_once_with({"wall_time_seconds": 0.05, "objective": 20.0})
    fake_mlflow.log_artifact.assert_called_once_with(str(artifact_path))


def test_log_solve_result_survives_mlflow_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.side_effect = RuntimeError("unreachable tracking server")
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)
    tracking._enabled = True

    tracking.log_solve_result(_result())
    # No exception propagates even though start_run blew up.
