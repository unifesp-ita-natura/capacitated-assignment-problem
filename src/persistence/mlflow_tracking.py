"""Log solve params, metrics, and artifacts to an MLflow tracking server."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from src.persistence.results import SolveResult

logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:  # mlflow is an optional dependency; see pyproject [tracking]
    mlflow = None

_enabled = False


def configure_tracking(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> bool:
    """Point this process at an MLflow tracking server. Call once at startup.

    Returns whether tracking is active, so callers can log a one-line notice
    instead of silently no-op-ing for an entire run.
    """
    global _enabled
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow is None or not uri:
        _enabled = False
        return False

    mlflow.set_tracking_uri(uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    _enabled = True
    return True


def tracking_enabled() -> bool:
    return _enabled


@contextmanager
def _safe(action: str, *, target: str = "") -> Iterator[None]:
    try:
        yield
    except Exception:
        logger.warning(
            "mlflow %s failed%s",
            action,
            f" for {target}" if target else "",
            exc_info=True,
        )


@contextmanager
def _run(run_name: str, tags: Mapping[str, str]) -> Iterator[None]:
    if not tracking_enabled():
        yield
        return
    try:
        run = mlflow.start_run(run_name=run_name, tags=dict(tags))
    except Exception:
        logger.warning("mlflow start_run failed for %s", run_name, exc_info=True)
        yield
        return
    with run:
        yield


def _run_tags(result: SolveResult, extra_tags: Mapping[str, str] | None) -> dict[str, str]:
    tags = {
        "status": result.status,
        "termination_condition": result.termination_condition,
        "solver": result.solver,
    }
    if result.git_commit is not None:
        tags["git_commit"] = result.git_commit
    tags.update(extra_tags or {})
    return tags


def _log_params(params: Mapping[str, object]) -> None:
    if not tracking_enabled() or not params:
        return
    with _safe("log_params"):
        mlflow.log_params({key: str(value) for key, value in params.items()})


def _log_metrics(result: SolveResult) -> None:
    if not tracking_enabled():
        return
    values = {"wall_time_seconds": result.wall_time_seconds, "objective": result.objective}
    metrics = {name: value for name, value in values.items() if value is not None}
    with _safe("log_metrics"):
        mlflow.log_metrics(metrics)


def _log_artifacts(artifact_paths: Sequence[str | Path] | None) -> None:
    if not tracking_enabled() or not artifact_paths:
        return
    for path in artifact_paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        with _safe("log_artifact", target=str(candidate)):
            mlflow.log_artifact(str(candidate))


def log_solve_result(
    result: SolveResult,
    *,
    params: Mapping[str, object] | None = None,
    tags: Mapping[str, str] | None = None,
    artifact_paths: Sequence[str | Path] | None = None,
) -> None:
    """Log one SolveResult as an MLflow run: tags, params, metrics, artifacts.

    No-op when tracking hasn't been configured (see `configure_tracking`).
    Logging failures are caught and warned, never raised, so a broken or
    unreachable tracking server cannot fail a solve or a comparison run.
    """
    with _run(result.model_name, _run_tags(result, tags)):
        _log_params(params or {})
        _log_metrics(result)
        _log_artifacts(artifact_paths)
