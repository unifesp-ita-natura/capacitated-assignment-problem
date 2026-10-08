"""Archive backtest inputs, effective candidate settings and execution metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter

from src.config.schema import ForecastParams, LightGBMParams
from src.forecasting.evaluation import RollingOriginSplit
from src.forecasting.features import feature_columns
from src.forecasting.model import REGISTRY

_ADAPTER = TypeAdapter(ForecastParams)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _features(params: ForecastParams) -> list[str]:
    if isinstance(params, LightGBMParams):
        return feature_columns(
            params.lags,
            params.rolling_windows,
            params.companion_lags,
            params.categorical_features,
        )
    return []


def _candidate_record(param: ForecastParams) -> dict:
    return {
        "name": REGISTRY.build(param).name,
        "model": param.model,
        "effective_parameters": param.model_dump(mode="json"),
        "features": _features(param),
        "input_kind": "feature_table" if isinstance(param, LightGBMParams) else "item_series",
        "scenario_inputs": ["opening_date", "window_start", "cycle_days"],
    }


def _candidate_records(entries: list[dict]) -> list[dict]:
    records = [_candidate_record(_ADAPTER.validate_python(entry)) for entry in entries]
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError(
            "candidate names must be unique; use distinct labels for LightGBM variants"
        )
    return records


def _record_directory(config: dict) -> Path:
    paths = config["paths"]
    if "run_dir" in paths:
        return Path(paths["run_dir"])
    return Path(paths["output_csv"]).parent / f"run_{uuid4().hex}"


def prepare_record(config_path: str | Path, config: dict) -> tuple[Path, dict]:
    """Validate settings and create a fresh run directory before evaluating any model."""
    source, base = Path(config_path).resolve(), Path(config["paths"]["base_csv"]).resolve()
    manifest = {
        "schema_version": 1,
        "run_id": uuid4().hex,
        "status": "running",
        "started_at": _now(),
        "timezone": "America/Sao_Paulo",
        "config_source": str(source),
        "config_sha256": _sha256(source),
        "data": {"path": str(base), "sha256": _sha256(base)},
        "split": asdict(RollingOriginSplit(**config["split"])),
        "daily": config.get("daily"),
        "candidates": _candidate_records(config["candidates"]),
        "code": {
            "commit": _git_output("rev-parse", "HEAD"),
            "working_tree_status": _git_output("status", "--porcelain"),
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                package: _package_version(package)
                for package in ("numpy", "pandas", "pydantic", "pyyaml", "lightgbm", "statsmodels")
            },
        },
        "artifacts": {
            "comparison": "comparison.csv",
            "predictions": "predictions.csv",
            "config": "config.yaml",
            "manifest": "manifest.json",
        },
        "parameter_scope": "Configured parameters including defaults, not per-fold fitted state.",
    }
    directory = _record_directory(config)
    _create_directory(directory)
    shutil.copyfile(source, directory / "config.yaml")
    update_record(directory, manifest, "running")
    return directory, manifest


def _create_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"A pasta da execução já existe: {directory}. Escolha outro paths.run_dir no YAML."
        ) from error


def update_record(directory: Path, manifest: dict, status: str, **details) -> None:
    """Write the run status and results without serializing nonstandard JSON NaN values."""
    record = {**manifest, **details, "status": status, "updated_at": _now()}
    (directory / "manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
