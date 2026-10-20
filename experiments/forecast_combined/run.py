"""Execute and archive combined level-shape forecasts from one YAML configuration."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schema import ForecastParams, LightGBMParams, NaiveParams
from src.forecasting.combined import forecast_combined
from src.forecasting.combined_validation import summarize_cities, validate_combined
from src.forecasting.dataset import load_demand_base
from src.forecasting.features import feature_columns
from src.forecasting.shape_evaluation import ShapeCandidate
from src.forecasting.shape_learning import BASE_FEATURES
from src.persistence.provenance import code_provenance, package_version, save_manifest, sha256_file


class CombinedPaths(BaseModel):
    """Source demand, explicit forecast windows and an exclusive output directory."""

    model_config = ConfigDict(extra="forbid")
    base_csv: Path
    windows_csv: Path | None = None
    run_dir: Path


class CombinedConfig(BaseModel):
    """A single chosen level-shape pair for future forecasting or historical replay."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["forecast", "validate"] = "forecast"
    as_of: datetime | None = None
    history_cycles: int = Field(default=20, ge=1)
    level: ForecastParams = Field(default_factory=lambda: NaiveParams(strategy="mean"))
    shape: ShapeCandidate = Field(
        default_factory=lambda: ShapeCandidate(name="sector", model="sector")
    )
    validation_cycles: list[str] = Field(default_factory=list)
    paths: CombinedPaths

    @model_validator(mode="after")
    def required_inputs(self):
        {"forecast": _require_future, "validate": _require_validation}[self.mode](self)
        return self


def _require_future(config: CombinedConfig) -> None:
    """Require forecast dates and an explicit future-window schedule."""
    if config.as_of is None or config.paths.windows_csv is None:
        raise ValueError("forecast mode requires as_of and paths.windows_csv")


def _require_validation(config: CombinedConfig) -> None:
    """Require named holdout cycles without accepting future-mode inputs."""
    if not config.validation_cycles:
        raise ValueError("validate mode requires validation_cycles")
    if config.as_of is not None or config.paths.windows_csv is not None:
        raise ValueError("validate mode derives as_of and recorded windows per cycle")


def _manifest(config: CombinedConfig) -> dict:
    """Record effective models, features, input identity and code revision."""
    return {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "parameters": config.model_dump(mode="json"),
        "base_sha256": sha256_file(config.paths.base_csv),
        "code": code_provenance(),
        "versions": {
            name: package_version(name)
            for name in ["pandas", "numpy", "pydantic", "lightgbm", "statsmodels"]
        },
        "shape_features": _shape_features(config.shape),
        "level_features": _level_features(config.level),
        "scenario_inputs": ["opening_date", "window_start", "cycle_days"],
        "formula": "items_pred = level_pred * share_pred",
        "target_keys": [
            "scenario_id",
            "cd_setor",
            "estado",
            "cidade",
            "CICLOS",
            "window_start",
            "date",
        ],
    }


def _shape_features(params: ShapeCandidate) -> list[str]:
    """Describe only predictors used by the selected shape family."""
    features = {
        "uniform": ["n_days"],
        "sector": ["cd_setor", "offset", "n_days", "actual_share"],
        "lightgbm": [*BASE_FEATURES, *params.learning.features],
    }
    return features[params.model]


def _level_features(params: ForecastParams) -> list[str]:
    """Record the effective feature list of a learned level candidate."""
    if isinstance(params, LightGBMParams):
        return feature_columns(
            params.lags, params.rolling_windows, params.companion_lags, params.categorical_features
        )
    return []


def _future(config: CombinedConfig, raw: pd.DataFrame, directory: Path) -> dict:
    """Save future intermediate outputs, keeping unavailable model totals explicitly missing."""
    windows = pd.read_csv(
        config.paths.windows_csv, dtype={"scenario_id": str, "cd_setor": str, "CICLOS": str}
    )
    result = forecast_combined(
        raw, windows, config.as_of, config.history_cycles, config.level, config.shape
    )
    for name, frame in [
        ("level_predictions", result.level),
        ("shape_predictions", result.shape),
        ("daily_predictions", result.daily),
    ]:
        frame.to_csv(directory / f"{name}.csv", index=False)
    return {
        "training_cycles": result.training_cycles,
        "n_daily_rows": len(result.daily),
        "n_missing_levels": int(result.level["level_pred"].isna().sum()),
    }


def _validation(config: CombinedConfig, raw: pd.DataFrame, directory: Path) -> dict:
    """Save a fixed-model replay and its daily actual-versus-predicted diagnostics."""
    result = validate_combined(
        raw, config.validation_cycles, config.history_cycles, config.level, config.shape
    )
    result.predictions.to_csv(directory / "daily_validation.csv", index=False)
    result.metrics.to_csv(directory / "validation_metrics.csv", index=False)
    summarize_cities(result.predictions).to_csv(directory / "validation_by_city.csv", index=False)
    result.windows.to_csv(directory / "validation_windows.csv", index=False)
    print(result.metrics.to_string(index=False))
    return {
        "n_validation_rows": len(result.predictions),
        "validation_cycles": config.validation_cycles,
    }


def _snapshot(config: CombinedConfig, contents: str, directory: Path, record: dict) -> None:
    """Keep the original YAML and the exact window schedule used in forecasting."""
    (directory / "config_snapshot.yaml").write_text(contents, encoding="utf-8")
    if config.paths.windows_csv is not None:
        data = config.paths.windows_csv.read_bytes()
        (directory / "windows_snapshot.csv").write_bytes(data)
        record["windows_sha256"] = hashlib.sha256(data).hexdigest()


def run(config_path: str | Path) -> Path:
    """Execute one configuration without overwriting prior runs or changing existing backtests."""
    contents = Path(config_path).read_text(encoding="utf-8-sig")
    config = CombinedConfig.model_validate(yaml.safe_load(contents))
    record = _manifest(config)
    directory = config.paths.run_dir
    directory.mkdir(parents=True, exist_ok=False)
    save_manifest(directory / "run_manifest.json", record)
    print(f"Combined output: {directory.resolve()}", flush=True)
    try:
        _snapshot(config, contents, directory, record)
        raw = load_demand_base(config.paths.base_csv)
        record.update(
            {"forecast": _future, "validate": _validation}[config.mode](config, raw, directory)
        )
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    else:
        record["status"] = "completed"
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        save_manifest(directory / "run_manifest.json", record)
    return directory


def main() -> None:
    """Accept the single configuration path used for this execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
