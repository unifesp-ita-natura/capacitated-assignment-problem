"""Read shape archives and prepare aligned daily plots and filtered metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.forecasting.shape_evaluation import compare_shapes

KEYS = ["cd_setor", "CICLOS", "origin_cycle"]
REQUIRED = [
    *KEYS,
    "candidate",
    "date",
    "actual",
    "actual_share",
    "share_pred",
    "items_pred",
    "total_actual",
]


def list_runs(root: Path) -> list[str]:
    """List direct run folders without following folder symlinks."""
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if _is_run(path))


def _is_run(path: Path) -> bool:
    """Recognize an archive folder without following a folder symlink."""
    return path.is_dir() and not path.is_symlink() and (path / "predictions.csv").is_file()


def run_path(root: Path, name: str) -> Path:
    """Resolve only listed run names inside the configured root."""
    if name not in list_runs(root):
        raise ValueError("Pasta de resultados inexistente ou inválida.")
    directory = (root / name).resolve()
    if directory.parent != root.resolve():
        raise ValueError("A pasta deve estar dentro da raiz de resultados.")
    return directory


def archive_file(directory: Path, name: str) -> Path:
    """Keep archive file symlinks from exposing files outside the run folder."""
    source = directory / name
    if source.resolve().parent != directory.resolve():
        raise ValueError("Arquivo fora da pasta da execução.")
    return source


@lru_cache(maxsize=4)
def _read_csv(path: str, modified: int) -> pd.DataFrame:
    """Cache predictions by modification time and validate their daily keys."""
    frame = pd.read_csv(path, dtype=dict.fromkeys([*KEYS, "candidate"], str))
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Resultados incompatíveis com shape diário; faltam: {sorted(missing)}")
    if frame[[*KEYS, "candidate", "date"]].duplicated().any():
        raise ValueError("Existem previsões duplicadas por modelo, origem, setor, ciclo e dia.")
    return _relative_days(frame)


def _relative_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Count days from each sector-cycle window opening, with the opening as day one."""
    dates = pd.to_datetime(frame["date"])
    starts = dates.groupby([frame[key] for key in [*KEYS, "candidate"]]).transform("min")
    if "window_start" in frame:
        starts = pd.to_datetime(frame["window_start"])
    return frame.assign(relative_day=(dates - starts).dt.days + 1)


def predictions(directory: Path) -> pd.DataFrame:
    """Read daily predictions without changing the archive or cached frame."""
    source = archive_file(directory, "predictions.csv")
    return _read_csv(str(source), source.stat().st_mtime_ns)


def records(frame: pd.DataFrame) -> list[dict]:
    """Serialize dataframe numbers and missing values as standard JSON."""
    return json.loads(frame.to_json(orient="records"))


def metadata(directory: Path) -> dict:
    """Read execution metadata and the original ranking when available."""
    manifest = archive_file(directory, "run_manifest.json")
    comparison = archive_file(directory, "comparison.csv")
    return {"manifest": _json_if_present(manifest), "official_metrics": _csv_if_present(comparison)}


def _json_if_present(path: Path) -> dict:
    """Read optional manifest data."""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def _csv_if_present(path: Path) -> list[dict]:
    """Read optional official metrics."""
    if path.is_file():
        return records(pd.read_csv(path))
    return []


def options(root: Path, name: str) -> dict:
    """List filters and valid cycle-origin pairs for one archive."""
    directory = run_path(root, name)
    frame = predictions(directory)
    return {
        "models": sorted(frame["candidate"].unique().tolist()),
        "sectors": sorted(frame["cd_setor"].unique().tolist()),
        "cycles": sorted(frame["CICLOS"].unique().tolist()),
        "origins_by_cycle": frame.groupby("CICLOS")["origin_cycle"].unique().map(sorted).to_dict(),
        **metadata(directory),
    }


def selected_points(root: Path, query: dict) -> pd.DataFrame:
    """Filter one cycle-origin and optional sector/model before aggregating."""
    frame = predictions(run_path(root, query["run"]))
    selected = frame[frame["CICLOS"].eq(query["cycle"]) & frame["origin_cycle"].eq(query["origin"])]
    return _filter(
        _filter(selected, "cd_setor", query.get("sector")), "candidate", query.get("model")
    )


def _filter(frame: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    """Treat empty filter selections as all values."""
    if value:
        return frame[frame[column].eq(value)].copy()
    return frame.copy()


def _common_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only daily keys available for all selected models."""
    counts = frame.groupby([*KEYS, "date"])["candidate"].transform("nunique")
    return frame[counts.eq(frame["candidate"].nunique())].copy()


def _series(frame: pd.DataFrame) -> list[dict]:
    """Align sectors by day since opening and derive weighted and cumulative shares."""
    daily = frame.groupby("relative_day", as_index=False)[["actual", "items_pred"]].sum()
    total = daily["actual"].sum()
    denominator = total if total else float("nan")
    daily["actual_pct"] = 100 * daily["actual"] / denominator
    daily["pred_pct"] = 100 * daily["items_pred"] / denominator
    daily["actual_cumulative_pct"] = daily["actual_pct"].cumsum()
    daily["pred_cumulative_pct"] = daily["pred_pct"].cumsum()
    daily["error"] = daily["items_pred"] - daily["actual"]
    return records(daily)


def view(root: Path, query: dict) -> dict:
    """Prepare the current slice and keep plot/table payloads bounded."""
    selected = _common_points(selected_points(root, query))
    if selected.empty:
        raise ValueError("Nenhum ponto comum encontrado para esses filtros.")
    selected["error"] = selected["items_pred"] - selected["actual"]
    selected["abs_error"] = selected["error"].abs()
    selected["abs_share_error"] = (selected["share_pred"] - selected["actual_share"]).abs()
    selected["uniform_fallback"] = selected.get("uniform_fallback", False)
    columns = [
        "candidate",
        *KEYS,
        "date",
        "relative_day",
        "actual_share",
        "share_pred",
        "actual",
        "items_pred",
        "error",
    ]
    return {
        "metrics": records(compare_shapes(selected)),
        "series": {name: _series(group) for name, group in selected.groupby("candidate")},
        "points": records(
            selected.sort_values(["candidate", "cd_setor", "date"])[columns].head(200)
        ),
        "n_points": len(selected),
        "n_sectors": selected["cd_setor"].nunique(),
    }
