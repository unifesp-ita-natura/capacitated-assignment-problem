"""Lê e grava os artefatos da calibração: CSV de execuções e traços, JSON do MIP e tabelas."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from src.calibration.runner import RunSpec, TraceRow

TRACE_COLUMNS = ("iteration", "energy", "best_energy", "temperature")
THOUSAND = 1_000


def runs_path(output_dir: str | Path, phase: str) -> Path:
    """CSV com uma linha por execução de uma fase."""
    return Path(output_dir) / f"runs_{phase}.csv"


def trace_path(output_dir: str | Path, spec: RunSpec) -> Path:
    """CSV do traço de uma execução: `sa_runs/{config}_{instance}_{seed}.csv`."""
    return Path(output_dir) / "sa_runs" / f"{spec.config_id}_{spec.instance}_{spec.seed}.csv"


def mip_path(output_dir: str | Path, instance: str) -> Path:
    """JSON do MIP de referência de uma instância."""
    return Path(output_dir) / "mip_optima" / f"{instance}.json"


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_trace(path: Path, rows: Iterable[TraceRow]) -> None:
    """Grava o traço de convergência de uma execução."""
    with open(_ensure_parent(path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_COLUMNS)
        writer.writerows(rows)


def read_trace(path: Path) -> pd.DataFrame:
    """Lê o traço de uma execução."""
    return pd.read_csv(path)


def append_run(path: Path, record: dict) -> None:
    """Acrescenta a linha de uma execução ao CSV da fase (cria o cabeçalho se for novo)."""
    is_new = not path.exists()
    with open(_ensure_parent(path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if is_new:
            writer.writeheader()
        writer.writerow(record)


def read_runs(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatena os CSVs de execução que existirem (frame vazio se nenhum existir)."""
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def done_keys(path: Path) -> set[tuple[str, str, str, int]]:
    """Chaves (fase, config, instância, seed) já gravadas — para retomar uma fase."""
    runs = read_runs([path])
    if runs.empty:
        return set()
    columns = runs[["phase", "config_id", "instance", "seed"]]
    return {(p, c, i, int(s)) for p, c, i, s in columns.itertuples(index=False)}


def write_json(path: Path, payload: dict) -> None:
    """Grava um JSON indentado (cria o diretório se preciso)."""
    with open(_ensure_parent(path), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    """Lê um JSON."""
    with open(path) as f:
        return json.load(f)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Grava a tabela em CSV e, ao lado, em Markdown (`.md`) para o relatório."""
    frame.to_csv(_ensure_parent(path), index=False)
    path.with_suffix(".md").write_text(to_markdown(frame))


def to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Tabela Markdown simples (sem depender do `tabulate`)."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join(_cell(v, digits) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def _cell(value, digits: int) -> str:
    """Números grandes com separador de milhar; os demais com `digits` algarismos."""
    if not isinstance(value, float):
        return str(value)
    return f"{value:,.0f}" if abs(value) >= THOUSAND else f"{value:.{digits}g}"
