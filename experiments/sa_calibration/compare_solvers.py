"""Driver da comparação Gurobi × HiGHS × SA: resolve os MIPs, roda o SA e agrega as tabelas."""

from __future__ import annotations

import argparse
import itertools
import os
import platform
import warnings
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.calibration import io
from src.calibration.comparison import (
    SolverComparisonConfig,
    best_lower_bound,
    group_by_instance,
    load_comparison_config,
    mip_record_name,
    mip_rows,
    proven_optimum,
    reference_upper_bound,
    sa_rows,
    solve_options,
)
from src.calibration.config import MipSettings
from src.calibration.instances import CalibrationInstance, ForecastChoice, build_instance
from src.calibration.mip import solve_mip
from src.calibration.runner import SARun, expand_specs, run_many, run_record

DEFAULT_CONFIG = "configs/experiments/sa_calibration/solver_comparison.yaml"
PHASE = "compare"
LOOSE_REFERENCE_SOLVER = "appsi_highs"  # a referência da calibração anterior (HiGHS, gap 0,01)


def build_instances(config: SolverComparisonConfig) -> dict[str, CalibrationInstance]:
    """Instâncias do YAML com a previsão fixada (nível, forma)."""
    choice = ForecastChoice(config.level_strategy, config.shape_strategy)
    return {name: build_instance(name, spec, choice) for name, spec in config.instances.items()}


def mip_path(output_dir: Path, instance: str, solver: str, gap: float) -> Path:
    """JSON de um solve do MIP em `outputs/solver_comparison/mip/`."""
    return output_dir / "mip" / f"{mip_record_name(instance, solver, gap)}.json"


def cmd_mip(config: SolverComparisonConfig, instances: dict[str, CalibrationInstance]) -> None:
    """Resolve cada (instância, solver, tolerância) que ainda não tem JSON, um por vez."""
    output_dir = Path(config.output_dir)
    io.write_json(output_dir / "env.json", environment())
    for name, solver, gap in itertools.product(instances, config.mip_solvers, config.mip_rel_gaps):
        path = mip_path(output_dir, name, solver, gap)
        if path.exists():
            continue
        extra = solve_options(config, solver, name, gap)
        settings = MipSettings(solver=solver, mip_rel_gap=gap, extra_options=extra)
        outcome = solve_mip(instances[name], settings, config.instances[name].mip_time_limit)
        io.write_json(path, {**outcome.to_dict(), "instance_metadata": instances[name].metadata()})
        print(
            f"[mip] {name} {solver} gap={gap:g}: {outcome.termination} "
            f"UB={outcome.upper_bound} LB={outcome.lower_bound} t={outcome.time:.1f}s",
            flush=True,
        )


def environment() -> dict:
    """Máquina e versões em que os MIPs rodaram (RAM, núcleos, Python, solvers)."""
    page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(page * pages / 1024**3, 2),
        "solvers": _solver_versions(),
    }


def _solver_versions() -> dict[str, str]:
    versions = {}
    for package in ("gurobipy", "highspy", "pyomo"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "ausente"
    return versions


def cmd_sa(config: SolverComparisonConfig, instances: dict[str, CalibrationInstance]) -> None:
    """Roda as configurações do SA × instâncias × seeds que ainda não estão no CSV."""
    output_dir = Path(config.output_dir)
    path = io.runs_path(output_dir, PHASE)
    specs = expand_specs(PHASE, config.sa_configs, instances, config.sa_seeds)
    done = io.done_keys(path)
    todo = [spec for spec in specs if spec.key not in done]
    print(f"[sa] {len(specs) - len(todo)} já feitas, {len(todo)} a rodar", flush=True)
    counter = itertools.count(1)

    def on_result(run: SARun) -> None:
        io.write_trace(io.trace_path(output_dir, run.spec), run.trace)
        io.append_run(path, run_record(run))
        print(
            f"[sa] {next(counter)}/{len(todo)} {run.spec.config_id} {run.spec.instance} "
            f"seed={run.spec.seed} obj={run.result.objective:.1f} "
            f"feasible={run.result.feasible} t={run.result.time:.1f}s",
            flush=True,
        )

    run_many(instances, todo, config.budget, config.n_workers, on_result)


def cmd_report(config: SolverComparisonConfig, instances: dict[str, CalibrationInstance]) -> None:
    """`comparison.csv`/`.md` (todas as linhas) e um JSON por par instância/método."""
    output_dir = Path(config.output_dir)
    records = [io.read_json(p) for p in sorted((output_dir / "mip").glob("*.json"))]
    z_ref, others = _references(config, group_by_instance(records, instances))
    runs = io.read_runs([io.runs_path(output_dir, PHASE)])
    table = pd.concat([mip_rows(records, z_ref), sa_rows(runs, z_ref, others)], ignore_index=True)
    table.insert(1, "z_ref", table["instance"].map(z_ref))
    table = table.sort_values(["instance", "method"], kind="stable").reset_index(drop=True)
    io.write_table(table, output_dir / "comparison.csv")
    _write_pair_jsons(output_dir, table)
    print(io.to_markdown(table))


def _references(
    config: SolverComparisonConfig, by_instance: dict[str, list[dict]]
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    """z_ref (ótimo provado) e as referências secundárias: UB do HiGHS frouxo e o melhor LB."""
    z_ref = {name: proven_optimum(rs, config.reference_gap) for name, rs in by_instance.items()}
    z_loose = {
        name: reference_upper_bound(rs, LOOSE_REFERENCE_SOLVER, config.loose_gap)
        for name, rs in by_instance.items()
    }
    z_lb = {name: best_lower_bound(rs) for name, rs in by_instance.items()}
    return z_ref, {f"highs{config.loose_gap:g}": z_loose, "lb": z_lb}


def _write_pair_jsons(output_dir: Path, table: pd.DataFrame) -> None:
    for row in table.to_dict(orient="records"):
        slug = "".join(c if c.isalnum() else "_" for c in row["method"]).strip("_")
        io.write_json(output_dir / "pairs" / f"{row['instance']}__{slug}.json", row)


def cmd_all(config: SolverComparisonConfig, instances: dict[str, CalibrationInstance]) -> None:
    """MIPs primeiro (um por vez, sem SA concorrendo pelos núcleos), depois o SA e o relatório."""
    cmd_mip(config, instances)
    cmd_sa(config, instances)
    cmd_report(config, instances)


COMMANDS = {"mip": cmd_mip, "sa": cmd_sa, "report": cmd_report, "all": cmd_all}


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m experiments.sa_calibration.compare_solvers <subcomando> [--config ...]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning, module="src.forecasting")
    config = load_comparison_config(args.config)
    COMMANDS[args.command](config, build_instances(config))


if __name__ == "__main__":
    main()
