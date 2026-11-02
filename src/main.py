"""Run the end-to-end pipeline: raw data -> forecast -> chosen solver -> explained metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RunConfig, load_config
from src.forecasting.dataset import load_demand_base
from src.persistence import write_results_csv
from src.pipeline import explain as explain_stage
from src.pipeline.forecast_stage import (
    HorizonPlan,
    forecast_accuracy,
    forecast_horizon,
    plan_horizon,
)
from src.pipeline.solve_stage import SOLVERS, Solved, run_solver
from src.solver.real_inputs import RealInstance, build_real_instance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pipeline.yaml"


def _assignment_table(solved: Solved, real: RealInstance, instance_as_is: dict[int, int]):
    """Sector, CD, As-Is and To-Be slot (combination id = slot day) as a frame."""
    to_be = solved.assignment or {}
    return pd.DataFrame(
        {
            "cd_setor": list(real.sector_ids),
            "cd_cd": [real.sector_to_cd[s] for s in real.sector_ids],
            "as_is_slot": [instance_as_is[i] for i in real.sector_ids.values()],
            "to_be_slot": [to_be.get(i) for i in real.sector_ids.values()],
        }
    )


def _build_instance(raw: pd.DataFrame, plan: HorizonPlan, config: RunConfig, daily, cycles):
    pipeline = config.pipeline
    return build_real_instance(
        daily,
        raw,
        plan.targets,
        as_is_cycle=cycles[-1],
        max_churn=pipeline.max_churn,
        capacity_multiplier=pipeline.capacity_multiplier,
        max_sectors=pipeline.max_sectors,
    )


def run_pipeline(config: RunConfig) -> explain_stage.Explanation:
    """Execute every stage for `config` and write the results under `config.paths.results`."""
    pipeline = config.pipeline
    level = config.forecast.params
    raw = load_demand_base(pipeline.base_csv)

    plan = plan_horizon(raw, pipeline.horizon_cycles, pipeline.history_cycles)
    print(f"Forecasting {plan.targets} as of {plan.as_of:%Y-%m-%d} with level={level.model}")
    forecast = forecast_horizon(raw, plan, pipeline, level)
    real = _build_instance(raw, plan, config, forecast.daily, forecast.training_cycles)
    instance = real.instance
    print(f"Instance: {len(instance.sectors)} sectors, {len(instance.days)} days")
    print(f"Solving with {pipeline.solver_family}")

    solved = run_solver(pipeline.solver_family, instance, config)
    accuracy = forecast_accuracy(raw, plan, pipeline, level) if pipeline.report_accuracy else {}
    explanation = explain_stage.explain(instance, solved, accuracy)

    results = Path(config.paths.results)
    write_results_csv([solved.result], results / "pipeline_solve.csv")
    explain_stage.write_explanation(explanation, results / "pipeline_explanation.csv")
    _assignment_table(solved, real, instance.current_assignment).to_csv(
        results / "pipeline_assignment.csv", index=False
    )
    print(explain_stage.render(explanation))
    return explanation


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--solver", choices=sorted(SOLVERS), help="overrides pipeline.solver_family"
    )
    parser.add_argument("--max-sectors", type=int, help="overrides pipeline.max_sectors")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the config, apply CLI overrides and run the pipeline."""
    args = _parse_args(argv)
    config = load_config(args.config)
    overrides = {"solver_family": args.solver, "max_sectors": args.max_sectors}
    updates = {key: value for key, value in overrides.items() if value is not None}
    config = config.model_copy(update={"pipeline": config.pipeline.model_copy(update=updates)})
    run_pipeline(config)


if __name__ == "__main__":
    main()
