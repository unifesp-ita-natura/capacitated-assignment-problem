"""Solve the example model with each configured solver and write a comparison table."""

from __future__ import annotations

import time

import pyomo.environ as pyo

from src.persistence import SolveResult, configure_tracking, log_solve_result, write_results_csv
from src.solver.mip.example import build_example_model, solve_example

# Add/remove Pyomo-registered solver names to compare different backends
# against the same model. To compare different *formulations* instead (or in
# addition), swap `build_example_model()` for a list of model builders and
# loop over both.
SOLVERS_TO_COMPARE = ["gurobi"]

OUTPUT_PATH = "experiments/compare_modeling/outputs/results.csv"


def run(solvers: list[str] = SOLVERS_TO_COMPARE) -> list[SolveResult]:
    results = []
    for solver_name in solvers:
        model = build_example_model()

        start = time.perf_counter()
        pyomo_results = solve_example(model, solver=solver_name)
        wall_time_seconds = time.perf_counter() - start

        results.append(
            SolveResult(
                model_name="example",
                solver=solver_name,
                status=str(pyomo_results.solver.status),
                termination_condition=str(pyomo_results.solver.termination_condition),
                objective=pyo.value(model.objective, exception=False),
                wall_time_seconds=wall_time_seconds,
            )
        )
        log_solve_result(results[-1], tags={"family": "mip"})
    return results


if __name__ == "__main__":
    # No-ops unless MLFLOW_TRACKING_URI is set (or configure_tracking is
    # called with an explicit uri) — see src/persistence/README.md.
    configure_tracking(experiment_name="compare_modeling")
    results = run()
    write_results_csv(results, OUTPUT_PATH)
    for result in results:
        print(result)
