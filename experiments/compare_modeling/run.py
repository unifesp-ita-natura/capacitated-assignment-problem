"""Solve the toy block-assignment model with each configured solver and write a comparison table."""

from __future__ import annotations

from src.persistence import SolveResult, configure_tracking, log_solve_result, write_results_csv
from src.solver.mip.block_assignment import run_block_assignment_mip

# Add/remove Pyomo-registered solver names to compare different backends
# against the same model. To compare different *formulations* instead (or in
# addition), swap the toy instance below for a list of instances/builders and
# loop over both.
SOLVERS_TO_COMPARE = ["gurobi", "appsi_highs"]

OUTPUT_PATH = "experiments/compare_modeling/outputs/results.csv"

TOY_INSTANCE = dict(
    sectors=[1, 2],
    combinations=[1, 2],
    days=[1, 2],
    cd_sectors={1: [1, 2]},
    daily_capacity={(1, 1): 100.0, (1, 2): 100.0},
    projected_demand={
        (1, 1, 1): 10.0,
        (1, 2, 1): 0.0,
        (1, 1, 2): 0.0,
        (1, 2, 2): 10.0,
        (2, 1, 1): 10.0,
        (2, 2, 1): 0.0,
        (2, 1, 2): 0.0,
        (2, 2, 2): 10.0,
    },
    current_assignment={(1, 1): 1, (2, 2): 1},
    max_churn=1,
    days_by_cycle={0: [1, 2]},
)


def run(solvers: list[str] = SOLVERS_TO_COMPARE) -> list[SolveResult]:
    results = []
    for solver_name in solvers:
        result = run_block_assignment_mip(**TOY_INSTANCE, solver=solver_name)
        results.append(result)
        log_solve_result(result, tags={"family": "mip"})
    return results


if __name__ == "__main__":
    # No-ops unless MLFLOW_TRACKING_URI is set (or configure_tracking is
    # called with an explicit uri) — see src/persistence/README.md.
    configure_tracking(experiment_name="compare_modeling")
    results = run()
    write_results_csv(results, OUTPUT_PATH)
    for result in results:
        print(result)
