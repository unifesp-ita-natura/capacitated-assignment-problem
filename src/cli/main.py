"""CLI entrypoint — runs a toy block-assignment solve against the default config."""

from __future__ import annotations

from src.config import load_config
from src.solver.inputs import ProblemInstance
from src.solver.mip import run_block_assignment_mip

TOY_INSTANCE = ProblemInstance(
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
    current_assignment={1: 1, 2: 2},
    max_churn=1,
    days_by_cycle={0: [1, 2]},
)


def main() -> None:
    config = load_config("configs/default.yaml")
    result = run_block_assignment_mip(TOY_INSTANCE, solver=config.solver)
    print(f"[{config.project_name}] solver={result.solver} status={result.status}")


if __name__ == "__main__":
    main()
