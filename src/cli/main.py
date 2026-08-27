"""CLI entrypoint — runs the example solve against the default config."""

from __future__ import annotations

from src.config import load_config
from src.solver.mip import solve_example


def main() -> None:
    config = load_config("configs/default.yaml")
    results = solve_example()
    print(f"[{config.project_name}] solver={config.solver} status={results.solver.status}")


if __name__ == "__main__":
    main()
