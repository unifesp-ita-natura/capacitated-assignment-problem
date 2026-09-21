"""Revalida as configurações finalistas com mais seeds e grava o traço de convergência.

python calibration/finalize.py run   --n-seeds 8
python calibration/finalize.py trace --config cal_T0_5000 --instance base40
"""

from __future__ import annotations

import argparse
import random
import time
import warnings

import numpy as np
import pandas as pd
import repo
from calibrate import evaluate_config, get_instance, make_params

warnings.filterwarnings("ignore")

sa = repo.simulated_annealing

# Finalistas da calibração, mais o default do repositório como linha de base.
CONFIGS = {
    "default_repo": dict(
        initial_temperature=1000.0,
        cooling_rate=0.99,
        min_temperature=0.01,
        sector_bias=0.8,
        destination_bias=0.7,
        penalty_coefficient=1e6,
    ),
    "cal_T0_500": dict(
        initial_temperature=500.0,
        cooling_rate=0.999,
        min_temperature=0.001,
        sector_bias=0.65,
        destination_bias=0.80,
        penalty_coefficient=1e6,
    ),
    "cal_T0_5000": dict(
        initial_temperature=5000.0,
        cooling_rate=0.999,
        min_temperature=0.001,
        sector_bias=0.80,
        destination_bias=0.80,
        penalty_coefficient=1e6,
    ),
    "cal_no_bias": dict(
        initial_temperature=622.0,
        cooling_rate=0.999,
        min_temperature=0.0129,
        sector_bias=0.50,
        destination_bias=0.50,
        penalty_coefficient=1e7,
    ),
    # Usa de fato o teto de 200_000 iterações: alfa calculado para esse orçamento.
    "cal_long": dict(
        initial_temperature=5000.0,
        cooling_rate=0.99993,
        min_temperature=0.001,
        sector_bias=0.80,
        destination_bias=0.80,
        penalty_coefficient=1e6,
    ),
}


def cmd_run(args) -> None:
    seeds = list(range(101, 101 + args.n_seeds))
    instances = args.instances.split(",")
    rows = []
    start = time.perf_counter()
    for name in args.configs.split(","):
        rows += evaluate_config(CONFIGS[name], instances, seeds, name)
        print(f"[validate] {name} pronto ({time.perf_counter() - start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[validate] escrito {args.out}")


def cmd_trace(args) -> None:
    """Grava a energia por iteração de algumas execuções, para o gráfico de convergência."""
    instance = get_instance(args.instance)
    params = make_params(CONFIGS[args.config])
    print(f"[trace] orçamento previsto: {sa.iterations_for_schedule(params)} iterações")
    traces = []
    for seed in range(101, 101 + args.n_seeds):
        result = sa.solve(
            **instance.sa_kwargs(), params=params, rng=random.Random(seed), trace=True
        )
        traces.append(result.trace)
        print(
            f"[trace] seed={seed} iters={result.iterations} obj={result.objective:.0f}",
            flush=True,
        )
    np.save(args.out, np.stack(traces))
    print(f"[trace] escrito {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="revalidação com mais seeds")
    run.add_argument("--configs", default="default_repo,cal_T0_500,cal_T0_5000,cal_no_bias")
    run.add_argument("--instances", default="base40,tight40,base80")
    run.add_argument("--n-seeds", type=int, default=8)
    run.add_argument("--out", default="results/validate.csv")
    run.set_defaults(func=cmd_run)

    trace = sub.add_parser("trace", help="traço de energia por iteração")
    trace.add_argument("--config", default="cal_T0_5000")
    trace.add_argument("--instance", default="base40")
    trace.add_argument("--n-seeds", type=int, default=3)
    trace.add_argument("--out", default="results/trace.npy")
    trace.set_defaults(func=cmd_trace)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
