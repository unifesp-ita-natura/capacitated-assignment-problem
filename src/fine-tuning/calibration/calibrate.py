"""Calibra os parâmetros do Simulated Annealing do CAP (varredura em grade + refinamento TPE).

Subcomandos:
    screen   varredura em grade embaralhada (hipercubo latino sobre os níveis)
    refine   refinamento sequencial tipo TPE (Parzen) sobre os melhores 20%
    bkv      execuções longas de referência (proxy do ótimo quando o MIP não roda)

Toda execução usa `simulated_annealing.calibration_params`, que desabilita a
parada por estagnação (`stagnation_window = max_iterations`, tolerância 0.0) e
fixa `max_iterations = 200_000`.

    python calibration/calibrate.py screen  --n-configs 60 --n-seeds 5
    python calibration/calibrate.py refine  --screen results/screen.csv --batches 3
    python calibration/calibrate.py bkv     --n-seeds 3
"""

from __future__ import annotations

import argparse
import random
import time
import warnings

import numpy as np
import pandas as pd
import repo
from instances import build_instance

warnings.filterwarnings("ignore")

sa = repo.simulated_annealing

MAX_ITERATIONS = 200_000

PARAM_NAMES = [
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "sector_bias",
    "destination_bias",
    "penalty_coefficient",
]

LEVELS = {
    "initial_temperature": [500.0, 1000.0, 2000.0, 5000.0],
    "cooling_rate": [0.90, 0.95, 0.98, 0.995, 0.999],
    "min_temperature": [0.001, 0.01, 0.1],
    "sector_bias": [0.5, 0.65, 0.8, 0.95],
    "destination_bias": [0.5, 0.65, 0.8, 0.95],
    "penalty_coefficient": [1e4, 1e5, 1e6, 1e7],
}

INSTANCE_SPECS = {
    "base40": dict(n_sectors=40, n_cycles_horizon=2, capacity_multiplier=1.0, seed=42),
    "tight40": dict(n_sectors=40, n_cycles_horizon=2, capacity_multiplier=0.003, seed=42),
    "base80": dict(n_sectors=80, n_cycles_horizon=3, capacity_multiplier=1.0, seed=7),
}

_INSTANCE_CACHE: dict[str, object] = {}


def get_instance(name: str):
    """A instância `name`, construída uma única vez por processo."""
    if name not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[name] = build_instance(name, **INSTANCE_SPECS[name])
    return _INSTANCE_CACHE[name]


def make_params(config: dict):
    """`AnnealingParams` da configuração, com a parada por estagnação desabilitada."""
    return sa.calibration_params(
        max_iterations=MAX_ITERATIONS, **{name: config[name] for name in PARAM_NAMES}
    )


def run_once(config: dict, instance_name: str, seed: int) -> dict:
    """Uma execução do SA, com o registro que vai para o CSV."""
    instance = get_instance(instance_name)
    start = time.perf_counter()
    result = sa.solve(**instance.sa_kwargs(), params=make_params(config), rng=random.Random(seed))
    elapsed = time.perf_counter() - start
    feasible = result.capacity_penalty == 0.0 and result.churn_penalty == 0.0
    return {
        "instance": instance_name,
        "seed": seed,
        "objective": result.objective,
        "capacity_penalty": result.capacity_penalty,
        "churn_penalty": result.churn_penalty,
        "feasible": bool(feasible),
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
        "seconds": elapsed,
        **config,
    }


def evaluate_config(config: dict, instances: list[str], seeds: list[int], config_id: str):
    """Roda a configuração em todas as instâncias e seeds pedidos."""
    rows = []
    for instance_name in instances:
        for seed in seeds:
            row = run_once(config, instance_name, seed)
            row["config_id"] = config_id
            rows.append(row)
    return rows


# --- etapa 1: varredura em grade embaralhada ---------------------------------


def shuffled_grid(n_configs: int, rng: np.random.Generator) -> list[dict]:
    """Amostra balanceada: cada nível de cada fator aparece ~n_configs/n_níveis vezes.

    É a alternativa viável ao fatorial completo (3.840 combinações nos níveis
    acima): mantém as marginais equilibradas, que é o que sustenta a leitura de
    efeitos principais.
    """
    columns = {}
    for name, levels in LEVELS.items():
        repeats = int(np.ceil(n_configs / len(levels)))
        pool = (levels * repeats)[:n_configs]
        rng.shuffle(pool)
        columns[name] = pool
    return [{name: columns[name][i] for name in LEVELS} for i in range(n_configs)]


# --- etapa 2: refinamento tipo TPE -------------------------------------------

BOUNDS = {
    "initial_temperature": (np.log10(500.0), np.log10(5000.0)),
    "cooling_rate": (np.log10(0.001), np.log10(0.10)),  # em log10(1 - alfa)
    "min_temperature": (-3.0, -1.0),
    "sector_bias": (0.5, 0.95),
    "destination_bias": (0.5, 0.95),
    "penalty_coefficient": (4.0, 7.0),
}


def to_latent(config: dict) -> np.ndarray:
    """Configuração -> espaço latente (log para escalas multiplicativas)."""
    return np.array(
        [
            np.log10(config["initial_temperature"]),
            np.log10(1.0 - config["cooling_rate"]),
            np.log10(config["min_temperature"]),
            config["sector_bias"],
            config["destination_bias"],
            np.log10(config["penalty_coefficient"]),
        ]
    )


def from_latent(x: np.ndarray) -> dict:
    """Espaço latente -> configuração, recortada nos limites da busca."""
    lower = np.array([BOUNDS[name][0] for name in PARAM_NAMES])
    upper = np.array([BOUNDS[name][1] for name in PARAM_NAMES])
    x = np.clip(x, lower, upper)
    return {
        "initial_temperature": float(10 ** x[0]),
        "cooling_rate": float(1.0 - 10 ** x[1]),
        "min_temperature": float(10 ** x[2]),
        "sector_bias": float(x[3]),
        "destination_bias": float(x[4]),
        "penalty_coefficient": float(10 ** x[5]),
    }


def tpe_suggest(history: pd.DataFrame, n_candidates: int, batch: int, rng: np.random.Generator):
    """Tree-structured Parzen Estimator: amostra de l(x) (top 20%) e maximiza l(x)/g(x).

    Substitui o Optuna quando ele não está disponível; o critério l/g é o mesmo
    do sampler padrão dele, com KDE gaussiana sobre o espaço latente.
    """
    from scipy.stats import gaussian_kde

    ranked = history.sort_values("score")
    n_good = max(4, int(np.ceil(0.2 * len(ranked))))
    good = np.array([to_latent(c) for c in ranked.head(n_good)["config"]]).T
    bad = np.array([to_latent(c) for c in ranked.tail(len(ranked) - n_good)["config"]]).T

    kde_good = gaussian_kde(good, bw_method=0.35)
    kde_bad = gaussian_kde(bad, bw_method=0.35) if bad.shape[1] > 6 else None

    candidates = kde_good.resample(n_candidates, seed=int(rng.integers(1e6)))
    scores = np.log(kde_good(candidates) + 1e-300)
    if kde_bad is not None:
        scores -= np.log(kde_bad(candidates) + 1e-300)
    order = np.argsort(-scores)[:batch]
    return [from_latent(candidates[:, index]) for index in order]


# --- métrica de qualidade -----------------------------------------------------


def current_bkv(all_rows: pd.DataFrame) -> dict[str, float]:
    """Melhor objetivo viável já visto por instância (proxy do ótimo do MIP)."""
    feasible = all_rows[all_rows["feasible"]]
    return feasible.groupby("instance")["objective"].min().to_dict()


def score_from_rows(rows: pd.DataFrame, bkv: dict[str, float]) -> float:
    """Gap percentual médio entre instâncias; execução inviável entra como 100%.

    A penalidade fixa por inviabilidade é o que impede uma configuração de
    "ganhar" reportando um objetivo baixo numa solução que estoura o churn.
    """
    per_instance = []
    for instance_name, group in rows.groupby("instance"):
        best = bkv.get(instance_name)
        gaps = [
            100.0 if not row["feasible"] else 100.0 * (row["objective"] - best) / best
            for _, row in group.iterrows()
        ]
        per_instance.append(float(np.mean(gaps)))
    return float(np.mean(per_instance))


# --- subcomandos --------------------------------------------------------------


def cmd_screen(args) -> None:
    rng = np.random.default_rng(args.seed)
    configs = shuffled_grid(args.n_configs, rng)
    seeds = list(range(1, args.n_seeds + 1))
    instances = args.instances.split(",")
    rows = []
    start = time.perf_counter()
    for index, config in enumerate(configs):
        rows += evaluate_config(config, instances, seeds, f"screen{index:03d}")
        if (index + 1) % 5 == 0:
            print(
                f"[screen] {index + 1}/{len(configs)} configs, {time.perf_counter() - start:.0f}s",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[screen] escrito {args.out} ({len(rows)} execuções)")


def _history_from_rows(runs: pd.DataFrame) -> list[dict]:
    """Uma entrada por configuração já avaliada, com seus parâmetros e execuções."""
    history = []
    for config_id, group in runs.groupby("config_id"):
        config = {name: float(group.iloc[0][name]) for name in PARAM_NAMES}
        history.append({"config_id": config_id, "config": config, "rows": group})
    return history


def cmd_refine(args) -> None:
    rng = np.random.default_rng(args.seed)
    previous = pd.concat([pd.read_csv(path) for path in args.screen.split(",")], ignore_index=True)
    seeds = list(range(1, args.n_seeds + 1))
    instances = args.instances.split(",")

    all_rows = previous.copy()
    history = _history_from_rows(previous)

    def rebuild_scores() -> None:
        bkv = current_bkv(all_rows)
        for entry in history:
            entry["score"] = score_from_rows(entry["rows"], bkv)

    rebuild_scores()
    new_rows = []
    start = time.perf_counter()
    for batch_index in range(args.batches):
        frame = pd.DataFrame([{"score": e["score"], "config": e["config"]} for e in history])
        for position, config in enumerate(tpe_suggest(frame, 400, args.batch_size, rng)):
            config_id = f"tpe{batch_index}_{position:02d}"
            rows = pd.DataFrame(evaluate_config(config, instances, seeds, config_id))
            new_rows.append(rows)
            all_rows = pd.concat([all_rows, rows], ignore_index=True)
            history.append({"config_id": config_id, "config": config, "rows": rows})
        rebuild_scores()
        best = min(history, key=lambda entry: entry["score"])
        print(
            f"[refine] lote {batch_index + 1}/{args.batches}, "
            f"{time.perf_counter() - start:.0f}s, melhor score={best['score']:.2f} "
            f"({best['config_id']})",
            flush=True,
        )
    pd.concat(new_rows, ignore_index=True).to_csv(args.out, index=False)
    print(f"[refine] escrito {args.out}")


def cmd_bkv(args) -> None:
    """Execuções longas de referência, para quando o MIP não estiver disponível."""
    config = dict(
        initial_temperature=5000.0,
        cooling_rate=0.9999,
        min_temperature=0.001,
        sector_bias=0.8,
        destination_bias=0.7,
        penalty_coefficient=1e6,
    )
    rows = []
    start = time.perf_counter()
    for instance_name in args.instances.split(","):
        for seed in range(901, 901 + args.n_seeds):
            row = run_once(config, instance_name, seed)
            row["config_id"] = "bkv_longrun"
            rows.append(row)
            print(
                f"[bkv] {instance_name} seed={seed} obj={row['objective']:.0f} "
                f"viável={row['feasible']} {time.perf_counter() - start:.0f}s",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[bkv] escrito {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    screen = sub.add_parser("screen", help="varredura em grade")
    screen.add_argument("--n-configs", type=int, default=60)
    screen.add_argument("--n-seeds", type=int, default=5)
    screen.add_argument("--instances", default="base40,tight40")
    screen.add_argument("--seed", type=int, default=0)
    screen.add_argument("--out", default="results/screen.csv")
    screen.set_defaults(func=cmd_screen)

    refine = sub.add_parser("refine", help="refinamento TPE")
    refine.add_argument(
        "--screen", default="results/screen.csv", help="CSV(s) separados por vírgula"
    )
    refine.add_argument("--batches", type=int, default=3)
    refine.add_argument("--batch-size", type=int, default=12)
    refine.add_argument("--n-seeds", type=int, default=5)
    refine.add_argument("--instances", default="base40,tight40")
    refine.add_argument("--seed", type=int, default=1)
    refine.add_argument("--out", default="results/refine.csv")
    refine.set_defaults(func=cmd_refine)

    bkv = sub.add_parser("bkv", help="execuções longas de referência")
    bkv.add_argument("--n-seeds", type=int, default=3)
    bkv.add_argument("--instances", default="base40,tight40,base80")
    bkv.add_argument("--out", default="results/bkv.csv")
    bkv.set_defaults(func=cmd_bkv)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
