"""Driver do experimento sa_calibration: lê o YAML e orquestra as fases via `src/calibration`."""

from __future__ import annotations

import argparse
import itertools
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.calibration import analysis, io, plots
from src.calibration.config import SACalibrationConfig, load_calibration_config
from src.calibration.gap import MipReference, mip_reference
from src.calibration.instances import CalibrationInstance, ForecastChoice, build_instance
from src.calibration.mip import min_feasible_multiplier, solve_mip
from src.calibration.runner import RunSpec, SARun, expand_specs, run_many, run_record
from src.calibration.space import ask_batch, create_study, lhs_configs, warm_start

DEFAULT_CONFIG = "configs/experiments/sa_calibration/full.yaml"
PHASES = ("screen", "tpe", "validate", "extras")
BEST = "best"


@dataclass(frozen=True)
class Context:
    """Config validada, diretório de saída e instâncias já construídas."""

    config: SACalibrationConfig
    output_dir: Path
    instances: dict[str, CalibrationInstance]

    @property
    def search_params(self) -> list[str]:
        """Nomes dos parâmetros do espaço de busca, na ordem do YAML."""
        return list(self.config.search_space)

    @property
    def tight_instances(self) -> list[str]:
        """Instâncias em que a capacidade restringe (calibram beta/gama/rho)."""
        return [name for name, spec in self.config.instances.items() if spec.tight]


def forecast_choice(config: SACalibrationConfig) -> ForecastChoice:
    """Estratégias de previsão pedidas no YAML (None -> seleção automática)."""
    return ForecastChoice(level=config.level_strategy, shape=config.shape_strategy)


def load_context(config_path: str) -> Context:
    """Lê o YAML e constrói todas as instâncias declaradas nele."""
    config = load_calibration_config(config_path)
    choice = forecast_choice(config)
    instances = {
        name: build_instance(name, spec, choice) for name, spec in config.instances.items()
    }
    return Context(config, Path(config.output_dir), instances)


# --- execução de fases do SA (com retomada) ---


def execute(ctx: Context, phase: str, specs: Sequence[RunSpec]) -> None:
    """Roda as execuções da fase que ainda não estão no CSV e grava cada uma ao terminar."""
    path = io.runs_path(ctx.output_dir, phase)
    done = io.done_keys(path)
    todo = [spec for spec in specs if spec.key not in done]
    print(f"[{phase}] {len(specs) - len(todo)} já feitas, {len(todo)} a rodar", flush=True)
    counter = itertools.count(1)

    def on_result(run: SARun) -> None:
        io.write_trace(io.trace_path(ctx.output_dir, run.spec), run.trace)
        io.append_run(path, run_record(run))
        _log_run(phase, next(counter), len(todo), run)

    run_many(ctx.instances, todo, ctx.config.budget, ctx.config.n_workers, on_result)


def _log_run(phase: str, index: int, total: int, run: SARun) -> None:
    result = run.result
    print(
        f"[{phase}] {index}/{total} {run.spec.config_id} {run.spec.instance} "
        f"seed={run.spec.seed} obj={result.objective:.1f} feasible={result.feasible} "
        f"it={result.iterations} t={result.time:.1f}s",
        flush=True,
    )


def load_mip_records(ctx: Context) -> dict[str, dict]:
    """O JSON do MIP de cada instância (`{"instance": nome}` se ainda não foi resolvido)."""
    return {name: _mip_record(ctx, name) for name in ctx.instances}


def _mip_record(ctx: Context, name: str) -> dict:
    path = io.mip_path(ctx.output_dir, name)
    return io.read_json(path) if path.exists() else {"instance": name}


def load_references(ctx: Context) -> dict[str, MipReference]:
    """Referência de gap de cada instância, a partir de `mip_optima/*.json`."""
    return {name: mip_reference(record) for name, record in load_mip_records(ctx).items()}


def phase_runs(ctx: Context, phases: Iterable[str], instances: Iterable[str]) -> pd.DataFrame:
    """Execuções das fases pedidas, restritas às instâncias pedidas, já com gap e grupo."""
    runs = io.read_runs([io.runs_path(ctx.output_dir, phase) for phase in phases])
    if runs.empty:
        return runs
    runs = runs[runs["instance"].isin(list(instances))]
    return analysis.prepare_runs(runs, load_references(ctx), ctx.tight_instances)


# --- subcomandos ---


def cmd_tighten(ctx: Context) -> None:
    """Acha o menor `capacity_multiplier` viável de cada instância apertada (vai para o YAML)."""
    settings, choice = ctx.config.tighten, forecast_choice(ctx.config)
    result = {}
    for name in ctx.tight_instances:
        spec = ctx.config.instances[name]
        minimum = min_feasible_multiplier(name, spec, choice, settings)
        suggested = minimum * settings.headroom
        result[name] = {"min_feasible": minimum, "suggested": suggested}
        print(f"[tighten] {name}: mínimo viável {minimum:.6f}, sugerido {suggested:.6f}")
    io.write_json(ctx.output_dir / "tighten.json", result)


def cmd_mip(ctx: Context) -> None:
    """Resolve o MIP de cada instância com HiGHS e grava `mip_optima/{instance}.json`."""
    for name, instance in ctx.instances.items():
        spec = ctx.config.instances[name]
        outcome = solve_mip(instance, ctx.config.mip, spec.mip_time_limit)
        record = {**outcome.to_dict(), "instance_metadata": instance.metadata()}
        io.write_json(io.mip_path(ctx.output_dir, name), record)
        print(
            f"[mip] {name}: {outcome.termination} obj={outcome.objective} "
            f"LB={outcome.lower_bound} gap={outcome.gap} t={outcome.time:.1f}s",
            flush=True,
        )


def cmd_screen(ctx: Context) -> None:
    """Fase 1: configurações em hipercubo latino x instâncias x seeds."""
    screen = ctx.config.screen
    points = lhs_configs(ctx.config.search_space, screen.n_configs, screen.sampler_seed)
    configs = {f"lhs_{i:02d}": point for i, point in enumerate(points)}
    io.write_json(ctx.output_dir / "screen_configs.json", configs)
    execute(ctx, "screen", expand_specs("screen", configs, screen.instances, screen.seeds))


def cmd_tpe(ctx: Context) -> None:
    """Fase 2: lotes de TPE (Optuna) aquecidos com tudo o que já foi avaliado."""
    tpe = ctx.config.tpe
    batches_path = ctx.output_dir / "tpe_batches.json"
    saved = io.read_json(batches_path) if batches_path.exists() else {}
    for batch in range(tpe.batches):
        configs = saved.get(str(batch)) or _ask_tpe_batch(ctx, batch)
        saved[str(batch)] = configs
        io.write_json(batches_path, saved)
        execute(ctx, "tpe", expand_specs("tpe", configs, tpe.instances, tpe.seeds))


def _ask_tpe_batch(ctx: Context, batch: int) -> dict[str, dict[str, float]]:
    """Novo lote do TPE, a partir do histórico (fases 1 e 2) nas instâncias do TPE."""
    tpe = ctx.config.tpe
    runs = phase_runs(ctx, ("screen", "tpe"), tpe.instances)
    study = create_study(tpe.sampler_seed + batch)
    warm_start(study, ctx.config.search_space, analysis.history_from_runs(runs, ctx.search_params))
    points = ask_batch(study, ctx.config.search_space, tpe.batch_size)
    return {f"tpe_b{batch}_{i:02d}": point for i, point in enumerate(points)}


def cmd_validate(ctx: Context) -> None:
    """Fase 3: finalistas (fases 1+2) + configurações de referência, com seeds novas."""
    phase = ctx.config.validate_phase
    ranking = ranking_for(ctx, ("screen", "tpe"), ctx.config.tpe.instances)
    finalists = list(ranking["config_id"].head(phase.n_finalists))
    runs = phase_runs(ctx, ("screen", "tpe"), ctx.config.tpe.instances)
    configs = analysis.configs_from_runs(runs[runs["config_id"].isin(finalists)], ctx.search_params)
    configs.update(named(ctx, phase.reference_configs))
    io.write_json(ctx.output_dir / "finalists.json", configs)
    execute(ctx, "validate", expand_specs("validate", configs, phase.instances, phase.seeds))


def cmd_extras(ctx: Context) -> None:
    """Varredura de T_min na melhor configuração + configurações nomeadas (relatório anterior)."""
    extras = ctx.config.extras
    _, config = recommended(ctx)
    best = {k: v for k, v in config.items() if k != "cooling_rate"}  # alfa re-derivado
    sweep = {f"tmin_{t:g}": {**best, "min_temperature": t} for t in extras.min_temperatures}
    configs = {**sweep, **named(ctx, extras.named_configs)}
    execute(ctx, "extras", expand_specs("extras", configs, extras.instances, extras.seeds))


def cmd_report(ctx: Context) -> None:
    """Tabelas, efeitos principais, convergência e recomendação em `outputs/report/`."""
    report_dir = ctx.output_dir / "report"
    records = load_mip_records(ctx).values()
    io.write_table(analysis.mip_table(records), report_dir / "mip_reference.csv")
    all_runs = phase_runs(ctx, PHASES, ctx.instances)
    io.write_table(
        analysis.summarize(all_runs, ("phase", "group", "config_id")),
        report_dir / "by_config_group.csv",
    )
    write_rankings(ctx, report_dir)
    write_effects(ctx, report_dir)
    write_convergence(ctx, report_dir)
    config_id, config = recommended(ctx)
    io.write_json(report_dir / "recommendation.json", {"config_id": config_id, "config": config})


def cmd_all(ctx: Context) -> None:
    """Todas as fases em sequência (o `tighten` fica de fora: seu resultado vai para o YAML)."""
    for command in (cmd_mip, cmd_screen, cmd_tpe, cmd_validate, cmd_extras, cmd_report):
        command(ctx)


# --- apoio aos subcomandos ---


def named(ctx: Context, names: Iterable[str]) -> dict[str, dict[str, float]]:
    """Configurações nomeadas do YAML (`default_repo`, `prev_*`...)."""
    return {name: dict(ctx.config.named_configs[name]) for name in names}


def ranking_for(ctx: Context, phases: Sequence[str], instances: Sequence[str]) -> pd.DataFrame:
    """Configurações das fases dadas, ordenadas pela regra de decisão."""
    runs = phase_runs(ctx, phases, instances)
    return analysis.ranked_summary(runs, ctx.config.tie_tolerance)


def recommended(ctx: Context) -> tuple[str, dict[str, float]]:
    """A configuração recomendada: topo da revalidação, ou das fases 1+2 se ela não rodou."""
    validated = io.runs_path(ctx.output_dir, "validate").exists()
    phases, instances = (
        (("validate",), ctx.config.validate_phase.instances)
        if validated
        else (("screen", "tpe"), ctx.config.tpe.instances)
    )
    return analysis.best_of(phase_runs(ctx, phases, instances), ctx.config.tie_tolerance)


def write_rankings(ctx: Context, report_dir: Path) -> None:
    """Ranking por fase e, na revalidação, separado por grupo (folgadas x apertadas)."""
    tolerance = ctx.config.tie_tolerance
    for phase in PHASES:
        runs = phase_runs(ctx, (phase,), ctx.instances)
        if not runs.empty:
            io.write_table(
                analysis.ranked_summary(runs, tolerance), report_dir / f"ranking_{phase}.csv"
            )
    validate = phase_runs(ctx, ("validate",), ctx.instances)
    for group, runs in validate.groupby("group"):
        io.write_table(
            analysis.ranked_summary(runs, tolerance), report_dir / f"ranking_validate_{group}.csv"
        )


def write_effects(ctx: Context, report_dir: Path) -> None:
    """Efeitos principais da fase 1 por grupo, com e sem as configurações que ficam inviáveis."""
    runs = phase_runs(ctx, ("screen",), ctx.instances)
    rho = analysis.feasibility_by(runs, "penalty_coefficient")
    io.write_table(rho, report_dir / "feasibility_by_rho.csv")
    _write_group_effects(report_dir, "all", runs)
    _write_group_effects(report_dir, "feasible", analysis.always_feasible(runs))


def _write_group_effects(report_dir: Path, tag: str, runs: pd.DataFrame) -> None:
    """Tabela e gráfico de efeitos principais de cada grupo (folgadas x apertadas)."""
    for group, group_runs in runs.groupby("group"):
        effects = analysis.main_effects(group_runs, analysis.SEARCH_PARAMS)
        name = f"effects_{tag}_{group}"
        io.write_table(effects, report_dir / f"{name}.csv")
        title = f"Fase 1 — instâncias {group} ({tag}): gap médio por quartil"
        plots.plot_main_effects(effects, report_dir / f"{name}.png", title)


def write_convergence(ctx: Context, report_dir: Path) -> None:
    """Energia e temperatura por iteração das configurações escolhidas no YAML."""
    settings = ctx.config.report
    ids = [_resolve_config_id(ctx, config_id) for config_id in settings.convergence_configs]
    traces = {
        config_id: io.read_trace(path)
        for config_id in ids
        if (path := _find_trace(ctx, config_id)).exists()
    }
    title = f"Convergência — {settings.convergence_instance}, seed {settings.convergence_seed}"
    plots.plot_convergence(traces, report_dir / "convergence.png", title)


def _resolve_config_id(ctx: Context, config_id: str) -> str:
    """Troca o apelido `best` pelo id da configuração recomendada na revalidação."""
    return recommended(ctx)[0] if config_id == BEST else config_id


def _find_trace(ctx: Context, config_id: str) -> Path:
    settings = ctx.config.report
    spec = RunSpec("", config_id, {}, settings.convergence_instance, settings.convergence_seed)
    return io.trace_path(ctx.output_dir, spec)


COMMANDS = {
    "tighten": cmd_tighten,
    "mip": cmd_mip,
    "screen": cmd_screen,
    "tpe": cmd_tpe,
    "validate": cmd_validate,
    "extras": cmd_extras,
    "report": cmd_report,
    "all": cmd_all,
}


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m experiments.sa_calibration.run <subcomando> [--config caminho.yaml]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning, module="src.forecasting")
    COMMANDS[args.command](load_context(args.config))


if __name__ == "__main__":
    main()
