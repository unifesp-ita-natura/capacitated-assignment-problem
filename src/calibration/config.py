"""Esquema validado (pydantic) do YAML do experimento `sa_calibration`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Status = Literal["exploratory", "in-progress", "confirmed", "superseded", "reference"]


class ExperimentDescription(BaseModel):
    """Goal/hypothesis/design-note/status exigidos por `configs/experiments/README.md`."""

    goal: str
    hypothesis: str
    design_note: str
    status: Status


class InstanceSpec(BaseModel):
    """Parâmetros do gerador sintético que definem uma instância do CAP."""

    n_sectors: int = Field(gt=0)
    n_cycles_horizon: int = Field(gt=0)
    seed: int
    capacity_multiplier: float = Field(default=1.0, gt=0)
    cycle_length: int = Field(default=21, gt=0)
    n_cycles_history: int = Field(default=6, gt=1)
    max_churn: float = Field(default=0.2, ge=0, le=1)  # fração dos setores que pode mudar
    tight: bool = False  # capacidade restringe de verdade -> calibra beta/gama/rho
    mip_time_limit: float = Field(default=600.0, gt=0)


class Dimension(BaseModel):
    """Um eixo do espaço de busca: intervalo [low, high], em escala log ou linear."""

    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> Dimension:
        if self.low >= self.high or (self.log and self.low <= 0):
            raise ValueError(f"intervalo inválido: [{self.low}, {self.high}] (log={self.log})")
        return self


class MipSettings(BaseModel):
    """Como o MIP de referência é resolvido."""

    solver: str = "appsi_highs"
    mip_rel_gap: float = Field(default=0.01, ge=0)


class BudgetSettings(BaseModel):
    """Orçamento fixo de toda execução do SA e resolução do traço gravado."""

    max_iterations: int = Field(default=200_000, gt=0)
    trace_every: int = Field(default=1_000, gt=0)


class ScreenPhase(BaseModel):
    """Fase 1: grade em hipercubo latino (LHS)."""

    n_configs: int = Field(gt=0)
    seeds: list[int]
    instances: list[str]
    sampler_seed: int = 0


class TpePhase(BaseModel):
    """Fase 2: lotes de TPE (Optuna) aquecidos com a fase 1."""

    batches: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    seeds: list[int]
    instances: list[str]
    sampler_seed: int = 0


class ValidatePhase(BaseModel):
    """Fase 3: revalidação dos finalistas com seeds novas e uma instância nunca vista."""

    n_finalists: int = Field(gt=0)
    seeds: list[int]
    instances: list[str]
    reference_configs: list[str]  # nomes em `named_configs` que sempre entram


class TightenSettings(BaseModel):
    """Bissecção do `capacity_multiplier` que define as instâncias apertadas."""

    low: float = Field(gt=0)
    high: float = Field(gt=0)
    steps: int = Field(default=8, gt=0)
    probe_time_limit: float = Field(default=60.0, gt=0)
    headroom: float = Field(default=1.15, ge=1)  # multiplicador usado = headroom x mínimo viável


class ReportSettings(BaseModel):
    """Quais execuções entram no gráfico de convergência."""

    convergence_instance: str
    convergence_seed: int
    convergence_configs: list[str]  # ids de configuração; "best" = recomendada


class ExtrasPhase(BaseModel):
    """Experimentos de resfriamento: varredura de T_min e as configs nomeadas."""

    min_temperatures: list[float]
    seeds: list[int]
    instances: list[str]
    named_configs: list[str]


class SACalibrationConfig(BaseModel):
    """O YAML inteiro de `configs/experiments/sa_calibration/*.yaml`."""

    experiment_name: str
    description: ExperimentDescription
    output_dir: str
    n_workers: int = Field(default=0, ge=0)  # 0 -> os.cpu_count()
    level_strategy: str | None = None  # None -> seleção automática do forecasting
    shape_strategy: str | None = None
    budget: BudgetSettings = BudgetSettings()
    mip: MipSettings = MipSettings()
    instances: dict[str, InstanceSpec]
    search_space: dict[str, Dimension]
    tie_tolerance: float = Field(default=0.5, ge=0)  # pontos percentuais de gap
    named_configs: dict[str, dict[str, float]]
    screen: ScreenPhase
    tpe: TpePhase
    validate_phase: ValidatePhase = Field(alias="validate")
    extras: ExtrasPhase
    tighten: TightenSettings
    report: ReportSettings

    @model_validator(mode="after")
    def _known_instances(self) -> SACalibrationConfig:
        phases = (self.screen, self.tpe, self.validate_phase, self.extras)
        missing = {n for phase in phases for n in phase.instances} - set(self.instances)
        if missing:
            raise ValueError(f"instâncias sem especificação: {sorted(missing)}")
        return self


def load_calibration_config(path: str | Path) -> SACalibrationConfig:
    """Lê e valida o YAML do experimento."""
    with open(path) as f:
        return SACalibrationConfig.model_validate(yaml.safe_load(f))
