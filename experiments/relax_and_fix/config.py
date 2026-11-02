"""Esquema validado (pydantic) do YAML do experimento `relax_and_fix`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.calibration.config import ExperimentDescription, MipSettings


class InstanceSettings(BaseModel):
    """Uma instância sintética: tamanho, aperto de capacidade e limite de tempo do MIP."""

    n_sectors: int = Field(gt=0)
    n_cycles_horizon: int = Field(default=2, gt=0)
    n_cycles_history: int = Field(default=6, gt=1)  # precisa ser > n_cycles_horizon
    seed: int = 7
    capacity: Literal["loose", "tight"] = "loose"
    # tight: multiplicador de capacidade = tight_headroom x (pico carga/capacidade do As-Is
    # com multiplicador 1.0), então o As-Is estoura o pior CD em (1 - tight_headroom).
    tight_headroom: float = Field(default=0.95, gt=0)
    mip_time_limit: float = Field(default=300.0, gt=0)


class WindowSettings(BaseModel):
    """Tamanho (W) e passo (P) das janelas de CDs."""

    window: int = Field(gt=0)
    step: int = Field(gt=0)
    max_group_sectors: int | None = Field(default=None, gt=0)  # None: um grupo por CD
    max_backtracks: int = Field(default=2, ge=0)  # tentativas extras de uma janela sem solução

    @model_validator(mode="after")
    def _step_fits_the_window(self) -> WindowSettings:
        if self.step > self.window:
            raise ValueError(f"step ({self.step}) deve ser <= window ({self.window})")
        return self


class RelaxAndFixConfig(BaseModel):
    """O YAML inteiro de `configs/experiments/relax_and_fix/*.yaml`."""

    experiment_name: str
    description: ExperimentDescription
    output_dir: str
    mip: MipSettings = MipSettings()
    subproblem_solver: str = "appsi_highs"  # solver dos subproblemas do R&F / F&O
    optimize_passes: int = Field(default=2, ge=0)
    subproblem_time_limit: float = Field(default=60.0, gt=0)
    instances: dict[str, InstanceSettings]
    windows: list[WindowSettings]


def load_config(path: str | Path) -> RelaxAndFixConfig:
    """Lê e valida o YAML do experimento."""
    with open(path) as f:
        return RelaxAndFixConfig.model_validate(yaml.safe_load(f))
