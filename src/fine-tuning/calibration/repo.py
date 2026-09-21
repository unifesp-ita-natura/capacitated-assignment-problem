"""Resolve os módulos do projeto, seja no repositório (`src.*`) ou na cópia standalone (`cap.*`)."""

from __future__ import annotations

import importlib
import importlib.util

PACKAGE = "src" if importlib.util.find_spec("src") is not None else "cap"


def _module(path: str):
    return importlib.import_module(f"{PACKAGE}.{path}")


generator = _module("generate.generator")
solver_inputs = _module("solver.inputs")
simulated_annealing = _module("solver.heuristics.simulated_annealing")
forecasting_data = _module("forecasting.data")
forecasting_model = _module("forecasting.model")
forecasting_scoring = _module("forecasting.scoring")
