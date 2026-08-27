"""Pydantic schema for the top-level run configuration."""

from __future__ import annotations

from pydantic import BaseModel


class Paths(BaseModel):
    instances: str = "data/instances"
    processed: str = "data/processed"
    results: str = "results"


class RunConfig(BaseModel):
    project_name: str = "experiments-template"
    modeling: str = "pyomo"
    solver: str = "gurobi"
    paths: Paths = Paths()
