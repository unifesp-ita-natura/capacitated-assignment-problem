"""Matheurística Relax-and-Fix + Fix-and-Optimize pro MIP de atribuição, por janelas de CDs."""

from __future__ import annotations

from src.solver.matheuristics.relax_and_fix import (
    RelaxAndFixParams,
    RelaxAndFixResult,
    run_relax_and_fix,
    solve,
)

__all__ = ["RelaxAndFixParams", "RelaxAndFixResult", "run_relax_and_fix", "solve"]
