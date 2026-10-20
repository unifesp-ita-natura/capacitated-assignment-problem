"""Expose the block-assignment Pyomo-based exact MIP solve entrypoint."""

from __future__ import annotations

from src.solver.mip.block_assignment import run_block_assignment_mip

__all__ = ["run_block_assignment_mip"]
