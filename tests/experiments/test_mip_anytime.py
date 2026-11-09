"""Tests for the Gurobi log parser and checkpoint table behind the mip_anytime experiment."""

import pytest

from experiments.mip_anytime.run import checkpoint_table
from src.solver.mip.gurobi_trace import TracePoint, parse_trace, state_at

LOG = """\
Variable types: 0 continuous, 8520 integer (8520 binary)

    Nodes    |    Current Node    |     Objective Bounds      |  Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 1189.84415    0   38 4294.72816 1189.84415  72.3%     -    0s
H    0     0                    2683.0630695 1189.84415  55.7%     -    1s
     0     0 1191.99823    0   35 1378.67449 1191.99823  13.5%  12.5   12s
     0     2 1191.99823    0   35 1378.67449 1191.99823  13.5%  12.5   15s
"""


def test_parse_trace_reads_incumbent_bound_and_time():
    assert parse_trace(LOG)[:2] == [
        TracePoint(0.0, 4294.72816, 1189.84415),
        TracePoint(1.0, 2683.0630695, 1189.84415),
    ]


def test_state_at_is_last_row_not_after_the_time():
    trace = parse_trace(LOG)
    assert state_at(trace, 12).incumbent == 1378.67449
    assert state_at(trace, 5).incumbent == 2683.0630695
    assert state_at([], 5) is None


def test_checkpoint_table_gap_to_bound():
    table = checkpoint_table(LOG)
    assert table[10]["incumbent"] == 2683.0630695
    assert table[30]["gap_to_bound"] == pytest.approx((1378.67449 - 1191.99823) / 1191.99823)


def test_highs_runners_solve_a_tiny_instance():
    from experiments.mip_anytime.run_highs import _matheuristic, _mip
    from tests.solver.heuristics.test_simulated_annealing import _instance

    instance = _instance()
    mip = _mip(instance, None, 30.0)
    assert mip["objective"] is not None
    assert _matheuristic(instance, None, 30.0)["objective"] == pytest.approx(mip["objective"])
