"""Preserve nonnegative hierarchy totals through proportional distribution or sparse WLS."""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix


def distribute(
    query: pd.DataFrame, root_values: np.ndarray, shares: dict[str, np.ndarray], levels: list[str]
) -> pd.Series:
    """Propagate normalized child shares from the configured starting level down to cities."""
    root = query[query["level"].eq(levels[0])]
    values = pd.Series(root_values, index=root["node"])
    for level in levels[1:]:
        children = query[query["level"].eq(level)]
        predicted = children["parent"].map(values).to_numpy() * shares[level]
        values = pd.concat([values, pd.Series(predicted, index=children["node"])])
    return values


def _path(node: str, parents: dict[str, str]) -> list[str]:
    result = []
    while node:
        if node in result:
            raise ValueError("hierarchy contains a cycle")
        result.append(node)
        node = parents[node]
    return result


def summing_matrix(query: pd.DataFrame, leaf_level: str) -> csr_matrix:
    """Map leaf values to every ancestor with a sparse matrix, never a dense city covariance."""
    positions = dict(zip(query["node"], range(len(query))))
    parents = dict(zip(query["node"], query["parent"]))
    leaves = query.loc[query["level"].eq(leaf_level), "node"]
    entries = [
        (positions[parent], column)
        for column, node in enumerate(leaves)
        for parent in _path(node, parents)
    ]
    rows, columns = zip(*entries, strict=True)
    return csr_matrix((np.ones(len(entries)), (rows, columns)), shape=(len(query), len(leaves)))


def reconcile(
    history: pd.DataFrame, query: pd.DataFrame, base: pd.Series, leaf_level: str, max_iter: int
) -> tuple[pd.Series, dict]:
    """Minimize scaled squared deviations jointly at all levels with nonnegative leaf values."""
    matrix = summing_matrix(query, leaf_level)
    deviations = history.groupby("node")["items"].std().fillna(1).clip(lower=1)
    scale = query["node"].map(deviations).fillna(1).to_numpy()
    leaf_scale = scale[query["level"].eq(leaf_level).to_numpy()]
    target = base.reindex(query["node"]).to_numpy()
    result = lsq_linear(
        matrix.multiply(leaf_scale).multiply((1 / scale)[:, None]).tocsr(),
        target / scale,
        bounds=(0, np.inf),
        tol=1e-6,
        lsmr_tol=1e-8,
        max_iter=max_iter,
        lsmr_maxiter=300,
    )
    if not result.success:
        raise RuntimeError(f"hierarchy WLS failed to converge: {result.message}")
    coherent = pd.Series(matrix @ (result.x * leaf_scale), index=query["node"])
    return coherent, {"method": "nonnegative_wls", "iterations": result.nit, "cost": result.cost}


def assert_coherent(query: pd.DataFrame, predictions: pd.Series, leaf_level: str) -> None:
    """Check every non-leaf against the sum of its immediate children."""
    frame = query.assign(predicted=query["node"].map(predictions))
    child_sums = frame.groupby("parent")["predicted"].sum()
    parents = frame[frame["level"].ne(leaf_level)]
    if not np.allclose(parents["predicted"], parents["node"].map(child_sums), atol=1e-6):
        raise ValueError("hierarchy totals do not conserve child predictions")
    if not np.isfinite(frame["predicted"]).all() or frame["predicted"].lt(-1e-9).any():
        raise ValueError("hierarchy predictions must be finite and nonnegative")
