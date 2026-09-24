"""Federated aggregation primitives (Federated Averaging).

``aggregate_weighted`` implements weighted FedAvg where each client's update is
weighted by the size of its local dataset (the research paper's weighted
aggregation). ``aggregate_uniform`` is used when secure aggregation is enabled,
because pairwise-masked updates only cancel under equal weights.
"""

from __future__ import annotations

import numpy as np

NDArrayLike = list[np.ndarray]


def aggregate_weighted(
    updates: list[tuple[NDArrayLike, int]], weights: list[int] | None = None
) -> NDArrayLike:
    """Weighted Federated Averaging (FedAvg).

    Parameters
    ----------
    updates:
        ``(parameters, n_examples)`` pairs from the participating clients.
    weights:
        Optional per-client weights; defaults to ``n_examples``.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
    if weights is None:
        weights = [n for _, n in updates]
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total aggregation weight must be positive.")

    first = updates[0][0]
    num_layers = len(first)
    aggregated: list[np.ndarray] = []
    for i in range(num_layers):
        acc = None
        for (params, _), w in zip(updates, weights):
            contrib = params[i].astype(np.float64) * w
            acc = contrib if acc is None else acc + contrib
        aggregated.append((acc / total).astype(np.float32))
    return aggregated


def aggregate_uniform(updates: list[NDArrayLike]) -> NDArrayLike:
    """Unweighted (uniform) averaging of parameter lists."""
    n = len(updates)
    if n == 0:
        raise ValueError("Cannot aggregate an empty list of updates.")
    return aggregate_weighted([(params, 1) for params in updates], weights=[1] * n)


def parameters_size_bytes(parameters) -> int:
    """Size in bytes of a Flower ``Parameters`` object (measured, not estimated)."""
    return sum(len(t) for t in parameters.tensors)


def count_parameters(ndarrays: NDArrayLike) -> int:
    return int(sum(a.size for a in ndarrays))