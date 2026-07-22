"""Accuracy and complexity measures, kept deliberately separate from recovery.

The separation is the point. A report that merges "fits well" and "found the right form" into one
number cannot express the finding that motivates this package, and every convenience function here
is named so that merging them requires a deliberate act.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The threshold the benchmark convention uses for an accuracy solution.
ACCURACY_R2_THRESHOLD = 0.999


def coefficient_of_determination(y: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R-squared on the finite predictions, or None when nothing is finite."""
    finite = np.isfinite(y_pred)
    if not finite.any():
        return None
    residual = y[finite] - y_pred[finite]
    variance = float(np.var(y[finite]))
    if variance <= 0:
        return None
    return float(1.0 - float(np.mean(residual * residual)) / variance)


def normalised_mse(y: np.ndarray, y_pred: np.ndarray) -> float | None:
    finite = np.isfinite(y_pred)
    if not finite.any():
        return None
    variance = float(np.var(y[finite]))
    if variance <= 0:
        return None
    residual = y[finite] - y_pred[finite]
    return float(np.mean(residual * residual) / variance)


def accuracy_solution(y: np.ndarray, y_pred: np.ndarray, *, threshold: float = ACCURACY_R2_THRESHOLD) -> bool:
    """Whether this counts as an ACCURACY solution.

    Named at length on purpose. An accuracy solution is not a recovery, and the gap between the two
    rates is the measurement this package was built to expose.
    """
    r2 = coefficient_of_determination(y, y_pred)
    return bool(r2 is not None and r2 >= threshold)


@dataclass(frozen=True)
class DescriptionLength:
    total: float
    structure: float
    constants: float
    residuals: float


def description_length(
    *,
    n_nodes: int,
    n_constants: int,
    y: np.ndarray,
    y_pred: np.ndarray,
    n_primitives: int,
    n_variables: int,
    precision_nats: float = math.log(2.0) * 16.0,
) -> DescriptionLength:
    """Description length in nats: the cost of the model plus the cost of the data given it.

    This is the selection rule this package recommends over best-accuracy. Selecting the most
    accurate member of a Pareto front reproduces the field's headline failure, because the most
    accurate member is routinely the most over-parameterised one.
    """
    alphabet = max(2, n_primitives + n_variables + 1)
    structure = n_nodes * math.log(alphabet)
    constants = n_constants * precision_nats
    finite = np.isfinite(y_pred)
    if not finite.any():
        residuals = float("inf")
    else:
        residual = y[finite] - y_pred[finite]
        sigma_squared = max(float(np.mean(residual * residual)), 1e-12)
        residuals = 0.5 * len(residual) * (math.log(2.0 * math.pi * sigma_squared) + 1.0)
    return DescriptionLength(
        total=structure + constants + residuals,
        structure=structure, constants=constants, residuals=residuals,
    )


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points of a (loss, complexity) set, both minimised.

    At equal complexity only the best loss survives, otherwise the front fills with structurally
    different expressions of identical size and stops being readable.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
    front: list[int] = []
    best = float("inf")
    for i in order:
        loss, _complexity = points[i]
        if loss < best:
            front.append(i)
            best = loss
    return front
