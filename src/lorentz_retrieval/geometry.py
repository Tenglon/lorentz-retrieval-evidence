"""Dependency-free reference equations for the final H10 retrieval head."""

from __future__ import annotations

import math
from typing import Sequence


Vector = Sequence[float]


def _dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimension")
    return math.fsum(a * b for a, b in zip(left, right))


def _norm(vector: Vector) -> float:
    return math.sqrt(max(0.0, _dot(vector, vector)))


def direct_lift(spatial: Vector, curvature: float) -> tuple[float, ...]:
    """Lift ``z`` to the curvature ``-c`` Lorentz hyperboloid.

    The result satisfies ``<x,x>_L = -1/c`` without a radius predictor or cap.
    """

    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature must be positive and finite")
    squared_norm = _dot(spatial, spatial)
    return (math.sqrt(1.0 / curvature + squared_norm), *map(float, spatial))


def lorentz_inner(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("ambient vectors must have equal dimension >= 2")
    return -left[0] * right[0] + _dot(left[1:], right[1:])


def lorentz_distance(left: Vector, right: Vector, curvature: float) -> float:
    """Geodesic distance on the curvature ``-c`` hyperboloid."""

    if curvature <= 0:
        raise ValueError("curvature must be positive")
    argument = max(1.0, -curvature * lorentz_inner(left, right))
    return math.acosh(argument) / math.sqrt(curvature)


def cosine(left: Vector, right: Vector) -> float:
    denominator = _norm(left) * _norm(right)
    return 0.0 if denominator == 0.0 else _dot(left, right) / denominator


def hybrid_score(
    left_spatial: Vector,
    right_spatial: Vector,
    *,
    curvature: float = 20.0,
    cosine_weight: float = 0.2,
) -> float:
    """H10 score: Lorentz distance plus cosine computed *before* the lift."""

    if not 0.0 <= cosine_weight <= 1.0:
        raise ValueError("cosine_weight must lie in [0, 1]")
    left = direct_lift(left_spatial, curvature)
    right = direct_lift(right_spatial, curvature)
    distance_stream = -math.sqrt(curvature) * lorentz_distance(
        left, right, curvature
    )
    return (1.0 - cosine_weight) * distance_stream + cosine_weight * cosine(
        left_spatial, right_spatial
    )
