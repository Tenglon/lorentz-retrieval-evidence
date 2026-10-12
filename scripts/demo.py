#!/usr/bin/env python3
"""Tiny no-dependency demonstration of the terminal H10 geometry."""

from lorentz_retrieval.geometry import direct_lift, hybrid_score, lorentz_inner


def main() -> None:
    query = (0.7, -0.2, 0.4)
    positive = (0.6, -0.1, 0.5)
    negative = (-0.4, 0.8, -0.1)
    curvature = 20.0
    point = direct_lift(query, curvature)
    print("Lorentz constraint:", lorentz_inner(point, point))
    print("Expected constraint:", -1.0 / curvature)
    print("positive score:", hybrid_score(query, positive, curvature=curvature))
    print("negative score:", hybrid_score(query, negative, curvature=curvature))


if __name__ == "__main__":
    main()
