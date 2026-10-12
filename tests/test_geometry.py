import math
import unittest

from lorentz_retrieval.geometry import (
    direct_lift,
    hybrid_score,
    lorentz_distance,
    lorentz_inner,
)


class GeometryTest(unittest.TestCase):
    def test_lift_is_on_hyperboloid(self):
        for curvature in (0.1, 1.0, 20.0):
            point = direct_lift((0.3, -1.2, 2.0), curvature)
            self.assertAlmostEqual(
                lorentz_inner(point, point), -1.0 / curvature, places=11
            )

    def test_distance_is_symmetric_and_zero_on_diagonal(self):
        left = direct_lift((0.2, 0.4), 20.0)
        right = direct_lift((-0.7, 0.1), 20.0)
        self.assertAlmostEqual(lorentz_distance(left, left, 20.0), 0.0)
        self.assertAlmostEqual(
            lorentz_distance(left, right, 20.0),
            lorentz_distance(right, left, 20.0),
        )

    def test_hybrid_prefers_identical_vector(self):
        query = (0.2, 0.8, -0.1)
        self.assertGreater(
            hybrid_score(query, query), hybrid_score(query, (-0.2, -0.8, 0.1))
        )

    def test_shell_identity_at_unit_curvature(self):
        radius, theta = 2.3, 0.17
        x = (math.cosh(radius), math.sinh(radius), 0.0)
        y = (
            math.cosh(radius),
            math.sinh(radius) * math.cos(theta),
            math.sinh(radius) * math.sin(theta),
        )
        distance = lorentz_distance(x, y, 1.0)
        expected = 1 + math.sinh(radius) ** 2 * (1 - math.cos(theta))
        self.assertAlmostEqual(math.cosh(distance), expected, places=11)


if __name__ == "__main__":
    unittest.main()
