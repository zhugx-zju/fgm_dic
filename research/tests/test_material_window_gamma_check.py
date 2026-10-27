"""Analytical checks of repeat-profile comparison on a common physical interval."""
import unittest

import numpy as np

from research.material_window_gamma_check import profile_distance


class ProfileDistanceTests(unittest.TestCase):
    def test_constant_profiles_use_symmetric_normalization(self):
        difference, bounds = profile_distance(np.array([0., 2.]), np.array([110., 110.]),
                                              np.array([1., 3.]), np.array([100., 100.]))
        self.assertEqual(bounds, (1., 2.))
        self.assertAlmostEqual(difference, 100*10/105)

    def test_same_affine_profile_is_invariant_to_knots_and_interval(self):
        first_x = np.array([-2., -.1, 3.])
        second_x = np.array([-1., 0., .4, 2.])
        difference, bounds = profile_distance(first_x, 1000+30*first_x,
                                              second_x, 1000+30*second_x)
        self.assertEqual(bounds, (-1., 2.))
        self.assertLess(difference, 1e-12)


if __name__ == '__main__':
    unittest.main()
