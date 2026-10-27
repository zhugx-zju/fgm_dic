"""Independent analytical check of the scalar field used in same-ROI comparisons."""
import unittest

import numpy as np

from fgm_asm.mesh import MeshInfo
from research.material_window_analysis import common_quadrature, scalar_values


class MaterialWindowAnalysisTests(unittest.TestCase):
    def test_common_partition_integrates_shifted_meshes_and_regions_exactly(self):
        mesh = MeshInfo(4., 2., 5, 3)
        old_mesh = MeshInfo(8., 4., 7, 4)
        points, weights = common_quadrature(mesh, old_mesh, np.array([1., .5]), [1.3])
        self.assertAlmostEqual(weights.sum(), 8.)
        left = points[:, 0] < 1.3
        self.assertAlmostEqual(weights[left].sum(), 2.6)
        # Independent continuous moments for E=x+2y on [0,4]×[0,2].
        field = old_mesh.coord[:, 0]-1 + 2*(old_mesh.coord[:, 1]-.5)
        values = scalar_values(old_mesh, field, points + [1., .5])
        mean = np.average(values, weights=weights)
        self.assertAlmostEqual(mean, 4.)
        self.assertAlmostEqual(np.average((values-mean)**2, weights=weights), 8/3)

    def test_bilinear_modulus_at_interiors_and_domain_corners(self):
        mesh = MeshInfo(4., 2., 4, 2)
        points = np.array([[0., 0.], [.3, .7], [3.7, 1.9], [4., 2.]])

        def modulus(x, y):
            return 1000 + 20*x - 30*y + 7*x*y

        field = modulus(mesh.coord[:, 0], mesh.coord[:, 1])
        np.testing.assert_allclose(scalar_values(mesh, field, points),
                                   modulus(points[:, 0], points[:, 1]),
                                   rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
