import unittest
import numpy as np
from fgm_asm import MeshInfo
from lib.uniform_sample_analysis import area_statistics


class AreaStatisticsTests(unittest.TestCase):
    def test_constant_field_has_zero_spatial_variability(self):
        mesh = MeshInfo(12., 4., 6, 2)
        result = area_statistics(mesh, np.full(mesh.n_nod, 1400.))
        self.assertAlmostEqual(result['mean_E_MPa'], 1400.)
        self.assertLess(result['spatial_CV_percent'], 1e-12)

    def test_affine_field_matches_exact_continuous_moments(self):
        mesh = MeshInfo(12., 4., 6, 2)
        result = area_statistics(mesh, 2.+mesh.X+2.*mesh.Y)
        self.assertAlmostEqual(result['mean_E_MPa'], 12.)
        self.assertAlmostEqual(result['spatial_sd_MPa'], np.sqrt(12.**2/12+4.**2/3))


if __name__ == '__main__':
    unittest.main()
