import unittest

import numpy as np

from archive.legacy_workflows import config
from fgm_asm.material import generate_fgm_modulus
from fgm_asm.mesh import MeshInfo
from fgm_asm.utils import summarize_x_only_reconstruction


class XOnlySyntheticTruthTests(unittest.TestCase):
    def test_default_forward_truth_has_no_y_dependence(self):
        settings = config.get_forward_config()
        self.assertEqual(settings.dis_type, "bil")
        mesh = MeshInfo(settings.geo_l, settings.geo_h, 4, 3)
        field, _ = generate_fgm_modulus(
            mesh, dis_type=settings.dis_type, Ex=settings.Ex, Ey=settings.Ey
        )
        np.testing.assert_allclose(field, np.broadcast_to(field[0], field.shape))
        np.testing.assert_allclose(field[0, [0, -1]], [1.0, settings.Ex])

    def test_profile_error_and_transverse_spread_are_separate(self):
        mesh = MeshInfo(4.0, 2.0, 4, 2)
        true_profile = np.linspace(1.0, 2.0, mesh.nods_x)
        truth = np.broadcast_to(true_profile, (mesh.nods_y, mesh.nods_x)).copy()
        reconstructed = truth + np.array([-0.1, 0.0, 0.1])[:, None]

        summary = summarize_x_only_reconstruction(mesh, truth, reconstructed.ravel())
        np.testing.assert_allclose(summary["reconstructed_mean"], true_profile)
        self.assertAlmostEqual(summary["profile_mae_pct"], 0.0)
        self.assertGreater(summary["transverse_rms_pct"], 0.0)
        self.assertAlmostEqual(summary["transverse_max_range_pct"], 20.0)

    def test_y_dependent_truth_is_rejected(self):
        mesh = MeshInfo(4.0, 2.0, 4, 2)
        truth = np.ones((mesh.nods_y, mesh.nods_x))
        truth[-1, -1] = 1.1
        with self.assertRaisesRegex(ValueError, "depend on x only"):
            summarize_x_only_reconstruction(mesh, truth, truth)


if __name__ == "__main__":
    unittest.main()
