import unittest

import numpy as np

from lib.dic_rot90_followup import (existing_orientation_registration, map_frame_forces,
                                    residual_energy, rigid_projection)
from lib.experiment_data import DICPlatform
from fgm_asm import MeshInfo
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import make_observation_case


class Rot90FollowupTests(unittest.TestCase):
    def test_projection_matches_shared_observation_kernel(self):
        mesh = MeshInfo(3., 2., 3, 2)
        xy = mesh.coord
        rng = np.random.default_rng(91)
        measured = rng.normal(size=xy.shape)
        predicted = rng.normal(size=xy.shape)
        case = make_observation_case(mesh, 'check', 120., 3.2, xy, measured,
                                      np.ones(len(xy)), .0003)
        state = evaluate_observation(case, predicted.ravel())
        np.testing.assert_allclose(rigid_projection(xy, measured-predicted).ravel(),
                                   state['residual'], atol=1e-13)

    def test_rigid_projection_and_energy_identity_without_independence(self):
        xy = np.array([[0., 0.], [3., 0.], [0., 2.], [3., 2.]])
        rigid = [2., -3.] + .02*np.column_stack([-xy[:, 1], xy[:, 0]])
        np.testing.assert_allclose(rigid_projection(xy, rigid), 0., atol=1e-14)
        values = np.array([[[1., 2.], [3., 4.]], [[3., 4.], [5., 6.]]])
        result = residual_energy(values)
        self.assertLess(result['decomposition_roundoff_mm2'], 1e-14)
        self.assertAlmostEqual(result['within_window_rms_mm'], 1.)

    def test_frame_force_mapping_applies_time_offset_and_bias(self):
        machine = np.array([[0., 60.], [1., 160.], [2., 260.]])
        forces = map_frame_forces(machine, np.array([5, 10]), 5., .5, 60.)
        np.testing.assert_allclose(forces, [50., 150.])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            map_frame_forces(machine, np.array([20]), 5., 0., 60.)

    def test_rot90_uses_existing_orientation_and_preserves_vector_pairs(self):
        xx, yy = np.meshgrid(np.arange(3.), np.arange(4.))
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        # Export roundoff in a nominally structured grid must not corrupt matching.
        xy[:, 1] += np.tile([0., 1e-13, -1e-13], 4)
        uv = np.column_stack([np.arange(12.), 20.+np.arange(12.)])
        ids = np.column_stack([np.repeat(np.arange(4), 3), np.tile(np.arange(3), 4)])
        platform = DICPlatform(xy, uv, np.zeros_like(uv), ids, {})
        a, b = existing_orientation_registration(platform, 70., 20., ('rot90',))
        np.testing.assert_array_equal(a, [[0., -1.], [1., 0.]])
        transformed = xy @ a.T+b
        np.testing.assert_allclose((transformed.min(axis=0)+transformed.max(axis=0))/2, [35., 10.])


if __name__ == '__main__':
    unittest.main()
