import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from fgm_asm import MeshInfo
from fgm_asm.research import compare_displacement, solve_forward
from archive.legacy_workflows.lib.dic_revision_study import assumed_registration, block_split, make_cases
from lib.experiment_data import DICPlatform


class RevisedDICStudyTests(unittest.TestCase):
    def test_assumed_transform_preserves_lengths_and_does_not_expand_cloud(self):
        xy = np.array([[-9., -31.], [9., 31.], [-9., 31.], [9., -31.]])
        for direction in (-1, 1):
            a, b = assumed_registration(xy, 70., 20., direction)
            mapped = xy @ a.T + b
            np.testing.assert_allclose(a @ a.T, np.eye(2))
            np.testing.assert_allclose(np.ptp(mapped, axis=0), [62., 18.])
            np.testing.assert_allclose(mapped.mean(axis=0), [35., 10.])

    def test_spatial_validation_has_guard_band(self):
        x, y = np.meshgrid(np.arange(4., 67., .5), np.arange(1., 19., .5))
        xy = np.column_stack([x.ravel(), y.ravel()])
        train, validation = block_split(xy, 6., 1.)
        self.assertFalse(np.any(train & validation))
        self.assertGreater(cKDTree(xy[validation]).query(xy[train])[0].min(), 1.)
        self.assertGreater(np.sum(~(train | validation)), 0)

    def test_conditional_cases_keep_physical_force_and_vector_transform(self):
        mesh = MeshInfo(70., 20., 7, 2)
        ids = np.column_stack([np.repeat(np.arange(3), 8), np.tile(np.arange(8), 3)])
        raw_xy = mesh.coord[:, ::-1] - [10., 35.]
        platforms, forces = {}, {}
        for name, force in [('180N', 120.), ('280N', 220.)]:
            u, _ = solve_forward(mesh, force, 3.2, .3, np.full(mesh.n_nod, 1700.))
            raw_u = u.reshape(-1, 2)[:, ::-1] + [.03, -.07]
            platforms[name] = DICPlatform(raw_xy, raw_u, np.zeros_like(raw_u), ids, {})
            forces[name] = {'mean_corrected_force_N': force}
        cfg = SimpleNamespace(GEO_L=70., GEO_H=20., THICKNESS=3.2, WORKING_SIGMA_MM=.0003,
                              REGISTRATION_STATUS='assumed', REGISTRATION_ASSUMPTION='test only',
                              DIC_COORDINATES_VERIFIED=False)
        a, b = assumed_registration(raw_xy, 70., 20.)
        cases = make_cases(cfg, mesh, platforms, forces, a, b)
        states = compare_displacement(mesh, np.full(mesh.n_nod, 1700.), cases, .3)
        for case, state in zip(cases, states):
            self.assertLess(state['weighted_rms'], 1e-12)
            self.assertFalse(case.metadata['coordinates_verified'])
            self.assertEqual(case.bc_info['tensile_end_force'], forces[case.name]['mean_corrected_force_N'])
        self.assertFalse(cfg.DIC_COORDINATES_VERIFIED)

    def test_reflected_free_field_cannot_establish_material_direction(self):
        mesh = MeshInfo(70., 20., 7, 2)
        modulus = 1400. + 10. * mesh.X + 2. * mesh.Y
        displacement, _ = solve_forward(mesh, 120., 3.2, .3, modulus)
        raw_xy = mesh.coord[:, ::-1] - [10., 35.]
        ids = np.column_stack([np.repeat(np.arange(3), 8), np.tile(np.arange(8), 3)])
        platform = DICPlatform(raw_xy, displacement.reshape(-1, 2)[:, ::-1],
                               np.zeros_like(raw_xy), ids, {})
        cfg = SimpleNamespace(GEO_L=70., GEO_H=20., THICKNESS=3.2, WORKING_SIGMA_MM=.0003,
                              REGISTRATION_STATUS='assumed', REGISTRATION_ASSUMPTION='test only')
        a, b = assumed_registration(raw_xy, 70., 20., direction=-1)
        cases = make_cases(cfg, mesh, {'180N': platform},
                           {'180N': {'mean_corrected_force_N': 120.}}, a, b)
        reflected = modulus.reshape(mesh.nods_y, mesh.nods_x)[:, ::-1].ravel()
        state = compare_displacement(mesh, reflected, cases, .3)[0]
        self.assertLess(state['weighted_rms'], 1e-12)


if __name__ == '__main__':
    unittest.main()
