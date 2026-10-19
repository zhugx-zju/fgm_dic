import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fgm_asm import MeshInfo
from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.load_case import evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import setup_boundary_conditions
from fgm_asm.research import make_observation_case, solve_forward
from lib.dic_case_workflow import (compare_dic_results, forward_calibrated, inverse_dic,
                                    prepare_dic, registration_transform)


class DICCaseWorkflowTests(unittest.TestCase):
    def test_landmarks_preserve_geometry_and_reject_span_fitting(self):
        raw = np.array([[0., 0.], [0., -70.], [20., 0.], [10., -35.]])
        matrix = np.array([[0., -1.], [1., 0.]])
        target = raw @ matrix.T+[3., 2.]
        spec = {'mode': 'landmarks', 'dic_points_mm': raw, 'fem_points_mm': target,
                'tolerance_mm': 1e-8, 'note': 'known physical control points'}
        a, b, verified = registration_transform(raw, spec)
        self.assertTrue(verified)
        np.testing.assert_allclose(a, matrix, atol=1e-14)
        np.testing.assert_allclose(b, [3., 2.], atol=1e-14)
        spec['fem_points_mm'] = target*np.array([.9, 1.])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            registration_transform(raw, spec)

    def test_alternate_fe_gauge_and_dic_rigid_reference_do_not_change_residual(self):
        mesh = MeshInfo(4., 2., 4, 2)
        bc = setup_boundary_conditions(mesh, 4., 2., 100., 3.2)
        alternate = dict(bc, fixdof=np.array([2*10, 2*10+1, 2*14+1]))
        material = MaterialInfo(nu=.26)
        material.update(1500.+100*mesh.X+30*mesh.Y)
        first = forward_solver(fem_assemble(mesh, material, bc))
        second = forward_solver(fem_assemble(mesh, material, alternate))
        measured = first.reshape(-1, 2)+[.08, -.04]+.003*np.column_stack([-mesh.Y, mesh.X])
        case = make_observation_case(mesh, 'gauge', 100., 3.2, mesh.coord, measured)
        for u in (first, second):
            state = evaluate_observation(case, u)
            self.assertLess(state['weighted_rms'], 1e-12)
        # Symmetric strain from the difference must vanish (rigid motion only).
        design = np.column_stack([np.ones(mesh.n_nod), mesh.coord])
        gradient = np.linalg.lstsq(design, (first-second).reshape(-1, 2), rcond=None)[0][1:].T
        np.testing.assert_allclose(gradient+gradient.T, 0., atol=1e-12)

    def _fixture(self, root):
        root = Path(root)
        mesh = MeshInfo(4., 2., 4, 2)
        x, y = np.meshgrid(np.arange(.25, 4., .5), np.arange(.25, 2., .5))
        xy = np.column_stack([x.ravel(), y.ravel()])
        platforms = {}
        for name, force, stages in [('fit', 100., (1, 2)), ('test', 200., (3, 4))]:
            folder = root/name
            folder.mkdir()
            u, _ = solve_forward(mesh, force, 3.2, .26, np.full(mesh.n_nod, 1700.))
            case = make_observation_case(mesh, name, force, 3.2, xy, np.zeros_like(xy))
            uv = (case.observation.H @ u).reshape(-1, 2)+[.08, -.03]+.002*np.column_stack([-xy[:, 1], xy[:, 0]])
            for stage in stages:
                lines = ['Type:,X,Y,DX,DY,D,EXX,EYY,EXY,']
                for i, (point, displacement) in enumerate(zip(xy, uv)):
                    row, col = divmod(i, 8)
                    lines.append(f',{point[0]+displacement[0]},{point[1]+displacement[1]},'
                                 f'{displacement[0]},{displacement[1]},0,0,0,0,{row},{col}')
                (folder/f'stage_{stage:05d}.comdat_comdata.csv').write_text('\n'.join(lines), encoding='gb18030')
            platforms[name] = {'folder': folder, 'stages': stages, 'force_N': force,
                               'force_note': 'controlled unloaded-reference actual force'}
        calibration = root/'calibration.csv'
        pd.DataFrame({'infill_percent': [70], 'batch': [1016], 'n': [5], 'mean_E_MPa': [1700.],
                      'sd_E_MPa': [20.], 'fit_method': ['independent']}).to_csv(calibration, index=False)
        return SimpleNamespace(
            CASE_ID='controlled', PLATFORMS=platforms, CALIBRATION_CSV=calibration,
            CALIBRATION_METHOD='independent', REGION_INFILL_PERCENT=(70,), REGION_BOUNDARIES_X=(),
            GEO_L=4., GEO_H=2., THICKNESS=3.2, NEL_X=4, NEL_Y=2,
            DIC_UNIT='mm', DIC_COORDINATE_STATE='deformed', DISPLACEMENT_REFERENCE='initial_unloaded_image',
            FORCE_REFERENCE='corrected_force_from_unloaded_state',
            REGISTRATION={'mode': 'explicit_verified', 'matrix': np.eye(2), 'offset_mm': [0., 0.],
                          'note': 'controlled analytical geometry'}, OUTSIDE_POINT_POLICY='reject',
            NU_VALUES=(.26,), PRIMARY_NU=.26, FIT_CASES=('fit',), PREDICT_CASES=('test',),
            E_REF=1500., NOISE_SCALE_MM=.0003, GAMMA_CANDIDATES=(1e6,),
            REGULARIZATION_MODE='affine_detrended', TGV_OPTIONS={}, MAX_ITER=500, MAX_FUN=2500,
            FTOL=1e-12, GTOL=1e-7, VALIDATION_BLOCK_MM=1., VALIDATION_BUFFER_MM=.1)

    def test_independent_stages_shared_mesh_and_partial_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._fixture(folder)
            destination = Path(folder)/'result'
            prepared, forward, inverse = (destination/'01_prepared', destination/'02_forward', destination/'03_inverse')
            prepare_dic(cfg, prepared)
            forward_calibrated(prepared, forward)
            inverse_dic(prepared, inverse)
            destination = destination/'04_comparison'
            # Comparing saved results must never restart mechanics or optimization.
            with patch('lib.dic_case_workflow.solve_forward', side_effect=AssertionError('no FE solve')):
                result = compare_dic_results(prepared, forward, inverse, destination, plots=False)
            self.assertEqual(result['status'], 'computed')
            self.assertEqual(result['physical_model_mm'][:2], [4., 2.])
            self.assertAlmostEqual(result['coverage'][0]['x_max_mm']-result['coverage'][0]['x_min_mm'], 3.5)
            self.assertTrue(result['shared_mesh']['same_for_forward_and_inverse'])
            table = pd.read_csv(destination/'comparison.csv')
            self.assertLess(table.rms_mm.max(), 1e-7)
            with np.load(destination/'nu_0.26'/'fields.npz') as fields:
                self.assertEqual(len(fields['E_inverse']), 15)
                np.testing.assert_allclose(fields['E_inverse'], 1700., rtol=1e-4)
            self.assertTrue((destination/'打开查看结果.html').exists())
            with self.assertRaises(FileExistsError):
                compare_dic_results(prepared, forward, inverse, destination, plots=False)
            # Refuse forward/inverse artifacts from different preparations.
            path = inverse/'run_metadata.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            record['prepared_metadata_sha256'] = 'different_case'
            path.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SAME'):
                compare_dic_results(prepared, forward, inverse, Path(folder)/'mixed', plots=False)

    def test_pending_registration_audits_but_never_starts_inverse(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._fixture(folder)
            cfg.REGISTRATION = {'mode': 'pending'}
            with patch('lib.dic_case_workflow._optimize', side_effect=AssertionError('must not optimize')):
                result = prepare_dic(cfg, Path(folder)/'pending')
            self.assertEqual(result['status'], 'needs_registration')
            self.assertFalse(result['coordinates_verified'])
            with self.assertRaisesRegex(ValueError, 'pending'):
                forward_calibrated(Path(folder)/'pending', Path(folder)/'forward')


if __name__ == '__main__':
    unittest.main()
