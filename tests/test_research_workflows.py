import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from fgm_asm import MeshInfo
from fgm_asm.research import solve_forward
from lib.research_workflows import run_experiment


class ResearchWorkflowTests(unittest.TestCase):
    def _inputs(self, folder):
        folder = Path(folder)
        mesh = MeshInfo(70., 20., 7, 2)
        settings = {}
        for name, force, setting, stages in (('180N', 100., 180, (1, 2)),
                                              ('280N', 200., 280, (3, 4))):
            target = folder / name
            target.mkdir()
            u, _ = solve_forward(mesh, force, 3.2, .3, np.full(mesh.n_nod, 1700.))
            uv = u.reshape(-1, 2) + [.04, -.03]
            for stage in stages:
                rows = ['Type:,X,Y,DX,DY,D,EXX,EYY,EXY,']
                for index, (point, displacement) in enumerate(zip(mesh.coord, uv)):
                    # DIC axes swapped relative to FEM, including components.
                    x, y = (point + displacement)[::-1]
                    dx, dy = displacement[::-1]
                    row, col = divmod(index, mesh.nods_x)
                    rows.append(f',{x},{y},{dx},{dy},0,0,0,0,{row},{col}')
                (target / f'stage_{stage:05d}.comdat_comdata.csv').write_text(
                    '\n'.join(rows), encoding='gb18030',
                )
            settings[name] = {'folder': target, 'stages': stages, 'setting_N': setting}
        calibration = folder / 'calibration.csv'
        pd.DataFrame({'infill_percent': range(40, 101, 10), 'batch': [1016]*7,
                      'n': [5]*7, 'mean_E_MPa': [1700.]*7, 'sd_E_MPa': [30.]*7,
                      'fit_method': ['independent']*7}).to_csv(calibration, index=False)
        forces = folder / 'force.csv'
        pd.DataFrame({'run': [3, 3], 'setting_N': [180, 280], 'stage_first': [1, 3],
                      'stage_last': [2, 4], 'mean_corrected_force_N': [100., 200.],
                      'full_force_coverage': [True, True]}).to_csv(forces, index=False)
        return SimpleNamespace(
            PLATFORMS=settings, FORCE_PAIRING_CSV=forces, CALIBRATION_CSV=calibration,
            CALIBRATION_METHOD='independent', GEO_L=70., GEO_H=20., THICKNESS=3.2, NU=.3,
            NEL_X=7, NEL_Y=2, FORWARD_NEL_X=70, FORWARD_NEL_Y=4,
            DIC_COORDINATES_VERIFIED=True, DIC_TO_FEM_MATRIX=[[0., 1.], [1., 0.]],
            DIC_TO_FEM_OFFSET=[0., 0.], NOISE_FLOOR_MM=.001,
            REGION_INFILL_PERCENT=(50, 60, 70, 80, 90),
            REGION_BOUNDARIES_X=(11., 27., 43., 59.),
            # Strong affine regularization is appropriate for this exact uniform
            # limiting case, whose true regularization value is zero. This is
            # an integration check, not an experimental gamma-selection rule.
            E_REF=1500., REGULARIZATION_MODE='affine_detrended', GAMMA=1e6,
            FIT_CASES=('180N',), PREDICT_CASES=('280N',),
            MAX_ITER=500, MAX_FUN=2000, FTOL=1e-12, GTOL=1e-8, TGV_OPTIONS={},
        )

    def test_calibrated_forward_and_dic_inverse_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._inputs(folder)
            forward_output = Path(folder) / 'forward'
            forward = run_experiment(cfg, forward_output, mode='forward')
            self.assertLess(forward['calibrated_forward_comparison'][0]['weighted_rms'], 1e-10)
            output = Path(folder) / 'inverse'
            result = run_experiment(cfg, output, mode='inverse')
            self.assertTrue(result['solver']['converged'])
            self.assertEqual(result['inverse_parameter_count'], 8*3)
            self.assertEqual(result['physical_roi']['region_lengths_mm'], [11., 16., 16., 16., 11.])
            self.assertEqual(result['fit_cases'], ('180N',))
            self.assertLess(result['inverse_comparison'][1]['weighted_rms'], 1e-7)
            table = pd.read_csv(output / 'regional_modulus_comparison.csv')
            self.assertEqual(table['infill_percent'].tolist(), [50, 60, 70, 80, 90])
            np.testing.assert_allclose(table['inverse_mean_E_MPa'], 1700., rtol=1e-4)
            self.assertTrue((output / '180N_calibrated_displacement.svg').exists())
            self.assertTrue((output / '280N_inverse_displacement.png').exists())
            self.assertTrue((output / 'x_profile.csv').exists())
            saved = json.loads((output / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertIn('source_sha256', saved)
            with self.assertRaises(FileExistsError):
                run_experiment(cfg, output, mode='inverse')

    def test_unverified_coordinates_stop_quantitative_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._inputs(folder)
            cfg.DIC_COORDINATES_VERIFIED = False
            with self.assertRaisesRegex(ValueError, 'unverified'):
                run_experiment(cfg, Path(folder) / 'unverified', mode='forward')

    def test_inspect_saves_registered_vectors_and_region_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._inputs(folder)
            output = Path(folder) / 'inspect'
            with patch('lib.research_workflows.plot_dic_registration'):
                summary = run_experiment(cfg, output, mode='inspect')
            report = summary['dic_registration']['180N']
            self.assertEqual(report['status'], 'verified')
            self.assertEqual(report['points_inside_roi'], 24)
            self.assertEqual(sum(report['points_per_region']), 24)
            with np.load(output / 'fields.npz') as fields:
                source = fields['180N_displacement_mean']
                np.testing.assert_allclose(fields['180N_fem_displacement_mm'], source[:, ::-1])
                np.testing.assert_allclose(np.ptp(fields['180N_fem_coordinates_mm'], axis=0),
                                           [70., 20.])

    def test_unconverged_run_preserves_results_with_explicit_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self._inputs(folder)
            cfg.MAX_ITER = 1
            output = Path(folder) / 'limited_budget'
            with patch('lib.research_workflows.plot_displacement_comparison'), \
                    patch('lib.research_workflows.plot_modulus_comparison'):
                result = run_experiment(cfg, output, mode='inverse')
            self.assertFalse(result['solver']['converged'])
            self.assertEqual(result['status'], 'not_converged')
            saved = json.loads((output / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'not_converged')
            self.assertTrue((output / 'fields.npz').exists())


if __name__ == '__main__':
    unittest.main()
