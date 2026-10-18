import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lib.experiment_data import (
    check_platform_reference, dic_registration_report, dic_roi_mask, load_calibration, load_dic_platform,
    transform_dic_platform,
)
from lib.io_utils import read_dic_csv


class ExperimentDataTests(unittest.TestCase):
    def test_calibration_selects_observed_five_regions_in_physical_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.csv'
            pd.DataFrame({'infill_percent': [100, 90, 80, 70, 60, 50, 40],
                          'batch': [1016]*7, 'n': [5]*7,
                          'mean_E_MPa': [2400, 2120, 1840, 1710, 1640, 1400, 1230],
                          'sd_E_MPa': [30.]*7,
                          'fit_method': ['independent']*7}).to_csv(path, index=False)
            selected = load_calibration(path, infill_percent=(50, 60, 70, 80, 90))
            self.assertEqual(selected['infill_percent'].tolist(), [50, 60, 70, 80, 90])
            np.testing.assert_allclose(selected['mean_E_MPa'], [1400, 1640, 1710, 1840, 2120])
            with self.assertRaises(ValueError):
                load_calibration(path, infill_percent=(50, 55))

    def _write(self, folder, stage, conflict=False, reference_coordinates=False):
        path = Path(folder) / f'stage_{stage:05d}.comdat_comdata.csv'
        rows = ['Type:,X,Y,DX,DY,D,EXX,EYY,EXY,']
        for row in range(2):
            for col in range(2):
                dx, dy = .1*stage, .2*stage
                x, y = (col, row) if reference_coordinates else (col+dx, row+dy)
                line = f',{x},{y},{dx},{dy},0,0,0,0,{row},{col}'
                rows.append(line)
                if row == col == 1:
                    rows.append(line if not conflict else
                                f',{col+dx},{row+dy},{dx+1},{dy},0,0,0,0,{row},{col}')
        path.write_text('\n'.join(rows), encoding='gb18030')
        return path

    def test_platform_reference_coordinates_duplicates_and_temporal_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1)
            self._write(folder, 2)
            data = load_dic_platform(Path(folder), 1, 2)
        self.assertEqual(data.coordinates.shape, (4, 2))
        np.testing.assert_allclose(np.ptp(data.coordinates, axis=0), [1., 1.])
        np.testing.assert_allclose(data.displacement, np.tile([.15, .3], (4, 1)))
        np.testing.assert_allclose(data.temporal_std, np.tile([.1, .2], (4, 1))/np.sqrt(2))
        self.assertEqual(data.metadata['sources'][0]['duplicates_removed'], 1)
        self.assertEqual(data.metadata['reference_coordinate_rule'], '[X-DX,Y-DY]')

    def test_reference_coordinate_exports_are_not_displaced_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1, reference_coordinates=True)
            self._write(folder, 2, reference_coordinates=True)
            data = load_dic_platform(Path(folder), 1, 2, coordinate_state='reference')
            np.testing.assert_allclose(data.coordinates, [[0, 0], [1, 0], [0, 1], [1, 1]])
            with self.assertRaisesRegex(ValueError, 'reference coordinates change'):
                load_dic_platform(Path(folder), 1, 2, coordinate_state='deformed')
            with self.assertRaisesRegex(ValueError, 'coordinate_state'):
                load_dic_platform(Path(folder), 1, 2, coordinate_state='unknown')

    def test_conflicting_duplicates_and_missing_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1, conflict=True)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                load_dic_platform(Path(folder), 1, 1)
            with self.assertRaisesRegex(ValueError, 'missing stages'):
                load_dic_platform(Path(folder), 1, 2)

    def test_coordinate_and_displacement_transform_use_the_same_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1)
            data = load_dic_platform(Path(folder), 1, 1)
        with self.assertRaisesRegex(ValueError, 'unverified'):
            transform_dic_platform(data, None, None)
        matrix = np.array([[0., -1.], [1., 0.]])
        xy, uv, _ = transform_dic_platform(data, matrix, [10., 20.], verified=True)
        np.testing.assert_allclose(xy, data.coordinates@matrix.T + [10., 20.])
        np.testing.assert_allclose(uv, data.displacement@matrix.T)
        with self.assertRaisesRegex(ValueError, 'uniform unit scale'):
            transform_dic_platform(data, [[2., 0.], [0., 1.]], [0., 0.], verified=True)
        with self.assertRaisesRegex(ValueError, 'uniform unit scale'):
            transform_dic_platform(data, [[2e-5, 0.], [0., 1e-5]], [0., 0.], verified=True)
        xy, uv, std = transform_dic_platform(data, 1e-5*matrix, [10., 20.], verified=True)
        np.testing.assert_allclose(xy, data.coordinates@(1e-5*matrix).T + [10., 20.])
        np.testing.assert_allclose(uv, data.displacement@(1e-5*matrix).T)
        np.testing.assert_allclose(std, 1e-5*np.linalg.norm(data.temporal_std, axis=1))

    def test_registration_reports_placement_and_boundary_roundoff(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1)
            data = load_dic_platform(Path(folder), 1, 1)
        pending = dic_registration_report(data, None, None, 1., 1., [.5])
        self.assertEqual(pending['status'], 'pending_physical_registration')
        self.assertNotIn('matrix', pending)
        report = dic_registration_report(data, [[0., -1.], [1., 0.]], [1., 0.],
                                         1., 1., [.5], verified=True)
        self.assertEqual(report['points_inside_roi'], 4)
        self.assertEqual(report['points_per_region'], [2, 2])
        np.testing.assert_array_equal(dic_roi_mask(np.array([[-1e-14, 0.], [1.01, .5]]),
                                                  1., 1.), [True, False])

    def test_platforms_share_one_initial_coordinate_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write(folder, 1)
            data = load_dic_platform(Path(folder), 1, 1)
        report = check_platform_reference({'first': data, 'second': data})
        self.assertEqual(report['second']['common_grid_points'], 4)
        shifted = replace(data, coordinates=data.coordinates + [.1, 0.])
        with self.assertRaisesRegex(ValueError, 'shared reference grid differs'):
            check_platform_reference({'first': data, 'second': shifted})

    def test_legacy_15_column_format_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.csv'
            path.write_text('header\n,1,2,0,.1,.2,0,.3,0,0,0,0,0,2,3', encoding='gb18030')
            df = read_dic_csv(path)
        self.assertAlmostEqual(df.loc[0, 'DX'], .1)
        self.assertEqual(df.loc[0, 'grid_col'], 3)


if __name__ == '__main__':
    unittest.main()
