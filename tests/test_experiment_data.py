import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.experiment_data import load_dic_platform, transform_dic_platform
from lib.io_utils import read_dic_csv


class ExperimentDataTests(unittest.TestCase):
    def _write(self, folder, stage, conflict=False):
        path = Path(folder) / f'stage_{stage:05d}.comdat_comdata.csv'
        rows = ['Type:,X,Y,DX,DY,D,EXX,EYY,EXY,']
        for row in range(2):
            for col in range(2):
                dx, dy = .1*stage, .2*stage
                line = f',{col+dx},{row+dy},{dx},{dy},0,0,0,0,{row},{col}'
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

    def test_legacy_15_column_format_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.csv'
            path.write_text('header\n,1,2,0,.1,.2,0,.3,0,0,0,0,0,2,3', encoding='gb18030')
            df = read_dic_csv(path)
        self.assertAlmostEqual(df.loc[0, 'DX'], .1)
        self.assertEqual(df.loc[0, 'grid_col'], 3)


if __name__ == '__main__':
    unittest.main()
