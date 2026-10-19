import unittest

import numpy as np

from lib.dic_noise import analyze_dic_series, temporal_projection_baseline
from lib.experiment_data import DICFrameSeries


class DICNoiseTests(unittest.TestCase):
    def _series(self, displacement=None):
        ids = np.array([(row, col) for row in range(3) for col in range(3)])
        xy = ids[:, ::-1].astype(float) - 1
        stages = np.arange(100, 115)
        if displacement is None:
            displacement = np.zeros((15, 9, 2))
        return DICFrameSeries(xy, displacement, ids,
                              {'sources': [{'stage': int(s)} for s in stages]})

    def test_known_rigid_motion_drift_and_quadratic_residual_are_separated(self):
        series = self._series()
        xy = series.coordinates
        time = np.arange(15) / 5.
        centered_time = time - time.mean()
        translations = np.column_stack([.2 * centered_time, -.3 * centered_time])
        rotations = .004 * centered_time
        slope = xy * [.002, -.004]
        # A symmetric quadratic is orthogonal to the linear time basis.
        temporal = centered_time**2 - np.mean(centered_time**2)
        amplitude = xy * [.01, .02]
        noise = temporal[:, None, None] * amplitude
        rotation_field = rotations[:, None, None] * np.column_stack([-xy[:, 1], xy[:, 0]])
        u = (.7 + translations[:, None, :] + rotation_field
             + centered_time[:, None, None] * slope + noise)
        summary, arrays = analyze_dic_series(self._series(u), 5.)
        np.testing.assert_allclose(arrays['translations'], translations, atol=1e-14)
        np.testing.assert_allclose(arrays['rotations'], rotations, atol=1e-14)
        np.testing.assert_allclose(arrays['slope'], slope, atol=1e-14)
        np.testing.assert_allclose(arrays['drift'], slope * 2.8, atol=1e-14)
        np.testing.assert_allclose(arrays['detrended'], noise, atol=1e-14)
        expected_sd = np.abs(amplitude) * np.sqrt(np.sum(temporal**2) / 13)
        np.testing.assert_allclose(arrays['detrended_sd'], expected_sd, atol=1e-14)
        np.testing.assert_allclose(arrays['mean_displacement'], u.mean(axis=0), atol=1e-14)
        self.assertAlmostEqual(summary['duration_s'], 2.8)

    def test_uniform_unit_scale_preserves_correlation_and_stage_selection(self):
        series = self._series()
        u = np.random.default_rng(12).normal(size=(15, 9, 2))
        first = DICFrameSeries(series.coordinates, u, series.grid_ids, series.metadata)
        second = DICFrameSeries(1e-5 * series.coordinates, 1e-5 * u,
                                series.grid_ids, series.metadata)
        original, _ = analyze_dic_series(first, 5.)
        scaled, _ = analyze_dic_series(second, 5.)
        self.assertEqual(original['representative_stage_after_rigid_removal'],
                         scaled['representative_stage_after_rigid_removal'])
        np.testing.assert_allclose(scaled['detrended_temporal_sd']['median'],
                                   1e-5 * np.array(original['detrended_temporal_sd']['median']))
        for a, b in zip(original['temporal_correlations'], scaled['temporal_correlations']):
            np.testing.assert_allclose(a['correlation'], b['correlation'], atol=1e-14)
        self.assertTrue(all(-1 <= value <= 1 for record in original['spatial_correlations']
                            for value in record['correlation']))

    def test_zero_signal_has_no_invented_correlation_or_uncertainty(self):
        summary, arrays = analyze_dic_series(self._series(), 5.)
        np.testing.assert_array_equal(arrays['detrended_sd'], np.zeros((9, 2)))
        self.assertTrue(all(record['correlation'] == [None, None]
                            for record in summary['temporal_correlations']))
        self.assertEqual(summary['linear_trend_energy_fraction_after_rigid_removal'], [0., 0.])

    def test_projection_null_moments_agree_with_independent_simulation(self):
        time = np.arange(15) / 5.
        centered_time = time - time.mean()
        rng = np.random.default_rng(25)
        centered = rng.normal(size=(15, 20000))
        centered -= centered.mean(axis=0)
        slope = centered_time @ centered / np.sum(centered_time**2)
        residual = centered - centered_time[:, None] * slope
        moments = temporal_projection_baseline(time)
        observed_trend_fraction = 1 - np.sum(residual**2) / np.sum(centered**2)
        self.assertAlmostEqual(observed_trend_fraction,
                               moments['linear_trend_expected_energy_fraction'], delta=.002)
        for record in moments['temporal_correlations']:
            values = centered if record['processing'] == 'cleaned' else residual
            lag = record['lag_frames']
            observed = np.sum(values[:-lag] * values[lag:]) / np.sqrt(
                np.sum(values[:-lag]**2) * np.sum(values[lag:]**2))
            self.assertAlmostEqual(observed, record['normalized_expected_cross_product'], delta=.01)

    def test_input_errors_are_actionable(self):
        series = self._series()
        for fps in (0., -1., np.nan):
            with self.assertRaisesRegex(ValueError, 'positive frame rate'):
                analyze_dic_series(series, fps)
        with self.assertRaisesRegex(ValueError, 'at least five frames'):
            analyze_dic_series(DICFrameSeries(series.coordinates, np.zeros((4, 9, 2)),
                                              series.grid_ids, series.metadata), 5.)


if __name__ == '__main__':
    unittest.main()
