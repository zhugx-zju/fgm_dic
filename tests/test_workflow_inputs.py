"""Small checks for the new frame/force boundary and portable configuration."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lib.config_utils import load_python_configuration
from lib.dic_frame_selection import read_timing, TimingData
from lib.frame_selection_workflow import analyze_timing

ROOT = Path(__file__).resolve().parents[1]


class WorkflowInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_python_configuration(str(ROOT/'config.py'))
        cls.spec = cls.cfg.TIMING_RUNS[0]
        cls.data = read_timing(cls.spec['folder'], cls.spec['fps'], force_file='force.csv')

    def test_known_events_recover_offset_window_and_net_force(self):
        record, _, _ = analyze_timing(self.data, self.spec, self.cfg.FRAME_SELECTION_OPTIONS)
        # Independent two-line events are located between samples; 5 Hz DIC
        # cannot support an exact continuous-time offset assertion.
        tolerance = .5/self.data.fps+.5*np.max(np.diff(self.data.machine[:, 0]))
        self.assertLessEqual(abs(record['machine_time_offset_s']+2.), tolerance+1e-9)
        self.assertEqual((record['stage_first'], record['stage_last']), (130, 144))
        self.assertEqual(record['status'], 'ready')
        self.assertAlmostEqual(record['mean_corrected_force_N'], 80., places=10)

    def test_missing_force_tail_is_never_extrapolated(self):
        data = TimingData(self.data.machine[self.data.machine[:, 0] < 23.], self.data.points,
                          self.data.point_labels, self.data.camera_time_s, self.data.fps)
        record, force, covered = analyze_timing(data, self.spec, self.cfg.FRAME_SELECTION_OPTIONS)
        self.assertEqual(record['status'], 'export_candidate_only')
        self.assertIsNone(record['mean_corrected_force_N'])
        self.assertTrue(np.isnan(force[~covered]).all())

    def test_reference_force_cannot_be_inferred_from_record_start(self):
        spec = dict(self.spec, reference_force_N=None)
        record, _, _ = analyze_timing(self.data, spec, self.cfg.FRAME_SELECTION_OPTIONS)
        self.assertTrue(record['full_force_coverage'])
        self.assertEqual(record['status'], 'export_candidate_only')
        self.assertIsNone(record['mean_corrected_force_N'])

    def test_sparse_force_record_does_not_satisfy_coverage_protocol(self):
        data = TimingData(self.data.machine[::10], self.data.points, self.data.point_labels,
                          self.data.camera_time_s, self.data.fps)
        record, _, _ = analyze_timing(data, self.spec, self.cfg.FRAME_SELECTION_OPTIONS)
        self.assertFalse(record['force_gap_check'])
        self.assertEqual(record['status'], 'export_candidate_only')

    def test_config_import_requires_no_data_or_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.py'
            path.write_bytes((ROOT/'config.py').read_bytes())
            cfg = load_python_configuration(str(path))
            self.assertEqual(cfg.PROJECT_ROOT, path.parent)
            self.assertFalse(cfg.CALIBRATION_CSV.exists())
            path.write_text(path.read_text(encoding='utf-8').replace("CASE_PROFILE = 'example'", "CASE_PROFILE = 'experiment'"), encoding='utf-8')
            cfg = load_python_configuration(str(path))
            self.assertEqual(cfg.CONFIGURED_ELEMENTS, (121, 19))
            self.assertFalse(cfg.CALIBRATION_CSV.exists())
            self.assertTrue(cfg.FORCE_ASSUMED)


if __name__ == '__main__':
    unittest.main()
