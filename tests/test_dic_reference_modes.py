"""Reference compatibility and fit-only input guards; no FE computation."""
import unittest
from types import SimpleNamespace

from lib.dic_case_workflow import _validate


def configuration():
    return SimpleNamespace(
        DIC_UNIT='mm', DISPLACEMENT_REFERENCE='initial_reference_image',
        FORCE_REFERENCE='force_increment_from_record_start',
        FORCE_REFERENCE_NOTE='Explicit record-start force increment; synchronization conditional.',
        GEO_L=112., GEO_H=18., THICKNESS=3.2, LOAD_SECTION_WIDTH=18.,
        NEL_X=121, NEL_Y=19, MAX_ITER=1500, MAX_FUN=12000,
        NU_VALUES=(.26,), PRIMARY_NU=.26,
        FIT_CASES=('run3',), PREDICT_CASES=(), PLATFORMS={'run3': {}},
        OUTSIDE_POINT_POLICY='reject', GAMMA_CANDIDATES=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
        REGULARIZATION_MODE='gradient', FTOL=1e-30, GTOL=1e-12,
        REGION_BOUNDARIES_X=(), REGION_INFILL_PERCENT=(60,))


class ReferenceModeTests(unittest.TestCase):
    def test_increment_fit_only_is_valid(self):
        _validate(configuration())

    def test_unloaded_reference_remains_valid(self):
        cfg = configuration()
        cfg.DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
        cfg.FORCE_REFERENCE = 'corrected_force_from_unloaded_state'
        _validate(cfg)

    def test_mismatched_force_and_image_are_rejected(self):
        cfg = configuration()
        cfg.DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
        with self.assertRaisesRegex(ValueError, 'matched unloaded'):
            _validate(cfg)

    def test_increment_requires_provenance(self):
        cfg = configuration()
        cfg.FORCE_REFERENCE_NOTE = ' '
        with self.assertRaisesRegex(ValueError, 'require a note'):
            _validate(cfg)

    def test_fit_and_prediction_overlap_is_rejected(self):
        cfg = configuration()
        cfg.PREDICT_CASES = ('run3',)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            _validate(cfg)

    def test_unassigned_platform_is_rejected(self):
        cfg = configuration()
        cfg.PLATFORMS['run1'] = {}
        with self.assertRaisesRegex(ValueError, 'Every platform'):
            _validate(cfg)


if __name__ == '__main__':
    unittest.main()
