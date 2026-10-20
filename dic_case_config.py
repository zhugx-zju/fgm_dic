"""One DIC experiment: four independent scripts consume this prepared case."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = 'sample2_run3_tikhonov_lcurve'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic'
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_ROOT / '3' / '180N_3', 'stages': (259, 273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT / '3' / '280N_3', 'stages': (436, 450), 'run': 3, 'setting_N': 280},
}
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'analysis_results' / 'dic_timing_20261018' / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'corrected_force_from_unloaded_state'
FORCE_REFERENCE_NOTE = 'Initial 60.1 N offset and unloaded DIC reference confirmed consistent for this experiment.'
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
DOMAIN_SOURCE = 'dic_reference_bounds'
# GEO_L/H are resolved from the fit platform before preparation is frozen.
GEO_L, GEO_H, THICKNESS = None, None, 3.2
LOAD_SECTION_WIDTH = 20.  # Actual specimen width, not the internal DIC ROI width.
INTERNAL_SIDE_BOUNDARY = 'traction_free_approximation'
# One mesh for BOTH independent forward and inverse; DIC points stay unchanged.
NEL_X, NEL_Y = 70, 20
REGION_INFILL_PERCENT = (50, 60, 70, 80, 90)
# Manufacturing interfaces relative to the specimen center; no span scaling.
REGION_BOUNDARIES_CENTERED_X = (-24., -8., 8., 24.)
REGION_BOUNDARIES_X = None
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGISTRATION = {
    'mode': 'assumed_center',
    'matrix': ((0., -1.), (1., 0.)),
    'cloud_center_fem_mm': None,  # Resolved to half the DIC reference span.
    'note': 'rot90 convention; ROI follows fit-platform reference bounds. Coincidence of ROI and specimen centers, and material orientation, remain assumptions.',
}
OUTSIDE_POINT_POLICY = 'reject'
FIT_CASES, PREDICT_CASES = ('180N',), ('280N',)
# Experiment-specific admissible range: 0.24 <= nu <= 0.33. The values below
# retain the frozen workflow's historical comparison points, not measured nu.
NU_VALUES = (.24, .26, .28)
PRIMARY_NU = .26  # Historical baseline; joint E/nu diagnosis is saved separately.
# Original area-weighted displacement mismatch + gamma/2 * integral(|grad E|^2).
# No reference modulus, noise normalization, or calibration-based initial field.
REGULARIZATION_MODE = 'gradient'
GAMMA_CANDIDATES = tuple(10.**(-12 + .5*i) for i in range(17))
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
