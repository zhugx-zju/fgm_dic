"""One DIC experiment: four independent scripts consume this prepared case."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = 'sample2_run3_tikhonov_lcurve'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_case_tikhonov_lcurve_20261019'
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_ROOT / '3' / '180N_3', 'stages': (259, 273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT / '3' / '280N_3', 'stages': (436, 450), 'run': 3, 'setting_N': 280},
}
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'analysis_results' / 'dic_timing_20261018' / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'corrected_force_from_unloaded_state'
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
GEO_L, GEO_H, THICKNESS = 70., 20., 3.2
# One mesh for BOTH independent forward and inverse; interfaces align at 1 mm.
NEL_X, NEL_Y = 70, 20
REGION_INFILL_PERCENT = (50, 60, 70, 80, 90)
REGION_BOUNDARIES_X = (11., 27., 43., 59.)
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGISTRATION = {
    'mode': 'assumed_center',
    'matrix': ((0., -1.), (1., 0.)),
    'cloud_center_fem_mm': (35., 10.),
    'note': 'rot90 working convention. Cloud center and material-zone placement are explicit assumptions; physical landmarks are not independently verified.',
}
OUTSIDE_POINT_POLICY = 'reject'
FIT_CASES, PREDICT_CASES = ('180N',), ('280N',)
NU_VALUES = (.24, .26, .28)
PRIMARY_NU = .26  # Prespecified middle of the user-provided range; not selected on 280N.
# Original area-weighted displacement mismatch + gamma/2 * integral(|grad E|^2).
# No reference modulus, noise normalization, or calibration-based initial field.
REGULARIZATION_MODE = 'gradient'
GAMMA_CANDIDATES = tuple(10.**(-12 + .5*i) for i in range(17))
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
