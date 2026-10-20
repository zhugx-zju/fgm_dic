"""One DIC experiment: four independent scripts consume this prepared case."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = 'sample2_run3_dic_mesh'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_mesh'
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_ROOT / '3' / '180N_2' / 'region', 'stages': (259, 273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT / '3' / '280N_2' / 'region', 'stages': (436, 450), 'run': 3, 'setting_N': 280},
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
# FE nodes are the complete DIC reference grid; counts resolve from fit data.
MESH_SOURCE = 'dic_grid'
NEL_X, NEL_Y = None, None
# Groups are ordered along increasing FEM x = raw +Y.
REGION_INFILL_PERCENT = (90, 80, 70, 60, 50)
# Manufacturing x increases with infill; zero is the specimen center.
REGION_BOUNDARIES_MANUFACTURING_X = (-24., -8., 8., 24.)
REGION_BOUNDARIES_X = None
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGISTRATION = {
    'mode': 'specimen_marks',
    'matrix': ((0., 1.), (-1., 0.)),  # x=+Y, y=-X; ux=DY, uy=-DX.
    'material_x_sign': -1,  # Manufacturing infill coordinate decreases along FEM x.
    'specimen_length_mm': 220.,  # Manufacturing nominal length.
    'low_infill_mark_end_distance_mm': 85.,
    'high_infill_mark_end_distance_mm': 73.,
    'mark_reference': 'roi_axial_extrema',
    'length_scale_source': 'independent_dic_calibration',
    'note': 'Adopted material-direction correction: infill increases along raw -Y; 73 mm mark at high-infill end, 85 mm at low-infill end; region ends correspond to the marks. Retain independently calibrated mm. Align mark midpoint without scaling; record the approximate mark-gap/export-span discrepancy.',
}
OUTSIDE_POINT_POLICY = 'reject'
FIT_CASES, PREDICT_CASES = ('180N',), ('280N',)
# Experiment-specific admissible range: 0.24 <= nu <= 0.33. The values below
# retain the frozen workflow's historical comparison points, not measured nu.
NU_VALUES = (.24, .26, .28, .33)
PRIMARY_NU = .26  # Historical baseline; joint E/nu diagnosis is saved separately.
# Original area-weighted displacement mismatch + gamma/2 * integral(|grad E|^2).
# No reference modulus, noise normalization, or calibration-based initial field.
REGULARIZATION_MODE = 'gradient'
# 50 logarithmically spaced candidates, including both endpoints.
GAMMA_CANDIDATES = tuple(10.**(-12 + 8*i/49) for i in range(50))
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
