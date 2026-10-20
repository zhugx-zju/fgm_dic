"""One DIC experiment: four independent scripts consume this prepared case."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = 'sample2_run3_full_coarse'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / 'coarse'
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_ROOT / '3' / '180N_2' / 'all_region', 'stages': (259, 273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT / '3' / '280N_2' / 'all_region', 'stages': (436, 450), 'run': 3, 'setting_N': 280},
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
# Preserve the former FE spacing on the same extended domain; DIC stays at its measured points.
MESH_SOURCE = 'configured_grid'
NEL_X, NEL_Y = 121, 19
# Groups are ordered along increasing FEM x = raw +Y.
REGION_INFILL_PERCENT = (100, 90, 80, 70, 60, 50, 40)
# Manufacturing x increases with infill; zero is the specimen center.
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24., -8., 8., 24., 40.)
# Seven contiguous 16 mm target sections; retain only actually exported points.
MATERIAL_WINDOW_MM = (-56., 56.)
REGION_BOUNDARIES_X = None
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGISTRATION = {
    'mode': 'material_coordinates',
    'matrix': ((0., 1.), (-1., 0.)),  # x=+Y, y=-X; ux=DY, uy=-DX.
    'material_x_sign': -1,  # Manufacturing infill coordinate decreases along FEM x.
    'specimen_length_mm': 220.,  # Manufacturing nominal length.
    'low_infill_mark_end_distance_mm': 85.,
    'high_infill_mark_end_distance_mm': 73.,
    'raw_axial_anchor_mm': -21.50749765093,  # Original region mark midpoint in raw Y0.
    'manufacturing_anchor_mm': 6.,  # Midpoint of manufacturing marks -25/+37 mm.
    'shared_reference_confirmed': True,
    'anchor_prepared_metadata_sha256': '4ea6dd2939ba290b60ee9c6fc89b04758b2d95386e9f45c365138d3d8204dbba',
    'length_scale_source': 'independent_dic_calibration',
    'note': 'all_region and region share the initial image and millimetre coordinates, as confirmed. Inherit the region mark-midpoint raw Y0=-21.50749765093 at manufacturing xi=6 mm. Retain material increase along raw -Y; same-point forward direction comparison saved in results/dic_full/direction_check. Select the continuous requested xi=[-56,56] window intersected with available data, preserving raw nodes and units.',
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
