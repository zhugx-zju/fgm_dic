"""Common H1 settings for separate sample_1 maximum-platform inversions."""
from dic_case_config import *

SAMPLE_ROOT = PROJECT_ROOT/'post_tensile_test'/'1016'/'sample_1'
# This is a different specimen/reference. Never inherit sample_2's material map.
del MATERIAL_WINDOW_MM
FORCE_ASSUMED = True
FORCE_REFERENCE_NOTE = (
    'User-authorized exploratory assumption: maximum setting 380 N minus '
    'each run initial machine force. No synchronous force covers these DIC '
    'windows; equivalence of the earlier DIC reference and initial machine '
    'load is assumed, not independently verified.')
REGISTRATION = {
    'mode': 'specimen_marks',
    'matrix': ((0., 1.), (-1., 0.)),
    'specimen_length_mm': 220.,
    'low_infill_mark_end_distance_mm': 53.,
    'high_infill_mark_end_distance_mm': 53.,
    'material_x_sign': -1,
    'mark_reference': 'roi_axial_extrema',
    'length_scale_source': 'independent_dic_calibration',
    'note': (
        'New specimen sample_1. User reports marks approximately 53 mm from '
        'each outer end and ROI boundaries approximately at those marks. '
        'Align their midpoint with specimen center; preserve measured mm '
        'span, without stretching to the nominal 114 mm gap. Infill direction '
        'is UNKNOWN. Descending infill along raw +Y is only comparison '
        'hypothesis A; reverse hypothesis B must also be evaluated. Neither '
        'partition nor calibration enters the free 2D inverse.')}
DIRECTION_UNCONFIRMED = True
MATERIAL_LOCATION_APPROXIMATE = True
DOMAIN_SOURCE = 'dic_reference_bounds'
GEOMETRY_REFERENCE_PLATFORM = '380N'
IDENTIFICATION_PLATFORM = '380N'
REGION_INFILL_PERCENT = (100, 90, 80, 70, 60, 50, 40)
REGION_BOUNDARIES_X = None
MATERIAL_PARAMETERIZATION = 'nodal_2d'
DATA_MODE = 'dic'
MESH_SOURCE = 'configured_grid'
REGULARIZATION_MODE = 'gradient'
DISPLACEMENT_COMPONENT_WEIGHTS = (1., 1.)
THICKNESS, LOAD_SECTION_WIDTH = 3.2, 20.
NU_VALUES, PRIMARY_NU = (.24, .26, .28, .33), .26
GAMMA_COUNT = 50
GAMMA_LOG10_RANGE = (-12., -4.)
GAMMA_CANDIDATES = tuple(10.**(-12.+8*i/49) for i in range(50))
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
FIT_CASES, PREDICT_CASES = ('380N',), ()
