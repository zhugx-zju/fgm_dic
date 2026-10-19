"""Exploratory revised-DIC study; physical landmarks remain unverified."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2' / '3'
FORCE_PAIRING_CSV = DATA_ROOT.parent / 'analysis_results' / 'dic_timing_20261018' / 'selected_dic_windows.csv'
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGION_INFILL_PERCENT = (50, 60, 70, 80, 90)
REGION_BOUNDARIES_X = (11., 27., 43., 59.)
GEO_L, GEO_H, THICKNESS, NU = 70., 20., 3.2, .3
NEL_X, NEL_Y = 35, 10
FORWARD_NEL_X, FORWARD_NEL_Y = 70, 20
E_REF = 1700.
REGULARIZATION_MODE = 'affine_detrended'
TGV_OPTIONS = {}
GAMMA_CANDIDATES = (100., 1000., 10000.)
MAX_ITER, MAX_FUN = 1200, 10000
FTOL, GTOL = 1e-11, 1e-7
WORKING_SIGMA_MM = .0003  # Common objective scale, not absolute uncertainty.
VALIDATION_BLOCK_MM = 6.
VALIDATION_BUFFER_MM = 1.
FIT_CASE, PREDICT_CASE = '180N', '280N'
STAGES = {'180N': (259, 273), '280N': (436, 450)}
FRAME_RATE_HZ = 5.
PRIMARY_REVISION = '_3'
DIC_COORDINATES_VERIFIED = False
REGISTRATION_STATUS = 'assumed_center_and_material_direction'
REGISTRATION_ASSUMPTION = (
    'Unit scale 1. Assume the measured cloud center coincides with the physical '
    '70x20 mm ROI center, and +Y points from 50% to 90% infill. '
    'This is an explicit exploratory assumption, not landmark verification. '
    'One transform from the 180N_3 reference is used for both loads and _2. '
    'No span normalization. Axial +/-4 mm and reversed direction are audited.'
)
