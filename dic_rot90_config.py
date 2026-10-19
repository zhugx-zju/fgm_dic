"""rot90 working convention and frame-level checks; landmarks still pending."""

from dic_revision_config import (
    PROJECT_ROOT, DATA_ROOT, FORCE_PAIRING_CSV, CALIBRATION_CSV,
    CALIBRATION_METHOD, REGION_INFILL_PERCENT, REGION_BOUNDARIES_X,
    GEO_L, GEO_H, THICKNESS, NU, NEL_X, NEL_Y, FORWARD_NEL_X, FORWARD_NEL_Y,
    E_REF, WORKING_SIGMA_MM, STAGES, FRAME_RATE_HZ, PRIMARY_REVISION,
    FIT_CASE, PREDICT_CASE, REGULARIZATION_MODE, TGV_OPTIONS, MAX_ITER, MAX_FUN,
    FTOL, GTOL,
)

DIC_ORIENTATION_OPERATIONS = ('rot90',)
DIC_COORDINATES_VERIFIED = False
REGISTRATION_STATUS = 'rot90_working_convention_center_assumed'
REGISTRATION_ASSUMPTION = (
    'Use existing orient_grid_field rot90: x=-Y_ref, y=X_ref, ux=-DY, uy=DX. '
    'Keep unit scale one and use the 180N_3 center for both platforms. '
    'Center placement within the 70x20 mm physical ROI and its assignment to '
    '50%-90% material zones are working assumptions, not landmark verification.'
)
TIMING_AUDIT_JSON = DATA_ROOT.parent / 'analysis_results' / 'dic_timing_20261018' / 'timing_checks.json'
MACHINE_FORCE_XLSB = DATA_ROOT.parent / 'sample_2_3.xlsb'
