"""Run 3 platforms; verify DIC scale and physical registration before solving."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_DIR / '3' / '180N', 'stages': (259, 273), 'setting_N': 180},
    '280N': {'folder': SAMPLE_DIR / '3' / '280N', 'stages': (436, 450), 'setting_N': 280},
}
FORCE_PAIRING_CSV = SAMPLE_DIR / 'analysis_results' / 'dic_timing_20261018' / 'selected_dic_windows.csv'
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'

GEO_L = 70.0  # Marks are 75 mm from either physical end of the 220 mm specimen
GEO_H = 20.0
THICKNESS = 3.2
NU = 0.3  # Assumption, to be checked by independent measurement/sensitivity analysis
NEL_X = 35
NEL_Y = 10
FORWARD_NEL_X = 70  # 1 mm cells align the five-region boundaries
FORWARD_NEL_Y = 20
SPECIMEN_ROI_X = (-35.0, 35.0)
REGION_INFILL_PERCENT = (50, 60, 70, 80, 90)
REGION_BOUNDARIES_X = (11., 27., 43., 59.)

# x_FEM = A @ [X_ref,Y_ref] + b; u_FEM = A @ [DX,DY].
# DIC Y is the loading direction. Supply the checked sign, uniform unit scale
# and offset using both measured marks and a checked width scale. The physical
# ROI is known, but export coordinates still need checking; never rescale axes
# independently just to force the numerical span to 70 x 20 mm.
DIC_COORDINATES_VERIFIED = False
DIC_COORDINATE_STATE = 'deformed'  # X_ref=X-DX; use 'reference' only if X/Y already refer to the initial image
DIC_TO_FEM_MATRIX = None
DIC_TO_FEM_OFFSET = None
DIC_REGISTRATION_NOTE = 'Export units and Y-positive infill order still require confirmation.'
NOISE_FLOOR_MM = 0.001  # Explicit provisional floor; not a measured uncertainty
FIT_CASES = ('180N',)
PREDICT_CASES = ('280N',)
E_REF = 1700.0  # MPa; does not impose the independent calibration on inversion
REGULARIZATION_MODE = 'affine_detrended'
GAMMA = 1.0  # Placeholder, not selected using the held-out platform
MAX_ITER = 2000
MAX_FUN = 15000
FTOL = 1e-12
GTOL = 1e-8
TGV_OPTIONS = {'length_scale': 20.0, 'ratio': 2.0, 'delta': 0.02}
