"""380N-only E(x) diagnostic; two-dimensional FE and full DIC are unchanged."""

from dic_380n_config import *

RUN_NAME = 'h1_380N_axial_20261025'
CASE_ID = f'sample2_run3_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
PLATFORMS = {'380N': PLATFORMS['380N']}
FIT_CASES, PREDICT_CASES = ('380N',), ()
MATERIAL_PARAMETERIZATION = 'axial'
NU_VALUES, PRIMARY_NU = (.26,), .26
# Same 121x19 mechanics, 50 gamma values and per-point budgets as the original
# 380N H1 run. Only 380N is computed, and calibration never selects gamma.
