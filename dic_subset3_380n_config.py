"""Exploratory 380N H1 inversion of contiguous 60%, 50%, and 40% regions."""
from dic_380n_config import *

RUN_NAME = 'load380_subset3_60_50_40_20261025'
CASE_ID = f'sample2_run3_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
GEOMETRY_REFERENCE_PLATFORM = '180N'
REGION_INFILL_PERCENT = (60, 50, 40)
MATERIAL_WINDOW_MM = (-56., -8.)
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24.)
NEL_X, NEL_Y = 49, 19
CONFIGURED_ELEMENTS = (NEL_X, NEL_Y)
REGISTRATION = dict(REGISTRATION, note=(
    'Shared original material map and millimetre scale retained. Select only '
    'xi=[-56,-8] mm (60%,50%,40%), intersected with actual DIC rows. The 40% '
    'region retains its original missing end coverage. No resampling or '
    'manufacturing-interface recentering. Internal cut ends use the original '
    'uniform-traction approximation; boundary equivalence is unverified.'))
SUBSET_SELECTION = {
    'source': 'results/dic_full/load380_20261022/h1/03_inverse/nu_0.26/regional_modulus_comparison.csv',
    'criterion': 'minimum equal-region mean absolute calibration difference among contiguous triples',
    'source_nu': .26,
    'scope': 'post hoc exploratory selection; calibration agreement is not independent validation',
    'uses_calibration_in_inverse_or_initialization': False,
}
