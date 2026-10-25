"""Exploratory 380N H1 inversion of the contiguous 60% and 50% regions."""
from dic_380n_config import *

RUN_NAME = 'load380_subset2_60_50_20261025'
CASE_ID = f'sample2_run3_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
GEOMETRY_REFERENCE_PLATFORM = '180N'
REGION_INFILL_PERCENT = (60, 50)
MATERIAL_WINDOW_MM = (-40., -8.)
REGION_BOUNDARIES_MANUFACTURING_X = (-24.,)
# ceil(actual retained axial span / original coarse spacing); y is unchanged.
NEL_X, NEL_Y = 37, 19
CONFIGURED_ELEMENTS = (NEL_X, NEL_Y)
REGISTRATION = dict(REGISTRATION, note=(
    'Shared original material map and millimetre scale retained. Select only '
    'xi=[-40,-8] mm (60%,50%), intersected with actual DIC rows, without '
    'resampling or recentering manufacturing interfaces. Internal cut ends '
    'use the original uniform-traction approximation; boundary equivalence '
    'is unverified.'))
SUBSET_SELECTION = {
    'source': 'results/dic_full/load380_20261022/h1/03_inverse/nu_0.26/regional_modulus_comparison.csv',
    'criterion': 'minimum equal-region mean absolute calibration difference among contiguous pairs',
    'source_nu': .26,
    'scope': 'post hoc exploratory selection; calibration agreement is not independent validation',
    'uses_calibration_in_inverse_or_initialization': False,
}
