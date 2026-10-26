"""sample_1 run 2: existing E(x) parameterization, same mechanics and data."""
from dic_sample1_run2_380n_config import *

RUN_NAME = 'sample1_run2_380n_axial_20261026'
CASE_ID = RUN_NAME
CASE_OUTPUT = PROJECT_ROOT/'results'/'dic_full'/RUN_NAME
MATERIAL_PARAMETERIZATION = 'axial'
NU_VALUES, PRIMARY_NU = (.26,), .26
