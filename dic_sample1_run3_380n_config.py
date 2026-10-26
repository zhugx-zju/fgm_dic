"""sample_1 run 3: free 2D H1, own gamma, assumed net force 325.3872 N."""
from dic_sample1_380n_common_config import *

RUN_NAME = 'sample1_run3_380n_h1_20261026'
CASE_ID = RUN_NAME
CASE_OUTPUT = PROJECT_ROOT/'results'/'dic_full'/RUN_NAME
PLATFORMS = {'380N': dict(folder=SAMPLE_ROOT/'3'/'380N', stages=(853,867),
    run=3, setting_N=380, force_N=325.3872,
    force_note='User-authorized assumption 380 - 54.6128 = 325.3872 N; no synchronous force; force/DIC event pairing inconsistent at confirmed 5 Hz.')}
NEL_X, NEL_Y = 130, 19
CONFIGURED_ELEMENTS = (NEL_X, NEL_Y)
