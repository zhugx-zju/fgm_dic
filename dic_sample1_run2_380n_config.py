"""sample_1 run 2: free 2D H1, own gamma, assumed net force 333.6728 N."""
from dic_sample1_380n_common_config import *

RUN_NAME = 'sample1_run2_380n_h1_20261026'
CASE_ID = RUN_NAME
CASE_OUTPUT = PROJECT_ROOT/'results'/'dic_full'/RUN_NAME
PLATFORMS = {'380N': dict(folder=SAMPLE_ROOT/'2'/'380N', stages=(680,694),
    run=2, setting_N=380, force_N=333.6728,
    force_note='User-authorized assumption 380 - 46.3272 = 333.6728 N; no synchronous maximum-platform force.')}
# ceil(measured extent / original coarse spacing 0.878534 x 0.971774 mm).
NEL_X, NEL_Y = 129, 19
CONFIGURED_ELEMENTS = (NEL_X, NEL_Y)
