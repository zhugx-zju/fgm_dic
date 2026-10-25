"""Exact-TV counterpart of the fixed 60%/50% local 380N H1 case."""
from dic_subset2_380n_config import *

RUN_NAME = 'load380_subset2_60_50_exact_tv_20261025'
CASE_ID = f'sample2_run3_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
METHOD = 'exact_tv'
REGULARIZATION_MODE = 'exact_tv'
GAMMA_COUNT = 50
GAMMA_LOG10_RANGE = (-10., -2.)
GAMMA_CANDIDATES = tuple(10.**(-10.+8*i/49) for i in range(50))
EXACT_TV_SOLVER = 'primal_dual'
EXACT_TV_PRIMAL_STEP = 2e8
EXACT_TV_COUPLING = .9
EXACT_TV_OPTIONS = dict(max_outer=12000, max_inner=150, stationarity_atol=1e-9,
    stationarity_rtol=1e-3, primal_atol=1e-6, primal_rtol=1e-4,
    gap_atol=1e-9, gap_rtol=1e-4, adapt_interval=10)
BASELINE_CASE = 'load380_subset2_60_50_20261025'
