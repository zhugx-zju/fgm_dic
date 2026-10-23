"""Small controlled support/solver pilot; not an L-curve selection run."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = 'support_precondition_pilot_20261023'
CASE_OUTPUT = PROJECT_ROOT/'results'/'unified_gradient_layered'/RUN_NAME
REFERENCE_RUN = PROJECT_ROOT/'results'/'unified_gradient_layered'/'380n_coarse_lcurve_20261023'
METHODS = ('smooth_full', 'sharp_full', 'sharp_local', 'local_projected', 'local_metric')
GAMMAS = (1e-4, 10.**(-12+8*24/49), 1e-12)
ELEMENTS = (121, 19)
N_REGIONS = 7
NU = .26
EPSILON_MM = .3
QUADRATURE_ORDER = 3
QUADRATURE_SUBDIVISIONS = 2
MODULUS_BOUNDS_MPA = (100., 6000.)
MAX_ITERATIONS = 400
MAX_EVALUATIONS = 600
FTOL = 1e-30
GTOL = 1e-9
MEAN_FD_LOG_STEP = 1e-4
EVALUATION_SHAPE = (129, 735)
