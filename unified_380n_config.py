"""380N DIC, fixed interfaces and intra-region H1; independent exploratory run."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = '380n_coarse_lcurve_20261023'
CASE_OUTPUT = PROJECT_ROOT/'results'/'unified_gradient_layered'/RUN_NAME
REFERENCE_PREPARED = PROJECT_ROOT/'results'/'dic_full'/'load380_20261022'/'h1'/'01_prepared'
PLATFORM = '380N'
ELEMENTS = (121, 19)
N_REGIONS = 7
NU = .26
EPSILON_MM = .3
QUADRATURE_ORDER = 3
QUADRATURE_SUBDIVISIONS = 2
GEOMETRY_PROTOCOL = 'fixed'
GAMMA_CANDIDATES = tuple(10.**(-12+8*i/49) for i in range(50))
MODULUS_BOUNDS_MPA = (100., 6000.)
MAX_ITERATIONS = 1500
MAX_EVALUATIONS = 3000
FTOL = 1e-30
GTOL = 1e-9
INITIALIZATION = 'uniform_then_data_region_constants'
EVALUATION_SHAPE = (129, 735)
