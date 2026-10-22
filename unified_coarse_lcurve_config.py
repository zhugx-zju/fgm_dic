"""121x19 unified material simulation gate before experimental inversion."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = 'coarse_lcurve_20261022'
CASE_OUTPUT = PROJECT_ROOT/'results'/'unified_gradient_layered'/RUN_NAME
REFERENCE_PREPARED = PROJECT_ROOT/'results'/'dic_full'/'coarse'/'01_prepared'
CASES = ('layered', 'continuous_2d', 'mixed_2d')
ELEMENTS = (121, 19)
N_REGIONS = 7
NU = .26
EPSILON_MM = .3
QUADRATURE_ORDER = 3
QUADRATURE_SUBDIVISIONS = 2
TRUTH_REFINEMENT = 2
TRUTH_CHECK_REFINEMENT = 3
# Fixed interfaces are explicitly known in the first protocol. Material
# coefficients always start from a uniform fit to this sample's observations.
GEOMETRY_PROTOCOL = 'fixed'
GAMMA_CANDIDATES = tuple(10.**(-12+8*i/49) for i in range(50))
MODULUS_BOUNDS_MPA = (100., 6000.)
MAX_ITERATIONS = 1500
MAX_EVALUATIONS = 3000
FTOL = 1e-30
GTOL = 1e-9
INITIALIZATION = 'uniform_then_data_region_constants'
SEED = 20261022
# Prespecified synthetic perturbation, not an estimate of pure DIC noise.
NOISE_STD_MM = .0005
VALIDATION_SEED = 20261023
GATE_MAX_FIELD_RELATIVE_RMS = .05
GATE_MAX_REGION_MEAN_RELATIVE_ERROR = .08
GATE_MAX_MEAN_REGION_RELATIVE_ERROR = .03
GATE_MAX_MULTISTART_FIELD_DIFFERENCE = .03
GATE_MAX_INTERFACE_CONTRAST_ERROR = .03
GATE_MAX_INTERFACE_POSITION_ERROR_MM = .44
UNKNOWN_START_SHIFTS = (-.2, .2)
EVALUATION_SHAPE = (129, 735)
