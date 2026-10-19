"""One-dimensional modulus benchmarks with free two-dimensional field inversion."""

GEO_L = 70.0
GEO_H = 20.0
THICKNESS = 3.2
NU = 0.3
NEL_X = 35
NEL_Y = 10
FORWARD_NEL_X = 70
FORWARD_NEL_Y = 20
E_REF = 1500.0  # MPa; independent numerical scale, also the uniform initial guess
TRUTH = 'partitioned_1d'  # 'uniform', 'linear_1d', 'exponential_1d', 'partitioned_1d'
REGION_BOUNDARIES_X = (11., 27., 43., 59.)
REGION_MODULI_MPA = (1400., 1640., 1710., 1840., 2120.)
# Only the synthetic truth is 1D. All nodal E(x,y) values remain independent.
FORCES_N = (119.77617870327234, 219.7354506515086)
N_POINTS = 400
NOISE_STD_MM = 0.0005
SEED = 42
RIGID_TRANSLATION_MM = (0.1, -0.07)
RIGID_ROTATION_RAD = 0.0002
REGULARIZATION_MODE = 'affine_detrended'
GAMMA = 1e-3  # Demonstration only; no experimental parameter selection implied
MAX_ITER = 1000
MAX_FUN = 15000
FTOL = 1e-14
GTOL = 1e-10
TGV_OPTIONS = {'length_scale': 20.0, 'ratio': 2.0, 'delta': 0.02}
