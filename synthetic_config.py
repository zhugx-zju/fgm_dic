"""Unified synthetic workflow, using the project's existing Python config style."""

GEO_L = 110.0
GEO_H = 20.0
THICKNESS = 3.2
NU = 0.3
NEL_X = 22
NEL_Y = 4
FORWARD_NEL_X = 44
FORWARD_NEL_Y = 8
E_REF = 1500.0  # MPa; independent numerical scale, also the uniform initial guess
TRUTH = 'smooth_2d'  # 'uniform', 'smooth_2d', 'seven_regions'
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
