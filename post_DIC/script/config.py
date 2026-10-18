from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgm_asm.config_types import InverseConfig, LCurveConfig


# ============================================================
# Preprocessing case settings
# ============================================================

# Base exported DIC filename without the trailing data-type suffix.
# Example:
#   "stage_00463.comdat_comdata"
DIC_EXPORT_BASENAME = "stage_00463.comdat_comdata"
CASE_PREFIX = DIC_EXPORT_BASENAME.split(".")[0]

# Corner used for the final fixed-point displacement reference.
# Allowed values:
#   None, "top_right", "bottom_right", "top_left", "bottom_left"
# For the current fliplr workflow, raw "bottom_right" becomes the FE
# "bottom_left" corner, which matches the FE y-reference point.
PREPROCESS_FIXED_POINT = "bottom_right"

# Physical side used for the fixed-end strip rigid-body removal.
# Allowed values:
#   "left", "right"
PREPROCESS_FIXED_END_SIDE = "right"


# ============================================================
# Preprocessing parameters
# ============================================================

# Ratio of the fixed-end strip width to the specimen length.
# Typical values:
#   0.05 to 0.20
PREPROCESS_STRIP_WIDTH_RATIO = 0.05

# Optional top/bottom margin removed from the strip when estimating
# the fixed-end rigid motion.
# Typical values:
#   0.00 to 0.15
PREPROCESS_STRIP_Y_MARGIN_RATIO = 0.05

# Ordered 2D orientation operations applied after rigid removal
# and fixed-point reference.
# Allowed items:
#   "rot0", "rot90", "rot180", "rot270", "fliplr", "flipud", "transpose"
# Typical examples:
#   ["rot180"]
#   ["fliplr"]
#   ["rot90", "transpose"]
#   []  -> no extra orientation transform
PREPROCESS_ORIENTATION_OPERATIONS = ["fliplr"]

# Optional whole-field sign control applied after orientation and
# before saving npy arrays / plotting.
# Allowed values:
#   1.0  -> keep sign
#   -1.0 -> flip sign
PREPROCESS_UX_SIGN = 1.0
PREPROCESS_UY_SIGN = 1.0

# Plot colormap used for saved displacement cloud plots.
# Common matplotlib choices:
#   "jet", "viridis", "plasma", "inferno", "coolwarm"
PREPROCESS_CMAP = "jet"

# Whether preprocessing is allowed to rewrite the inverse geometry values
# in this config file. Keep this disabled for reproducible inverse runs.
PREPROCESS_UPDATE_CONFIG = False


# ============================================================
# Optional inverse settings
# ============================================================

# FE geometry size used by the inverse workflow.
GEO_L = 69.9012598386
GEO_H = 49.39872496765

# CASE solves equilibrium on the physical specimen domain. The DIC subset may
# be an interior region, so it can be embedded in a larger CASE domain.
CASE_DOMAIN_GEO_L = GEO_L
CASE_DOMAIN_GEO_H = 50.0
CASE_DIC_X_OFFSET = 0.0
CASE_DIC_Y_OFFSET = 0.5 * (CASE_DOMAIN_GEO_H - GEO_H)

# Structured FE mesh density used by the inverse workflow.
# Typical values:
#   positive integers such as 20, 39, 50
NEL_X = 39
NEL_Y = 39

# Poisson ratio.
# Typical values:
#   0.0 to 0.49
NU = 0.2

# Elastic modulus bounds used by the inverse solver.
E_MIN = 0.001
E_MAX = 1000.0
EHAT_MIN = 0.05
EHAT_MAX = 5.0

# Regularization and optimization parameters.
# The defaults below are intentionally strict and may keep descending for
# a long time on real DIC data. For quick engineering checks, a practical
# starting point is often FTOL = 1e-12 and GTOL = 1e-4.
GAMMA = 3.933892640241518e2
GAMMA_MIN = 1
GAMMA_MAX = 1e5
N_GAMMA = 50
MAX_ITER = 500
FTOL = 1e-14
GTOL = 1e-7

# Optional path to a true modulus field file for comparison.
# Allowed values:
#   None
#   absolute or project-relative file path string
TRUE_MODULUS_FILE = None

# Experimental loading values used by the inverse workflow.
TENSILE_END_FORCE = 281.3719
PRESCRIBED_RIGHT_EDGE_DISPLACEMENT = 0.035283444320249276

# Observation-operator settings used by the inverse workflow.
# Allowed values:
#   "uniform_subgrid"
#   "fe_gauss_nearest"
OBSERVATION_SAMPLING_MODE = "uniform_subgrid"
OBSERVATION_UNIFORM_SUBGRID_NX = NEL_X + 1
OBSERVATION_UNIFORM_SUBGRID_NY = NEL_Y + 1
# Component weights are used in r^T W r.  For local_max mode, the scalar
# values below are further multiplied by
#   1 / max(epsilon * max(abs(u_component)), abs(u_component_i))^2
OBSERVATION_UX_COMPONENT_WEIGHT = 1.0
OBSERVATION_UY_COMPONENT_WEIGHT = 1.0
OBSERVATION_COMPONENT_WEIGHT_MODE = "local_max"
OBSERVATION_LOCAL_MAX_EPSILON = 0.10
OBSERVATION_COLLAPSE_DUPLICATES = True

# Results directory used by the inverse workflow.
# Allowed values:
#   directory name such as "results"
#   absolute or project-relative directory path string
RESULTS_ROOT_DIR = "results"

FIXED_POINT = PREPROCESS_FIXED_POINT
USE_CORRECTED_CSV = True
CASE_NAME = f"{CASE_PREFIX}_{FIXED_POINT}_inverse"


def get_data_dir() -> Path:
    return PROJECT_ROOT / "data"


def resolve_project_file(file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and len(candidate.parts) > 1:
        return PROJECT_ROOT / candidate
    return get_data_dir() / candidate


def get_prepared_dic_dir() -> Path:
    return get_data_dir()


def get_case_prefix() -> str:
    return CASE_PREFIX


def get_grid_dic_csv_path() -> Path:
    return resolve_project_file(f"{DIC_EXPORT_BASENAME}_grid.csv")


def get_pixel_dic_csv_path() -> Path:
    return resolve_project_file(f"{DIC_EXPORT_BASENAME}_pixel.csv")


def get_grid_case_name() -> str:
    return f"{get_case_prefix()}_grid_preprocessed"


def get_pixel_case_name() -> str:
    return f"{get_case_prefix()}_pixel_preprocessed"


def get_raw_dic_csv_path() -> Path:
    return resolve_project_file(f"{DIC_EXPORT_BASENAME}.csv")


def get_corrected_dic_csv_path() -> Path:
    fixed_point = PREPROCESS_FIXED_POINT
    return resolve_project_file(f"{DIC_EXPORT_BASENAME}_corrected_fixed_point_{fixed_point}.csv")


def get_case_results_dir() -> Path:
    return PROJECT_ROOT / RESULTS_ROOT_DIR


def get_inverse_case_name() -> str:
    return f"{get_case_prefix()}_{PREPROCESS_FIXED_POINT}_inverse"


def get_prepared_grid_paths() -> dict[str, Path]:
    prepared_dir = get_prepared_dic_dir()
    case_name = get_grid_case_name()
    return {
        "ux_grid": prepared_dir / f"{case_name}_ux_grid.npy",
        "uy_grid": prepared_dir / f"{case_name}_uy_grid.npy",
        "x_grid": prepared_dir / f"{case_name}_x_grid.npy",
        "y_grid": prepared_dir / f"{case_name}_y_grid.npy",
    }


def get_observation_case_name() -> str:
    return f"{get_inverse_case_name()}_{OBSERVATION_SAMPLING_MODE}_observation"


def get_observation_package_paths() -> dict[str, Path]:
    prepared_dir = get_prepared_dic_dir()
    case_name = get_observation_case_name()
    return {
        "x_meas": prepared_dir / f"{case_name}_x_meas.npy",
        "y_meas": prepared_dir / f"{case_name}_y_meas.npy",
        "ux_meas": prepared_dir / f"{case_name}_ux_meas.npy",
        "uy_meas": prepared_dir / f"{case_name}_uy_meas.npy",
        "u_measured": prepared_dir / f"{case_name}_u_measured.npy",
        "point_weights": prepared_dir / f"{case_name}_point_weights.npy",
        "weight_vector": prepared_dir / f"{case_name}_weight_vector.npy",
        "H": prepared_dir / f"{case_name}_H.npz",
        "W": prepared_dir / f"{case_name}_W.npz",
    }


def get_inverse_config() -> InverseConfig:
    return InverseConfig(
        gamma=GAMMA,
        E_min=E_MIN,
        E_max=E_MAX,
        Ehat_min=EHAT_MIN,
        Ehat_max=EHAT_MAX,
        max_iter=MAX_ITER,
        ftol=FTOL,
        gtol=GTOL,
        nu=NU,
    )


def get_lcurve_config() -> LCurveConfig:
    return LCurveConfig(
        gamma_min=GAMMA_MIN,
        gamma_max=GAMMA_MAX,
        n_gamma=N_GAMMA,
        E_min=E_MIN,
        E_max=E_MAX,
        Ehat_min=EHAT_MIN,
        Ehat_max=EHAT_MAX,
        max_iter=MAX_ITER,
        ftol=FTOL,
        gtol=GTOL,
        nu=NU,
    )
