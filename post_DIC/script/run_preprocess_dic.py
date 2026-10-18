from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from fgm_asm import MeshInfo, prepare_observation_package_from_grids, save_observation_package
from lib.config_utils import ensure_output_dirs, get_project_paths
from lib.dic_preprocessing import (
    compute_preprocessed_boundary_diagnostics,
    compute_preprocessed_grid_summary,
    plot_preprocessed_displacement_fields,
    preprocess_structured_dic,
    save_preprocessed_arrays,
)


paths = get_project_paths()
ensure_output_dirs(paths)

print("=" * 70)
print("Preparing structured DIC preprocessing outputs")
print("=" * 70)

grid_arrays = None
jobs = [
    ("grid", cfg.get_grid_dic_csv_path(), cfg.get_grid_case_name()),
    # ("pixel", cfg.get_pixel_dic_csv_path(), cfg.get_pixel_case_name()),
]

prepared_dir = cfg.get_prepared_dic_dir()

for label, csv_path, case_name in jobs:
    print(f"[{label}] source CSV: {csv_path}")
    arrays = preprocess_structured_dic(
        csv_path=csv_path,
        fixed_point=cfg.PREPROCESS_FIXED_POINT,
        fixed_end_side=cfg.PREPROCESS_FIXED_END_SIDE,
        strip_width_ratio=cfg.PREPROCESS_STRIP_WIDTH_RATIO,
        strip_y_margin_ratio=cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO,
        displacement_signs=(cfg.PREPROCESS_UX_SIGN, cfg.PREPROCESS_UY_SIGN),
        orientation_operations=cfg.PREPROCESS_ORIENTATION_OPERATIONS,
    )
    saved_paths = save_preprocessed_arrays(prepared_dir, case_name, arrays)
    plot_preprocessed_displacement_fields(prepared_dir, case_name, arrays, cfg.PREPROCESS_CMAP)
    print(f"[{label}] grid shape: {arrays['UX_grid'].shape}")
    print(f"[{label}] saved files: {list(saved_paths.values())}")
    print(f"[{label}] fixed-point: {arrays['metadata']['fixed_point_reference']['fixed_point']}")
    print(f"[{label}] rigid strip side: {arrays['metadata']['rigid']['strip_side']}")
    boundary = compute_preprocessed_boundary_diagnostics(arrays)
    print(
        "[{label}] FE-left ux mean/min/max: "
        "{mean:.6e} / {min:.6e} / {max:.6e}".format(
            label=label,
            mean=boundary["ux_left"]["mean"],
            min=boundary["ux_left"]["min"],
            max=boundary["ux_left"]["max"],
        )
    )
    print(
        "[{label}] FE-right ux mean/min/max: "
        "{mean:.6e} / {min:.6e} / {max:.6e}".format(
            label=label,
            mean=boundary["ux_right"]["mean"],
            min=boundary["ux_right"]["min"],
            max=boundary["ux_right"]["max"],
        )
    )
    print(
        f"[{label}] FE-bottom-left (ux, uy): "
        f"({boundary['bottom_left']['ux']:.6e}, {boundary['bottom_left']['uy']:.6e})"
    )
    left_reference_ratio = abs(boundary["ux_left"]["mean"]) / max(abs(boundary["ux_right"]["mean"]), 1e-15)
    if left_reference_ratio > 0.10:
        print(
            f"[{label}] WARNING: FE-left mean ux is still {left_reference_ratio:.2%} "
            "of the loading-side displacement. Check the preprocessing reference."
        )
    if label == "grid":
        grid_arrays = arrays

if grid_arrays is None:
    raise RuntimeError("Grid preprocessing output is required to update geometry and boundary displacement.")

summary = compute_preprocessed_grid_summary(grid_arrays)
geo_l = summary["geo_l"]
geo_h = summary["geo_h"]
prescribed_right_edge_displacement = summary["prescribed_right_edge_displacement"]

if cfg.PREPROCESS_UPDATE_CONFIG:
    config_path = PROJECT_ROOT / "script" / "config.py"
    config_text = config_path.read_text(encoding="utf-8")
    config_text = re.sub(r"^GEO_L = .*$", f"GEO_L = {geo_l!r}", config_text, flags=re.MULTILINE)
    config_text = re.sub(r"^GEO_H = .*$", f"GEO_H = {geo_h!r}", config_text, flags=re.MULTILINE)
    config_text = re.sub(
        r"^PRESCRIBED_RIGHT_EDGE_DISPLACEMENT = .*$",
        f"PRESCRIBED_RIGHT_EDGE_DISPLACEMENT = {prescribed_right_edge_displacement!r}",
        config_text,
        flags=re.MULTILINE,
    )
    config_path.write_text(config_text, encoding="utf-8")

    print(f"[grid] updated GEO_L in config.py to: {geo_l:.12f}")
    print(f"[grid] updated GEO_H in config.py to: {geo_h:.12f}")
    print(
        "[grid] updated PRESCRIBED_RIGHT_EDGE_DISPLACEMENT in config.py to: "
        f"{prescribed_right_edge_displacement:.12f}"
    )
else:
    print(f"[grid] inferred GEO_L: {geo_l:.12f}")
    print(f"[grid] inferred GEO_H: {geo_h:.12f}")
    print(
        "[grid] inferred PRESCRIBED_RIGHT_EDGE_DISPLACEMENT: "
        f"{prescribed_right_edge_displacement:.12f}"
    )
    print("[grid] config.py was not rewritten; set PREPROCESS_UPDATE_CONFIG = True to enable that.")

observation_mesh = MeshInfo(geo_l, geo_h, cfg.NEL_X, cfg.NEL_Y)
observation_package, sampled = prepare_observation_package_from_grids(
    mesh_info=observation_mesh,
    x_grid=grid_arrays["X_grid"],
    y_grid=grid_arrays["Y_grid"],
    ux_grid=grid_arrays["UX_grid"],
    uy_grid=grid_arrays["UY_grid"],
    sampling_mode=cfg.OBSERVATION_SAMPLING_MODE,
    uniform_subgrid_nx=cfg.OBSERVATION_UNIFORM_SUBGRID_NX,
    uniform_subgrid_ny=cfg.OBSERVATION_UNIFORM_SUBGRID_NY,
    ux_component_weight=cfg.OBSERVATION_UX_COMPONENT_WEIGHT,
    uy_component_weight=cfg.OBSERVATION_UY_COMPONENT_WEIGHT,
    component_weight_mode=cfg.OBSERVATION_COMPONENT_WEIGHT_MODE,
    local_max_epsilon=cfg.OBSERVATION_LOCAL_MAX_EPSILON,
    collapse_duplicates=cfg.OBSERVATION_COLLAPSE_DUPLICATES,
)
observation_saved_paths = save_observation_package(
    prepared_dir,
    cfg.get_observation_case_name(),
    observation_package,
)

print(f"[observation] sampling mode: {cfg.OBSERVATION_SAMPLING_MODE}")
print(f"[observation] measurement points: {observation_package.x_meas.size}")
print(f"[observation] H shape: {observation_package.H.shape}")
print(f"[observation] saved files: {list(observation_saved_paths.values())}")
if "n_duplicate_points_merged" in sampled:
    print(f"[observation] merged duplicate points: {int(sampled['n_duplicate_points_merged'])}")
