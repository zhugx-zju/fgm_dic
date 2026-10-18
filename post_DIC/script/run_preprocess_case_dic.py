from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from fgm_asm import MeshInfo, prepare_observation_package_from_grids, save_observation_package
from lib.dic_preprocessing import (
    compute_preprocessed_grid_summary,
    plot_preprocessed_displacement_fields,
    preprocess_structured_dic,
    save_preprocessed_arrays,
)


def get_case_grid_name() -> str:
    return f"{cfg.get_case_prefix()}_case_grid_preprocessed"


def get_case_observation_name() -> str:
    return f"{cfg.get_case_prefix()}_case_{cfg.OBSERVATION_SAMPLING_MODE}_observation"


def get_case_observation_package_paths() -> dict[str, Path]:
    prepared_dir = cfg.get_prepared_dic_dir()
    case_name = get_case_observation_name()
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


def embed_case_dic_grid(arrays: dict[str, object]) -> dict[str, object]:
    embedded = dict(arrays)
    embedded["X_grid"] = np.asarray(arrays["X_grid"], dtype=float) + cfg.CASE_DIC_X_OFFSET
    embedded["Y_grid"] = np.asarray(arrays["Y_grid"], dtype=float) + cfg.CASE_DIC_Y_OFFSET

    x_min = float(np.nanmin(embedded["X_grid"]))
    x_max = float(np.nanmax(embedded["X_grid"]))
    y_min = float(np.nanmin(embedded["Y_grid"]))
    y_max = float(np.nanmax(embedded["Y_grid"]))
    tol = 1e-9 * max(cfg.CASE_DOMAIN_GEO_L, cfg.CASE_DOMAIN_GEO_H, 1.0)
    if x_min < -tol or x_max > cfg.CASE_DOMAIN_GEO_L + tol:
        raise ValueError("Embedded CASE DIC x-grid lies outside the CASE physical domain.")
    if y_min < -tol or y_max > cfg.CASE_DOMAIN_GEO_H + tol:
        raise ValueError("Embedded CASE DIC y-grid lies outside the CASE physical domain.")

    metadata = dict(arrays.get("metadata", {}))
    metadata["case_embedding"] = {
        "domain_geo_l": float(cfg.CASE_DOMAIN_GEO_L),
        "domain_geo_h": float(cfg.CASE_DOMAIN_GEO_H),
        "dic_x_offset": float(cfg.CASE_DIC_X_OFFSET),
        "dic_y_offset": float(cfg.CASE_DIC_Y_OFFSET),
    }
    embedded["metadata"] = metadata
    return embedded


def main() -> None:
    prepared_dir = cfg.get_prepared_dic_dir()
    arrays = preprocess_structured_dic(
        csv_path=cfg.get_grid_dic_csv_path(),
        fixed_point=None,
        fixed_end_side=cfg.PREPROCESS_FIXED_END_SIDE,
        strip_width_ratio=cfg.PREPROCESS_STRIP_WIDTH_RATIO,
        strip_y_margin_ratio=cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO,
        displacement_signs=(cfg.PREPROCESS_UX_SIGN, cfg.PREPROCESS_UY_SIGN),
        orientation_operations=cfg.PREPROCESS_ORIENTATION_OPERATIONS,
    )
    arrays = embed_case_dic_grid(arrays)
    saved_grid_paths = save_preprocessed_arrays(prepared_dir, get_case_grid_name(), arrays)
    plot_preprocessed_displacement_fields(prepared_dir, get_case_grid_name(), arrays, cfg.PREPROCESS_CMAP)

    summary = compute_preprocessed_grid_summary(arrays)
    mesh_info = MeshInfo(cfg.CASE_DOMAIN_GEO_L, cfg.CASE_DOMAIN_GEO_H, cfg.NEL_X, cfg.NEL_Y)
    observation_package, sampled = prepare_observation_package_from_grids(
        mesh_info=mesh_info,
        x_grid=arrays["X_grid"],
        y_grid=arrays["Y_grid"],
        ux_grid=arrays["UX_grid"],
        uy_grid=arrays["UY_grid"],
        sampling_mode=cfg.OBSERVATION_SAMPLING_MODE,
        uniform_subgrid_nx=cfg.OBSERVATION_UNIFORM_SUBGRID_NX,
        uniform_subgrid_ny=cfg.OBSERVATION_UNIFORM_SUBGRID_NY,
        ux_component_weight=cfg.OBSERVATION_UX_COMPONENT_WEIGHT,
        uy_component_weight=cfg.OBSERVATION_UY_COMPONENT_WEIGHT,
        component_weight_mode=cfg.OBSERVATION_COMPONENT_WEIGHT_MODE,
        local_max_epsilon=cfg.OBSERVATION_LOCAL_MAX_EPSILON,
        collapse_duplicates=cfg.OBSERVATION_COLLAPSE_DUPLICATES,
    )
    saved_observation_paths = save_observation_package(
        prepared_dir,
        get_case_observation_name(),
        observation_package,
    )

    print("=" * 70)
    print("CASE DIC preprocessing complete")
    print("=" * 70)
    print(f"source CSV: {cfg.get_grid_dic_csv_path()}")
    print("case preprocessing: fixed-end rigid removal only, no fixed-point reference")
    print(f"fixed-end side: {cfg.PREPROCESS_FIXED_END_SIDE}")
    print(f"strip width ratio: {cfg.PREPROCESS_STRIP_WIDTH_RATIO}")
    print(f"strip y-margin ratio: {cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO}")
    print(f"CASE domain GEO_L: {cfg.CASE_DOMAIN_GEO_L:.12f}")
    print(f"CASE domain GEO_H: {cfg.CASE_DOMAIN_GEO_H:.12f}")
    print(f"DIC x offset in CASE domain: {cfg.CASE_DIC_X_OFFSET:.12f}")
    print(f"DIC y offset in CASE domain: {cfg.CASE_DIC_Y_OFFSET:.12f}")
    print(f"grid files: {list(saved_grid_paths.values())}")
    print(f"observation files: {list(saved_observation_paths.values())}")
    print(f"DIC ROI span L: {summary['geo_l']:.12f}")
    print(f"DIC ROI span H: {summary['geo_h']:.12f}")
    print(f"observation points: {observation_package.x_meas.size}")
    if "n_duplicate_points_merged" in sampled:
        print(f"merged duplicate points: {int(sampled['n_duplicate_points_merged'])}")


if __name__ == "__main__":
    main()
