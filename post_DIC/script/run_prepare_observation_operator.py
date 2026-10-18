from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgm_asm import (
    MeshInfo,
    load_preprocessed_dic_grids,
    prepare_observation_package_from_grids,
    save_observation_package,
)
from script import config


def main() -> None:
    mesh_info = MeshInfo(config.GEO_L, config.GEO_H, config.NEL_X, config.NEL_Y)
    grids = load_preprocessed_dic_grids(config.get_prepared_grid_paths())

    package, sampled = prepare_observation_package_from_grids(
        mesh_info=mesh_info,
        x_grid=grids["x_grid"],
        y_grid=grids["y_grid"],
        ux_grid=grids["ux_grid"],
        uy_grid=grids["uy_grid"],
        sampling_mode=config.OBSERVATION_SAMPLING_MODE,
        uniform_subgrid_nx=config.OBSERVATION_UNIFORM_SUBGRID_NX,
        uniform_subgrid_ny=config.OBSERVATION_UNIFORM_SUBGRID_NY,
        ux_component_weight=config.OBSERVATION_UX_COMPONENT_WEIGHT,
        uy_component_weight=config.OBSERVATION_UY_COMPONENT_WEIGHT,
        collapse_duplicates=config.OBSERVATION_COLLAPSE_DUPLICATES,
    )

    output_name = config.get_observation_case_name()
    saved_paths = save_observation_package(
        output_dir=config.get_prepared_dic_dir(),
        case_name=output_name,
        package=package,
    )

    print("=" * 70)
    print("Prepared observation operator package")
    print("=" * 70)
    print(f"Sampling mode: {config.OBSERVATION_SAMPLING_MODE}")
    print(f"Measurement points: {package.x_meas.size}")
    print(f"H shape: {package.H.shape}")
    print(f"W shape: {package.W.shape}")
    print(f"Sum of point weights: {np.sum(package.point_weights):.6e}")
    print(f"Saved H: {saved_paths['H']}")
    print(f"Saved W: {saved_paths['W']}")
    print(f"Saved u_measured: {saved_paths['u_measured']}")

    if "n_duplicate_points_merged" in sampled:
        print(f"Merged duplicate DIC points: {int(sampled['n_duplicate_points_merged'])}")


if __name__ == "__main__":
    main()
