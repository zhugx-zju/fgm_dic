from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from fgm_asm.case_inverse import case_inverse_solver_scipy
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.observation import load_observation_package
from fgm_asm.visualization import plot_gradient_field, plot_iteration_history, plot_reconstruction_results


CASE_GAMMA = 1e2


def get_case_observation_package_paths() -> dict[str, Path]:
    prepared_dir = cfg.get_prepared_dic_dir()
    case_name = f"{cfg.get_case_prefix()}_case_{cfg.OBSERVATION_SAMPLING_MODE}_observation"
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


def main() -> None:
    results_dir = PROJECT_ROOT / "results_case"
    results_dir.mkdir(parents=True, exist_ok=True)

    mesh_info = MeshInfo(cfg.CASE_DOMAIN_GEO_L, cfg.CASE_DOMAIN_GEO_H, cfg.NEL_X, cfg.NEL_Y)
    bc_info = setup_boundary_conditions(
        mesh_info=mesh_info,
        geo_l=cfg.CASE_DOMAIN_GEO_L,
        disp_amp=cfg.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT,
    )
    observation_paths = get_case_observation_package_paths()
    missing_paths = [path for path in observation_paths.values() if not path.exists()]
    if missing_paths:
        missing_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Prepared CASE observation package not found. Missing files:\n"
            f"{missing_str}\n"
            "Run script/run_preprocess_case_dic.py first."
        )
    observation_package = load_observation_package(observation_paths)

    results = case_inverse_solver_scipy(
        mesh_info=mesh_info,
        bc_info=bc_info,
        observation_package=observation_package,
        tensile_end_force=cfg.TENSILE_END_FORCE,
        gamma=CASE_GAMMA,
        max_iter=cfg.MAX_ITER,
        ftol=cfg.FTOL,
        gtol=cfg.GTOL,
        nu=cfg.NU,
        ehat_min=cfg.EHAT_MIN,
        ehat_max=cfg.EHAT_MAX,
    )

    save_data = {
        "results": results,
        "mesh_info": mesh_info,
        "bc_info": bc_info,
        "observation_package_paths": observation_paths,
        "gamma_used": CASE_GAMMA,
        "nu": cfg.NU,
        "method": "strong_stabilized_CASE",
    }
    with (results_dir / "inverse_results.pkl").open("wb") as file:
        pickle.dump(save_data, file)

    plot_reconstruction_results(mesh_info, None, results["E_final"], None, 0.0, results_dir)
    plot_iteration_history(results, save_path=results_dir)
    try:
        plot_gradient_field(mesh_info, results, save_path=results_dir)
    except ValueError as exc:
        print(f"Skipped CASE gradient plot: {exc}")

    print(f"CASE results saved under {results_dir}")


if __name__ == "__main__":
    main()
