"""
Shared workflow helpers for solver entry flows.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .observation import load_observation_package
from .l_curve import find_optimal_gamma_lcurve, plot_lcurve_results
from .inverse_solver import lbfgs_inverse_solver_scipy
from .mesh import MeshInfo, setup_boundary_conditions
from .results_io import save_inverse_results, save_lcurve_analysis, write_python_config_snapshot
from .visualization import (
    plot_gradient_field,
    plot_iteration_history,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
)
def run_real_data_inverse_lcurve(config_module) -> dict[str, Any]:
    """
    Run the real-data inverse workflow using configuration from ``script/config.py``.
    """
    lcurve_config = config_module.get_lcurve_config()

    print("=" * 70)
    print("Inverse Problem Solver with L-curve Analysis")
    print("=" * 70)

    mesh_info = MeshInfo(config_module.GEO_L, config_module.GEO_H, config_module.NEL_X, config_module.NEL_Y)
    bc_info = setup_boundary_conditions(
        mesh_info=mesh_info,
        geo_l=config_module.GEO_L,
        disp_amp=config_module.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT,
    )
    observation_paths = config_module.get_observation_package_paths()
    missing_paths = [path for path in observation_paths.values() if not path.exists()]
    if missing_paths:
        missing_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Prepared observation package not found. Missing files:\n"
            f"{missing_str}\n"
            "Run script/run_preprocess_dic.py first."
        )
    observation_package = load_observation_package(observation_paths)

    print(f"Case name: {config_module.CASE_NAME}")
    print(f"Observation package: {config_module.get_observation_case_name()}")
    print(f"Observation sampling mode: {observation_package.sampling_mode}")
    print(f"Observation points: {observation_package.x_meas.size}")
    print(f"Measured force: {config_module.TENSILE_END_FORCE:.6e}")
    print(f"Prescribed right-edge displacement: {config_module.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT:.6e}")

    case_results_dir = config_module.get_case_results_dir()
    case_results_dir.mkdir(parents=True, exist_ok=True)

    start_time_lcurve = time.time()
    gamma_optimal, lcurve_results = find_optimal_gamma_lcurve(
        mesh_info=mesh_info,
        bc_info=bc_info,
        observation_package=observation_package,
        tensile_end_force=config_module.TENSILE_END_FORCE,
        nu=lcurve_config.nu,
        gamma_min=lcurve_config.gamma_min,
        gamma_max=lcurve_config.gamma_max,
        n_gamma=lcurve_config.n_gamma,
        E_max=lcurve_config.E_max,
        max_iter=lcurve_config.max_iter,
        ftol=lcurve_config.ftol,
        gtol=lcurve_config.gtol,
        ehat_min=lcurve_config.Ehat_min,
        ehat_max=lcurve_config.Ehat_max,
    )
    elapsed_time_lcurve = time.time() - start_time_lcurve

    optimal_idx = lcurve_results["optimal_idx"]
    scan_results = lcurve_results["all_results"][optimal_idx]
    scan_E_reconstructed = lcurve_results["E_solutions"][optimal_idx]

    print("\n" + "=" * 70)
    print("Re-running inverse solve with selected gamma from default initialization")
    print("=" * 70)

    start_time_rerun = time.time()
    results = lbfgs_inverse_solver_scipy(
        mesh_info=mesh_info,
        bc_info=bc_info,
        observation_package=observation_package,
        tensile_end_force=config_module.TENSILE_END_FORCE,
        raw_init=None,
        gamma=gamma_optimal,
        E_max=lcurve_config.E_max,
        max_iter=lcurve_config.max_iter,
        ftol=lcurve_config.ftol,
        gtol=lcurve_config.gtol,
        nu=lcurve_config.nu,
        ehat_min=lcurve_config.Ehat_min,
        ehat_max=lcurve_config.Ehat_max,
    )
    elapsed_time_rerun = time.time() - start_time_rerun

    comparison_summary = {
        "scan_final_cost": float(scan_results["final_cost"]),
        "rerun_final_cost": float(results["final_cost"]),
        "modulus_diff_l2": float(np.linalg.norm(results["E_final"] - scan_E_reconstructed)),
        "modulus_diff_rel_l2": float(
            np.linalg.norm(results["E_final"] - scan_E_reconstructed) /
            (np.linalg.norm(scan_E_reconstructed) + 1e-15)
        ),
    }

    print(f"\nOptimal gamma: {gamma_optimal:.6e}")
    print(f"Final cost: {results['final_cost']:.6e}")
    print(f"Weighted RMS residual: {results['weighted_rms_residual']:.6e}")
    print(f"Relative weighted RMS residual: {100.0 * results['relative_weighted_rms_residual']:.2f}%")
    print(f"Iterations: {results['n_iterations']}")
    print(f"L-curve time: {elapsed_time_lcurve:.2f} s")
    print(f"Final rerun time: {elapsed_time_rerun:.2f} s")

    lcurve_save_path = save_lcurve_analysis(
        lcurve_results,
        case_results_dir,
        extra_data={
            "case_name": config_module.CASE_NAME,
            "selection_method": "lcurve_max_curvature",
            "gamma_optimal": gamma_optimal,
            "optimal_idx": optimal_idx,
        },
    )
    print(f"  L-curve analysis saved to {lcurve_save_path}")

    inverse_save_path = save_inverse_results(
        results=results,
        errors=None,
        E_true=None,
        output_dir=case_results_dir,
        extra_data={
            "case_name": config_module.CASE_NAME,
            "mesh_info": mesh_info,
            "bc_info": bc_info,
            "observation_sampling_mode": observation_package.sampling_mode,
            "n_measurements": int(observation_package.x_meas.size),
            "observation_package_paths": {key: str(path) for key, path in observation_paths.items()},
            "gamma_used": gamma_optimal,
            "elapsed_time_lcurve_seconds": elapsed_time_lcurve,
            "elapsed_time_total_seconds": elapsed_time_rerun,
            "result_source": "real_dic_final_rerun_after_lcurve",
            "gamma_selection_method": "lcurve_max_curvature",
            "lcurve_analysis_file": lcurve_save_path.name,
            "scan_optimal": {
                "gamma_used": gamma_optimal,
                "optimal_idx": optimal_idx,
                "result_source": "lcurve_scan_optimal",
                "E_reconstructed": scan_E_reconstructed,
                "errors": None,
                "results": scan_results,
            },
            "comparison_summary": comparison_summary,
        },
    )
    print(f"  Inverse results saved to {inverse_save_path}")

    config_snapshot_path = write_python_config_snapshot(
        case_results_dir,
        [
            (
                "Run Metadata",
                {
                    "WORKFLOW": "inverse_l_curve",
                    "CASE_NAME": config_module.CASE_NAME,
                    "RAW_DIC_CSV": str(config_module.get_raw_dic_csv_path()),
                    "CORRECTED_DIC_CSV": str(config_module.get_corrected_dic_csv_path()) if config_module.USE_CORRECTED_CSV else None,
                    "PREPARED_DIC_DIR": str(config_module.get_prepared_dic_dir()),
                    "RESULTS_DIR": str(case_results_dir),
                    "TENSILE_END_FORCE": float(config_module.TENSILE_END_FORCE),
                    "PRESCRIBED_RIGHT_EDGE_DISPLACEMENT": float(config_module.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT),
                    "GAMMA_USED": float(gamma_optimal),
                    "SCAN_OPTIMAL_FINAL_COST": float(scan_results["final_cost"]),
                    "FINAL_RERUN_FINAL_COST": float(results["final_cost"]),
                    "CONVERGED": bool(results["converged"]),
                    "MESSAGE": str(results["message"]),
                    "N_ITERATIONS": int(results["n_iterations"]),
                },
            ),
            (
                "Geometry And Mesh",
                {
                    "GEO_L": config_module.GEO_L,
                    "GEO_H": config_module.GEO_H,
                    "NEL_X": config_module.NEL_X,
                    "NEL_Y": config_module.NEL_Y,
                },
            ),
            (
                "Inverse Configuration",
                {
                    "NU": config_module.NU,
                    "E_MIN": config_module.E_MIN,
                    "E_MAX": config_module.E_MAX,
                    "EHAT_MIN": config_module.EHAT_MIN,
                    "EHAT_MAX": config_module.EHAT_MAX,
                    "GAMMA": config_module.GAMMA,
                    "GAMMA_MIN": config_module.GAMMA_MIN,
                    "GAMMA_MAX": config_module.GAMMA_MAX,
                    "N_GAMMA": config_module.N_GAMMA,
                    "MAX_ITER": config_module.MAX_ITER,
                    "FTOL": config_module.FTOL,
                    "GTOL": config_module.GTOL,
                },
            ),
            (
                "DIC Configuration",
                {
                    "USE_CORRECTED_CSV": bool(config_module.USE_CORRECTED_CSV),
                    "FIXED_POINT": config_module.FIXED_POINT,
                    "OBSERVATION_SAMPLING_MODE": observation_package.sampling_mode,
                    "OBSERVATION_UNIFORM_SUBGRID_NX": int(config_module.OBSERVATION_UNIFORM_SUBGRID_NX),
                    "OBSERVATION_UNIFORM_SUBGRID_NY": int(config_module.OBSERVATION_UNIFORM_SUBGRID_NY),
                    "OBSERVATION_UX_COMPONENT_WEIGHT": float(config_module.OBSERVATION_UX_COMPONENT_WEIGHT),
                    "OBSERVATION_UY_COMPONENT_WEIGHT": float(config_module.OBSERVATION_UY_COMPONENT_WEIGHT),
                    "TRUE_MODULUS_FILE": config_module.TRUE_MODULUS_FILE,
                },
            ),
        ],
    )
    print(f"  Config snapshot saved to {config_snapshot_path}")

    plot_lcurve_results(lcurve_results, save_path=case_results_dir)
    plot_reconstruction_results(
        mesh_info=mesh_info,
        E_true=None,
        E_reconstructed=results["E_final"],
        errors=None,
        noise_level=0.0,
        save_path=case_results_dir,
        filename_stem="reconstruction_results",
    )
    plot_iteration_history(
        results,
        save_path=case_results_dir,
        noise_level=0.0,
        filename_stem="iteration_history",
    )
    plot_gradient_field(
        mesh_info,
        results,
        noise_level=0.0,
        save_path=case_results_dir,
        filename_stem="gradient_field",
    )
    plot_reconstruction_comparison(
        mesh_info=mesh_info,
        E_true=None,
        scan_E_reconstructed=scan_E_reconstructed,
        scan_errors=None,
        rerun_E_reconstructed=results["E_final"],
        rerun_errors=None,
        noise_level=0.0,
        save_path=case_results_dir,
    )

    print("\n" + "=" * 70)
    print("Inverse problem solved successfully with real DIC data.")
    print(f"Results saved under {case_results_dir}")
    print("=" * 70)
    plt.show()

    return {
        "mesh_info": mesh_info,
        "bc_info": bc_info,
        "observation_package": observation_package,
        "gamma_optimal": gamma_optimal,
        "lcurve_results": lcurve_results,
        "results": results,
        "output_dir": case_results_dir,
        "inverse_results_path": inverse_save_path,
        "lcurve_path": lcurve_save_path,
        "comparison_summary": comparison_summary,
    }
