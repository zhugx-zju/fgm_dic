from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from fgm_asm import MeshInfo, load_observation_package, setup_boundary_conditions
from fgm_asm.fem_forward import compute_tensile_end_force, fem_assemble, forward_solver
from fgm_asm.inverse_solver import _compute_data_misfit_state
from fgm_asm.material import MaterialInfo


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def _component_stats(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    print(
        f"{name}: mean={np.mean(values):.6e}, "
        f"rms={_rms(values):.6e}, min={np.min(values):.6e}, max={np.max(values):.6e}"
    )


def main() -> None:
    mesh_info = MeshInfo(cfg.GEO_L, cfg.GEO_H, cfg.NEL_X, cfg.NEL_Y)
    bc_info = setup_boundary_conditions(
        mesh_info=mesh_info,
        geo_l=cfg.GEO_L,
        disp_amp=cfg.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT,
    )
    observation_package = load_observation_package(cfg.get_observation_package_paths())

    material_info = MaterialInfo(nu=cfg.NU)
    material_info.update(np.ones(mesh_info.n_nod), iteration=1)
    fem_info = fem_assemble(mesh_info, material_info, bc_info)
    forward_u = forward_solver(fem_info)
    data_state = _compute_data_misfit_state(
        mesh_info=mesh_info,
        forw_U=forward_u,
        observation_package=observation_package,
    )

    prediction = data_state["measurement_prediction"]
    residual = data_state["residual"]
    force_unit = compute_tensile_end_force(fem_info, forward_u)
    homogeneous_alpha = cfg.TENSILE_END_FORCE / force_unit

    print("=" * 70)
    print("Inverse input diagnostics")
    print("=" * 70)
    print(f"Observation package: {cfg.get_observation_case_name()}")
    print(f"Observation points: {observation_package.x_meas.size}")
    print(f"Observation x range: {np.min(observation_package.x_meas):.6e} to {np.max(observation_package.x_meas):.6e}")
    print(f"Observation y range: {np.min(observation_package.y_meas):.6e} to {np.max(observation_package.y_meas):.6e}")
    print(f"Observation weight sum: {np.sum(observation_package.weight_vector):.6e}")
    print(f"Prescribed right-edge displacement: {cfg.PRESCRIBED_RIGHT_EDGE_DISPLACEMENT:.6e}")
    print(f"Measured tensile force: {cfg.TENSILE_END_FORCE:.6e}")
    print(f"Homogeneous unit-E force: {force_unit:.6e}")
    print(f"Homogeneous force scale alpha: {homogeneous_alpha:.6e}")
    print(
        "Homogeneous baseline weighted RMS residual: "
        f"{data_state['weighted_rms_residual']:.6e}"
    )
    print(
        "Homogeneous baseline relative weighted RMS residual: "
        f"{100.0 * data_state['relative_weighted_rms_residual']:.2f}%"
    )
    print("-" * 70)
    _component_stats("Measured ux", observation_package.ux_meas)
    _component_stats("Predicted ux", prediction[0::2])
    _component_stats("Residual ux", residual[0::2])
    _component_stats("Measured uy", observation_package.uy_meas)
    _component_stats("Predicted uy", prediction[1::2])
    _component_stats("Residual uy", residual[1::2])


if __name__ == "__main__":
    main()
