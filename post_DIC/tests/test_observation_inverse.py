import tempfile
import unittest
from pathlib import Path

import numpy as np

from fgm_asm import (
    MeshInfo,
    MaterialInfo,
    build_observation_package,
    compute_tensile_end_force,
    fem_assemble,
    forward_solver,
    lbfgs_inverse_solver_scipy,
    load_observation_package,
    save_observation_package,
    setup_boundary_conditions,
)


class ObservationPackageTests(unittest.TestCase):
    def test_save_and_load_round_trip_preserves_observation_data(self):
        mesh_info = MeshInfo(geo_l=2.0, geo_h=1.0, nel_x=2, nel_y=1)
        x_meas = np.array([0.0, 0.5, 1.5, 2.0], dtype=float)
        y_meas = np.array([0.0, 0.25, 0.75, 1.0], dtype=float)
        ux_meas = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
        uy_meas = np.array([0.0, -0.1, -0.2, -0.3], dtype=float)
        point_weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        package = build_observation_package(
            mesh_info=mesh_info,
            x_meas=x_meas,
            y_meas=y_meas,
            ux_meas=ux_meas,
            uy_meas=uy_meas,
            point_weights=point_weights,
            sampling_mode="uniform_subgrid",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_observation_package(
                output_dir=Path(tmpdir),
                case_name="roundtrip_uniform_subgrid_observation",
                package=package,
            )
            loaded = load_observation_package(saved_paths)

        np.testing.assert_allclose(loaded.x_meas, x_meas)
        np.testing.assert_allclose(loaded.y_meas, y_meas)
        np.testing.assert_allclose(loaded.ux_meas, ux_meas)
        np.testing.assert_allclose(loaded.uy_meas, uy_meas)
        np.testing.assert_allclose(loaded.point_weights, point_weights)
        np.testing.assert_allclose(loaded.u_measured, package.u_measured)
        np.testing.assert_allclose(loaded.weight_vector, package.weight_vector)
        np.testing.assert_allclose((loaded.H @ np.arange(mesh_info.n_dof, dtype=float)),
                                   (package.H @ np.arange(mesh_info.n_dof, dtype=float)))
        self.assertEqual(loaded.sampling_mode, "uniform_subgrid")


class InverseSolverObservationTests(unittest.TestCase):
    def test_solver_accepts_pointwise_observation_package(self):
        mesh_info = MeshInfo(geo_l=2.0, geo_h=1.0, nel_x=2, nel_y=1)
        bc_info = setup_boundary_conditions(mesh_info=mesh_info, geo_l=2.0, disp_amp=0.1)

        material_info = MaterialInfo(nu=0.3)
        material_info.update(np.ones(mesh_info.n_nod), iteration=1)
        fem_info = fem_assemble(mesh_info, material_info, bc_info)
        u_true = forward_solver(fem_info)
        tensile_end_force = compute_tensile_end_force(fem_info, u_true)

        package = build_observation_package(
            mesh_info=mesh_info,
            x_meas=mesh_info.X,
            y_meas=mesh_info.Y,
            ux_meas=u_true[0::2],
            uy_meas=u_true[1::2],
            point_weights=np.ones(mesh_info.n_nod, dtype=float),
            sampling_mode="manual",
        )

        results = lbfgs_inverse_solver_scipy(
            mesh_info=mesh_info,
            bc_info=bc_info,
            observation_package=package,
            tensile_end_force=tensile_end_force,
            raw_init=np.zeros(mesh_info.n_nod),
            gamma=1e-6,
            max_iter=5,
            ftol=1e-18,
            gtol=1e-12,
            nu=0.3,
        )

        self.assertEqual(results["data_mode"], "observation_operator:manual")
        self.assertEqual(results["n_measurements"], mesh_info.n_nod)
        self.assertLess(results["final_cost"], 1e-12)
        self.assertLess(results["weighted_rms_residual"], 1e-12)
        self.assertLess(results["relative_weighted_rms_residual"], 1e-12)
        np.testing.assert_allclose(results["measurement_prediction_final"], package.u_measured, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
