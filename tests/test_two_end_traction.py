import unittest

import numpy as np

from fgm_asm.fem_forward import compute_reaction_forces, fem_assemble, forward_solver
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions


class TwoEndTractionTests(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshInfo(4.0, 2.0, 4, 2)
        self.bc = setup_boundary_conditions(self.mesh, 4.0, 2.0, F_tot=2.0, thickness=2.0)

    def test_equal_opposite_edge_resultants_and_minimal_gauge(self):
        x = self.mesh.coord[:, 0]
        left = np.isclose(x, 0.0)
        right = np.isclose(x, 4.0)
        force_x = self.bc["force"][0::2]

        self.assertEqual(len(self.bc["fixdof"]), 3)
        self.assertEqual(self.bc["loading_mode"], "two_end_traction")
        self.assertAlmostEqual(force_x[left].sum(), -1.0)
        self.assertAlmostEqual(force_x[right].sum(), 1.0)
        self.assertAlmostEqual(force_x.sum(), 0.0)
        np.testing.assert_allclose(self.bc["force"][1::2], 0.0)
        np.testing.assert_allclose(force_x[left], [-0.25, -0.5, -0.25])
        np.testing.assert_allclose(force_x[right], [0.25, 0.5, 0.25])

    def test_homogeneous_solution_matches_uniform_stress(self):
        modulus = 100.0
        nu = 0.3
        material = MaterialInfo(nu=nu)
        material.update(np.full(self.mesh.n_nod, modulus))
        fem = fem_assemble(self.mesh, material, self.bc)
        displacement = forward_solver(fem)
        strain_x = self.bc["traction_magnitude"] / modulus

        np.testing.assert_allclose(displacement[0::2], strain_x * self.mesh.X, atol=1e-12)
        np.testing.assert_allclose(displacement[1::2], -nu * strain_x * self.mesh.Y, atol=1e-12)
        np.testing.assert_allclose(
            compute_reaction_forces(fem, displacement)[self.bc["fixdof"]],
            0.0,
            atol=1e-11,
        )

    def test_adjoint_modulus_gradient_matches_finite_difference(self):
        self.mesh.assemble_mass_matrix()
        self.mesh.get_regularization_matrix()
        material = MaterialInfo(nu=0.3)
        true_modulus = 80.0 + 5.0 * self.mesh.X / self.mesh.geo_l
        material.update(true_modulus)
        measured = forward_solver(fem_assemble(self.mesh, material, self.bc))
        trial_modulus = np.full(self.mesh.n_nod, 90.0)

        def state(modulus):
            return _evaluate_inverse_state(
                self.mesh, self.bc, measured, material, 0.0, modulus, iteration=1
            )

        analytical = state(trial_modulus)["grad_tar_E"]
        for node in (0, self.mesh.n_nod // 2, self.mesh.n_nod - 1):
            step = 1e-4
            upper = trial_modulus.copy()
            lower = trial_modulus.copy()
            upper[node] += step
            lower[node] -= step
            numerical = (state(upper)["cost_tar"] - state(lower)["cost_tar"]) / (2 * step)
            np.testing.assert_allclose(analytical[node], numerical, rtol=1e-5, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
