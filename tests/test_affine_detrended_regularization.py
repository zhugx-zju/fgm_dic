import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from archive.legacy_workflows import config

from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.l_curve import find_optimal_gamma_lcurve
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.regularization import (
    get_affine_detrended_gradient,
    get_affine_detrended_regularization,
)


class AffineDetrendedRegularizationTests(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshInfo(4.0, 2.0, 4, 2)
        self.material = MaterialInfo(nu=0.3)

    def test_any_affine_2d_field_has_zero_penalty_and_gradient(self):
        modulus = 2.0 + 0.2 * self.mesh.X - 0.1 * self.mesh.Y
        self.material.update(modulus)
        self.assertAlmostEqual(
            get_affine_detrended_regularization(self.mesh, self.material), 0.0, places=12
        )
        np.testing.assert_allclose(
            get_affine_detrended_gradient(self.mesh, self.material), 0.0, atol=1e-12
        )

    def test_nonaffine_gradient_matches_finite_difference(self):
        modulus = 2.0 + 0.2 * self.mesh.X + 0.03 * self.mesh.X**2
        self.material.update(modulus)
        analytical = get_affine_detrended_gradient(self.mesh, self.material)
        self.assertGreater(
            get_affine_detrended_regularization(self.mesh, self.material), 0.0
        )
        for node in (0, self.mesh.n_nod // 2, self.mesh.n_nod - 1):
            step = 1e-5
            upper = modulus.copy()
            lower = modulus.copy()
            upper[node] += step
            lower[node] -= step
            self.material.update(upper)
            upper_value = get_affine_detrended_regularization(self.mesh, self.material)
            self.material.update(lower)
            lower_value = get_affine_detrended_regularization(self.mesh, self.material)
            numerical = (upper_value - lower_value) / (2 * step)
            np.testing.assert_allclose(analytical[node], numerical, rtol=1e-7, atol=1e-9)

    def test_full_adjoint_objective_gradient_matches_log_modulus_difference(self):
        boundary = setup_boundary_conditions(self.mesh, 4.0, 2.0, 2.0, thickness=2.0)
        true_modulus = 2.0 + 0.2 * self.mesh.X
        self.material.update(true_modulus)
        measured = forward_solver(fem_assemble(self.mesh, self.material, boundary))
        self.mesh.assemble_mass_matrix()
        self.mesh.get_regularization_matrix()
        trial_log_modulus = np.log(2.2 + 0.1 * self.mesh.X + 0.02 * self.mesh.Y**2)

        def evaluate(log_modulus):
            return _evaluate_inverse_state(
                self.mesh, boundary, measured, self.material, 1e-3,
                np.exp(log_modulus), iteration=1,
                regularization_mode='affine_detrended',
            )

        analytical = evaluate(trial_log_modulus)['grad_m']
        for node in (0, self.mesh.n_nod // 2, self.mesh.n_nod - 1):
            step = 1e-5
            upper = trial_log_modulus.copy()
            lower = trial_log_modulus.copy()
            upper[node] += step
            lower[node] -= step
            numerical = (evaluate(upper)['objective'] - evaluate(lower)['objective']) / (2 * step)
            np.testing.assert_allclose(analytical[node], numerical, rtol=1e-6, atol=1e-9)

    def test_flat_lcurve_does_not_claim_optimal_gamma(self):
        identical_result = {
            'E_final': np.ones(self.mesh.n_nod),
            'residual_norm': 1e-5,
            'regularization_norm': 1e-5,
            'converged': True,
            'n_iterations': 0,
        }
        with patch('fgm_asm.l_curve.lbfgs_inverse_solver_scipy', return_value=identical_result), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'no resolved corner'):
                find_optimal_gamma_lcurve(
                    self.mesh, {}, np.zeros(self.mesh.n_dof), config.get_forward_config(),
                    n_gamma=5, regularization_mode='affine_detrended',
                )


if __name__ == "__main__":
    unittest.main()
