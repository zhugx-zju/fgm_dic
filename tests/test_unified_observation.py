import tempfile
import unittest
from pathlib import Path

import numpy as np

from fgm_asm import MeshInfo, MaterialInfo, lbfgs_inverse_solver_scipy
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.load_case import evaluate_observation
from fgm_asm.observation import (
    build_observation_operator, load_observation_package, save_observation_package,
)
from fgm_asm.research import make_observation_case, partition_modulus, solve_forward
from fgm_asm.tgv import HuberTGV


class UnifiedObservationTests(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshInfo(4., 2., 4, 2)
        rng = np.random.default_rng(54)
        self.xy = rng.uniform([0., 0.], [4., 2.], size=(35, 2))
        self.cases = []
        for index, force in enumerate((2., 3.)):
            u, _ = solve_forward(self.mesh, force, 2., .3,
                                 2. + .1*self.mesh.X + .05*self.mesh.Y)
            uv = (build_observation_operator(self.mesh, *self.xy.T) @ u).reshape(-1, 2)
            uv += [.3, -.1] + .002 * np.column_stack([-self.xy[:, 1], self.xy[:, 0]])
            case = make_observation_case(self.mesh, str(index), force, 2., self.xy, uv,
                                         point_weights=np.linspace(.1, 2., len(uv)),
                                         noise_std=np.array([.5, 1.]))
            self.cases.append(case)

    def test_rigid_fit_removes_only_rigid_modes(self):
        case = self.cases[0]
        u, _ = solve_forward(self.mesh, 2., 2., .3, 2. + .1*self.mesh.X + .05*self.mesh.Y)
        state = evaluate_observation(case, u)
        np.testing.assert_allclose(state['residual'], 0., atol=1e-15)
        different, _ = solve_forward(self.mesh, 2., 2., .3, np.full(self.mesh.n_nod, 5.))
        self.assertGreater(evaluate_observation(case, different)['cost_tar'], 1e-4)

    def test_multiload_adjoint_gradient_with_rigid_fit_and_regularization(self):
        material = MaterialInfo(.3)
        initial = np.log(2.4 + .02*self.mesh.X*self.mesh.Y)
        directions = np.random.default_rng(77).normal(size=(4, self.mesh.n_nod))
        for mode in ('gradient', 'affine_detrended'):
            def evaluate(m):
                return _evaluate_inverse_state(
                    self.mesh, None, None, material, .03, np.exp(m), 1,
                    regularization_mode=mode, observation_cases=self.cases, modulus_scale=2.,
                )
            state = evaluate(initial)
            for direction in directions:
                direction = direction / np.linalg.norm(direction)
                step = 1e-6
                fd = (evaluate(initial + step*direction)['objective'] -
                      evaluate(initial - step*direction)['objective'])/(2*step)
                np.testing.assert_allclose(state['grad_m'] @ direction, fd, rtol=2e-5, atol=2e-9)

    def test_tgv_joint_gradient_with_observation_cases(self):
        n = self.mesh.n_nod
        tgv = HuberTGV(self.mesh, length_scale=2., modulus_scale=2.)
        rng = np.random.default_rng(73)
        vector = np.r_[np.log(2.4 + .02*self.mesh.X*self.mesh.Y), rng.normal(0., .01, 2*n)]
        material = MaterialInfo(.3)

        def evaluate(z):
            e = np.exp(z[:n])
            state = _evaluate_inverse_state(self.mesh, None, None, material, 0., e, 1,
                                            observation_cases=self.cases)
            value, ge, gv = tgv.value_gradient(e, z[n:])
            return state['objective'] + .01*value, np.r_[state['grad_m'] + .01*ge*e, .01*gv]

        _, gradient = evaluate(vector)
        for direction in rng.normal(size=(4, 3*n)):
            direction /= np.linalg.norm(direction)
            step = 1e-6
            fd = (evaluate(vector+step*direction)[0] - evaluate(vector-step*direction)[0])/(2*step)
            np.testing.assert_allclose(gradient@direction, fd, rtol=2e-5, atol=2e-9)

    def test_operator_preserves_affine_field_near_right_boundary(self):
        xy = np.array([[4.-1e-6, .7], [4., 2.], [0., 0.]])
        u = np.column_stack([2*self.mesh.X + self.mesh.Y,
                             -.2*self.mesh.X + 3*self.mesh.Y]).ravel()
        actual = (build_observation_operator(self.mesh, *xy.T) @ u).reshape(-1, 2)
        np.testing.assert_allclose(actual, np.column_stack([2*xy[:, 0]+xy[:, 1],
                                                          -.2*xy[:, 0]+3*xy[:, 1]]), atol=1e-12)

    def test_observation_serialization_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = save_observation_package(Path(folder), 'observed', self.cases[0].observation)
            restored = load_observation_package(paths)
        np.testing.assert_allclose(restored.H.toarray(), self.cases[0].observation.H.toarray())
        np.testing.assert_allclose(restored.weight_vector, self.cases[0].observation.weight_vector)

    def test_both_solver_paths_accept_pointwise_data_and_recover_uniform_modulus(self):
        fine = MeshInfo(4., 2., 8, 4)
        u, _ = solve_forward(fine, 2., 2., .3, np.full(fine.n_nod, 3.))
        uv = (build_observation_operator(fine, *self.xy.T) @ u).reshape(-1, 2) + [.2, -.3]
        case = make_observation_case(self.mesh, 'uniform', 2., 2., self.xy, uv)
        for mode in ('affine_detrended', 'tgv'):
            result = lbfgs_inverse_solver_scipy(
                self.mesh, observation_cases=[case], E_init=np.full(self.mesh.n_nod, 2.5),
                gamma=.01, modulus_scale=3., max_iter=500, ftol=1e-14, gtol=1e-9,
                regularization_mode=mode, tgv_options={'length_scale': 2.},
            )
            self.assertTrue(result['converged'], result['message'])
            np.testing.assert_allclose(result['E_final'], 3., rtol=.005)
            self.assertEqual(result['data_mode'], 'observation_cases')

    def test_sharp_calibrated_material_is_integrated_without_nodal_smoothing(self):
        # nu=0 and axial layers have a 1D exact compliance independent of y.
        mesh = MeshInfo(4., 2., 4, 2)
        field = partition_modulus([2., 4.], boundaries=[2.])
        u, _ = solve_forward(mesh, 2., 1., 0., field)
        # stress=1 MPa: total extension = 2/2 + 2/4 = 1.5 mm.
        np.testing.assert_allclose(u[0::2][np.isclose(mesh.X, 4.)], 1.5, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
