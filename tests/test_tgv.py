import unittest

import numpy as np

from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.material import MaterialInfo
from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.tgv import HuberTGV


class TGVTests(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshInfo(4., 2., 4, 2)
        self.tgv = HuberTGV(self.mesh, length_scale=2., delta=.02)

    def test_affine_nullspace_with_local_slopes(self):
        e = 2. + .2*self.mesh.X - .1*self.mesh.Y
        v = np.r_[np.full(self.mesh.n_nod, .4), np.full(self.mesh.n_nod, -.2)]
        value, ge, gv = self.tgv.value_gradient(e, v)
        self.assertLess(value, 1e-25)
        np.testing.assert_allclose(ge, 0, atol=1e-12)
        np.testing.assert_allclose(gv, 0, atol=1e-12)

    def test_full_joint_adjoint_gradient_in_both_huber_branches(self):
        rng = np.random.default_rng(51)
        n = self.mesh.n_nod
        material = MaterialInfo(nu=.3)
        material.update(2 + .2*self.mesh.X)
        bc = setup_boundary_conditions(self.mesh, 4., 2., 2., thickness=2.)
        measured = forward_solver(fem_assemble(self.mesh, material, bc))
        self.mesh.assemble_mass_matrix()
        for size in (.001, .3):
            vector = np.r_[np.log(2 + size*(.1*self.mesh.X + .02*self.mesh.Y**2)),
                           size*rng.normal(size=2*n)]

            def evaluate(x):
                e = np.exp(x[:n])
                state = _evaluate_inverse_state(self.mesh, bc, measured, material, 0., e, 1)
                value, ge, gv = self.tgv.value_gradient(e, x[n:])
                return state['objective'] + .01*value, np.r_[state['grad_m']+.01*ge*e, .01*gv]

            _, gradient = evaluate(vector)
            # All components and three independent directions exercise tensor shear,
            # interpolation, boundary assembly and the log-E chain rule.
            for direction in np.r_[np.eye(3*n), rng.normal(size=(3, 3*n))]:
                direction /= np.linalg.norm(direction)
                h = 1e-6
                fd = (evaluate(vector+h*direction)[0]-evaluate(vector-h*direction)[0])/(2*h)
                np.testing.assert_allclose(gradient@direction, fd, atol=2e-9, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
