"""Huber TGV2 on Q1 scalar and vector fields; no directional constraint on E.

Coordinates are xi=x/length_scale, z=E/modulus_scale, and v is a nodal
two-component slope in xi coordinates. Integrals use dxi=dA/length_scale**2.
The auxiliary field is optimized jointly, rather than replaced by a global mean.
"""

import numpy as np
from scipy.sparse import coo_matrix, vstack, hstack, csr_matrix


class HuberTGV:
    """R(z,v)=int rho(|grad z-v|)+ratio*rho(|sym grad v|) dxi.

    rho(s)=s*s/(2*delta) for s<=delta, otherwise s-delta/2.
    Continuous Q1 v is a conforming approximation; jumps in nodal E are
    represented by transitions across at least one element.
    """

    def __init__(self, mesh, length_scale=20.0, modulus_scale=1.0,
                 ratio=2.0, delta=0.02):
        if not all(np.isfinite(t) and t > 0 for t in
                   (length_scale, modulus_scale, ratio, delta)):
            raise ValueError('TGV scales, ratio and delta must be finite and positive')
        self.modulus_scale = modulus_scale
        self.length_scale = length_scale
        self.ratio = ratio
        self.delta = delta
        self.n = mesh.n_nod
        q = mesh.gauss_N.shape[0] * mesh.n_el
        rows = np.repeat(np.arange(q), 4)
        cols = np.broadcast_to(mesh.ele_nods_id[None, :, :] - 1,
                               mesh.gauss_dNdx.shape).ravel()

        def matrix(values):
            return coo_matrix((values.ravel(), (rows, cols)),
                              shape=(q, self.n)).tocsr()

        dx = matrix(mesh.gauss_dNdx * length_scale)
        dy = matrix(mesh.gauss_dNdy * length_scale)
        interp = matrix(np.broadcast_to(mesh.gauss_N[:, None, :],
                                        mesh.gauss_dNdx.shape))
        zero = csr_matrix((q, self.n))
        self.gradient = vstack([dx, dy], format='csr')
        self.interpolate = vstack([hstack([interp, zero]),
                                   hstack([zero, interp])], format='csr')
        # sqrt(2)*epsilon_xy accounts for both off-diagonal tensor entries.
        self.symmetric_gradient = vstack([
            hstack([dx, zero]), hstack([zero, dy]),
            hstack([dy / np.sqrt(2), dx / np.sqrt(2)]),
        ], format='csr')
        self.weights = (mesh.gauss_abs_det_j * mesh.gauss_w[:, None]).ravel() / length_scale**2

    def _huber(self, components):
        magnitude = np.linalg.norm(components, axis=0)
        value = np.where(magnitude <= self.delta, magnitude**2 / (2*self.delta),
                         magnitude - self.delta/2)
        derivative = components * (self.weights / np.maximum(magnitude, self.delta))
        return float(self.weights @ value), derivative.ravel()

    def value_gradient(self, modulus, slopes):
        """Return R, dR/dE [n], dR/dv [2n]; v order is [all vx, all vy]."""
        modulus = np.asarray(modulus)
        slopes = np.asarray(slopes)
        if modulus.shape != (self.n,) or slopes.shape != (2*self.n,):
            raise ValueError('Expected modulus shape (n,) and slopes shape (2*n,)')
        residual = self.gradient @ (modulus / self.modulus_scale) - self.interpolate @ slopes
        curvature = self.symmetric_gradient @ slopes
        first, df = self._huber(residual.reshape(2, -1))
        second, ds = self._huber(curvature.reshape(3, -1))
        grad_e = np.asarray(self.gradient.T @ df).ravel() / self.modulus_scale
        grad_v = np.asarray(-self.interpolate.T @ df +
                            self.ratio * (self.symmetric_gradient.T @ ds)).ravel()
        return first + self.ratio * second, grad_e, grad_v


def tgv_inverse_solver(mesh_info, bc_info, U_measured, E_init=None, gamma=1e-6,
                       E_min=0.001, max_iter=2000, ftol=1e-12, gtol=1e-8,
                       nu=0.3, tgv_options=None, max_fun=15000):
    """Minimize data cost + gamma/2 * R(E,v) in [log E,v] with L-BFGS-B.

    E positivity follows from the log parametrization; regularization is on E,
    not log E. The same forward and adjoint code as the existing solver is used.
    """
    from scipy.optimize import minimize
    from .material import MaterialInfo
    from .inverse_solver import _evaluate_inverse_state

    if mesh_info.M is None:
        mesh_info.assemble_mass_matrix()
    tgv = HuberTGV(mesh_info, **(tgv_options or {}))
    initial = (np.full(mesh_info.n_nod, E_min + 0.5) if E_init is None
               else np.asarray(E_init, dtype=float))
    if initial.shape != (mesh_info.n_nod,) or not np.all(np.isfinite(initial) & (initial > 0)):
        raise ValueError('Initial modulus must be a positive finite nodal vector')
    material = MaterialInfo(nu=nu)
    n = mesh_info.n_nod
    initial_vector = np.concatenate([np.log(initial), np.zeros(2*n)])
    history = []
    cached = {}
    evaluations = 0

    def evaluate(vector):
        nonlocal evaluations
        if 'x' in cached and np.array_equal(vector, cached['x']):
            return cached['state']
        evaluations += 1
        modulus = np.exp(vector[:n])
        state = _evaluate_inverse_state(mesh_info, bc_info, U_measured, material,
                                        0.0, modulus, evaluations)
        value, grad_e, grad_v = tgv.value_gradient(modulus, vector[n:])
        state['cost_reg'] = 0.5 * gamma * value
        state['objective'] = state['cost_tar'] + state['cost_reg']
        state['reg_value'] = value
        state['grad_reg_m'] = 0.5 * gamma * grad_e * modulus
        state['grad_m'] = state['grad_tar_m'] + state['grad_reg_m']
        state['joint_gradient'] = np.concatenate([state['grad_m'], 0.5 * gamma * grad_v])
        cached.update(x=vector.copy(), state=state)
        return state

    def objective(vector):
        state = evaluate(vector)
        return state['objective'], state['joint_gradient']

    def callback(vector):
        state = evaluate(vector)
        history.append((state['objective'], state['cost_tar'], state['cost_reg'],
                        np.linalg.norm(state['joint_gradient'])))
        if (len(history)-1) % 100 == 0:
            print(f"TGV iteration {len(history)-1}: cost={state['objective']:.8e}, "
                  f"gradient={history[-1][-1]:.3e}", flush=True)

    callback(initial_vector)
    result = minimize(objective, initial_vector, jac=True, method='L-BFGS-B', callback=callback,
                      options=dict(maxiter=max_iter, maxfun=max_fun, ftol=ftol, gtol=gtol,
                                   maxls=50, maxcor=10))
    state = evaluate(result.x)
    history = np.asarray(history)
    return dict(E_final=state['E_vec'], U_final=state['forw_U'],
                auxiliary_slopes=result.x[n:].copy(),
                cost_history=history[:, 0], cost_tar_history=history[:, 1],
                cost_reg_history=history[:, 2], grad_norm_history=history[:, 3],
                gradient_norms=history[:, 3], cond_H_history=np.ones(len(history)),
                n_iterations=int(result.nit), n_evaluations=evaluations,
                converged=bool(result.success), message=str(result.message),
                final_cost=state['objective'], residual_norm=state['residual_norm'],
                regularization_norm=float(np.sqrt(state['reg_value'])),
                regularization_mode='tgv', final_gradient=state['grad_m'],
                final_grad_tar=state['grad_tar_m'], final_grad_reg=state['grad_reg_m'],
                joint_gradient=state['joint_gradient'])
